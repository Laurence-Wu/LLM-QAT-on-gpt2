"""
QAT Training Module - Fixed memory leak version
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import gc


def train_qat(model, train_loader, val_loader, config, model_config):
    """QAT training with single precision and fake quantization - memory optimized."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Clear cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.adam_betas,
        eps=config.adam_epsilon
    )

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_iterations)

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if config.use_amp else None

    # Training metrics
    best_val_loss = float('inf')
    print(f"\nStarting QAT training ({model_config.quantization_bits}-bit)")
    print(f"Iterations: {config.num_iterations}, Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")

    # Create data iterator
    train_iter = iter(train_loader)

    # Main training loop
    progress_bar = tqdm(range(config.num_iterations), desc="QAT")

    for iteration in progress_bar:
        model.train()

        # Clear gradients only at the start of accumulation
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0

        # Accumulate gradients over multiple steps
        for step in range(config.gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)

            # Forward with AMP
            with torch.amp.autocast('cuda', enabled=config.use_amp):
                outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
                loss = outputs['loss'] / config.gradient_accumulation_steps

            # Get loss value before backward (detach to avoid keeping graph)
            loss_value = loss.detach().item()
            total_loss += loss_value

            # Backward - accumulates gradients
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # CRITICAL FIX: Clean up intermediate tensors immediately
            del outputs, loss, input_ids
            if attention_mask is not None:
                del attention_mask
            # Force Python to release the batch dict
            batch.clear()
            del batch

        # Optimizer step - after all gradient accumulation
        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

        scheduler.step()

        # CRITICAL: Clear gradients after optimizer step to free memory
        optimizer.zero_grad(set_to_none=True)

        # AGGRESSIVE memory cleanup - every 10 iterations
        if iteration % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Update progress bar with memory info
        if iteration % 20 == 0:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                progress_bar.set_postfix({
                    'loss': f'{total_loss:.4f}',
                    'gpu_alloc': f'{allocated:.1f}GB',
                    'gpu_res': f'{reserved:.1f}GB'
                })

        # Evaluation with memory cleanup
        if iteration % config.eval_interval == 0 and iteration > 0:
            val_loss = evaluate(model, val_loader, device, config.use_amp)
            print(f"\n[Iter {iteration}] Train: {total_loss:.4f}, Val: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Force cleanup after eval
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return model


def evaluate(model, val_loader, device, use_amp):
    """Quick evaluation on validation set - memory optimized."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if num_batches >= 5:  # Reduce eval batches to save memory
                break

            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
                loss = outputs['loss']

            total_loss += loss.item()
            num_batches += 1

            # Clean up immediately
            del outputs, loss, input_ids
            if attention_mask is not None:
                del attention_mask
            batch.clear()
            del batch

    # Clear cache after eval
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return total_loss / max(num_batches, 1)
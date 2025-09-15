"""
QAT Training Module - Fixed memory leak version
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import gc
import json
import time


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
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_iterations,  # Maximum number of iterations
        eta_min=0  # Minimum learning rate
    )

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if config.use_amp else None

    # Training metrics
    best_val_loss = float('inf')
    best_iteration = 0
    print(f"\nStarting QAT training ({model_config.quantization_bits}-bit)")
    print(f"Iterations: {config.num_iterations}, Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")

    # Initialize training statistics dictionary
    training_stats = {
        'iteration_losses': [],
        'validation_losses': [],
        'bit_width_usage': [],
        'learning_rates': [],
        'memory_usage': [],
        'best_val_loss': None,
        'best_iteration': None
    }

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

            # Detach loss immediately to prevent graph retention
            loss_value = loss.detach().item()
            total_loss += loss_value

            # Backward with retain_graph=False (default)
            if scaler:
                scaler.scale(loss).backward(retain_graph=False)
            else:
                loss.backward(retain_graph=False)

            # Clean up intermediate tensors immediately
            del outputs, loss, input_ids
            if attention_mask is not None:
                del attention_mask
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

        # Clear gradients after optimizer step to free memory
        optimizer.zero_grad(set_to_none=True)

        # Store training statistics
        training_stats['iteration_losses'].append(total_loss)
        training_stats['bit_width_usage'].append(model_config.quantization_bits)
        training_stats['learning_rates'].append(optimizer.param_groups[0]['lr'])
        if torch.cuda.is_available():
            training_stats['memory_usage'].append(torch.cuda.memory_allocated() / 1024**2)  # MB
        else:
            training_stats['memory_usage'].append(0)

        # Periodic memory cleanup to prevent accumulation
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

            # Store validation loss
            training_stats['validation_losses'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iteration = iteration
                training_stats['best_val_loss'] = best_val_loss
                training_stats['best_iteration'] = best_iteration

            # Force cleanup after eval
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    # Save training statistics to JSON file
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    stats_path = f'qat_training_stats_{timestamp}.json'

    with open(stats_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        stats_to_save = {}
        for key, value in training_stats.items():
            if hasattr(value, 'tolist'):
                stats_to_save[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], 'tolist'):
                stats_to_save[key] = [v.tolist() if hasattr(v, 'tolist') else v for v in value]
            else:
                stats_to_save[key] = value
        json.dump(stats_to_save, f, indent=2)

    print(f"Training statistics saved to {stats_path}")

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
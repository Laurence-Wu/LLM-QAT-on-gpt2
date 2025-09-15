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
import random


def get_next_bitwidth(iteration, model_config):
    """Determine next bit-width based on switching strategy."""
    if model_config.switch_strategy == 'cyclic':
        # Cycle through bit-widths
        cycle_position = (iteration // model_config.switch_interval) % len(model_config.bit_widths)
        return model_config.bit_widths[cycle_position]
    elif model_config.switch_strategy == 'random':
        # Random selection
        return random.choice(model_config.bit_widths)
    elif model_config.switch_strategy == 'curriculum':
        # Curriculum learning - start high, go low
        schedule_idx = min(iteration // 50, len(model_config.curriculum_schedule) - 1)
        return model_config.curriculum_schedule[schedule_idx]
    else:
        return 8  # Default


def train_qat(model, train_loader, val_loader, config, model_config):
    """QAT training with switchable precision support."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Check if using switchable model
    is_switchable = hasattr(model, 'set_precision') and model_config.use_switchable

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

    # cosine Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_iterations,  # Maximum number of iterations
        eta_min=0  # Minimum learning rate
    )

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if config.use_amp else None

    # Training metrics
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
        'losses_per_bit': {4: [], 8: [], 16: []} if is_switchable else {}
    }

    # Create data iterator
    train_iter = iter(train_loader)

    # Main training loop
    progress_bar = tqdm(range(config.num_iterations), desc="QAT")

    for iteration in progress_bar:
        model.train()

        # Switch bit-width if using switchable model
        current_bits = model_config.quantization_bits  # Default
        if is_switchable:
            current_bits = get_next_bitwidth(iteration, model_config)
            model.set_precision(current_bits)

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
        training_stats['bit_width_usage'].append(current_bits)  # Track actual bit-width used
        training_stats['learning_rates'].append(optimizer.param_groups[0]['lr'])
        if torch.cuda.is_available():
            training_stats['memory_usage'].append(torch.cuda.memory_allocated() / 1024**2)  # MB
        else:
            training_stats['memory_usage'].append(0)

        # Track loss per bit-width if switchable
        if is_switchable:
            training_stats['losses_per_bit'][current_bits].append(total_loss)

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
                postfix_dict = {
                    'loss': f'{total_loss:.4f}',
                    'gpu_alloc': f'{allocated:.1f}GB',
                    'gpu_res': f'{reserved:.1f}GB'
                }
                if is_switchable:
                    postfix_dict['bits'] = current_bits
                progress_bar.set_postfix(postfix_dict)

        # Evaluation with memory cleanup
        if iteration % config.eval_interval == 0 and iteration > 0:
            val_loss = evaluate(model, val_loader, device, config.use_amp)
            print(f"\n[Iter {iteration}] Train: {total_loss:.4f}, Val: {val_loss:.4f}")

            # Store validation loss
            training_stats['validation_losses'].append(val_loss)

            # Force cleanup after eval
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print(f"\nTraining complete.")

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

        # Add configuration information
        stats_to_save['model_config'] = {
            'model_name': model_config.model_name,
            'output_model_path': model_config.output_model_path,
            'quantization_bits': model_config.quantization_bits,
            'use_gradient_checkpointing': model_config.use_gradient_checkpointing
        }

        stats_to_save['training_config'] = {
            'train_split': config.train_split,
            'val_split': config.val_split,
            'batch_size': config.batch_size,
            'max_seq_length': config.max_seq_length,
            'doc_stride': config.doc_stride,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'adam_epsilon': config.adam_epsilon,
            'adam_betas': config.adam_betas,
            'max_grad_norm': config.max_grad_norm,
            'num_iterations': config.num_iterations,
            'gradient_accumulation_steps': config.gradient_accumulation_steps,
            'eval_interval': config.eval_interval,
            'save_interval': config.save_interval,
            'use_amp': config.use_amp,
            'num_workers': config.num_workers
        }

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
"""
Main training script for Cyclic Precision Training (CPT).
Implements the key CPT training loop with precision cycling within each step.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
import numpy as np
from tqdm import tqdm
import time
import os
import argparse
from typing import Dict, Optional

from config_cpt import get_config
from cpt_model import CPTModel
from cyclic_scheduler import CyclicPrecisionScheduler, PrecisionRangeTest
from deploy import save_cpt_checkpoint, save_final_models
from dataset import WikiTextDataset


def train_cycle_with_cyclic_precision(
    model: CPTModel,
    batch: Dict[str, torch.Tensor],
    optimizer: optim.Optimizer,
    precision_scheduler: CyclicPrecisionScheduler,
    lr_scheduler: optim.lr_scheduler._LRScheduler,
    device: str
) -> Dict[str, float]:
    """
    Single training step with cyclic precision.
    This is the KEY function that cycles through all precisions within one step.
    """
    model.train()

    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)

    # Get all precisions for this cycle
    cycle_precisions = precision_scheduler.get_cycle_precisions()

    total_loss = 0
    losses_per_precision = {}

    # Zero gradients once
    optimizer.zero_grad()

    # CRITICAL: Cycle through ALL precisions within this single training step
    for i, precision in enumerate(cycle_precisions):
        # Set model to current precision
        model.set_precision(precision)

        # Forward pass with current precision
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Average loss over the cycle
        loss_scaled = loss / len(cycle_precisions)

        # Backward pass - accumulate gradients
        loss_scaled.backward()

        # Track losses
        total_loss += loss.item()
        losses_per_precision[f'{precision}bit'] = loss.item()

    # Single optimizer step after full cycle
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Step learning rate scheduler for EACH precision in the cycle
    # This ensures smooth LR decay across all precision iterations
    for _ in cycle_precisions:
        lr_scheduler.step()

    # Advance precision scheduler position
    precision_scheduler.global_cycle += 1

    return {
        'total_loss': total_loss / len(cycle_precisions),
        'losses_per_precision': losses_per_precision,
        'cycle_info': precision_scheduler.get_current_cycle_info()
    }


def evaluate(model: CPTModel, dataloader: DataLoader, device: str, precision: int = 8) -> Dict[str, float]:
    """Evaluate model at specific precision."""
    model.eval()
    model.set_precision(precision)

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating at {precision}-bit"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return {
        'loss': avg_loss,
        'perplexity': perplexity
    }


def load_pretrained_weights(model: CPTModel, device: str):
    """Load pretrained GPT-2 weights into CPT model and return GPT-2 config."""
    print("Loading pretrained GPT-2 weights...")

    # Load GPT-2 model and config
    gpt2 = GPT2Model.from_pretrained('gpt2')
    gpt2_config = GPT2Config.from_pretrained('gpt2')
    gpt2_state = gpt2.state_dict()
    d_model = gpt2_config.n_embd

    # Direct weight mapping
    weight_mapping = {
        # Embeddings
        'wte.weight': 'wte.weight',
        'wpe.weight': 'wpe.weight',
        # Final layer norm
        'ln_f.weight': 'ln_f.weight',
        'ln_f.bias': 'ln_f.bias'
    }

    # Add transformer block mappings
    for i in range(len(model.h)):
        # Layer norms
        weight_mapping.update({
            f'h.{i}.ln_1.weight': f'h.{i}.ln_1.weight',
            f'h.{i}.ln_1.bias': f'h.{i}.ln_1.bias',
            f'h.{i}.ln_2.weight': f'h.{i}.ln_2.weight',
            f'h.{i}.ln_2.bias': f'h.{i}.ln_2.bias',
            # Output projection
            f'h.{i}.attn.c_proj.weight': f'h.{i}.attn.out_proj.linear.weight',
            # MLP weights
            f'h.{i}.mlp.c_fc.weight': f'h.{i}.mlp.fc_in.linear.weight',
            f'h.{i}.mlp.c_proj.weight': f'h.{i}.mlp.fc_out.linear.weight'
        })

    # Load weights
    loaded_count = 0
    for src_key, dst_key in weight_mapping.items():
        if src_key in gpt2_state:
            # Get source and destination tensors
            src_tensor = gpt2_state[src_key].to(device)
            dst_attrs = dst_key.split('.')

            # Navigate to the destination
            target = model
            for attr in dst_attrs[:-1]:
                if attr.isdigit():
                    target = target[int(attr)]
                else:
                    target = getattr(target, attr)

            # Set the weight
            setattr(target, dst_attrs[-1], nn.Parameter(src_tensor))
            loaded_count += 1

    # Handle QKV weights separately (need splitting)
    for i in range(len(model.h)):
        qkv_key = f'h.{i}.attn.c_attn.weight'
        if qkv_key in gpt2_state:
            qkv_weight = gpt2_state[qkv_key].to(device)
            model.h[i].attn.q_proj.linear.weight.data = qkv_weight[:d_model, :]
            model.h[i].attn.k_proj.linear.weight.data = qkv_weight[d_model:2*d_model, :]
            model.h[i].attn.v_proj.linear.weight.data = qkv_weight[2*d_model:, :]
            loaded_count += 3

    print(f"Loaded {loaded_count} weight tensors from GPT-2")
    return gpt2_config


def main(args):
    """Main training function."""
    # Load configuration
    config = get_config()
    training_config = config['training']
    model_config = config['model']
    cpt_config = config['cpt']

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # datasets
    print("Loading datasets...")
    train_dataset = WikiTextDataset(
        training_config.train_split,
        tokenizer,
        training_config.max_seq_length,
        training_config.doc_stride
    )
    val_dataset = WikiTextDataset(
        training_config.val_split,
        tokenizer,
        training_config.max_seq_length,
        training_config.doc_stride
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers
    )

    # model
    print("Creating CPT model...")
    model = CPTModel(config).to(device)

    # Load pretrained weights if specified
    if args.load_pretrained:
        gpt2_config = load_pretrained_weights(model, device)
        # Optionally update model config with GPT-2 config values
        # This ensures compatibility with pretrained weights
        print(f"GPT-2 config: hidden_size={gpt2_config.n_embd}, num_layers={gpt2_config.n_layer}, num_heads={gpt2_config.n_head}")

    # Create cyclic precision scheduler
    precision_scheduler = CyclicPrecisionScheduler(
        bit_widths=model_config.bit_widths,
        schedule_type=cpt_config.schedule_type,
        cycle_length=cpt_config.cycle_length
    )

    # Run Precision Range Test (only during training, not evaluation)
    if not args.eval_only:
        print("Running Precision Range Test...")
        prt = PrecisionRangeTest(
            model,
            start_bits=cpt_config.prt_start_bits,
            max_bits=max(model_config.bit_widths),  # Use the maximum bit width from config
            threshold=cpt_config.prt_threshold,
            test_iterations=cpt_config.prt_iterations,
            target_bits=training_config.target_bits
        )
        lower_bound, upper_bound = prt.find_bounds(train_loader, nn.CrossEntropyLoss())
        print(f"PRT Results: Lower bound = {lower_bound}-bit, Upper bound = {upper_bound}-bit")
        # Update precision scheduler with PRT results
        precision_scheduler.min_bits = lower_bound
        precision_scheduler.max_bits = upper_bound

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=training_config.adam_betas,
        eps=training_config.adam_epsilon,
        weight_decay=training_config.weight_decay
    )

    # Calculate total cycles for learning rate scheduler
    # Total cycles = num_epochs * steps_per_epoch * cycle_length
    # This ensures LR updates for every precision change
    steps_per_epoch = len(train_loader)
    cycle_length = cpt_config.cycle_length
    total_lr_cycles = training_config.num_epochs * steps_per_epoch * cycle_length

    # Create cosine learning rate scheduler
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_lr_cycles,
        eta_min=1e-6  # Minimum learning rate
    )
    print(f"Created LR scheduler with {total_lr_cycles:,} total cycles")
    print(f"  ({training_config.num_epochs} epochs * {steps_per_epoch} batches * {cycle_length} precisions)")

    # Training loop
    print("Starting CPT training...")
    global_cycle = 0
    best_val_loss = float('inf')

    for epoch in range(training_config.num_epochs):
        epoch_losses = []
        epoch_start_time = time.time()

        # Training
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_config.num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # CRITICAL: Train with cyclic precision
            cycle_results = train_cycle_with_cyclic_precision(
                model, batch, optimizer, precision_scheduler, lr_scheduler, device
            )

            epoch_losses.append(cycle_results['total_loss'])
            global_cycle += 1

            # Update progress bar
            cycle_info = cycle_results['cycle_info']
            current_lr = lr_scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{cycle_results['total_loss']:.4f}",
                'precision': f"{cycle_info['current_precision']}bit",
                'cycle': cycle_info['cycle_count'],
                'lr': f"{current_lr:.2e}"
            })

            # Log detailed info periodically
            if global_cycle % training_config.log_interval == 0:
                precision_losses = cycle_results['losses_per_precision']
                print(f"\nCycle {global_cycle} - Losses per precision: {precision_losses}")

            # Clear cache periodically
            if global_cycle % training_config.empty_cache_interval == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Epoch statistics
        avg_epoch_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.2f}s")

        # Validation
        if (epoch + 1) % training_config.eval_interval == 0:
            print("Running validation...")
            # Evaluate at different precisions
            for precision in model_config.bit_widths:
                val_results = evaluate(model, val_loader, device, precision)
                print(f"Validation at {precision}-bit - Loss: {val_results['loss']:.4f}, "
                      f"Perplexity: {val_results['perplexity']:.2f}")

                # Track best model at 8-bit
                if precision == 8 and val_results['loss'] < best_val_loss:
                    best_val_loss = val_results['loss']
                    print(f"New best validation loss: {best_val_loss:.4f}")
                    # Save best checkpoint
                    save_cpt_checkpoint(
                        model, optimizer, lr_scheduler, epoch, global_cycle,
                        best_val_loss, config, 'checkpoints/best_model.pth'
                    )

        # Save checkpoint
        if (epoch + 1) % training_config.save_interval == 0:
            save_cpt_checkpoint(
                model, optimizer, lr_scheduler, epoch, global_cycle,
                avg_epoch_loss, config, f'checkpoints/checkpoint_epoch{epoch+1}.pth'
            )

    # Save final models at all precisions
    print("Saving final models...")
    save_final_models(model, config, 'final_models')

    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cyclic Precision Training')
    parser.add_argument('--load_pretrained', action='store_true',
                        help='Load pretrained GPT-2 weights')
    parser.add_argument('--skip_prt', action='store_true',
                        help='Skip Precision Range Test')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate the model without training')

    args = parser.parse_args()
    main(args)
"""
Main training script for Cyclic Precision Training (CPT).
Implements the key CPT training loop with precision cycling within each step.
"""

import os
import sys

# Fix import path to ensure we use part2 modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# Remove parent directory from path if it exists to avoid importing from part1
if parent_dir in sys.path:
    sys.path.remove(parent_dir)
# Insert current directory at the beginning to prioritize local imports
sys.path.insert(0, current_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
import numpy as np
from tqdm import tqdm
import time
import gc
import argparse
from typing import Dict, Optional

from config_cpt import get_config
from cpt_model import CPTModel
from cyclic_scheduler import CyclicPrecisionScheduler, PrecisionRangeTest
from calibration import CalibrationManager
import deploy as cpt_deploy
import dataset as cpt_dataset


def train_epoch_with_cpt(
    model: CPTModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    precision: int,
    device: str,
    max_grad_norm: float = 1.0
) -> float:
    """Train one epoch at specified precision (CPT paper approach)."""
    model.train()
    model.set_precision(precision)

    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


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


def load_pretrained_weights(model, model_config):
    """Load pretrained GPT-2 weights into CPT model and freeze all except LoRA adapters."""
    print("Loading pretrained GPT-2 weights...")

    import gc
    from transformers import GPT2LMHeadModel
    pretrained = GPT2LMHeadModel.from_pretrained('gpt2')

    # Load embeddings and freeze them
    model.wte.weight.data = pretrained.transformer.wte.weight.data.clone()
    model.wte.weight.requires_grad = False
    model.wpe.weight.data = pretrained.transformer.wpe.weight.data.clone()
    model.wpe.weight.requires_grad = False

    # Load layer-specific weights for each transformer block
    for i in range(len(pretrained.transformer.h)):
        # Layer normalization weights - same for all bit widths, freeze them
        for bit_width in model.h[i].bit_widths:
            bit_key = str(bit_width)
            model.h[i].ln_1.weights[bit_key].data = pretrained.transformer.h[i].ln_1.weight.data.clone()
            model.h[i].ln_1.biases[bit_key].data = pretrained.transformer.h[i].ln_1.bias.data.clone()
            model.h[i].ln_1.weights[bit_key].requires_grad = False
            model.h[i].ln_1.biases[bit_key].requires_grad = False

            model.h[i].ln_2.weights[bit_key].data = pretrained.transformer.h[i].ln_2.weight.data.clone()
            model.h[i].ln_2.biases[bit_key].data = pretrained.transformer.h[i].ln_2.bias.data.clone()
            model.h[i].ln_2.weights[bit_key].requires_grad = False
            model.h[i].ln_2.biases[bit_key].requires_grad = False

        # Attention weights - load combined QKV projection directly (like part1)
        model.h[i].attn.c_attn.linear.weight.data = pretrained.transformer.h[i].attn.c_attn.weight.data.t().contiguous()
        model.h[i].attn.c_attn.linear.bias.data = pretrained.transformer.h[i].attn.c_attn.bias.data.clone()
        model.h[i].attn.c_attn.linear.weight.requires_grad = False
        model.h[i].attn.c_attn.linear.bias.requires_grad = False

        # Output projection - also needs transpose and freeze
        model.h[i].attn.c_proj.linear.weight.data = pretrained.transformer.h[i].attn.c_proj.weight.data.t().contiguous()
        model.h[i].attn.c_proj.linear.bias.data = pretrained.transformer.h[i].attn.c_proj.bias.data.clone()
        model.h[i].attn.c_proj.linear.weight.requires_grad = False
        model.h[i].attn.c_proj.linear.bias.requires_grad = False

        # MLP weights - also need transpose and freeze
        model.h[i].mlp['fc_in'].linear.weight.data = pretrained.transformer.h[i].mlp.c_fc.weight.data.t().contiguous()
        model.h[i].mlp['fc_in'].linear.bias.data = pretrained.transformer.h[i].mlp.c_fc.bias.data.clone()
        model.h[i].mlp['fc_in'].linear.weight.requires_grad = False
        model.h[i].mlp['fc_in'].linear.bias.requires_grad = False

        model.h[i].mlp['fc_out'].linear.weight.data = pretrained.transformer.h[i].mlp.c_proj.weight.data.t().contiguous()
        model.h[i].mlp['fc_out'].linear.bias.data = pretrained.transformer.h[i].mlp.c_proj.bias.data.clone()
        model.h[i].mlp['fc_out'].linear.weight.requires_grad = False
        model.h[i].mlp['fc_out'].linear.bias.requires_grad = False

    # Load final layer norm and freeze
    for bit_width in model_config.bit_widths:
        bit_key = str(bit_width)
        model.ln_f.weights[bit_key].data = pretrained.transformer.ln_f.weight.data.clone()
        model.ln_f.biases[bit_key].data = pretrained.transformer.ln_f.bias.data.clone()
        model.ln_f.weights[bit_key].requires_grad = False
        model.ln_f.biases[bit_key].requires_grad = False

    # Load language modeling head and freeze
    model.lm_head.linear.weight.data = pretrained.lm_head.weight.data.clone()
    model.lm_head.linear.weight.requires_grad = False

    # Now enable only LoRA adapter parameters
    lora_count = 0
    target_modules = ['c_attn', 'c_proj', 'fc_in', 'fc_out', 'lm_head']

    for name, module in model.named_modules():
        # Check if this is a module with LoRA adapters
        if not any(target in name for target in target_modules):
            continue
        if not hasattr(module, 'lora_adapters'):
            continue

        # Enable gradients for LoRA adapters across all bit widths
        for bit_key in module.lora_adapters.keys():
            lora_layer = module.lora_adapters[bit_key]
            if hasattr(lora_layer, 'lora_A'):
                lora_layer.lora_A.requires_grad = True
                lora_layer.lora_B.requires_grad = True
                lora_count += 1

    print(f"Enabled {lora_count} LoRA adapter pairs for training")

    del pretrained
    torch.cuda.empty_cache()
    gc.collect()

    # Print summary of parameter counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + frozen_params

    print("Pretrained weights loaded and frozen successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    print(f"  Trainable (LoRA) parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")


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
    train_dataset = cpt_dataset.WikiTextDataset(
        training_config.train_split,
        tokenizer,
        training_config.max_seq_length,
        training_config.doc_stride
    )
    val_dataset = cpt_dataset.WikiTextDataset(
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
    model = CPTModel(config)  # Create model on CPU first

    # Load pretrained weights if specified (while model is on CPU)
    load_pretrained_weights(model, model_config)

    # Move model to device AFTER loading pretrained weights (like part1)
    model = model.to(device)
    print(f"Model moved to {device}")

    # Create calibration manager and calibrate all precisions
    print("\nInitializing calibration manager...")
    calib_mgr = CalibrationManager(model, train_loader, device)
    student_bits = [b for b in model_config.bit_widths if b < 32]
    print("Calibrating all precision levels...")
    calib_mgr.calibrate_all_precisions(student_bits)

    # Create cyclic precision scheduler
    precision_scheduler = CyclicPrecisionScheduler(
        bit_widths=model_config.bit_widths,
        schedule_type=cpt_config.schedule_type,
        total_epochs=training_config.num_epochs,
        total_cycles=cpt_config.total_cycles
    )
    print(f"CPT: {cpt_config.total_cycles} cycles over {training_config.num_epochs} epochs")
    print(f"Cycle length: {precision_scheduler.cycle_length_epochs} epochs")

    # Run Precision Range Test (only during training, not evaluation)
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

    # Create cosine learning rate scheduler (steps once per epoch)
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=training_config.num_epochs,
        eta_min=1e-6
    )
    print(f"LR scheduler: T_max={training_config.num_epochs} epochs")

    # Training loop
    print("Starting CPT training...")
    best_val_loss = float('inf')

    for epoch in range(training_config.num_epochs):
        epoch_start_time = time.time()

        # Calculate precision for THIS epoch using CPT
        current_precision = precision_scheduler.get_precision_for_epoch(epoch)
        print(f"\nEpoch {epoch+1}/{training_config.num_epochs} - Precision: {current_precision}-bit")

        # Train one epoch at this precision
        avg_epoch_loss = train_epoch_with_cpt(
            model, train_loader, optimizer, current_precision,
            device, training_config.max_grad_norm
        )

        # Step LR scheduler once per epoch
        lr_scheduler.step()

        epoch_time = time.time() - epoch_start_time
        current_lr = lr_scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} - Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.2e}, Time: {epoch_time:.2f}s")

        # Clear cache periodically
        if (epoch + 1) % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Validation
        if (epoch + 1) % training_config.eval_interval == 0:
            print("Running validation...")
            for precision in [current_precision, training_config.target_bits]:
                if precision in model_config.bit_widths:
                    calib_mgr.ensure_calibrated(precision)
                    val_results = evaluate(model, val_loader, device, precision)
                    print(f"  {precision}-bit - Loss: {val_results['loss']:.4f}, PPL: {val_results['perplexity']:.2f}")

                    if precision == training_config.target_bits and val_results['loss'] < best_val_loss:
                        best_val_loss = val_results['loss']
                        print(f"  New best: {best_val_loss:.4f}")

    print("\n" + "="*60)
    print("Training completed! Saving target model...")
    target_bits = training_config.target_bits
    print(f"Target precision: {target_bits}-bit")

    saved_path = cpt_deploy.save_target_model(model, config, target_bits, 'final_models')
    if saved_path:
        print(f"✅ Saved: {saved_path}")
    else:
        print("❌ Save failed")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cyclic Precision Training')
    parser.add_argument('--load_pretrained', action='store_true',
                        help='Load pretrained GPT-2 weights')
    parser.add_argument('--skip_prt', action='store_true',
                        help='Skip Precision Range Test')
    # No checkpoint functionality - CPT only saves final target model
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate the model without training')

    args = parser.parse_args()
    main(args)
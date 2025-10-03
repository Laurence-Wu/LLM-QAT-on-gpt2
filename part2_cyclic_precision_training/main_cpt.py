import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir in sys.path:
    sys.path.remove(parent_dir)
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
    max_grad_norm: float = 1.0,
    lr_scheduler = None
) -> float:
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
        lr_scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model: CPTModel, dataloader: DataLoader, device: str, precision: int = 8) -> Dict[str, float]:
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
    print("Loading pretrained GPT-2 weights...")

    from transformers import GPT2LMHeadModel
    pretrained = GPT2LMHeadModel.from_pretrained('gpt2')

    model.wte.weight.data = pretrained.transformer.wte.weight.data.clone()
    model.wte.weight.requires_grad = False
    model.wpe.weight.data = pretrained.transformer.wpe.weight.data.clone()
    model.wpe.weight.requires_grad = False

    for i in range(len(pretrained.transformer.h)):
        model.h[i].ln_1.weight.data = pretrained.transformer.h[i].ln_1.weight.data.clone()
        model.h[i].ln_1.bias.data = pretrained.transformer.h[i].ln_1.bias.data.clone()
        model.h[i].ln_1.weight.requires_grad = True
        model.h[i].ln_1.bias.requires_grad = True

        model.h[i].ln_2.weight.data = pretrained.transformer.h[i].ln_2.weight.data.clone()
        model.h[i].ln_2.bias.data = pretrained.transformer.h[i].ln_2.bias.data.clone()
        model.h[i].ln_2.weight.requires_grad = True
        model.h[i].ln_2.bias.requires_grad = True

        model.h[i].attn.c_attn.linear.weight.data = pretrained.transformer.h[i].attn.c_attn.weight.data.t().contiguous()
        model.h[i].attn.c_attn.linear.bias.data = pretrained.transformer.h[i].attn.c_attn.bias.data.clone()
        model.h[i].attn.c_attn.linear.weight.requires_grad = False
        model.h[i].attn.c_attn.linear.bias.requires_grad = False

        model.h[i].attn.c_proj.linear.weight.data = pretrained.transformer.h[i].attn.c_proj.weight.data.t().contiguous()
        model.h[i].attn.c_proj.linear.bias.data = pretrained.transformer.h[i].attn.c_proj.bias.data.clone()
        model.h[i].attn.c_proj.linear.weight.requires_grad = False
        model.h[i].attn.c_proj.linear.bias.requires_grad = False

        model.h[i].mlp['fc_in'].linear.weight.data = pretrained.transformer.h[i].mlp.c_fc.weight.data.t().contiguous()
        model.h[i].mlp['fc_in'].linear.bias.data = pretrained.transformer.h[i].mlp.c_fc.bias.data.clone()
        model.h[i].mlp['fc_in'].linear.weight.requires_grad = False
        model.h[i].mlp['fc_in'].linear.bias.requires_grad = False

        model.h[i].mlp['fc_out'].linear.weight.data = pretrained.transformer.h[i].mlp.c_proj.weight.data.t().contiguous()
        model.h[i].mlp['fc_out'].linear.bias.data = pretrained.transformer.h[i].mlp.c_proj.bias.data.clone()
        model.h[i].mlp['fc_out'].linear.weight.requires_grad = False
        model.h[i].mlp['fc_out'].linear.bias.requires_grad = False

    model.ln_f.weight.data = pretrained.transformer.ln_f.weight.data.clone()
    model.ln_f.bias.data = pretrained.transformer.ln_f.bias.data.clone()
    model.ln_f.weight.requires_grad = True
    model.ln_f.bias.requires_grad = True

    model.lm_head.linear.weight.data = pretrained.lm_head.weight.data.clone()
    model.lm_head.linear.weight.requires_grad = False

    lora_count = 0
    from cpt_model import CPTLinear

    for name, module in model.named_modules():
        if isinstance(module, CPTLinear):
            if hasattr(module, 'shared_lora') and module.shared_lora is not None:
                if hasattr(module.shared_lora, 'lora_A'):
                    module.shared_lora.lora_A.requires_grad = True
                    module.shared_lora.lora_B.requires_grad = True
                    lora_count += 1

    print(f"Enabled {lora_count} shared LoRA adapter pairs for training")

    del pretrained
    torch.cuda.empty_cache()
    gc.collect()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + frozen_params

    print("Pretrained weights loaded and frozen successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    print(f"  Trainable (LoRA) parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")


def main(args):
    config = get_config()
    training_config = config['training']
    model_config = config['model']
    cpt_config = config['cpt']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

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

    print("Creating CPT model...")
    model = CPTModel(config)

    load_pretrained_weights(model, model_config)

    model = model.to(device)
    print(f"Model moved to {device}")

    print("\nInitializing calibration manager...")
    calib_mgr = CalibrationManager(model, train_loader, device)

    if not calib_mgr.gradient_calibrated:
        calib_mgr.calibrate_gradient_quantizers()
        calib_mgr.gradient_calibrated = True

    precision_scheduler = CyclicPrecisionScheduler(
        bit_widths=model_config.bit_widths,
        schedule_type=cpt_config.schedule_type,
        total_epochs=training_config.num_epochs,
        total_cycles=cpt_config.total_cycles
    )
    print(f"CPT: {cpt_config.total_cycles} cycles over {training_config.num_epochs} epochs")
    print(f"Cycle length: {precision_scheduler.epochs_per_cycle} epochs")

    print("Running Precision Range Test...")
    prt = PrecisionRangeTest(
        model,
        start_bits=cpt_config.prt_start_bits,
        max_bits=max(model_config.bit_widths),
        threshold=cpt_config.prt_threshold,
        test_iterations=cpt_config.prt_iterations,
        target_bits=training_config.target_bits
    )
    lower_bound, upper_bound = prt.find_bounds(train_loader, nn.CrossEntropyLoss())
    print(f"PRT Results: Lower bound = {lower_bound}-bit, Upper bound = {upper_bound}-bit")
    precision_scheduler.min_bits = lower_bound
    precision_scheduler.max_bits = upper_bound

    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=training_config.adam_betas,
        eps=training_config.adam_epsilon,
        weight_decay=training_config.weight_decay
    )

    total_steps = training_config.num_epochs * len(train_loader)

    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6
    )
    print(f"LR scheduler: T_max={total_steps} steps ({training_config.num_epochs} epochs, {len(train_loader)} batches/epoch)")

    # CRITICAL: Pre-calibrate ALL precisions that will be used during CPT
    print("\n" + "="*70)
    print("Pre-calibrating all precisions in CPT range...")
    print(f"Range: [{lower_bound}, {upper_bound}] bits")

    precisions_to_calibrate = [b for b in model_config.bit_widths if lower_bound <= b <= upper_bound]
    print(f"Precisions to calibrate: {precisions_to_calibrate}")

    for bits in precisions_to_calibrate:
        # Calibrate weight and input quantizers
        if bits not in calib_mgr.calibrated_bits:
            print(f"  Calibrating {bits}-bit weight/input quantizers...")
            calib_mgr.ensure_calibrated(bits)

        # Calibrate LoRA weight quantizers
        if bits not in calib_mgr.lora_calibrated_bits:
            print(f"  Calibrating {bits}-bit LoRA weight quantizers...")
            calib_mgr.calibrate_lora_weight_quantizers([bits])
            calib_mgr.lora_calibrated_bits.add(bits)

    print(f"\n✓ Pre-calibration complete!")
    print(f"  Weight/input calibrated: {sorted(calib_mgr.calibrated_bits)}")
    print(f"  LoRA weight calibrated: {sorted(calib_mgr.lora_calibrated_bits)}")
    print("="*70)

    print("\nStarting CPT training with pre-calibrated quantizers...")
    best_val_loss = float('inf')
    prev_loss = None
    loss_history = []

    for epoch in range(training_config.num_epochs):
        epoch_start_time = time.time()

        current_precision = precision_scheduler.get_precision_for_epoch(epoch)
        cycle_num = int(epoch / precision_scheduler.epochs_per_cycle)
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{training_config.num_epochs} - Precision: {current_precision}-bit (Cycle {cycle_num+1}/{cpt_config.total_cycles})")
        print(f"{'='*70}")

        calib_mgr.ensure_calibrated(current_precision)

        if current_precision not in calib_mgr.lora_calibrated_bits:
            print(f"  Calibrating LoRA weight quantizers for {current_precision}-bit...")
            calib_mgr.calibrate_lora_weight_quantizers([current_precision])
            calib_mgr.lora_calibrated_bits.add(current_precision)

        model.set_precision(current_precision)

        avg_epoch_loss = train_epoch_with_cpt(
            model, train_loader, optimizer, current_precision,
            device, training_config.max_grad_norm, lr_scheduler
        )

        epoch_time = time.time() - epoch_start_time
        current_lr = lr_scheduler.get_last_lr()[0]

        loss_history.append(avg_epoch_loss)
        loss_change = avg_epoch_loss - prev_loss if prev_loss is not None else 0.0
        prev_loss = avg_epoch_loss

        print(f"  Loss: {avg_epoch_loss:.4f} (Δ={loss_change:+.4f}), LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")

        if loss_change > 0 and epoch > 0:
            print(f"  ⚠️ Loss increased by {loss_change:.4f}")

        if len(loss_history) >= 3:
            recent_losses = loss_history[-3:]
            if all(recent_losses[i] < recent_losses[i+1] for i in range(len(recent_losses)-1)):
                print(f"  ⚠️ WARNING: Loss increasing for 3 consecutive epochs")

        if (epoch + 1) % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    print("\n" + "="*70)
    print("Training completed!")
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print(f"  Final Training Loss: {loss_history[-1]:.4f}")
    if len(loss_history) > 1:
        loss_improvement = loss_history[0] - loss_history[-1]
        print(f"  Loss Improvement: {loss_improvement:.4f}")

    print(f"\n{'='*70}")
    print("Preparing to save target model...")
    target_bits = training_config.target_bits
    print(f"Target precision: {target_bits}-bit")

    # Ensure all quantizers are calibrated for target_bits before saving
    print(f"Ensuring calibration for target precision {target_bits}-bit...")
    calib_mgr.ensure_calibrated(target_bits)

    # CRITICAL: Ensure LoRA weight quantizers are calibrated for target_bits
    if target_bits not in calib_mgr.lora_calibrated_bits:
        print(f"  Calibrating LoRA weight quantizers for target precision {target_bits}-bit...")
        calib_mgr.calibrate_lora_weight_quantizers([target_bits])
        calib_mgr.lora_calibrated_bits.add(target_bits)
        print(f"  ✓ LoRA weight quantizers calibrated for {target_bits}-bit")
    else:
        print(f"  ✓ LoRA weight quantizers already calibrated for {target_bits}-bit")

    print("Saving target model...")
    saved_path = cpt_deploy.save_target_model(model, config, target_bits, 'final_models')
    if saved_path:
        print(f"✅ Saved: {saved_path}")
    else:
        print("❌ Save failed")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cyclic Precision Training')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate the model without training')

    args = parser.parse_args()
    main(args)
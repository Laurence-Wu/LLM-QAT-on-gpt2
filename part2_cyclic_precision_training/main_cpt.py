"""
Main training script for Cyclic Precision Training (CPT).
Implements the key CPT training loop with precision cycling within each step.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2Model
from datasets import load_dataset
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


class WikiTextDataset(torch.utils.data.Dataset):
    """WikiText dataset for language modeling."""

    def __init__(self, split: str, tokenizer, max_seq_length: int = 256, doc_stride: int = 128):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride

        # Load dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        self.texts = dataset['text']

        # Tokenize and create sequences
        self.sequences = self._create_sequences()

    def _create_sequences(self):
        """Create overlapping sequences from texts."""
        sequences = []
        for text in self.texts:
            if len(text.strip()) == 0:
                continue

            # Tokenize
            tokens = self.tokenizer(
                text,
                truncation=False,
                return_tensors='pt'
            )['input_ids'][0]

            # Create overlapping sequences
            for i in range(0, len(tokens) - self.max_seq_length + 1, self.doc_stride):
                seq = tokens[i:i + self.max_seq_length]
                if len(seq) == self.max_seq_length:
                    sequences.append(seq)

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            'input_ids': seq[:-1],
            'labels': seq[1:]
        }


def train_step_with_cyclic_precision(
    model: CPTModel,
    batch: Dict[str, torch.Tensor],
    optimizer: optim.Optimizer,
    scheduler: CyclicPrecisionScheduler,
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
    cycle_precisions = scheduler.get_cycle_precisions()

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

    # Advance scheduler position
    scheduler.global_step += 1

    return {
        'total_loss': total_loss / len(cycle_precisions),
        'losses_per_precision': losses_per_precision,
        'cycle_info': scheduler.get_current_cycle_info()
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
    """Load pretrained GPT-2 weights into CPT model."""
    print("Loading pretrained GPT-2 weights...")

    # Load GPT-2 model
    gpt2 = GPT2Model.from_pretrained('gpt2')
    gpt2_state = gpt2.state_dict()

    # Map weights to CPT model
    model_state = model.state_dict()
    loaded_keys = []

    # Load embeddings
    if 'wte.weight' in gpt2_state:
        model.wte.weight.data = gpt2_state['wte.weight'].to(device)
        loaded_keys.append('wte.weight')
    if 'wpe.weight' in gpt2_state:
        model.wpe.weight.data = gpt2_state['wpe.weight'].to(device)
        loaded_keys.append('wpe.weight')

    # Load transformer blocks
    for i in range(len(model.h)):
        # Attention weights
        if f'h.{i}.attn.c_attn.weight' in gpt2_state:
            # Split concatenated QKV weights
            qkv_weight = gpt2_state[f'h.{i}.attn.c_attn.weight'].to(device)
            d_model = model.config['model'].n_embd
            model.h[i].attn.q_proj.linear.weight.data = qkv_weight[:d_model, :]
            model.h[i].attn.k_proj.linear.weight.data = qkv_weight[d_model:2*d_model, :]
            model.h[i].attn.v_proj.linear.weight.data = qkv_weight[2*d_model:, :]

        if f'h.{i}.attn.c_proj.weight' in gpt2_state:
            model.h[i].attn.out_proj.linear.weight.data = gpt2_state[f'h.{i}.attn.c_proj.weight'].to(device)

        # MLP weights
        if f'h.{i}.mlp.c_fc.weight' in gpt2_state:
            model.h[i].mlp['fc_in'].linear.weight.data = gpt2_state[f'h.{i}.mlp.c_fc.weight'].to(device)
        if f'h.{i}.mlp.c_proj.weight' in gpt2_state:
            model.h[i].mlp['fc_out'].linear.weight.data = gpt2_state[f'h.{i}.mlp.c_proj.weight'].to(device)

        # Layer norms (convert to Range LayerNorm parameters)
        if f'h.{i}.ln_1.weight' in gpt2_state:
            model.h[i].ln_1.weight.data = gpt2_state[f'h.{i}.ln_1.weight'].to(device)
            model.h[i].ln_1.bias.data = gpt2_state[f'h.{i}.ln_1.bias'].to(device)
        if f'h.{i}.ln_2.weight' in gpt2_state:
            model.h[i].ln_2.weight.data = gpt2_state[f'h.{i}.ln_2.weight'].to(device)
            model.h[i].ln_2.bias.data = gpt2_state[f'h.{i}.ln_2.bias'].to(device)

    # Final layer norm
    if 'ln_f.weight' in gpt2_state:
        model.ln_f.weight.data = gpt2_state['ln_f.weight'].to(device)
        model.ln_f.bias.data = gpt2_state['ln_f.bias'].to(device)

    print(f"Loaded {len(loaded_keys)} weight matrices from GPT-2")


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

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create datasets
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

    # Create model
    print("Creating CPT model...")
    model = CPTModel(config).to(device)

    # Load pretrained weights if specified
    if args.load_pretrained:
        load_pretrained_weights(model, device)

    # Create cyclic precision scheduler
    scheduler = CyclicPrecisionScheduler(
        bit_widths=model_config.bit_widths,
        schedule_type=cpt_config.schedule_type,
        cycle_length=cpt_config.cycle_length
    )

    # Optional: Run Precision Range Test
    if cpt_config.use_prt and not args.skip_prt:
        print("Running Precision Range Test...")
        prt = PrecisionRangeTest(
            model,
            start_bits=cpt_config.prt_start_bits,
            threshold=cpt_config.prt_threshold,
            test_iterations=cpt_config.prt_iterations
        )
        lower_bound, upper_bound = prt.find_bounds(train_loader, nn.CrossEntropyLoss())
        print(f"PRT Results: Lower bound = {lower_bound}-bit, Upper bound = {upper_bound}-bit")
        # Update scheduler with PRT results
        scheduler.min_bits = lower_bound
        scheduler.max_bits = upper_bound

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=training_config.adam_betas,
        eps=training_config.adam_epsilon,
        weight_decay=training_config.weight_decay
    )

    # Training loop
    print("Starting CPT training...")
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(training_config.num_epochs):
        epoch_losses = []
        epoch_start_time = time.time()

        # Training
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_config.num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # CRITICAL: Train with cyclic precision
            step_results = train_step_with_cyclic_precision(
                model, batch, optimizer, scheduler, device
            )

            epoch_losses.append(step_results['total_loss'])
            global_step += 1

            # Update progress bar
            cycle_info = step_results['cycle_info']
            progress_bar.set_postfix({
                'loss': f"{step_results['total_loss']:.4f}",
                'precision': f"{cycle_info['current_precision']}bit",
                'cycle': cycle_info['cycle_count']
            })

            # Log detailed info periodically
            if global_step % training_config.log_interval == 0:
                precision_losses = step_results['losses_per_precision']
                print(f"\nStep {global_step} - Losses per precision: {precision_losses}")

            # Clear cache periodically
            if global_step % training_config.empty_cache_interval == 0 and torch.cuda.is_available():
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
                        model, optimizer, epoch, global_step,
                        best_val_loss, config, 'checkpoints/best_model.pth'
                    )

        # Save checkpoint
        if (epoch + 1) % training_config.save_interval == 0:
            save_cpt_checkpoint(
                model, optimizer, epoch, global_step,
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

    args = parser.parse_args()
    main(args)
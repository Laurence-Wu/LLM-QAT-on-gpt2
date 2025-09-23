#!/usr/bin/env python3
"""
Part 2: Cyclic Precision Training (CPT)
Implementation based on CPT paper - cycles through different bit-widths during training.
"""

import os
import sys
import torch
import gc
import time
import json
from transformers import GPT2Config, GPT2TokenizerFast, GPT2Model

# Imports are now local to this folder

# Memory optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# Import CPT components
from models.cpt_model import CPTModel, CPTLMHeadModel
from training.cpt_scheduler import CPTScheduler
from utils.dataset import create_dataloaders

# Import local configurations and training
from config_cyclic import ModelConfig, CyclicTrainingConfig, CyclicPrecisionConfig
from train_cyclic import train_with_cpt


def initialize_model(model_config, device):
    """
    Initialize GPT-2 model for CPT training.
    """
    print("\nInitializing model for CPT...")

    # Create GPT-2 configuration
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        layer_norm_epsilon=model_config.layer_norm_epsilon,
        embd_pdrop=model_config.embd_pdrop,
    )

    # Add LoRA configuration to the config object
    gpt2_config.lora_rank = model_config.lora_rank
    gpt2_config.lora_alpha = model_config.lora_alpha

    # Initialize model
    model = CPTLMHeadModel(gpt2_config)

    # Load pretrained weights if requested
    if model_config.use_pretrained:
        load_pretrained_weights(model)

    # Enable gradient checkpointing
    model.use_gradient_checkpointing = model_config.use_gradient_checkpointing

    # Move to device
    model = model.to(device)

    print(f"Model initialized: {model_config.n_layer} layers")
    print(f"Gradient checkpointing: {model.use_gradient_checkpointing}")

    return model


def load_pretrained_weights(model):
    """Load pretrained GPT-2 weights."""
    print("Loading pretrained GPT-2 weights...")
    pretrained = GPT2Model.from_pretrained('gpt2')

    # Copy embeddings
    model.wte.weight.data = pretrained.wte.weight.data.clone()
    # Only copy the position embeddings we need (model might have fewer positions than pretrained)
    min_positions = min(model.wpe.weight.shape[0], pretrained.wpe.weight.shape[0])
    model.wpe.weight.data[:min_positions] = pretrained.wpe.weight.data[:min_positions].clone()
    if model.wpe.weight.shape[0] != pretrained.wpe.weight.shape[0]:
        print(f"Adjusted position embeddings from {pretrained.wpe.weight.shape[0]} to {model.wpe.weight.shape[0]}")

    # Copy transformer blocks
    for i in range(min(len(model.h), len(pretrained.h))):
        # Layer normalizations
        model.h[i].ln_1.weight.data = pretrained.h[i].ln_1.weight.data.clone()
        model.h[i].ln_1.bias.data = pretrained.h[i].ln_1.bias.data.clone()
        model.h[i].ln_2.weight.data = pretrained.h[i].ln_2.weight.data.clone()
        model.h[i].ln_2.bias.data = pretrained.h[i].ln_2.bias.data.clone()

        # Attention weights
        if hasattr(model.h[i].attn.c_attn, 'linear'):
            model.h[i].attn.c_attn.linear.weight.data = \
                pretrained.h[i].attn.c_attn.weight.data.t().contiguous()
            if pretrained.h[i].attn.c_attn.bias is not None:
                model.h[i].attn.c_attn.linear.bias.data = \
                    pretrained.h[i].attn.c_attn.bias.data.clone()

        if hasattr(model.h[i].attn.c_proj, 'linear'):
            model.h[i].attn.c_proj.linear.weight.data = \
                pretrained.h[i].attn.c_proj.weight.data.t().contiguous()
            if pretrained.h[i].attn.c_proj.bias is not None:
                model.h[i].attn.c_proj.linear.bias.data = \
                    pretrained.h[i].attn.c_proj.bias.data.clone()

        # MLP weights
        if hasattr(model.h[i].mlp.c_fc, 'linear'):
            model.h[i].mlp.c_fc.linear.weight.data = \
                pretrained.h[i].mlp.c_fc.weight.data.t().contiguous()
            if pretrained.h[i].mlp.c_fc.bias is not None:
                model.h[i].mlp.c_fc.linear.bias.data = \
                    pretrained.h[i].mlp.c_fc.bias.data.clone()

        if hasattr(model.h[i].mlp.c_proj, 'linear'):
            model.h[i].mlp.c_proj.linear.weight.data = \
                pretrained.h[i].mlp.c_proj.weight.data.t().contiguous()
            if pretrained.h[i].mlp.c_proj.bias is not None:
                model.h[i].mlp.c_proj.linear.bias.data = \
                    pretrained.h[i].mlp.c_proj.bias.data.clone()

    # Final layer normalization
    model.ln_f.weight.data = pretrained.ln_f.weight.data.clone()
    model.ln_f.bias.data = pretrained.ln_f.bias.data.clone()

    # Clean up
    del pretrained
    torch.cuda.empty_cache()
    gc.collect()

    print("Pretrained weights loaded")


def main():
    """Main CPT training function."""

    # Setup device
    device = torch.device('cuda')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()
        gc.collect()

    # Load configurations
    model_config = ModelConfig()
    training_config = CyclicTrainingConfig()
    cyclic_config = CyclicPrecisionConfig()

    # Initialize model
    model = initialize_model(model_config, device)

    # Setup tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create data loaders
    print("\nPreparing datasets...")
    train_loader, val_loader = create_dataloaders(
        tokenizer,
        train_split=training_config.train_split,
        val_split=training_config.val_split,
        batch_size=training_config.batch_size,
        max_length=training_config.max_seq_length,
        doc_stride=training_config.doc_stride
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Run CPT training
    print("\n" + "="*60)
    print("Starting Cyclic Precision Training (CPT)")
    print("="*60)
    print(f"Iterations: {training_config.num_cpt_iterations}")
    print(f"Cycle length: {cyclic_config.cycle_length}")
    print(f"Bit-width pattern: {cyclic_config.bit_width_pattern}")
    print(f"Batch size: {training_config.batch_size}")
    print(f"Gradient accumulation: {training_config.gradient_accumulation_steps}")

    # Train with CPT
    trained_model, training_stats = train_with_cpt(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_config=training_config,
        cyclic_config=cyclic_config,
        n_layers=model_config.n_layer
    )

    print("\n" + "="*60)
    print("CPT Training Complete!")
    print("="*60)

    # Save model and statistics
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # Save model
    model_path = f"cpt_gpt2_{timestamp}.pth"
    print(f"\nSaving model to {model_path}")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'model_config': model_config.__dict__,
        'training_config': training_config.__dict__,
        'cyclic_config': cyclic_config.__dict__,
        'timestamp': timestamp
    }, model_path)

    # Save training statistics
    stats_path = f"cpt_training_stats_{timestamp}.json"
    print(f"Saving statistics to {stats_path}")
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
            'train_split': training_config.train_split,
            'val_split': training_config.val_split,
            'batch_size': training_config.batch_size,
            'max_seq_length': training_config.max_seq_length,
            'doc_stride': training_config.doc_stride,
            'learning_rate': training_config.learning_rate,
            'weight_decay': training_config.weight_decay,
            'adam_epsilon': training_config.adam_epsilon,
            'adam_betas': training_config.adam_betas,
            'max_grad_norm': training_config.max_grad_norm,
            'num_iterations': training_config.num_iterations,
            'gradient_accumulation_steps': training_config.gradient_accumulation_steps,
            'eval_interval': training_config.eval_interval,
            'use_amp': training_config.use_amp,
            'num_workers': training_config.num_workers
        }

        stats_to_save['cyclic_config'] = {
            'bit_widths': cyclic_config.bit_widths,
            'cycle_length': cyclic_config.cycle_length,
            'annealing_type': cyclic_config.annealing_type,
            'temperature': cyclic_config.temperature,
            'min_temperature': cyclic_config.min_temperature,
            'temperature_decay': cyclic_config.temperature_decay
        }

        json.dump(stats_to_save, f, indent=2)

    # Print final statistics
    print("\n📊 Training Summary:")
    print(f"  Final loss: {training_stats.get('final_loss', 'N/A'):.4f}")
    if 'best_val_loss' in training_stats:
        print(f"  Best validation loss: {training_stats['best_val_loss']:.4f}")

    if 'cycle_metrics' in training_stats and training_stats['cycle_metrics']:
        print(f"\n🔄 Cycle Statistics:")
        print(f"  Total cycles completed: {len(training_stats['cycle_metrics'])}")
        cycle_losses = [m['avg_loss'] for m in training_stats['cycle_metrics']]
        print(f"  Best cycle avg loss: {min(cycle_losses):.4f}")
        print(f"  Final cycle avg loss: {cycle_losses[-1]:.4f}")

    print("\n✅ CPT training completed successfully!")

    return trained_model, training_stats


if __name__ == "__main__":
    main()
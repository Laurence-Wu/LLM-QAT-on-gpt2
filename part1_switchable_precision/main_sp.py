#!/usr/bin/env python3
"""
Part 1: Switchable Precision (SP)
Multi-precision training with separate LoRA adapters for each bit-width.
"""

import os
import sys
import torch
import gc
import json
from transformers import GPT2Config, GPT2TokenizerFast, GPT2Model

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Memory optimizations for efficient training
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# Import shared components
# Use the new separated model file for Switchable Precision
from shared.models_sp import SPModel, SPLMHeadModel
from shared.dataset import create_dataloaders
from shared.deploy import save_int8_checkpoint

# Use try/except to handle both direct execution and import cases
try:
    from config_sp import ModelConfig, TrainingConfig
    from train_sp import train_sp
except ImportError:
    from .config_sp import ModelConfig, TrainingConfig
    from .train_sp import train_sp


def initialize_model(model_config, device):
    # Validate required config attributes for switchable precision
    # Try to access each required attribute, throw error if any are missing
    try:
        _ = model_config.lora_rank_per_bit
        _ = model_config.lora_alpha_per_bit
        _ = model_config.activation_bits_per_bit
        _ = model_config.kv_cache_bits_per_bit
        _ = model_config.kv_cache_bits  # Single kv_cache_bits attribute
    except AttributeError as e:
        print(f"Error: ModelConfig missing required attribute: {e}")
        print("Required attributes: lora_rank_per_bit, lora_alpha_per_bit, activation_bits_per_bit, kv_cache_bits_per_bit, kv_cache_bits")
        print("These should be defined in ModelConfig class in config_sp.py")
        raise AttributeError(f"ModelConfig missing required switchable precision attribute: {e}")

    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        layer_norm_epsilon=model_config.layer_norm_epsilon,
        embd_pdrop=model_config.embd_pdrop,
        quantization_bits=model_config.quantization_bits,
        lora_rank=model_config.lora_rank,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        kv_cache_bits=model_config.kv_cache_bits
    )

    # Add switchable precision specific configs
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    gpt2_config.activation_bits_per_bit = model_config.activation_bits_per_bit
    gpt2_config.kv_cache_bits_per_bit = model_config.kv_cache_bits_per_bit

    # Print configuration being used
    print(f"Initializing SP Model with configurations:")
    print(f"  Bit widths: {model_config.bit_widths}")
    print(f"  LoRA rank per bit: {model_config.lora_rank_per_bit}")
    print(f"  LoRA alpha per bit: {model_config.lora_alpha_per_bit}")
    print(f"  Activation bits per bit: {model_config.activation_bits_per_bit}")
    print(f"  KV cache bits per bit: {model_config.kv_cache_bits_per_bit}")
    print(f"  Default KV cache bits: {model_config.kv_cache_bits}")

    # Use switchable model if configured
    gpt2_config.bit_widths = model_config.bit_widths
    model = SPLMHeadModel(gpt2_config)


    # Explicitly enable gradient checkpointing
    model.use_gradient_checkpointing = True

    # initialize all the layers with apply() function
    # layerNorm

    # weights of QKV, projection of the output of transformer
    # and the two feedforward layers
    load_pretrained_weights(model)

    # Model should be on the GPU
    model = model.to(device)

    print(f"SP Model: {model_config.n_layer} layers, bit-widths: {model_config.bit_widths}")
    print(f"Gradient checkpointing: {model.use_gradient_checkpointing}")

    return model


def load_pretrained_weights(model):
    print("Loading pretrained GPT-2 weights")
    pretrained = GPT2Model.from_pretrained('gpt2')

    # Copy embeddings
    model.transformer.wte.weight.data = pretrained.wte.weight.data.clone()
    # Only copy the position embeddings we need (model might have fewer positions than pretrained)
    min_positions = min(model.transformer.wpe.weight.shape[0], pretrained.wpe.weight.shape[0])
    model.transformer.wpe.weight.data[:min_positions] = pretrained.wpe.weight.data[:min_positions].clone()
    if model.transformer.wpe.weight.shape[0] != pretrained.wpe.weight.shape[0]:
        print(f"Adjusted position embeddings from {pretrained.wpe.weight.shape[0]} to {model.transformer.wpe.weight.shape[0]}")

    # Copy transformer blocks
    for i in range(min(len(model.transformer.h), len(pretrained.h))):
        # Layer normalizations
        model.transformer.h[i].ln_1.weight.data = pretrained.h[i].ln_1.weight.data.clone()
        model.transformer.h[i].ln_1.bias.data = pretrained.h[i].ln_1.bias.data.clone()
        model.transformer.h[i].ln_2.weight.data = pretrained.h[i].ln_2.weight.data.clone()
        model.transformer.h[i].ln_2.bias.data = pretrained.h[i].ln_2.bias.data.clone()

        # Attention QKV weights
        model.transformer.h[i].attn.c_attn.linear.weight.data = pretrained.h[i].attn.c_attn.weight.data.t().contiguous()
        model.transformer.h[i].attn.c_attn.linear.bias.data = pretrained.h[i].attn.c_attn.bias.data.clone()
        #  for attention layer projection
        model.transformer.h[i].attn.c_proj.linear.weight.data = pretrained.h[i].attn.c_proj.weight.data.t().contiguous()
        model.transformer.h[i].attn.c_proj.linear.bias.data = pretrained.h[i].attn.c_proj.bias.data.clone()

        # feedforward projection matrix to a higher dimension.
        model.transformer.h[i].mlp.c_fc.linear.weight.data = pretrained.h[i].mlp.c_fc.weight.data.t().contiguous()
        model.transformer.h[i].mlp.c_fc.linear.bias.data = pretrained.h[i].mlp.c_fc.bias.data.clone()

        # feedforward projection matrix from a higher dimension.
        model.transformer.h[i].mlp.c_proj.linear.weight.data = pretrained.h[i].mlp.c_proj.weight.data.t().contiguous()
        model.transformer.h[i].mlp.c_proj.linear.bias.data = pretrained.h[i].mlp.c_proj.bias.data.clone()

    # Final layer normalization
    model.transformer.ln_f.weight.data = pretrained.ln_f.weight.data.clone()
    model.transformer.ln_f.bias.data = pretrained.ln_f.bias.data.clone()

    # Delete pretrained model to free memory immediately
    del pretrained
    torch.cuda.empty_cache()
    gc.collect()

    print("pretrained weights loaded successfully.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Switchable Precision Training Script')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Training requires CUDA.")

    device = torch.device('cuda')
    print(f"Using device: {device}")
    torch.cuda.empty_cache()
    gc.collect()

    # Use configuration from config_qat.py
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Initialize model to gpu
    model = initialize_model(model_config, device)

    # Setup tokenizer to cpu
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token #use the end token as the padding to each sentences
    
    # Create data loaders in cpu
    print("\nDatasets")
    train_loader, val_loader = create_dataloaders(
        tokenizer,
        train_split=training_config.train_split,
        val_split=training_config.val_split,
        batch_size=training_config.batch_size,
        max_length=training_config.max_seq_length,
        doc_stride=training_config.doc_stride
    )
    
    ## print the current gpu
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    trained_model, training_stats = train_sp(
        model,
        train_loader,
        val_loader,
        training_config,
        model_config
    )
    
    print("Training complete")

    # Save both FP32 and INT8 models
    try:
        import time
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        # Save FP32 model (standard checkpoint with ALL configurations)
        fp32_filename = f"qat_gpt2_{model_config.quantization_bits}bit_fp32_{timestamp}.pth"

        # Create comprehensive model configuration dictionary
        model_config_dict = model_config.__dict__.copy()

        # Ensure critical SP configurations are included
        critical_configs = [
            'lora_rank_per_bit', 'lora_alpha_per_bit',
            'activation_bits_per_bit', 'kv_cache_bits_per_bit',
            'bit_widths', 'switch_strategy', 'switch_interval', 'curriculum_schedule'
        ]

        for config_key in critical_configs:
            if hasattr(model_config, config_key):
                model_config_dict[config_key] = getattr(model_config, config_key)
            else:
                print(f"Warning: Configuration '{config_key}' not found in model_config")

        # Verify lora_rank_per_bit and lora_alpha_per_bit are present and not None
        if 'lora_rank_per_bit' not in model_config_dict or model_config_dict['lora_rank_per_bit'] is None:
            raise ValueError("Critical configuration 'lora_rank_per_bit' is missing or None!")
        if 'lora_alpha_per_bit' not in model_config_dict or model_config_dict['lora_alpha_per_bit'] is None:
            raise ValueError("Critical configuration 'lora_alpha_per_bit' is missing or None!")

        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'model_config': model_config_dict,
            'training_config': training_config.__dict__,
            'timestamp': timestamp,
            'bit_widths': getattr(model_config, 'bit_widths', [4, 8, 16])  # Add for easy access
        }, fp32_filename)
        print(f"Saved FP32 model to {fp32_filename}")

        # Save INT8 model (quantized for deployment)
        int8_filename = f"sp_gpt2_int8_{timestamp}.pth"
        save_int8_checkpoint(trained_model, int8_filename, model_config, training_config)

    except Exception as e:
        # Don't silently catch - re-raise the error
        print(f"Error saving models: {e}")
        raise

    return trained_model

if __name__ == "__main__":
    main()

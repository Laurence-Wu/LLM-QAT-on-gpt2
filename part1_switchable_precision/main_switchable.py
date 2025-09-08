#!/usr/bin/env python3
"""
Part 1: Switchable Precision Training
This module implements training GPT-2 with switchable precision across different bit widths.
The model can dynamically switch between different quantization levels during training.
"""

import os
import sys
import torch
import gc
from transformers import GPT2Config, GPT2TokenizerFast, GPT2Model

# Add shared folder to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

# Memory optimizations for efficient training
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# Import shared components
from models import SwitchableQuantizedGPT2
from dataset import create_dataloaders

# Import local configurations and training functions
from config_switchable import ModelConfig, TrainingConfig
from train_switchable import train_switchable_quantization


def initialize_model(model_config, device):
    print("\nInitializing\n")
    
    # Create GPT-2 configuration with quantization support
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        layer_norm_epsilon=model_config.layer_norm_epsilon,
        embd_pdrop=model_config.embd_pdrop,
        bit_widths=model_config.bit_widths
    )
    
    # Initialize model with switchable quantization
    model = SwitchableQuantizedGPT2(gpt2_config).to(device)
    
    load_pretrained_weights(model)
    
    print(f"Model initialized with {model_config.n_layer} layers")
    print(f"Switchable bit widths: {model_config.bit_widths}")
    
    return model


def load_pretrained_weights(model):
    """
    Load pretrained GPT-2 weights into the switchable quantized model.
    
    Args:
        model: SwitchableQuantizedGPT2 model to load weights into
    """
    try:
        print("Loading pretrained GPT-2 weights...")
        pretrained = GPT2Model.from_pretrained('gpt2')
        
        # Copy embeddings
        model.wte.weight.data = pretrained.wte.weight.data.clone()
        model.wpe.weight.data = pretrained.wpe.weight.data.clone()
        
        # Copy transformer blocks
        for i in range(min(len(model.h), len(pretrained.h))):
            # Layer normalizations
            model.h[i].ln_1.weight.data = pretrained.h[i].ln_1.weight.data.clone()
            model.h[i].ln_1.bias.data = pretrained.h[i].ln_1.bias.data.clone()
            model.h[i].ln_2.weight.data = pretrained.h[i].ln_2.weight.data.clone()
            model.h[i].ln_2.bias.data = pretrained.h[i].ln_2.bias.data.clone()
            
            # Attention weights (handling Conv1D transpose)
            if hasattr(model.h[i].attn.c_attn, 'quantized_linear'):
                model.h[i].attn.c_attn.quantized_linear.weight.data = \
                    pretrained.h[i].attn.c_attn.weight.data.t().contiguous()
                if pretrained.h[i].attn.c_attn.bias is not None:
                    model.h[i].attn.c_attn.quantized_linear.bias.data = \
                        pretrained.h[i].attn.c_attn.bias.data.clone()
            
            if hasattr(model.h[i].attn.c_proj, 'quantized_linear'):
                model.h[i].attn.c_proj.quantized_linear.weight.data = \
                    pretrained.h[i].attn.c_proj.weight.data.t().contiguous()
                if pretrained.h[i].attn.c_proj.bias is not None:
                    model.h[i].attn.c_proj.quantized_linear.bias.data = \
                        pretrained.h[i].attn.c_proj.bias.data.clone()
            
            # MLP weights
            if hasattr(model.h[i].mlp.c_fc, 'quantized_linear'):
                model.h[i].mlp.c_fc.quantized_linear.weight.data = \
                    pretrained.h[i].mlp.c_fc.weight.data.t().contiguous()
                if pretrained.h[i].mlp.c_fc.bias is not None:
                    model.h[i].mlp.c_fc.quantized_linear.bias.data = \
                        pretrained.h[i].mlp.c_fc.bias.data.clone()
            
            if hasattr(model.h[i].mlp.c_proj, 'quantized_linear'):
                model.h[i].mlp.c_proj.quantized_linear.weight.data = \
                    pretrained.h[i].mlp.c_proj.weight.data.t().contiguous()
                if pretrained.h[i].mlp.c_proj.bias is not None:
                    model.h[i].mlp.c_proj.quantized_linear.bias.data = \
                        pretrained.h[i].mlp.c_proj.bias.data.clone()
        
        # Final layer normalization
        model.ln_f.weight.data = pretrained.ln_f.weight.data.clone()
        model.ln_f.bias.data = pretrained.ln_f.bias.data.clone()
        
        print("Pretrained weights successfully loaded")
        
    except Exception as e:
        print(f"Warning: Could not load pretrained weights: {e}")
        print("Continuing with random initialization...")


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Initialize model
    model = initialize_model(model_config, device)
    
    if torch.cuda.is_available():
        print(f"Model GPU Memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
    
    # Setup tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create data loaders
    print("\nLoading datasets...")
    train_loader, val_loader = create_dataloaders(
        tokenizer,
        train_split=training_config.train_split,
        val_split=training_config.val_split,
        batch_size=training_config.batch_size,
        max_length=training_config.max_seq_length,
        doc_stride=training_config.doc_stride
    )
    
    # Phase 1: Train with switchable quantization
    
    trained_model = train_switchable_quantization(
        model, 
        train_loader, 
        val_loader, 
        training_config,
        model_config,
        n_layers=model_config.n_layer
    )
    
    print("Training returned")
    
    # Skip model saving to avoid crash
    print("Skipping model save")

    return trained_model, results


if __name__ == "__main__":
    try:
        model, results = main()
    except Exception as e:
        import traceback
        traceback.print_exc()
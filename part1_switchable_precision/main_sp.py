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
from transformers import GPT2Config, GPT2TokenizerFast

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Memory optimizations for efficient training
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# Import shared components
# Use the new separated model file for Switchable Precision
from models_sp import SPModel, SPLMHeadModel
from dataset import create_dataloaders
from deploy import save_int8_checkpoint, save_sp_checkpoints

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
    except AttributeError as e:
        print(f"Error: ModelConfig missing required attribute: {e}")
        print("Required attributes: lora_rank_per_bit, lora_alpha_per_bit, activation_bits_per_bit")
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
        lora_dropout=model_config.lora_dropout
    )

    # Add switchable precision specific configs
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    gpt2_config.activation_bits_per_bit = model_config.activation_bits_per_bit
    gpt2_config.quantizer_per_bit = getattr(model_config, 'quantizer_per_bit', None)

    # Print configuration being used
    print(f"Initializing SP Model with configurations:")
    print(f"  Bit widths: {model_config.bit_widths}")
    print(f"  LoRA rank per bit: {model_config.lora_rank_per_bit}")
    print(f"  LoRA alpha per bit: {model_config.lora_alpha_per_bit}")
    print(f"  Activation bits per bit: {model_config.activation_bits_per_bit}")
    if gpt2_config.quantizer_per_bit:
        print(f"  Quantizer per bit: {gpt2_config.quantizer_per_bit}")

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

    # Set to 32-bit precision to unfreeze teacher weights after loading
    # This is critical - load_pretrained_weights freezes everything,
    # but 32-bit teacher needs unfrozen weights for training
    model.set_precision(32)
    print("Set initial precision to 32-bit (teacher mode) - weights unfrozen for teacher training")

    print(f"SP Model: {model_config.n_layer} layers, bit-widths: {model_config.bit_widths}")
    print(f"Gradient checkpointing: {model.use_gradient_checkpointing}")

    return model


def load_pretrained_weights(model):
    print("Loading pretrained GPT-2 weights")
    from transformers import GPT2LMHeadModel  # Import the correct class with LM head
    pretrained = GPT2LMHeadModel.from_pretrained('gpt2')  # Load model WITH language model head

    import torch.nn as nn

    # Copy embeddings - frozen (never trained)
    model.transformer.wte.weight.data = pretrained.transformer.wte.weight.data.clone()
    model.transformer.wte.weight.requires_grad = False  # Keep frozen

    # Only copy the position embeddings we need (model might have fewer positions than pretrained)
    min_positions = min(model.transformer.wpe.weight.shape[0], pretrained.transformer.wpe.weight.shape[0])
    model.transformer.wpe.weight.data[:min_positions] = pretrained.transformer.wpe.weight.data[:min_positions].clone()
    model.transformer.wpe.weight.requires_grad = False  # Keep frozen

    if model.transformer.wpe.weight.shape[0] != pretrained.transformer.wpe.weight.shape[0]:
        print(f"Adjusted position embeddings from {pretrained.transformer.wpe.weight.shape[0]} to {model.transformer.wpe.weight.shape[0]}")

    # Copy LM head weights - frozen by default (will be unfrozen for 32-bit teacher)
    model.lm_head.weight.data = pretrained.lm_head.weight.data.clone()
    model.lm_head.weight.requires_grad = False  # Frozen by default

    # Copy transformer blocks - frozen by default (will be unfrozen for 32-bit teacher)
    for i in range(min(len(model.transformer.h), len(pretrained.transformer.h))):
        # Layer normalizations - load into ALL precision-specific layers
        # For SwitchableLayerNorm, copy weights to each precision's LayerNorm
        for ln_key in model.transformer.h[i].ln_1.ln_layers:
            model.transformer.h[i].ln_1.ln_layers[ln_key].weight.data = pretrained.transformer.h[i].ln_1.weight.data.clone()
            model.transformer.h[i].ln_1.ln_layers[ln_key].bias.data = pretrained.transformer.h[i].ln_1.bias.data.clone()
            model.transformer.h[i].ln_1.ln_layers[ln_key].weight.requires_grad = False
            model.transformer.h[i].ln_1.ln_layers[ln_key].bias.requires_grad = False

        for ln_key in model.transformer.h[i].ln_2.ln_layers:
            model.transformer.h[i].ln_2.ln_layers[ln_key].weight.data = pretrained.transformer.h[i].ln_2.weight.data.clone()
            model.transformer.h[i].ln_2.ln_layers[ln_key].bias.data = pretrained.transformer.h[i].ln_2.bias.data.clone()
            model.transformer.h[i].ln_2.ln_layers[ln_key].weight.requires_grad = False
            model.transformer.h[i].ln_2.ln_layers[ln_key].bias.requires_grad = False

        # Attention QKV weights - transpose and freeze by default
        model.transformer.h[i].attn.c_attn.linear.weight.data = pretrained.transformer.h[i].attn.c_attn.weight.data.t().contiguous()
        model.transformer.h[i].attn.c_attn.linear.bias.data = pretrained.transformer.h[i].attn.c_attn.bias.data.clone()
        model.transformer.h[i].attn.c_attn.linear.weight.requires_grad = False
        model.transformer.h[i].attn.c_attn.linear.bias.requires_grad = False

        # Attention projection - transpose and freeze by default
        model.transformer.h[i].attn.c_proj.linear.weight.data = pretrained.transformer.h[i].attn.c_proj.weight.data.t().contiguous()
        model.transformer.h[i].attn.c_proj.linear.bias.data = pretrained.transformer.h[i].attn.c_proj.bias.data.clone()
        model.transformer.h[i].attn.c_proj.linear.weight.requires_grad = False
        model.transformer.h[i].attn.c_proj.linear.bias.requires_grad = False

        # MLP feedforward projection to higher dimension - transpose and freeze by default
        model.transformer.h[i].mlp.c_fc.linear.weight.data = pretrained.transformer.h[i].mlp.c_fc.weight.data.t().contiguous()
        model.transformer.h[i].mlp.c_fc.linear.bias.data = pretrained.transformer.h[i].mlp.c_fc.bias.data.clone()
        model.transformer.h[i].mlp.c_fc.linear.weight.requires_grad = False
        model.transformer.h[i].mlp.c_fc.linear.bias.requires_grad = False

        # MLP feedforward projection from higher dimension - transpose and freeze by default
        model.transformer.h[i].mlp.c_proj.linear.weight.data = pretrained.transformer.h[i].mlp.c_proj.weight.data.t().contiguous()
        model.transformer.h[i].mlp.c_proj.linear.bias.data = pretrained.transformer.h[i].mlp.c_proj.bias.data.clone()
        model.transformer.h[i].mlp.c_proj.linear.weight.requires_grad = False
        model.transformer.h[i].mlp.c_proj.linear.bias.requires_grad = False

    # Final layer normalization - frozen by default
    # Copy to all precision-specific LayerNorms
    for ln_key in model.transformer.ln_f.ln_layers:
        model.transformer.ln_f.ln_layers[ln_key].weight.data = pretrained.transformer.ln_f.weight.data.clone()
        model.transformer.ln_f.ln_layers[ln_key].bias.data = pretrained.transformer.ln_f.bias.data.clone()
        model.transformer.ln_f.ln_layers[ln_key].weight.requires_grad = False
        model.transformer.ln_f.ln_layers[ln_key].bias.requires_grad = False


    # Just ensure LoRA parameters are trainable (they're already initialized in LoRALayer)
    lora_count = 0
    for name, module in model.named_modules():
        try:
            if module.lora_adapters:  # Will throw AttributeError if not present
                for bit_key, lora_layer in module.lora_adapters.items():
                    if isinstance(lora_layer.lora_A, nn.Parameter):
                        lora_layer.lora_A.requires_grad = True
                        lora_count += 1
                    if isinstance(lora_layer.lora_B, nn.Parameter):
                        lora_layer.lora_B.requires_grad = True
        except AttributeError:
            # Module doesn't have lora_adapters, continue
            pass

    print(f"Enabled {lora_count} LoRA adapter pairs for training")

    # Delete pretrained model to free memory immediately
    del pretrained
    torch.cuda.empty_cache()
    gc.collect()

    # Count trainable vs frozen parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + frozen_params

    print(f"Pretrained weights loaded and frozen successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    print(f"  Trainable (LoRA) parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

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

    # Add calibration configuration if not present
    if not hasattr(training_config, 'calibration_samples'):
        training_config.calibration_samples = 10  # Default calibration samples

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
    print(f"Calibration samples per bit-width: {training_config.calibration_samples}")

    trained_model, training_stats = train_sp(
        model,
        train_loader,
        val_loader,
        training_config,
        model_config
    )
    
    print("Training complete")

    # Save checkpoints for all configured bit widths
    try:
        # Use the new save_sp_checkpoints function to save models for each bit width
        # This will save:
        # - INT8 checkpoints for student models (6, 8, 16-bit)
        # - FP32 checkpoint for teacher model (32-bit)
        saved_checkpoints = save_sp_checkpoints(
            trained_model,
            base_filename="sp_gpt2",
            model_config=model_config,
            training_config=training_config
        )

        print("\nCheckpoint Summary:")
        for bits, filepath in saved_checkpoints.items():
            print(f"  {bits}-bit: {filepath}")

    except Exception as e:
        # Don't silently catch - re-raise the error
        print(f"Error saving models: {e}")
        raise

    return trained_model

if __name__ == "__main__":
    main()

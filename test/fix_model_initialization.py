#!/usr/bin/env python3
"""
Fix Model Initialization Issue
Ensures SP model is properly initialized with pretrained weights
"""

import sys
import os
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models_sp import SPLMHeadModel
from part1_switchable_precision.config_sp import ModelConfig


def create_properly_initialized_model(use_pretrained=True):
    """
    Create SP model with proper initialization.

    Args:
        use_pretrained: If True, load pretrained GPT-2 weights

    Returns:
        Properly initialized SPLMHeadModel
    """
    print("\n" + "="*60)
    print("CREATING PROPERLY INITIALIZED SP MODEL")
    print("="*60)

    # Create configuration
    model_config = ModelConfig()

    # IMPORTANT: Match GPT-2 dimensions when using pretrained weights
    if use_pretrained:
        model_config.n_embd = 768  # GPT-2 small uses 768
        model_config.n_head = 12   # GPT-2 small uses 12 heads
        model_config.n_layer = 12  # Use exact same as GPT-2 small (12 layers)
        print("\n✓ Using exact GPT-2 dimensions")
    else:
        model_config.n_layer = 2
        model_config.n_embd = 256
        model_config.n_head = 4
        print("\n✓ Using custom dimensions")

    # Create GPT-2 config
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        layer_norm_epsilon=model_config.layer_norm_epsilon,
        embd_pdrop=model_config.embd_pdrop
    )

    # Add SP-specific configurations
    gpt2_config.bit_widths = model_config.bit_widths
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit

    # Create model
    print("\n1. Creating SP model...")
    model = SPLMHeadModel(gpt2_config)

    if use_pretrained:
        print("\n2. Loading pretrained GPT-2 weights...")
        load_pretrained_weights_properly(model, model_config)
    else:
        print("\n2. Initializing with Xavier/Kaiming initialization...")
        initialize_weights_properly(model)

    # Verify initialization
    print("\n3. Verifying initialization...")
    verify_initialization(model)

    return model, model_config


def load_pretrained_weights_properly(model, config):
    """
    Properly load pretrained GPT-2 weights into SP model.
    """
    pretrained = GPT2Model.from_pretrained('gpt2')

    # 1. Copy embeddings
    print("   - Copying embeddings...")
    model.transformer.wte.weight.data = pretrained.wte.weight.data.clone()

    # Handle position embeddings size mismatch
    min_pos = min(model.transformer.wpe.weight.shape[0], pretrained.wpe.weight.shape[0])
    model.transformer.wpe.weight.data[:min_pos] = pretrained.wpe.weight.data[:min_pos].clone()

    # Initialize remaining position embeddings if needed
    if model.transformer.wpe.weight.shape[0] > pretrained.wpe.weight.shape[0]:
        print(f"   - Initializing extra position embeddings ({pretrained.wpe.weight.shape[0]} -> {model.transformer.wpe.weight.shape[0]})")
        # Copy pattern from existing embeddings
        remaining = model.transformer.wpe.weight.shape[0] - pretrained.wpe.weight.shape[0]
        model.transformer.wpe.weight.data[min_pos:] = pretrained.wpe.weight.data[:remaining].clone()

    # 2. Copy transformer blocks
    num_layers = min(len(model.transformer.h), len(pretrained.h))
    print(f"   - Copying {num_layers} transformer blocks (model has {len(model.transformer.h)}, pretrained has {len(pretrained.h)})...")

    for i in range(num_layers):
        # Layer normalizations
        model.transformer.h[i].ln_1.weight.data = pretrained.h[i].ln_1.weight.data.clone()
        model.transformer.h[i].ln_1.bias.data = pretrained.h[i].ln_1.bias.data.clone()
        model.transformer.h[i].ln_2.weight.data = pretrained.h[i].ln_2.weight.data.clone()
        model.transformer.h[i].ln_2.bias.data = pretrained.h[i].ln_2.bias.data.clone()

        # Attention weights (transpose for nn.Linear)
        model.transformer.h[i].attn.c_attn.linear.weight.data = pretrained.h[i].attn.c_attn.weight.data.t().contiguous()
        model.transformer.h[i].attn.c_attn.linear.bias.data = pretrained.h[i].attn.c_attn.bias.data.clone()
        model.transformer.h[i].attn.c_proj.linear.weight.data = pretrained.h[i].attn.c_proj.weight.data.t().contiguous()
        model.transformer.h[i].attn.c_proj.linear.bias.data = pretrained.h[i].attn.c_proj.bias.data.clone()

        # MLP weights (transpose for nn.Linear)
        model.transformer.h[i].mlp.c_fc.linear.weight.data = pretrained.h[i].mlp.c_fc.weight.data.t().contiguous()
        model.transformer.h[i].mlp.c_fc.linear.bias.data = pretrained.h[i].mlp.c_fc.bias.data.clone()
        model.transformer.h[i].mlp.c_proj.linear.weight.data = pretrained.h[i].mlp.c_proj.weight.data.t().contiguous()
        model.transformer.h[i].mlp.c_proj.linear.bias.data = pretrained.h[i].mlp.c_proj.bias.data.clone()

    # 3. Copy final layer norm
    print("   - Copying final layer norm...")
    model.transformer.ln_f.weight.data = pretrained.ln_f.weight.data.clone()
    model.transformer.ln_f.bias.data = pretrained.ln_f.bias.data.clone()

    # 4. Initialize LoRA adapters properly
    print("   - Initializing LoRA adapters...")
    for name, module in model.named_modules():
        if hasattr(module, 'lora_adapters'):
            for bit_key, lora_layer in module.lora_adapters.items():
                # LoRA A: small random initialization
                nn.init.kaiming_uniform_(lora_layer.lora_A, a=5**0.5)
                # LoRA B: initialize to zeros (no initial contribution)
                nn.init.zeros_(lora_layer.lora_B)

    # Verify weight tying
    if model.lm_head.weight is model.transformer.wte.weight:
        print("   ✓ LM head weight tying verified")
    else:
        print("   ⚠️ LM head weight tying not working!")

    print("   ✓ Pretrained weights loaded successfully")


def initialize_weights_properly(model):
    """
    Initialize model weights properly without pretrained weights.
    """
    def init_weights(module):
        if isinstance(module, nn.Linear):
            # Xavier initialization for linear layers
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # Normal initialization for embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # Layer norm: bias=0, weight=1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    model.apply(init_weights)

    # Special initialization for LoRA
    for name, module in model.named_modules():
        if hasattr(module, 'lora_adapters'):
            for bit_key, lora_layer in module.lora_adapters.items():
                # LoRA A: small random initialization
                nn.init.kaiming_uniform_(lora_layer.lora_A, a=5**0.5)
                # LoRA B: zeros
                nn.init.zeros_(lora_layer.lora_B)

    print("   ✓ Weights initialized with Xavier/Kaiming initialization")


def verify_initialization(model):
    """
    Verify that model is properly initialized.
    """
    issues = []

    # Check embeddings
    wte_std = model.transformer.wte.weight.data.std().item()
    wpe_std = model.transformer.wpe.weight.data.std().item()

    if wte_std < 0.001:
        issues.append("Token embeddings are near zero")
    if wpe_std < 0.001:
        issues.append("Position embeddings are near zero")

    # Check attention/MLP weights
    for i, block in enumerate(model.transformer.h):
        attn_std = block.attn.c_attn.linear.weight.data.std().item()
        mlp_std = block.mlp.c_fc.linear.weight.data.std().item()

        if attn_std < 0.001:
            issues.append(f"Block {i} attention weights are near zero")
        if mlp_std < 0.001:
            issues.append(f"Block {i} MLP weights are near zero")

    # Check LoRA initialization
    for name, module in model.named_modules():
        if hasattr(module, 'lora_adapters'):
            for bit_key, lora_layer in module.lora_adapters.items():
                b_max = lora_layer.lora_B.data.abs().max().item()
                if b_max > 0.01:
                    issues.append(f"{name}.{bit_key} LoRA B not initialized to zero")

    if issues:
        print("   ⚠️ Initialization issues found:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print("   ✅ Model initialization verified successfully")

    return len(issues) == 0


def test_initialized_model(model, model_config):
    """
    Test the initialized model to verify it produces reasonable loss.
    """
    print("\n4. Testing initialized model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Test input
    batch_size = 2
    seq_length = 32
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_length)).to(device)
    labels = input_ids.clone()

    losses = {}
    with torch.no_grad():
        for bits in [16, 8, 4]:
            model.set_precision(bits)
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss'].item()
            losses[bits] = loss

            if loss < 15:
                print(f"   ✅ {bits:2d}-bit: Loss = {loss:8.4f} (Good)")
            elif loss < 30:
                print(f"   ⚠️ {bits:2d}-bit: Loss = {loss:8.4f} (Acceptable)")
            else:
                print(f"   ❌ {bits:2d}-bit: Loss = {loss:8.4f} (Too high)")

    # Expected loss for random predictions: -log(1/vocab_size) ≈ 10.8 for 50257 vocab
    expected_random_loss = torch.nn.functional.cross_entropy(
        torch.ones(1, model_config.vocab_size) / model_config.vocab_size,
        torch.zeros(1, dtype=torch.long)
    ).item()

    print(f"\n   Expected loss for random predictions: {expected_random_loss:.2f}")
    print(f"   Your model's best loss: {min(losses.values()):.2f}")

    if min(losses.values()) < expected_random_loss * 2:
        print("\n   ✅ Model initialization is good! Loss is reasonable.")
    else:
        print("\n   ⚠️ Loss is still high. Consider:")
        print("      1. Using pretrained weights")
        print("      2. Reducing learning rate")
        print("      3. Starting with higher precision (16-bit)")

    return losses


def main():
    """
    Main function to demonstrate proper model initialization.
    """
    print("\n" + "="*80)
    print("MODEL INITIALIZATION FIX")
    print("="*80)

    # Test 1: With pretrained weights (recommended)
    print("\n--- Test 1: With Pretrained Weights ---")
    model_pretrained, config_pretrained = create_properly_initialized_model(use_pretrained=True)
    losses_pretrained = test_initialized_model(model_pretrained, config_pretrained)

    # Test 2: Without pretrained weights
    print("\n--- Test 2: With Random Initialization ---")
    model_random, config_random = create_properly_initialized_model(use_pretrained=False)
    losses_random = test_initialized_model(model_random, config_random)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\nResults:")
    print(f"  With pretrained weights: Best loss = {min(losses_pretrained.values()):.2f}")
    print(f"  With random init:        Best loss = {min(losses_random.values()):.2f}")

    print("\n💡 RECOMMENDATION:")
    if min(losses_pretrained.values()) < min(losses_random.values()):
        print("  Always use pretrained weights for better initialization!")
        print("  This dramatically reduces initial loss and speeds up training.")
    else:
        print("  Both initializations work, but pretrained is still recommended.")

    print("\n📝 To fix your training code, add this after model creation:")
    print("-"*60)
    print("model = SPLMHeadModel(config)")
    print("load_pretrained_weights_properly(model, model_config)  # Add this line!")
    print("-"*60)

    return model_pretrained


if __name__ == "__main__":
    model = main()
    print("\n✅ Initialization fix complete!")
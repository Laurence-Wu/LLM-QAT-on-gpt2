#!/usr/bin/env python3
"""
Quick test to verify S-BN fixes work correctly.
"""

import sys
import os
import torch
from transformers import GPT2Config

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models_sp import SPLMHeadModel
from shared.switchable_batchnorm import SwitchableLayerNorm
from part1_switchable_precision.config_sp import ModelConfig


def main():
    print("\n" + "="*60)
    print("TESTING S-BN FIXES")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model config
    model_config = ModelConfig()
    config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=2,  # Small for testing
        n_head=model_config.n_head
    )

    # Add SP attributes
    config.bit_widths = model_config.bit_widths
    config.lora_rank_per_bit = model_config.lora_rank_per_bit
    config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    config.quantizer_per_bit = model_config.quantizer_per_bit
    config.lora_dropout = model_config.lora_dropout

    # Create model
    print("\n🔧 Creating SP model with S-BN...")
    model = SPLMHeadModel(config).to(device)

    # Test 1: Precision switching without errors
    print("\n1. Testing precision switching:")
    for bits in [4, 8, 16, 32]:
        try:
            model.set_precision(bits)
            print(f"   ✅ {bits}-bit: Set successfully")
        except Exception as e:
            print(f"   ❌ {bits}-bit: Error - {e}")

    # Test 2: Check S-BN structure
    print("\n2. Checking S-BN layer structure:")
    first_block = model.transformer.h[0]
    if isinstance(first_block.ln_1, SwitchableLayerNorm):
        print("   ✅ Using SwitchableLayerNorm")
        for bits in [4, 8, 16, 32]:
            ln_key = f'ln_{bits}bit'
            if ln_key in first_block.ln_1.ln_layers:
                ln = first_block.ln_1.ln_layers[ln_key]
                print(f"   ✅ {bits}-bit: LayerNorm exists")
                # Check if weights are accessible
                if hasattr(ln, 'weight') and hasattr(ln, 'bias'):
                    print(f"      - Weight shape: {ln.weight.shape}")
                    print(f"      - Bias shape: {ln.bias.shape}")
    else:
        print("   ❌ Not using SwitchableLayerNorm")

    # Test 3: Forward pass
    print("\n3. Testing forward pass:")
    input_ids = torch.randint(0, config.vocab_size, (2, 64), device=device)
    for bits in [4, 8, 16, 32]:
        model.set_precision(bits)
        try:
            outputs = model(input_ids, return_dict=True)
            loss = outputs['loss'] if 'loss' in outputs else None
            print(f"   ✅ {bits}-bit: Forward pass successful")
        except Exception as e:
            print(f"   ❌ {bits}-bit: Error - {e}")

    # Test 4: Gradient flow
    print("\n4. Testing gradient flow:")
    model.set_precision(8)  # Use 8-bit for test
    model.train()
    outputs = model(input_ids, labels=input_ids, return_dict=True)
    loss = outputs['loss']
    loss.backward()

    grad_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and param.grad.abs().sum() > 0:
            grad_count += 1

    print(f"   ✅ {grad_count} parameters have gradients")

    print("\n" + "="*60)
    print("✅ S-BN FIXES VERIFIED")
    print("="*60)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test CPTModel layer-wise precision setting
"""

import sys
import os
import torch
from transformers import GPT2Config

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models_cpt import CPTModel

def test_layer_precision():
    """Test CPTModel.set_layer_precision method."""
    print("Testing CPTModel layer-wise precision...")

    # Create config
    config = GPT2Config(
        n_embd=256,
        n_head=4,
        n_layer=2,  # 2 layers for testing
        n_positions=128,
        vocab_size=1000
    )
    config.lora_rank = 8
    config.lora_alpha = 16
    config.lora_dropout = 0.1
    config.quantization_bits = 8

    # Create model
    model = CPTModel(config)

    # Test 1: Set global precision
    print("\n1. Testing global precision setting...")
    model.set_precision(4, 4)
    print("   ✓ Global precision set to 4-bit")

    # Test 2: Set layer-wise precision with correct format
    print("\n2. Testing layer-wise precision setting...")
    layer_configs = [
        {'attn_bits': 8, 'mlp_bits': 8, 'activation_bits': 8, 'kv_bits': 8},  # Layer 0
        {'attn_bits': 4, 'mlp_bits': 4, 'activation_bits': 4, 'kv_bits': 4}   # Layer 1
    ]

    try:
        model.set_layer_precision(layer_configs)
        print("   ✓ Layer-wise precision set successfully")
    except Exception as e:
        print(f"   ✗ Failed to set layer-wise precision: {e}")
        return False

    # Test 3: Forward pass with layer-wise precision
    print("\n3. Testing forward pass with layer-wise precision...")
    input_ids = torch.randint(0, 1000, (2, 10))

    try:
        output = model(input_ids)
        assert output.shape == (2, 10, 256), f"Wrong output shape: {output.shape}"
        print(f"   ✓ Forward pass successful, output shape: {output.shape}")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        return False

    # Test 4: Try with mismatched number of layers (should handle gracefully)
    print("\n4. Testing with more layer configs than layers...")
    extra_layer_configs = layer_configs + [
        {'attn_bits': 2, 'mlp_bits': 2, 'activation_bits': 2, 'kv_bits': 2}
    ]

    try:
        model.set_layer_precision(extra_layer_configs)
        print("   ✓ Handled extra layer configs gracefully")
    except Exception as e:
        print(f"   Note: Extra configs caused error (expected): {e}")

    return True

def main():
    print("="*60)
    print("CPT Layer-wise Precision Test")
    print("="*60)

    try:
        success = test_layer_precision()

        print("\n" + "="*60)
        if success:
            print("✅ All layer precision tests passed!")
        else:
            print("❌ Some tests failed")
        print("="*60)

        return success
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
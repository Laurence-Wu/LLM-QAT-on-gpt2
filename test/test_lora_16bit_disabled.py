#!/usr/bin/env python3
"""
Test to verify LoRA is properly disabled for 16-bit mode.
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part1_switchable_precision.config_sp import ModelConfig
from shared.lora import LoRALayer, SPLinearWithLoRA


def test_lora_disabled_16bit():
    """Test that LoRA is disabled for 16-bit mode."""
    print("="*60)
    print("Testing LoRA Disabled for 16-bit Mode")
    print("="*60)

    # Load config
    config = ModelConfig()

    print("\n1. Checking configuration...")
    print(f"   16-bit LoRA rank: {config.lora_rank_per_bit.get(16, 'Not set')}")
    print(f"   16-bit LoRA alpha: {config.lora_alpha_per_bit.get(16, 'Not set')}")

    if config.lora_rank_per_bit.get(16, 1) != 0:
        print("   ❌ FAILED: 16-bit rank should be 0")
        return False
    else:
        print("   ✅ PASSED: 16-bit rank is 0")

    # Test LoRALayer directly
    print("\n2. Testing LoRALayer with 16-bit...")
    lora_16bit = LoRALayer(
        in_features=768,
        out_features=768,
        rank=config.lora_rank_per_bit.get(16, 0),
        alpha=config.lora_alpha_per_bit.get(16, 0),
        bits=16
    )

    if lora_16bit.enabled:
        print(f"   ❌ FAILED: LoRA enabled={lora_16bit.enabled} (should be False)")
        return False
    else:
        print(f"   ✅ PASSED: LoRA enabled={lora_16bit.enabled}")

    if lora_16bit.scaling != 0:
        print(f"   ❌ FAILED: LoRA scaling={lora_16bit.scaling} (should be 0)")
        return False
    else:
        print(f"   ✅ PASSED: LoRA scaling={lora_16bit.scaling}")

    # Test with SPLinearWithLoRA
    print("\n3. Testing SPLinearWithLoRA...")
    sp_layer = SPLinearWithLoRA(
        in_features=768,
        out_features=768,
        bit_widths=[4, 8, 16],
        lora_rank_per_bit=config.lora_rank_per_bit,
        lora_alpha_per_bit=config.lora_alpha_per_bit
    )

    # Set to 16-bit mode
    sp_layer.set_precision(16)

    # Check the 16-bit LoRA adapter
    lora_16bit_adapter = sp_layer.lora_adapters['16bit']

    if lora_16bit_adapter.enabled:
        print(f"   ❌ FAILED: SPLinearWithLoRA 16-bit LoRA enabled (should be disabled)")
        return False
    else:
        print(f"   ✅ PASSED: SPLinearWithLoRA 16-bit LoRA disabled")

    # Test forward pass to ensure no LoRA contribution
    print("\n4. Testing forward pass with 16-bit...")
    x = torch.randn(2, 10, 768)

    # Get output with 16-bit (no LoRA)
    sp_layer.set_precision(16)
    out_16bit = sp_layer(x)

    # Manually compute without LoRA for comparison
    with torch.no_grad():
        expected_out = torch.nn.functional.linear(x, sp_layer.linear.weight, sp_layer.linear.bias)

    # Check if outputs match (16-bit should be pure linear, no LoRA)
    if torch.allclose(out_16bit, expected_out, atol=1e-5):
        print(f"   ✅ PASSED: 16-bit output matches pure linear (no LoRA contribution)")
    else:
        diff = (out_16bit - expected_out).abs().max().item()
        print(f"   ❌ FAILED: 16-bit output differs from pure linear by {diff}")
        return False

    # Compare with 8-bit to ensure LoRA is active there
    print("\n5. Verifying LoRA is active for 8-bit...")
    sp_layer.set_precision(8)
    lora_8bit_adapter = sp_layer.lora_adapters['8bit']

    if not lora_8bit_adapter.enabled:
        print(f"   ❌ FAILED: 8-bit LoRA should be enabled")
        return False
    else:
        print(f"   ✅ PASSED: 8-bit LoRA is enabled")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED: LoRA properly disabled for 16-bit")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_lora_disabled_16bit()
    sys.exit(0 if success else 1)
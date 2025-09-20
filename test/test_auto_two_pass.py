#!/usr/bin/env python3
"""
Test automatic two-pass quantization in forward pass.
"""

import sys
import os
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.quantization import LearnableFakeQuantize
from shared.lora import SPLinearWithLoRA
from part1_switchable_precision.config_sp import ModelConfig, TrainingConfig


def test_auto_two_pass():
    """Test automatic two-pass behavior in forward pass."""
    print("="*60)
    print("Testing Automatic Two-Pass Quantization")
    print("="*60)

    # Setup
    config = ModelConfig()
    training_config = TrainingConfig()
    gradient_accumulation_steps = training_config.gradient_accumulation_steps

    print(f"\n1. Configuration:")
    print(f"   Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"   Expected Pass 1 batches: {gradient_accumulation_steps}")
    print(f"   Expected Pass 2 batches: {gradient_accumulation_steps}")

    # Create a simple layer with automatic two-pass
    layer = SPLinearWithLoRA(
        in_features=768,
        out_features=768,
        bit_widths=[4, 8, 16],
        lora_rank_per_bit=config.lora_rank_per_bit,
        lora_alpha_per_bit=config.lora_alpha_per_bit,
        gradient_accumulation_steps=gradient_accumulation_steps
    )

    # Set to 8-bit and training mode
    layer.set_precision(8)
    layer.train()

    # Get the 8-bit quantizer for monitoring
    quantizer = layer.quantizers_weight['8bit']

    print(f"\n2. Initial state:")
    print(f"   Training mode: {layer.training}")
    print(f"   Current precision: {layer.current_bits}-bit")
    print(f"   Quantizer calibrated: {quantizer.calibrated}")
    print(f"   Collecting stats: {quantizer.collecting_stats}")
    print(f"   Stats frozen: {quantizer.stats_frozen}")
    print(f"   Forward counter: {quantizer.forward_counter}")

    # Track scale changes
    scale_history = []

    # Simulate forward passes
    x = torch.randn(2, 10, 768)

    print(f"\n3. Running {2 * gradient_accumulation_steps} forward passes...")

    for i in range(2 * gradient_accumulation_steps):
        # Forward pass
        output = layer(x)

        # Record scale
        current_scale = quantizer.scale.clone().mean().item()
        scale_history.append(current_scale)

        # Print status at key points
        if i == 0:
            print(f"\n   After batch 1 (Pass 1 start):")
            print(f"     Collecting stats: {quantizer.collecting_stats}")
            print(f"     Stats frozen: {quantizer.stats_frozen}")
            print(f"     Forward counter: {quantizer.forward_counter}")

        elif i == gradient_accumulation_steps - 1:
            print(f"\n   After batch {gradient_accumulation_steps} (Pass 1 end):")
            print(f"     Collecting stats: {quantizer.collecting_stats}")
            print(f"     Stats frozen: {quantizer.stats_frozen}")
            print(f"     Calibrated: {quantizer.calibrated}")
            print(f"     Forward counter: {quantizer.forward_counter}")
            print(f"     Scale: {current_scale:.6f}")

        elif i == gradient_accumulation_steps:
            print(f"\n   After batch {gradient_accumulation_steps + 1} (Pass 2 start):")
            print(f"     Collecting stats: {quantizer.collecting_stats}")
            print(f"     Stats frozen: {quantizer.stats_frozen}")
            print(f"     Forward counter: {quantizer.forward_counter}")
            print(f"     Scale: {current_scale:.6f}")

        elif i == 2 * gradient_accumulation_steps - 1:
            print(f"\n   After batch {2 * gradient_accumulation_steps} (Pass 2 end):")
            print(f"     Collecting stats: {quantizer.collecting_stats}")
            print(f"     Stats frozen: {quantizer.stats_frozen}")
            print(f"     Forward counter: {quantizer.forward_counter}")
            print(f"     Scale: {current_scale:.6f}")

    # Check scale stability during Pass 2
    print(f"\n4. Verifying scale stability during Pass 2:")
    pass1_scales = scale_history[:gradient_accumulation_steps]
    pass2_scales = scale_history[gradient_accumulation_steps:]

    # Pass 1 should show 0 or changing scales
    print(f"   Pass 1 scales: {pass1_scales[:3]} ... {pass1_scales[-1:]}")

    # Pass 2 should show constant non-zero scale
    print(f"   Pass 2 scales: {pass2_scales[:3]} ... {pass2_scales[-1:]}")

    # Check if scales were frozen during Pass 2
    pass2_unique = len(set(pass2_scales))
    if pass2_unique == 1 and pass2_scales[0] > 0:
        print(f"   ✅ PASSED: Scale frozen during Pass 2 (value: {pass2_scales[0]:.6f})")
    else:
        print(f"   ❌ FAILED: Scale changed during Pass 2 ({pass2_unique} unique values)")
        return False

    # Run another cycle to verify reset
    print(f"\n5. Testing cycle reset (batch {2 * gradient_accumulation_steps + 1}):")
    output = layer(x)

    print(f"   Collecting stats: {quantizer.collecting_stats}")
    print(f"   Stats frozen: {quantizer.stats_frozen}")
    print(f"   Forward counter: {quantizer.forward_counter}")

    if quantizer.forward_counter == 1 and quantizer.collecting_stats:
        print(f"   ✅ PASSED: Cycle properly reset for new Pass 1")
    else:
        print(f"   ❌ FAILED: Cycle did not reset properly")
        return False

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED: Automatic two-pass working correctly")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_auto_two_pass()
    sys.exit(0 if success else 1)
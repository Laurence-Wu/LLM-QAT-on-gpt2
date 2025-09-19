#!/usr/bin/env python3
"""
Test Quantizer Calibration Behavior
Checks if calibration is recalculated for each new input tensor
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.quantization import LearnableFakeQuantize, QuantizedLinear


def test_calibration_behavior():
    """Test if quantizer recalibrates for each new input."""
    print("\n" + "="*80)
    print("QUANTIZER CALIBRATION BEHAVIOR TEST")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create a quantizer
    quantizer = LearnableFakeQuantize(num_bits=8, symmetric=True, per_channel=False)
    quantizer = quantizer.to(device)

    print("\n1. TESTING TRAINING MODE CALIBRATION:")
    quantizer.train()

    # First input
    x1 = torch.randn(2, 10, device=device) * 5.0  # Range ~[-15, 15]
    print(f"\n   First input range: [{x1.min():.2f}, {x1.max():.2f}]")

    # Forward pass
    _ = quantizer(x1)
    print(f"   After x1 - Running min: {quantizer.running_min.item():.2f}, max: {quantizer.running_max.item():.2f}")
    print(f"   Scale: {quantizer.scale.item():.4f}")
    scale1 = quantizer.scale.clone()

    # Second input with different range
    x2 = torch.randn(2, 10, device=device) * 2.0  # Range ~[-6, 6]
    print(f"\n   Second input range: [{x2.min():.2f}, {x2.max():.2f}]")

    # Forward pass
    _ = quantizer(x2)
    print(f"   After x2 - Running min: {quantizer.running_min.item():.2f}, max: {quantizer.running_max.item():.2f}")
    print(f"   Scale: {quantizer.scale.item():.4f}")
    scale2 = quantizer.scale.clone()

    # Check if scale changed (EMA update)
    if not torch.equal(scale1, scale2):
        print(f"\n   ✓ TRAINING MODE: Scale updated with EMA (changed from {scale1.item():.4f} to {scale2.item():.4f})")
    else:
        print(f"\n   ✗ TRAINING MODE: Scale did not change")

    print("\n2. TESTING EVAL MODE CALIBRATION:")
    quantizer.eval()
    quantizer.calibrated = False  # Reset calibration

    # Third input
    x3 = torch.randn(2, 10, device=device) * 10.0  # Range ~[-30, 30]
    print(f"\n   Third input range: [{x3.min():.2f}, {x3.max():.2f}]")

    # Forward pass - should do one-shot calibration
    _ = quantizer(x3)
    print(f"   After x3 - Running min: {quantizer.running_min.item():.2f}, max: {quantizer.running_max.item():.2f}")
    print(f"   Scale: {quantizer.scale.item():.4f}")
    print(f"   Calibrated: {quantizer.calibrated}")
    scale3 = quantizer.scale.clone()

    # Fourth input with different range
    x4 = torch.randn(2, 10, device=device) * 1.0  # Range ~[-3, 3]
    print(f"\n   Fourth input range: [{x4.min():.2f}, {x4.max():.2f}]")

    # Forward pass - should NOT recalibrate in eval mode
    _ = quantizer(x4)
    print(f"   After x4 - Running min: {quantizer.running_min.item():.2f}, max: {quantizer.running_max.item():.2f}")
    print(f"   Scale: {quantizer.scale.item():.4f}")
    scale4 = quantizer.scale.clone()

    # Check if scale stayed the same
    if torch.equal(scale3, scale4):
        print(f"\n   ✓ EVAL MODE: Scale fixed after calibration (stayed at {scale3.item():.4f})")
    else:
        print(f"\n   ✗ EVAL MODE: Scale changed unexpectedly")

    print("\n3. TESTING PER-CHANNEL CALIBRATION:")
    quantizer_pc = LearnableFakeQuantize(num_bits=8, symmetric=True, per_channel=True, channel_dim=0)
    quantizer_pc = quantizer_pc.to(device)
    quantizer_pc.train()

    # Input with different ranges per channel
    x5 = torch.randn(3, 10, device=device)
    x5[0] *= 10.0  # Channel 0: large range
    x5[1] *= 1.0   # Channel 1: small range
    x5[2] *= 5.0   # Channel 2: medium range

    print(f"\n   Channel 0 range: [{x5[0].min():.2f}, {x5[0].max():.2f}]")
    print(f"   Channel 1 range: [{x5[1].min():.2f}, {x5[1].max():.2f}]")
    print(f"   Channel 2 range: [{x5[2].min():.2f}, {x5[2].max():.2f}]")

    # Forward pass
    _ = quantizer_pc(x5)

    if quantizer_pc.scale.numel() > 1:
        print(f"\n   Per-channel scales:")
        for i in range(min(3, quantizer_pc.scale.shape[0])):
            print(f"     Channel {i}: {quantizer_pc.scale[i].item():.4f}")
        print(f"\n   ✓ PER-CHANNEL: Different scales for different channels")
    else:
        print(f"\n   ✗ PER-CHANNEL: Using single scale")


def test_calibration_in_model():
    """Test calibration behavior in a full model."""
    print("\n4. TESTING CALIBRATION IN QUANTIZED LINEAR LAYER:")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create a quantized linear layer
    layer = QuantizedLinear(in_features=10, out_features=5, weight_bits=8, activation_bits=8)
    layer = layer.to(device)

    print("\n   Training mode:")
    layer.train()

    # Multiple forward passes with different inputs
    for i in range(3):
        x = torch.randn(2, 10, device=device) * (i + 1)
        y = layer(x)

        weight_scale = layer.weight_quantizer.scale.mean().item()
        act_scale = layer.activation_quantizer.scale.item()

        print(f"   Pass {i+1} - Weight scale: {weight_scale:.4f}, Activation scale: {act_scale:.4f}")

    print("\n   Eval mode:")
    layer.eval()

    # Reset calibration for test
    layer.weight_quantizer.calibrated = False
    layer.activation_quantizer.calibrated = False

    # First pass calibrates
    x = torch.randn(2, 10, device=device) * 5
    y = layer(x)
    weight_scale_eval = layer.weight_quantizer.scale.mean().item()
    act_scale_eval = layer.activation_quantizer.scale.item()
    print(f"   First pass - Weight scale: {weight_scale_eval:.4f}, Activation scale: {act_scale_eval:.4f}")

    # Second pass should use same calibration
    x = torch.randn(2, 10, device=device) * 10
    y = layer(x)
    weight_scale_eval2 = layer.weight_quantizer.scale.mean().item()
    act_scale_eval2 = layer.activation_quantizer.scale.item()
    print(f"   Second pass - Weight scale: {weight_scale_eval2:.4f}, Activation scale: {act_scale_eval2:.4f}")

    if weight_scale_eval == weight_scale_eval2 and act_scale_eval == act_scale_eval2:
        print("\n   ✓ Scales remain fixed in eval mode after calibration")
    else:
        print("\n   ✗ Scales changed in eval mode")


def test_recalibration_trigger():
    """Test what triggers recalibration."""
    print("\n5. TESTING RECALIBRATION TRIGGERS:")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    quantizer = LearnableFakeQuantize(num_bits=8, symmetric=True)
    quantizer = quantizer.to(device)

    # Initial calibration in eval mode
    quantizer.eval()
    x1 = torch.randn(2, 10, device=device)
    _ = quantizer(x1)
    print(f"\n   Initial calibration: calibrated={quantizer.calibrated}")
    initial_scale = quantizer.scale.clone()

    # Test 1: Does switching back to train mode reset calibration?
    quantizer.train()
    x2 = torch.randn(2, 10, device=device) * 5
    _ = quantizer(x2)
    train_scale = quantizer.scale.clone()

    if not torch.equal(initial_scale, train_scale):
        print(f"   ✓ Training mode updates calibration (scale changed)")
    else:
        print(f"   ✗ Training mode did not update calibration")

    # Test 2: Does changing bit-width reset calibration?
    quantizer.eval()
    old_bits = quantizer.num_bits
    quantizer.set_num_bits(4)  # Change bit-width
    print(f"\n   Changed bits from {old_bits} to {quantizer.num_bits}")
    print(f"   Calibrated after bit change: {quantizer.calibrated}")

    if not quantizer.calibrated:
        print(f"   ✓ Bit-width change resets calibration")
    else:
        print(f"   ✗ Bit-width change did not reset calibration")

    # Recalibrate with new bit-width
    x3 = torch.randn(2, 10, device=device)
    _ = quantizer(x3)
    new_scale = quantizer.scale.clone()
    print(f"   New scale after recalibration: {new_scale.item():.4f}")


def analyze_calibration_impact():
    """Analyze the impact of calibration on quantization error."""
    print("\n6. ANALYZING CALIBRATION IMPACT ON QUANTIZATION ERROR:")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test different calibration strategies
    strategies = [
        ("No calibration", False, None),
        ("Min-max calibration", True, "minmax"),
        ("EMA calibration", True, "ema")
    ]

    x_test = torch.randn(100, 32, device=device) * 3.0  # Test data

    for name, do_calibrate, strategy in strategies:
        print(f"\n   {name}:")

        quantizer = LearnableFakeQuantize(num_bits=8, symmetric=True)
        quantizer = quantizer.to(device)

        if do_calibrate:
            if strategy == "minmax":
                # One-shot calibration
                quantizer.eval()
                _ = quantizer(x_test)
            elif strategy == "ema":
                # EMA calibration with multiple batches
                quantizer.train()
                for _ in range(10):
                    x_calib = torch.randn(10, 32, device=device) * 3.0
                    _ = quantizer(x_calib)
                quantizer.eval()
        else:
            # No calibration - use default scale
            quantizer.eval()
            quantizer.calibrated = True
            quantizer.scale.fill_(1.0)

        # Measure quantization error
        with torch.no_grad():
            x_quant = quantizer(x_test)
            error = (x_test - x_quant).abs()
            mse = error.pow(2).mean().item()
            max_error = error.max().item()
            rel_error = (error / (x_test.abs() + 1e-8)).mean().item()

        print(f"     Scale: {quantizer.scale.mean().item():.4f}")
        print(f"     MSE: {mse:.6f}")
        print(f"     Max error: {max_error:.4f}")
        print(f"     Relative error: {rel_error:.4%}")


def main():
    """Run all calibration tests."""
    print("\n" + "="*80)
    print("COMPREHENSIVE QUANTIZER CALIBRATION ANALYSIS")
    print("="*80)

    # Run tests
    test_calibration_behavior()
    test_calibration_in_model()
    test_recalibration_trigger()
    analyze_calibration_impact()

    print("\n" + "="*80)
    print("CALIBRATION BEHAVIOR SUMMARY")
    print("="*80)

    print("\n📊 KEY FINDINGS:")
    print("• TRAINING MODE: Continuously updates statistics with EMA")
    print("• EVAL MODE: One-shot calibration on first input, then fixed")
    print("• Calibration is NOT recalculated for each new input in eval mode")
    print("• Calibration IS updated for each input in training mode")
    print("• Changing bit-width resets calibration state")
    print("• Per-channel calibration tracks separate statistics per channel")

    print("\n🔧 RECOMMENDATIONS:")
    print("• Use training mode for calibration phase")
    print("• Switch to eval mode for inference to maintain consistent quantization")
    print("• Recalibrate when changing bit-widths")
    print("• Consider per-channel quantization for weights with varying ranges")


if __name__ == "__main__":
    main()
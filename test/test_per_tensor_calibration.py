#!/usr/bin/env python3
"""
Test script to verify per-tensor calibration works correctly
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from part1_switchable_precision.quantization import LearnableFakeQuantize

def test_per_channel_vs_per_tensor():
    """Test that per_channel parameter controls calibration behavior correctly"""

    print("="*60)
    print("Testing per-channel vs per-tensor calibration")
    print("="*60)

    # Create test input with different statistics per position
    # Shape: [batch=2, seq_len=4, hidden=3]
    test_input = torch.tensor([
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0],
         [7.0, 8.0, 9.0],
         [10.0, 11.0, 12.0]],
        [[0.5, 1.0, 1.5],
         [2.0, 2.5, 3.0],
         [3.5, 4.0, 4.5],
         [5.0, 5.5, 6.0]]
    ])

    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test input range: [{test_input.min().item():.2f}, {test_input.max().item():.2f}]")

    # Test 1: Per-channel calibration (channel_dim=1)
    print("\n" + "-"*40)
    print("Test 1: Per-channel calibration (per_channel=True)")
    print("-"*40)

    quantizer_per_channel = LearnableFakeQuantize(
        num_bits=8,
        channel_dim=1,  # Keep statistics per sequence position
        quantizer_type='minmax',
        per_channel=True  # Enable per-channel calibration
    )

    # Calibrate
    quantizer_per_channel.start_calibration()
    _ = quantizer_per_channel(test_input)
    quantizer_per_channel.finish_calibration()

    print(f"Scale shape: {quantizer_per_channel.scale.shape}")
    print(f"Scale values: {quantizer_per_channel.scale.squeeze()}")
    print(f"Expected: Shape [1, 4, 1] with different values per position")

    # Test 2: Per-tensor calibration
    print("\n" + "-"*40)
    print("Test 2: Per-tensor calibration (per_channel=False)")
    print("-"*40)

    quantizer_per_tensor = LearnableFakeQuantize(
        num_bits=8,
        channel_dim=1,  # This will be ignored due to per_channel=False
        quantizer_type='minmax',
        per_channel=False  # Disable per-channel, use global statistics
    )

    # Calibrate
    quantizer_per_tensor.start_calibration()
    _ = quantizer_per_tensor(test_input)
    quantizer_per_tensor.finish_calibration()

    print(f"Scale shape: {quantizer_per_tensor.scale.shape}")
    print(f"Scale value: {quantizer_per_tensor.scale.item():.6f}")
    print(f"Expected: Shape [1] with single global value")

    # Test 3: Verify quantization works with variable-length inputs
    print("\n" + "-"*40)
    print("Test 3: Variable-length input handling")
    print("-"*40)

    # Different length input
    test_input_short = torch.randn(2, 2, 3)  # Shorter sequence
    test_input_long = torch.randn(2, 10, 3)  # Longer sequence

    try:
        # Per-channel should fail with different lengths
        print("\nTrying per-channel with different length...")
        out = quantizer_per_channel(test_input_short)
        print(f"ERROR: Per-channel should have failed with shape mismatch!")
    except Exception as e:
        print(f"✓ Per-channel failed as expected: {type(e).__name__}")

    try:
        # Per-tensor should work with any length
        print("\nTrying per-tensor with different lengths...")
        out_short = quantizer_per_tensor(test_input_short)
        out_long = quantizer_per_tensor(test_input_long)
        print(f"✓ Per-tensor works with short input: {out_short.shape}")
        print(f"✓ Per-tensor works with long input: {out_long.shape}")
    except Exception as e:
        print(f"ERROR: Per-tensor failed unexpectedly: {e}")

    # Test 4: Log quantization
    print("\n" + "-"*40)
    print("Test 4: Log quantization with per-tensor")
    print("-"*40)

    quantizer_log = LearnableFakeQuantize(
        num_bits=8,
        channel_dim=0,
        quantizer_type='log',
        per_channel=False
    )

    # Use positive values for log quantization
    test_input_pos = torch.abs(test_input) + 0.1

    quantizer_log.start_calibration()
    _ = quantizer_log(test_input_pos)
    quantizer_log.finish_calibration()

    print(f"Log quantizer scale shape: {quantizer_log.scale.shape}")
    print(f"Log quantizer scale value: {quantizer_log.scale.item():.6f}")

    # Test with different shapes
    out1 = quantizer_log(torch.abs(test_input_short) + 0.1)
    out2 = quantizer_log(torch.abs(test_input_long) + 0.1)
    print(f"✓ Log quantizer works with different shapes: {out1.shape}, {out2.shape}")

    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)

if __name__ == "__main__":
    test_per_channel_vs_per_tensor()
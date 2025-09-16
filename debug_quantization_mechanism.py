"""
Debug if quantization is actually working correctly.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.quantization import LearnableFakeQuantize, QuantizationFunction


def test_basic_quantization():
    """Test if basic quantization is working."""

    print("="*60)
    print("TESTING BASIC QUANTIZATION MECHANISM")
    print("="*60)

    # Test data
    x = torch.randn(1, 100)
    print(f"\nOriginal tensor:")
    print(f"  Shape: {x.shape}")
    print(f"  Unique values: {len(torch.unique(x))}")
    print(f"  Min: {x.min().item():.6f}")
    print(f"  Max: {x.max().item():.6f}")
    print(f"  Mean: {x.mean().item():.6f}")
    print(f"  Std: {x.std().item():.6f}")

    # Test symmetric quantizer
    print("\n" + "-"*40)
    print("SYMMETRIC QUANTIZATION (8-bit)")
    print("-"*40)

    sym_quantizer = LearnableFakeQuantize(num_bits=8, symmetric=True)
    sym_quantizer.eval()

    print(f"Initial state:")
    print(f"  Scale: {sym_quantizer.scale.item():.6f}")
    print(f"  Zero point: {sym_quantizer.zero_point.item():.6f}")
    print(f"  Running min: {sym_quantizer.running_min.item():.6f}")
    print(f"  Running max: {sym_quantizer.running_max.item():.6f}")
    print(f"  Calibrated: {sym_quantizer.calibrated}")
    print(f"  Quant range: [{sym_quantizer.quant_min}, {sym_quantizer.quant_max}]")

    # Try quantizing without calibration
    x_quant_uncalib = sym_quantizer(x)
    print(f"\nWithout calibration:")
    print(f"  Unique values after quantization: {len(torch.unique(x_quant_uncalib))}")
    print(f"  Min: {x_quant_uncalib.min().item():.6f}")
    print(f"  Max: {x_quant_uncalib.max().item():.6f}")
    print(f"  All zeros? {torch.all(x_quant_uncalib == 0).item()}")

    # Manually set scale and zero_point for testing
    print("\nManually setting scale based on data range...")
    with torch.no_grad():
        data_max = max(abs(x.min()), abs(x.max()))
        sym_quantizer.scale.fill_(data_max / 127)  # For 8-bit symmetric
        sym_quantizer.zero_point.fill_(0)
        sym_quantizer.running_min.fill_(-data_max)
        sym_quantizer.running_max.fill_(data_max)

    print(f"After manual calibration:")
    print(f"  Scale: {sym_quantizer.scale.item():.6f}")
    print(f"  Zero point: {sym_quantizer.zero_point.item():.6f}")

    x_quant_manual = sym_quantizer(x)
    print(f"\nWith manual calibration:")
    print(f"  Unique values after quantization: {len(torch.unique(x_quant_manual))}")
    print(f"  Min: {x_quant_manual.min().item():.6f}")
    print(f"  Max: {x_quant_manual.max().item():.6f}")
    print(f"  Mean abs diff: {(x - x_quant_manual).abs().mean().item():.6f}")

    # Test asymmetric quantizer
    print("\n" + "-"*40)
    print("ASYMMETRIC QUANTIZATION (8-bit)")
    print("-"*40)

    asym_quantizer = LearnableFakeQuantize(num_bits=8, symmetric=False)
    asym_quantizer.eval()

    print(f"Initial state:")
    print(f"  Scale: {asym_quantizer.scale.item():.6f}")
    print(f"  Zero point: {asym_quantizer.zero_point.item():.6f}")
    print(f"  Running min: {asym_quantizer.running_min.item():.6f}")
    print(f"  Running max: {asym_quantizer.running_max.item():.6f}")
    print(f"  Calibrated: {asym_quantizer.calibrated}")
    print(f"  Quant range: [{asym_quantizer.quant_min}, {asym_quantizer.quant_max}]")

    # Without calibration
    x_quant_asym_uncalib = asym_quantizer(x)
    print(f"\nWithout calibration:")
    print(f"  Unique values: {len(torch.unique(x_quant_asym_uncalib))}")
    print(f"  All zeros? {torch.all(x_quant_asym_uncalib == 0).item()}")

    # Manual calibration
    print("\nManually calibrating...")
    with torch.no_grad():
        asym_quantizer.running_min.fill_(x.min().item())
        asym_quantizer.running_max.fill_(x.max().item())
        data_range = x.max().item() - x.min().item()
        asym_quantizer.scale.fill_(data_range / 255)  # For 8-bit asymmetric
        asym_quantizer.zero_point.fill_(round(-x.min().item() / asym_quantizer.scale.item()))

    print(f"After manual calibration:")
    print(f"  Scale: {asym_quantizer.scale.item():.6f}")
    print(f"  Zero point: {asym_quantizer.zero_point.item():.6f}")
    print(f"  Running min: {asym_quantizer.running_min.item():.6f}")
    print(f"  Running max: {asym_quantizer.running_max.item():.6f}")

    x_quant_asym_manual = asym_quantizer(x)
    print(f"\nWith manual calibration:")
    print(f"  Unique values: {len(torch.unique(x_quant_asym_manual))}")
    print(f"  Min: {x_quant_asym_manual.min().item():.6f}")
    print(f"  Max: {x_quant_asym_manual.max().item():.6f}")
    print(f"  Mean abs diff: {(x - x_quant_asym_manual).abs().mean().item():.6f}")


def test_quantization_function_directly():
    """Test the QuantizationFunction directly."""

    print("\n" + "="*60)
    print("TESTING QUANTIZATION FUNCTION DIRECTLY")
    print("="*60)

    x = torch.randn(1, 100)

    # Test with proper scale and zero_point
    scale = torch.tensor([0.02])  # Reasonable scale for [-3, 3] range
    zero_point = torch.tensor([128.0])  # Middle of [0, 255] range

    print(f"\nInput stats:")
    print(f"  Unique values: {len(torch.unique(x))}")
    print(f"  Range: [{x.min().item():.3f}, {x.max().item():.3f}]")

    print(f"\nQuantization params:")
    print(f"  Scale: {scale.item()}")
    print(f"  Zero point: {zero_point.item()}")

    # Test symmetric
    x_quant_sym = QuantizationFunction.apply(x, scale, torch.tensor([0.0]), 8, True)
    print(f"\nSymmetric quantization (8-bit):")
    print(f"  Unique values: {len(torch.unique(x_quant_sym))}")
    print(f"  Range: [{x_quant_sym.min().item():.3f}, {x_quant_sym.max().item():.3f}]")
    print(f"  Diff: {(x - x_quant_sym).abs().mean().item():.6f}")

    # Test asymmetric
    x_quant_asym = QuantizationFunction.apply(x, scale, zero_point, 8, False)
    print(f"\nAsymmetric quantization (8-bit):")
    print(f"  Unique values: {len(torch.unique(x_quant_asym))}")
    print(f"  Range: [{x_quant_asym.min().item():.3f}, {x_quant_asym.max().item():.3f}]")
    print(f"  Diff: {(x - x_quant_asym).abs().mean().item():.6f}")

    # Test with zero scale (should cause issues)
    print("\nTesting with zero/unit scale:")
    bad_scale = torch.tensor([1.0])
    bad_zero = torch.tensor([0.0])
    x_quant_bad = QuantizationFunction.apply(x, bad_scale, bad_zero, 8, False)
    print(f"  Unique values: {len(torch.unique(x_quant_bad))}")
    print(f"  Range: [{x_quant_bad.min().item():.3f}, {x_quant_bad.max().item():.3f}]")


def test_training_mode_calibration():
    """Test if training mode properly calibrates."""

    print("\n" + "="*60)
    print("TESTING TRAINING MODE CALIBRATION")
    print("="*60)

    quantizer = LearnableFakeQuantize(num_bits=8, symmetric=False)
    quantizer.train()  # Put in training mode

    print("Initial state (training mode):")
    print(f"  Calibrated: {quantizer.calibrated}")
    print(f"  Scale: {quantizer.scale.item():.6f}")
    print(f"  Running min: {quantizer.running_min.item():.6f}")
    print(f"  Running max: {quantizer.running_max.item():.6f}")

    # Pass some data through
    for i in range(10):
        x = torch.randn(1, 100) * 2  # Varied data
        _ = quantizer(x)

        if i == 0:
            print(f"\nAfter first batch:")
            print(f"  Calibrated: {quantizer.calibrated}")
            print(f"  Scale: {quantizer.scale.item():.6f}")
            print(f"  Running min: {quantizer.running_min.item():.6f}")
            print(f"  Running max: {quantizer.running_max.item():.6f}")

    print(f"\nAfter 10 batches:")
    print(f"  Calibrated: {quantizer.calibrated}")
    print(f"  Scale: {quantizer.scale.item():.6f}")
    print(f"  Running min: {quantizer.running_min.item():.6f}")
    print(f"  Running max: {quantizer.running_max.item():.6f}")

    # Now test in eval mode
    quantizer.eval()
    test_data = torch.randn(1, 100)
    quant_data = quantizer(test_data)

    print(f"\nEval mode quantization:")
    print(f"  Input unique values: {len(torch.unique(test_data))}")
    print(f"  Output unique values: {len(torch.unique(quant_data))}")
    print(f"  Mean diff: {(test_data - quant_data).abs().mean().item():.6f}")


if __name__ == "__main__":
    test_basic_quantization()
    test_quantization_function_directly()
    test_training_mode_calibration()
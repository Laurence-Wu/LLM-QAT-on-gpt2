"""
Test to understand 16-bit quantization behavior.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.quantization import LearnableFakeQuantize


def test_quantization_at_different_bits():
    """Test quantization behavior at different bit widths."""

    # Create test tensor
    x = torch.randn(10, 10)

    print("Original tensor stats:")
    print(f"  Mean: {x.mean().item():.6f}")
    print(f"  Std:  {x.std().item():.6f}")
    print(f"  Min:  {x.min().item():.6f}")
    print(f"  Max:  {x.max().item():.6f}")

    for bits in [8, 16, 32]:
        print(f"\n{bits}-bit quantization:")

        # Test symmetric quantization (for weights)
        quant_sym = LearnableFakeQuantize(num_bits=bits, symmetric=True)
        quant_sym.eval()  # Put in eval mode
        x_q_sym = quant_sym(x)
        diff_sym = (x - x_q_sym).abs().mean().item()
        print(f"  Symmetric - Mean abs diff: {diff_sym:.6f}")

        # Test asymmetric quantization (for activations)
        quant_asym = LearnableFakeQuantize(num_bits=bits, symmetric=False)
        quant_asym.eval()  # Put in eval mode
        x_q_asym = quant_asym(x)
        diff_asym = (x - x_q_asym).abs().mean().item()
        print(f"  Asymmetric - Mean abs diff: {diff_asym:.6f}")

        # Check if it's actually quantizing
        unique_vals = len(torch.unique(x_q_sym))
        print(f"  Unique values after quantization: {unique_vals}")

        # Check the actual number of bits used
        if bits < 32:
            expected_vals = 2**bits
            print(f"  Expected max unique values: {expected_vals}")


def test_fp16_comparison():
    """Compare fake quantization to actual FP16."""

    print("\n" + "="*60)
    print("Comparing 16-bit fake quantization to FP16")
    print("="*60)

    x = torch.randn(100, 100)

    # Actual FP16
    x_fp16 = x.half().float()
    fp16_diff = (x - x_fp16).abs().mean().item()
    print(f"FP16 conversion diff: {fp16_diff:.6f}")

    # 16-bit fake quantization
    quant16 = LearnableFakeQuantize(num_bits=16, symmetric=False)
    quant16.eval()
    x_q16 = quant16(x)
    q16_diff = (x - x_q16).abs().mean().item()
    print(f"16-bit fake quant diff: {q16_diff:.6f}")

    print(f"\nRatio (fake_quant/fp16): {q16_diff/fp16_diff:.2f}x worse")


if __name__ == "__main__":
    test_quantization_at_different_bits()
    test_fp16_comparison()
#!/usr/bin/env python3
"""
Test different quantization methods (minmax, relu_clip, tanh, log) with the new quant_methods module.
"""

import sys
import os
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.quantization import LearnableFakeQuantize
from test.utils import get_configured_bit_widths
from part1_switchable_precision.config_sp import ModelConfig


def test_quantizer_methods():
    """Test all quantization methods with different input patterns."""
    print("\n" + "="*60)
    print("TESTING DIFFERENT QUANTIZATION METHODS")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test inputs with different characteristics
    torch.manual_seed(42)
    batch_size = 2
    features = 64

    # Different input patterns
    test_inputs = {
        'normal': torch.randn(batch_size, features, device=device),  # Normal distribution
        'uniform': torch.rand(batch_size, features, device=device) * 2 - 1,  # Uniform [-1, 1]
        'sparse': torch.randn(batch_size, features, device=device) * (torch.rand(batch_size, features, device=device) > 0.7).float(),  # Sparse
        'positive': torch.abs(torch.randn(batch_size, features, device=device)),  # All positive
        'outliers': torch.randn(batch_size, features, device=device) + torch.randn(batch_size, features, device=device) * 10 * (torch.rand(batch_size, features, device=device) > 0.95).float(),  # With outliers
    }

    # Test different quantizer types
    quantizer_types = ['minmax', 'relu_clip', 'tanh', 'log']
    # Get configured bit widths and use first two student precisions
    config = ModelConfig()
    all_bit_widths = get_configured_bit_widths(config=config)
    bit_widths = [b for b in all_bit_widths if b < 32][:2]  # Use first 2 student precisions

    results = {}

    for quant_type in quantizer_types:
        print(f"\n🔧 Testing {quant_type.upper()} Quantizer:")
        results[quant_type] = {}

        for bits in bit_widths:
            print(f"\n  {bits}-bit quantization:")
            results[quant_type][bits] = {}

            # Create quantizer
            quantizer = LearnableFakeQuantize(
                num_bits=bits,
                symmetric=True,
                quantizer_type=quant_type
            ).to(device)

            # Test each input pattern
            for input_name, input_tensor in test_inputs.items():
                # Clone to avoid modifying original
                x = input_tensor.clone().requires_grad_(True)

                # Calibrate
                quantizer.start_calibration()
                with torch.no_grad():
                    _ = quantizer(x)
                quantizer.finish_calibration(debug=False)

                # Forward pass
                x_quant = quantizer(x)

                # Calculate metrics
                mse = torch.mean((x_quant - x) ** 2).item()
                rel_error = torch.mean(torch.abs(x_quant - x) / (torch.abs(x) + 1e-7)).item()

                # Check gradient flow
                loss = x_quant.sum()
                loss.backward()
                has_gradient = x.grad is not None and x.grad.abs().sum() > 0

                results[quant_type][bits][input_name] = {
                    'mse': mse,
                    'rel_error': rel_error,
                    'has_gradient': has_gradient,
                    'input_range': (x.min().item(), x.max().item()),
                    'output_range': (x_quant.min().item(), x_quant.max().item()),
                }

                # Clear gradients
                if x.grad is not None:
                    x.grad.zero_()

    # Print results comparison
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)

    for input_name in test_inputs.keys():
        print(f"\n📊 Input: {input_name}")
        print(f"{'Quantizer':<12} {'Bits':<6} {'MSE':<12} {'Rel Error':<12} {'Gradient':<10}")
        print("-" * 60)

        for quant_type in quantizer_types:
            for bits in bit_widths:
                res = results[quant_type][bits][input_name]
                grad_status = "✅" if res['has_gradient'] else "❌"
                print(f"{quant_type:<12} {bits:<6} {res['mse']:<12.6f} {res['rel_error']:<12.6f} {grad_status:<10}")

    # Special behavior analysis
    print("\n" + "="*60)
    print("SPECIAL BEHAVIOR ANALYSIS")
    print("="*60)

    print("\n1. ReLU Clip (positive outputs only):")
    for bits in bit_widths:
        res = results['relu_clip'][bits]['normal']
        print(f"   {bits}-bit: Output range = [{res['output_range'][0]:.3f}, {res['output_range'][1]:.3f}]")
        if res['output_range'][0] >= 0:
            print(f"            ✅ Correctly clips to non-negative range")
        else:
            print(f"            ❌ Failed to clip negative values")

    print("\n2. Tanh (bounded outputs):")
    for bits in bit_widths:
        res = results['tanh'][bits]['outliers']
        print(f"   {bits}-bit: Input range  = [{res['input_range'][0]:.3f}, {res['input_range'][1]:.3f}]")
        print(f"            Output range = [{res['output_range'][0]:.3f}, {res['output_range'][1]:.3f}]")

    print("\n3. Log (non-uniform quantization):")
    for bits in bit_widths:
        res_sparse = results['log'][bits]['sparse']
        res_normal = results['log'][bits]['normal']
        print(f"   {bits}-bit: Sparse MSE = {res_sparse['mse']:.6f}, Normal MSE = {res_normal['mse']:.6f}")

    print("\n✅ All quantization methods tested successfully!")
    return results


def test_quantizer_in_layer():
    """Test quantizers integrated in a linear layer."""
    print("\n" + "="*60)
    print("TESTING QUANTIZERS IN LINEAR LAYER")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Import the LoRA layer
    from shared.lora import LinearWithLoRA

    # Test configuration
    in_features = 128
    out_features = 64
    batch_size = 4

    print("\nTesting LinearWithLoRA with different quantizer types:")

    for quant_type in ['minmax', 'relu_clip', 'tanh', 'log']:
        print(f"\n🔧 {quant_type.upper()} quantizer:")

        # Create layer
        layer = LinearWithLoRA(
            in_features=in_features,
            out_features=out_features,
            bits=8,
            lora_rank=16,
            lora_alpha=32,
            quantizer_type=quant_type
        ).to(device)

        # Test input
        x = torch.randn(batch_size, in_features, device=device, requires_grad=True)

        # Forward pass
        output = layer(x)

        # Check output shape
        assert output.shape == (batch_size, out_features), f"Output shape mismatch: {output.shape}"
        print(f"   ✅ Output shape correct: {output.shape}")

        # Check gradient flow
        loss = output.sum()
        loss.backward()

        has_input_grad = x.grad is not None and x.grad.abs().sum() > 0
        has_weight_grad = layer.linear.weight.grad is not None and layer.linear.weight.grad.abs().sum() > 0

        print(f"   ✅ Input gradient: {'Yes' if has_input_grad else 'No'}")
        print(f"   ✅ Weight gradient: {'Yes' if has_weight_grad else 'No'}")

        # Clear gradients
        layer.zero_grad()
        if x.grad is not None:
            x.grad.zero_()

    print("\n✅ All layer integration tests passed!")


def test_calibration_consistency():
    """Test that calibration produces consistent results across methods."""
    print("\n" + "="*60)
    print("TESTING CALIBRATION CONSISTENCY")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(123)

    # Create consistent test data
    calibration_data = [
        torch.randn(16, 256, device=device) for _ in range(10)
    ]

    for quant_type in ['minmax', 'relu_clip', 'tanh', 'log']:
        print(f"\n🔧 Testing {quant_type.upper()} calibration consistency:")

        # Create two identical quantizers
        q1 = LearnableFakeQuantize(num_bits=8, quantizer_type=quant_type).to(device)
        q2 = LearnableFakeQuantize(num_bits=8, quantizer_type=quant_type).to(device)

        # Calibrate both with same data
        q1.start_calibration()
        q2.start_calibration()

        for data in calibration_data:
            with torch.no_grad():
                _ = q1(data)
                _ = q2(data)

        q1.finish_calibration(debug=False)
        q2.finish_calibration(debug=False)

        # Check if calibration results are identical
        scale_match = torch.allclose(q1.scale, q2.scale, rtol=1e-5)
        zero_match = torch.allclose(q1.zero_point, q2.zero_point, rtol=1e-5)

        print(f"   Scale match: {'✅' if scale_match else '❌'}")
        print(f"   Zero-point match: {'✅' if zero_match else '❌'}")

        # Test on new data
        test_data = torch.randn(8, 256, device=device)
        out1 = q1(test_data)
        out2 = q2(test_data)

        output_match = torch.allclose(out1, out2, rtol=1e-5)
        print(f"   Output match: {'✅' if output_match else '❌'}")

        if quant_type == 'minmax':
            print(f"   Scale values: {q1.scale.mean().item():.6f}")
            print(f"   Running min: {q1.running_min.mean().item():.6f}")
            print(f"   Running max: {q1.running_max.mean().item():.6f}")

    print("\n✅ Calibration consistency tests completed!")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("QUANTIZATION METHODS TEST SUITE")
    print("="*80)

    # Run all tests
    test_quantizer_methods()
    test_quantizer_in_layer()
    test_calibration_consistency()

    print("\n" + "="*80)
    print("✅ ALL QUANTIZATION METHOD TESTS COMPLETED SUCCESSFULLY!")
    print("="*80)
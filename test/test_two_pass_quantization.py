"""
Test two-pass quantization implementation for switchable precision training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from shared.quantization import LearnableFakeQuantize
from shared.lora import SPLinearWithLoRA


def test_two_pass_quantizer():
    """Test basic two-pass quantizer functionality."""
    print("\n=== Testing Two-Pass Quantizer ===")

    # Create quantizer
    quantizer = LearnableFakeQuantize(num_bits=4, symmetric=True)

    # Create test data
    x1 = torch.randn(2, 8, 16)
    x2 = torch.randn(2, 8, 16) * 2  # Different scale
    x3 = torch.randn(2, 8, 16) * 0.5

    # Pass 1: Collect statistics
    print("\nPass 1: Collecting statistics...")
    quantizer.start_stats_collection()

    # Process multiple batches
    _ = quantizer(x1)
    print(f"  After batch 1: num_batches = {quantizer.num_batches_collected}")

    _ = quantizer(x2)
    print(f"  After batch 2: num_batches = {quantizer.num_batches_collected}")

    _ = quantizer(x3)
    print(f"  After batch 3: num_batches = {quantizer.num_batches_collected}")

    # Finalize statistics
    quantizer.stop_stats_collection()
    print(f"  Stats frozen: {quantizer.stats_frozen}")
    print(f"  Scale computed: {quantizer.scale.mean().item():.6f}")

    # Pass 2: Apply quantization with fixed parameters
    print("\nPass 2: Applying quantization with fixed parameters...")
    scale_before = quantizer.scale.clone()

    y1 = quantizer(x1)
    y2 = quantizer(x2)
    y3 = quantizer(x3)

    scale_after = quantizer.scale.clone()

    # Verify scale didn't change during Pass 2
    scale_unchanged = torch.allclose(scale_before, scale_after)
    print(f"  Scale unchanged during Pass 2: {scale_unchanged}")

    # Unfreeze for next iteration
    quantizer.unfreeze_stats()
    print(f"  Stats unfrozen: {not quantizer.stats_frozen}")

    print("\n✅ Two-pass quantizer test passed!" if scale_unchanged else "❌ Test failed!")
    return scale_unchanged


def test_sp_linear_two_pass():
    """Test SPLinearWithLoRA with two-pass quantization."""
    print("\n=== Testing SPLinearWithLoRA Two-Pass ===")

    # Create layer
    layer = SPLinearWithLoRA(
        in_features=32,
        out_features=16,
        bit_widths=[4, 8, 16],
        lora_rank_per_bit={4: 4, 8: 8, 16: 0}
    )

    # Test data
    x1 = torch.randn(4, 32)
    x2 = torch.randn(4, 32) * 1.5

    # Test 4-bit mode
    layer.set_precision(4)
    print(f"\nTesting 4-bit mode...")

    # Pass 1: Collect statistics
    layer.start_stats_collection()
    with torch.no_grad():
        _ = layer(x1)
        _ = layer(x2)
    layer.stop_stats_collection()

    # Get scale after statistics collection
    weight_scale = layer.quantizers_weight['4bit'].scale.mean().item()
    input_scale = layer.quantizers_input['4bit'].scale.mean().item()
    print(f"  Weight scale: {weight_scale:.6f}")
    print(f"  Input scale: {input_scale:.6f}")

    # Pass 2: Forward and backward with fixed parameters
    layer.train()
    y = layer(x1)
    loss = y.sum()
    loss.backward()

    # Verify gradients exist
    has_gradients = layer.linear.weight.grad is not None
    print(f"  Gradients computed: {has_gradients}")

    # Unfreeze
    layer.unfreeze_stats()

    # Test 8-bit mode
    layer.set_precision(8)
    print(f"\nTesting 8-bit mode...")

    # Pass 1: Collect statistics
    layer.start_stats_collection()
    with torch.no_grad():
        _ = layer(x1)
        _ = layer(x2)
    layer.stop_stats_collection()

    # Get scale after statistics collection
    weight_scale = layer.quantizers_weight['8bit'].scale.mean().item()
    input_scale = layer.quantizers_input['8bit'].scale.mean().item()
    print(f"  Weight scale: {weight_scale:.6f}")
    print(f"  Input scale: {input_scale:.6f}")

    # Test 16-bit mode (should bypass two-pass)
    layer.set_precision(16)
    print(f"\nTesting 16-bit mode (should bypass)...")

    # These should be no-ops for 16-bit
    layer.start_stats_collection()
    y = layer(x1)
    layer.stop_stats_collection()

    print(f"  16-bit forward completed (no quantization)")

    print("\n✅ SPLinearWithLoRA two-pass test passed!" if has_gradients else "❌ Test failed!")
    return has_gradients


def test_gradient_consistency():
    """Test that gradients are consistent with fixed quantization parameters."""
    print("\n=== Testing Gradient Consistency ===")

    # Create a simple model
    model = nn.Sequential(
        SPLinearWithLoRA(16, 32, bit_widths=[4, 8, 16]),
        nn.ReLU(),
        SPLinearWithLoRA(32, 16, bit_widths=[4, 8, 16])
    )

    # Set to 4-bit
    for layer in model:
        if isinstance(layer, SPLinearWithLoRA):
            layer.set_precision(4)

    # Test data
    x = torch.randn(8, 16, requires_grad=True)
    target = torch.randn(8, 16)

    # Pass 1: Collect statistics
    print("\nPass 1: Collecting statistics...")
    for layer in model:
        if isinstance(layer, SPLinearWithLoRA):
            layer.start_stats_collection()

    with torch.no_grad():
        _ = model(x)

    for layer in model:
        if isinstance(layer, SPLinearWithLoRA):
            layer.stop_stats_collection()

    # Pass 2: Compute gradients with fixed parameters
    print("Pass 2: Computing gradients...")
    model.zero_grad()

    output1 = model(x)
    loss1 = nn.MSELoss()(output1, target)
    loss1.backward(retain_graph=True)

    # Save gradients
    grad1 = x.grad.clone()

    # Compute again with same fixed parameters
    model.zero_grad()
    x.grad = None

    output2 = model(x)
    loss2 = nn.MSELoss()(output2, target)
    loss2.backward()

    grad2 = x.grad.clone()

    # Check consistency
    gradients_consistent = torch.allclose(grad1, grad2, rtol=1e-5)
    print(f"  Gradients consistent: {gradients_consistent}")
    print(f"  Max gradient difference: {(grad1 - grad2).abs().max().item():.2e}")

    # Unfreeze
    for layer in model:
        if isinstance(layer, SPLinearWithLoRA):
            layer.unfreeze_stats()

    print("\n✅ Gradient consistency test passed!" if gradients_consistent else "❌ Test failed!")
    return gradients_consistent


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Two-Pass Quantization Implementation")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run tests
    results = []
    results.append(("Basic Two-Pass", test_two_pass_quantizer()))
    results.append(("SPLinearWithLoRA", test_sp_linear_two_pass()))
    results.append(("Gradient Consistency", test_gradient_consistency()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30} {status}")

    all_passed = all(r[1] for r in results)
    print("\n" + ("✅ ALL TESTS PASSED!" if all_passed else "❌ SOME TESTS FAILED!"))

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
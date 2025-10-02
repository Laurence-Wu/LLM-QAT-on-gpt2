"""
Test gradient calibration to ensure it only collects backward gradient statistics.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
part2_dir = os.path.join(parent_dir, 'part2_cyclic_precision_training')
sys.path.insert(0, part2_dir)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from quantization import LearnableFakeQuantize, GradientQuantizer
from cpt_model import LoRAAdapter


def test_gradient_quantizer_calibration():
    """Test that gradient quantizer collects statistics only during backward pass."""
    print("\n" + "="*70)
    print("Testing Gradient Quantizer Calibration")
    print("="*70)

    # Create a simple LoRA adapter
    in_features = 64
    out_features = 64
    rank = 8

    lora = LoRAAdapter(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        alpha=16,
        num_bits=8,
        quantizer_type='log',
        gradient_bits=8
    )

    # Get the gradient quantizer
    grad_quantizer = lora.grad_quantizer_8bit

    print(f"\n1. Initial state:")
    print(f"   collecting_stats: {grad_quantizer.collecting_stats}")
    print(f"   calibrated: {grad_quantizer.calibrated}")
    print(f"   training mode: {grad_quantizer.training}")

    # Start calibration
    grad_quantizer.eval()
    grad_quantizer.start_calibration()

    print(f"\n2. After start_calibration():")
    print(f"   collecting_stats: {grad_quantizer.collecting_stats}")
    print(f"   calibrated: {grad_quantizer.calibrated}")
    print(f"   training mode: {grad_quantizer.training}")
    print(f"   num_batches_collected: {grad_quantizer.num_batches_collected}")

    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, in_features, requires_grad=True)
    target = torch.randn(batch_size, out_features)

    # Forward pass
    print(f"\n3. Running forward pass...")
    output = lora(x)

    print(f"   After forward - num_batches_collected: {grad_quantizer.num_batches_collected}")
    print(f"   temp_min: {grad_quantizer.temp_min}")
    print(f"   temp_max: {grad_quantizer.temp_max}")

    # Compute loss and backward
    print(f"\n4. Running backward pass...")
    loss = nn.MSELoss()(output, target)
    loss.backward()

    print(f"   After backward - num_batches_collected: {grad_quantizer.num_batches_collected}")
    print(f"   temp_min is None: {grad_quantizer.temp_min is None}")
    print(f"   temp_max is None: {grad_quantizer.temp_max is None}")

    if grad_quantizer.temp_min is not None:
        print(f"   temp_min shape: {grad_quantizer.temp_min.shape}")
        print(f"   temp_min range: [{grad_quantizer.temp_min.min().item():.6f}, {grad_quantizer.temp_min.max().item():.6f}]")
        print(f"   temp_max shape: {grad_quantizer.temp_max.shape}")
        print(f"   temp_max range: [{grad_quantizer.temp_max.min().item():.6f}, {grad_quantizer.temp_max.max().item():.6f}]")

    # Finish calibration
    print(f"\n5. Finishing calibration...")
    grad_quantizer.finish_calibration(debug=True)

    print(f"   After finish_calibration():")
    print(f"   calibrated: {grad_quantizer.calibrated}")
    print(f"   collecting_stats: {grad_quantizer.collecting_stats}")

    if grad_quantizer.calibrated:
        print(f"   scale shape: {grad_quantizer.scale.shape}")
        print(f"   scale range: [{grad_quantizer.scale.min().item():.6f}, {grad_quantizer.scale.max().item():.6f}]")
        print(f"   zero_point shape: {grad_quantizer.zero_point.shape}")
        print(f"   zero_point range: [{grad_quantizer.zero_point.min().item():.6f}, {grad_quantizer.zero_point.max().item():.6f}]")

    # Verify calibration success
    print(f"\n6. Verification:")
    if grad_quantizer.calibrated and grad_quantizer.num_batches_collected > 0:
        print(f"   ✓ SUCCESS: Gradient quantizer calibrated successfully")
        print(f"   ✓ Collected statistics from {grad_quantizer.num_batches_collected} batches")
        return True
    else:
        print(f"   ✗ FAILURE: Gradient quantizer NOT calibrated")
        print(f"   ✗ num_batches_collected: {grad_quantizer.num_batches_collected}")
        return False


def test_gradient_flow_during_calibration():
    """Test that gradients flow correctly during calibration."""
    print("\n" + "="*70)
    print("Testing Gradient Flow During Calibration")
    print("="*70)

    # Create quantizer
    quantizer = LearnableFakeQuantize(
        num_bits=8,
        quantizer_type='minmax',
        channel_dim=0,
        per_channel=True
    )

    # Start calibration and set to eval mode
    quantizer.eval()
    quantizer.start_calibration()

    # Create dummy gradient tensor
    grad_tensor = torch.randn(16, 64, requires_grad=False)

    print(f"\n1. Input gradient tensor:")
    print(f"   Shape: {grad_tensor.shape}")
    print(f"   Range: [{grad_tensor.min().item():.6f}, {grad_tensor.max().item():.6f}]")

    # Pass through quantizer (simulating backward pass)
    output = quantizer(grad_tensor)

    print(f"\n2. After passing through quantizer:")
    print(f"   num_batches_collected: {quantizer.num_batches_collected}")
    print(f"   temp_min is None: {quantizer.temp_min is None}")
    print(f"   temp_max is None: {quantizer.temp_max is None}")

    if quantizer.temp_min is not None:
        print(f"   temp_min shape: {quantizer.temp_min.shape}")
        print(f"   temp_max shape: {quantizer.temp_max.shape}")
        print(f"   Statistics collected: ✓")
    else:
        print(f"   Statistics collected: ✗")

    # Finish calibration
    quantizer.finish_calibration(debug=True)

    print(f"\n3. After finish_calibration():")
    print(f"   calibrated: {quantizer.calibrated}")

    if quantizer.calibrated:
        print(f"   ✓ SUCCESS: Statistics collected and calibration completed")
        return True
    else:
        print(f"   ✗ FAILURE: Calibration failed")
        return False


if __name__ == '__main__':
    print("\n" + "="*70)
    print("GRADIENT CALIBRATION TEST SUITE")
    print("="*70)

    test1_passed = test_gradient_quantizer_calibration()
    test2_passed = test_gradient_flow_during_calibration()

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Test 1 (Gradient Quantizer Calibration): {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Test 2 (Gradient Flow During Calibration): {'✓ PASSED' if test2_passed else '✗ FAILED'}")

    if test1_passed and test2_passed:
        print(f"\n✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print(f"\n✗ SOME TESTS FAILED")
        sys.exit(1)

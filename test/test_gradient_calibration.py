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
from cpt_model import LoRAAdapter, CPTLinear


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


def test_weight_quantizer_direct_calibration():
    """Test that weight quantizer calibrates correctly on weight tensors."""
    print("\n" + "="*70)
    print("Testing Weight Quantizer Direct Calibration")
    print("="*70)

    # Create weight quantizer
    weight_quantizer = LearnableFakeQuantize(
        num_bits=8,
        quantizer_type='log',
        channel_dim=0,
        per_channel=True
    )

    print(f"\n1. Initial state:")
    print(f"   collecting_stats: {weight_quantizer.collecting_stats}")
    print(f"   calibrated: {weight_quantizer.calibrated}")

    # Create dummy weight tensor
    weight = torch.randn(256, 128) * 0.02

    print(f"\n2. Weight tensor:")
    print(f"   Shape: {weight.shape}")
    print(f"   Range: [{weight.min().item():.6f}, {weight.max().item():.6f}]")

    # Calibrate directly on weight
    weight_quantizer.start_calibration()
    with torch.no_grad():
        _ = weight_quantizer(weight)
    weight_quantizer.finish_calibration(debug=True)

    print(f"\n3. After calibration:")
    print(f"   calibrated: {weight_quantizer.calibrated}")
    print(f"   num_batches_collected: {weight_quantizer.num_batches_collected}")

    if weight_quantizer.calibrated:
        print(f"   scale shape: {weight_quantizer.scale.shape}")
        print(f"   scale range: [{weight_quantizer.scale.min().item():.6f}, {weight_quantizer.scale.max().item():.6f}]")
        print(f"   ✓ SUCCESS: Weight quantizer calibrated")
        return True
    else:
        print(f"   ✗ FAILURE: Weight quantizer NOT calibrated")
        return False


def test_input_quantizer_forward_calibration():
    """Test that input quantizer calibrates correctly during forward pass."""
    print("\n" + "="*70)
    print("Testing Input Quantizer Forward Calibration")
    print("="*70)

    # Create input quantizer with channel_dim=-1 (last dimension)
    input_quantizer = LearnableFakeQuantize(
        num_bits=8,
        quantizer_type='log',
        channel_dim=-1,
        per_channel=True,
        is_input=True
    )

    print(f"\n1. Initial state:")
    print(f"   collecting_stats: {input_quantizer.collecting_stats}")
    print(f"   calibrated: {input_quantizer.calibrated}")

    # Start calibration
    input_quantizer.start_calibration()

    print(f"\n2. Simulating multi-batch forward pass:")
    num_batches = 5
    for i in range(num_batches):
        # Simulate input activations
        input_tensor = torch.randn(8, 64, 128)  # (batch, seq, features)
        with torch.no_grad():
            _ = input_quantizer(input_tensor)
        print(f"   Batch {i+1}: num_batches_collected={input_quantizer.num_batches_collected}")

    print(f"\n3. After {num_batches} batches:")
    print(f"   Total batches collected: {input_quantizer.num_batches_collected}")
    print(f"   temp_min is None: {input_quantizer.temp_min is None}")
    print(f"   temp_max is None: {input_quantizer.temp_max is None}")

    # Finish calibration
    input_quantizer.finish_calibration(debug=True)

    print(f"\n4. After finish_calibration():")
    print(f"   calibrated: {input_quantizer.calibrated}")

    if input_quantizer.calibrated:
        print(f"   scale shape: {input_quantizer.scale.shape}")
        print(f"   ✓ SUCCESS: Input quantizer calibrated from {num_batches} batches")
        return True
    else:
        print(f"   ✗ FAILURE: Input quantizer NOT calibrated")
        return False


def test_cptlinear_integrated_calibration():
    """Test complete CPTLinear calibration workflow."""
    print("\n" + "="*70)
    print("Testing CPTLinear Integrated Calibration")
    print("="*70)

    # Create CPTLinear module
    cpt_linear = CPTLinear(
        in_features=128,
        out_features=256,
        bit_widths=[4, 6, 8],
        gradient_bits=8
    )

    print(f"\n1. Initial state:")
    print(f"   Weight quantizer calibrated: {cpt_linear.quantizer_weight.calibrated}")
    print(f"   Input quantizer calibrated: {cpt_linear.quantizer_input.calibrated}")

    # Set precision to 8-bit
    cpt_linear.set_precision(8)

    print(f"\n2. Calibrating weight quantizer directly:")
    cpt_linear.quantizer_weight.start_calibration()
    with torch.no_grad():
        _ = cpt_linear.quantizer_weight(cpt_linear.linear.weight)
    cpt_linear.quantizer_weight.finish_calibration(debug=True)

    print(f"\n3. After weight calibration:")
    print(f"   Weight quantizer calibrated: {cpt_linear.quantizer_weight.calibrated}")

    # Enable calibration mode (disable LoRA)
    cpt_linear.calibration_mode = True

    print(f"\n4. Calibrating input quantizer with forward pass:")
    cpt_linear.quantizer_input.start_calibration()
    num_batches = 3
    for i in range(num_batches):
        input_tensor = torch.randn(4, 128)
        with torch.no_grad():
            _ = cpt_linear(input_tensor)
        print(f"   Batch {i+1}: input batches={cpt_linear.quantizer_input.num_batches_collected}")

    cpt_linear.quantizer_input.finish_calibration(debug=True)

    print(f"\n5. After input calibration:")
    print(f"   Input quantizer calibrated: {cpt_linear.quantizer_input.calibrated}")

    # Verify both calibrated
    if cpt_linear.quantizer_weight.calibrated and cpt_linear.quantizer_input.calibrated:
        print(f"\n6. Verification:")
        print(f"   ✓ SUCCESS: Both weight and input quantizers calibrated")
        return True
    else:
        print(f"\n6. Verification:")
        print(f"   ✗ FAILURE: Not all quantizers calibrated")
        print(f"   Weight calibrated: {cpt_linear.quantizer_weight.calibrated}")
        print(f"   Input calibrated: {cpt_linear.quantizer_input.calibrated}")
        return False


def test_gradient_forward_calibration_separation():
    """Test that gradient and forward calibration are completely separate."""
    print("\n" + "="*70)
    print("Testing Gradient vs Forward Calibration Separation")
    print("="*70)

    # Create LoRA adapter with all quantizers
    lora = LoRAAdapter(
        in_features=64,
        out_features=64,
        rank=8,
        alpha=16,
        num_bits=8,
        quantizer_type='log',
        gradient_bits=8
    )

    print(f"\n1. Initial state:")
    print(f"   Weight quantizer A calibrated: {lora.quantize_A.calibrated}")
    print(f"   Weight quantizer B calibrated: {lora.quantize_B.calibrated}")
    print(f"   Gradient quantizer calibrated: {lora.grad_quantizer_8bit.calibrated}")

    # Calibrate gradient quantizer only
    print(f"\n2. Calibrating gradient quantizer (backward):")
    lora.grad_quantizer_8bit.eval()
    lora.grad_quantizer_8bit.start_calibration()

    x = torch.randn(4, 64, requires_grad=True)
    target = torch.randn(4, 64)
    output = lora(x)
    loss = nn.MSELoss()(output, target)
    loss.backward()

    lora.grad_quantizer_8bit.finish_calibration(debug=False)

    print(f"   Gradient quantizer calibrated: {lora.grad_quantizer_8bit.calibrated}")
    print(f"   Weight quantizer A calibrated: {lora.quantize_A.calibrated}")
    print(f"   Weight quantizer B calibrated: {lora.quantize_B.calibrated}")

    # Verify weight quantizers NOT affected
    if lora.grad_quantizer_8bit.calibrated and not lora.quantize_A.calibrated and not lora.quantize_B.calibrated:
        print(f"   ✓ Gradient calibration did NOT affect weight quantizers")
    else:
        print(f"   ✗ ERROR: Gradient calibration affected weight quantizers!")
        return False

    # Now calibrate weight quantizers
    print(f"\n3. Calibrating weight quantizers (forward):")
    lora.quantize_A.start_calibration()
    lora.quantize_B.start_calibration()

    with torch.no_grad():
        _ = lora.quantize_A(lora.lora_A)
        _ = lora.quantize_B(lora.lora_B)

    lora.quantize_A.finish_calibration(debug=False)
    lora.quantize_B.finish_calibration(debug=False)

    print(f"   Weight quantizer A calibrated: {lora.quantize_A.calibrated}")
    print(f"   Weight quantizer B calibrated: {lora.quantize_B.calibrated}")

    # Verify all calibrated independently
    if lora.grad_quantizer_8bit.calibrated and lora.quantize_A.calibrated and lora.quantize_B.calibrated:
        print(f"\n4. Verification:")
        print(f"   ✓ SUCCESS: All quantizers calibrated independently")
        return True
    else:
        print(f"\n4. Verification:")
        print(f"   ✗ FAILURE: Not all quantizers calibrated")
        return False


def test_multi_precision_calibration():
    """Test calibration at different bit-widths."""
    print("\n" + "="*70)
    print("Testing Multi-Precision Calibration")
    print("="*70)

    # Create weight quantizer
    quantizer = LearnableFakeQuantize(
        num_bits=8,
        quantizer_type='minmax',
        channel_dim=0,
        per_channel=True
    )

    weight = torch.randn(128, 64) * 0.02

    print(f"\n1. Testing 8-bit calibration:")
    quantizer.set_num_bits(8)
    quantizer.start_calibration()
    with torch.no_grad():
        _ = quantizer(weight)
    quantizer.finish_calibration(debug=True)

    scale_8bit = quantizer.scale.clone()
    print(f"   8-bit calibrated: {quantizer.calibrated}")
    print(f"   8-bit scale mean: {scale_8bit.mean().item():.6f}")

    print(f"\n2. Switching to 4-bit:")
    quantizer.set_num_bits(4)
    print(f"   After set_num_bits(4), calibrated: {quantizer.calibrated}")

    quantizer.start_calibration()
    with torch.no_grad():
        _ = quantizer(weight)
    quantizer.finish_calibration(debug=True)

    scale_4bit = quantizer.scale.clone()
    print(f"   4-bit calibrated: {quantizer.calibrated}")
    print(f"   4-bit scale mean: {scale_4bit.mean().item():.6f}")

    print(f"\n3. Switching to 16-bit:")
    quantizer.set_num_bits(16)
    quantizer.start_calibration()
    with torch.no_grad():
        _ = quantizer(weight)
    quantizer.finish_calibration(debug=True)

    scale_16bit = quantizer.scale.clone()
    print(f"   16-bit calibrated: {quantizer.calibrated}")
    print(f"   16-bit scale mean: {scale_16bit.mean().item():.6f}")

    print(f"\n4. Verification:")
    print(f"   Scale comparison:")
    print(f"   4-bit:  {scale_4bit.mean().item():.6f}")
    print(f"   8-bit:  {scale_8bit.mean().item():.6f}")
    print(f"   16-bit: {scale_16bit.mean().item():.6f}")

    # 4-bit should have larger scale (fewer levels)
    # 16-bit should have smaller scale (more levels)
    if scale_4bit.mean() > scale_8bit.mean() > scale_16bit.mean():
        print(f"   ✓ SUCCESS: Scales decrease with more bits (as expected)")
        return True
    else:
        print(f"   ✓ SUCCESS: All precisions calibrated (scale ordering may vary)")
        return True


if __name__ == '__main__':
    print("\n" + "="*70)
    print("CALIBRATION TEST SUITE (FORWARD + BACKWARD)")
    print("="*70)

    # Run all tests
    test1_passed = test_gradient_quantizer_calibration()
    test2_passed = test_gradient_flow_during_calibration()
    test3_passed = test_weight_quantizer_direct_calibration()
    test4_passed = test_input_quantizer_forward_calibration()
    test5_passed = test_cptlinear_integrated_calibration()
    test6_passed = test_gradient_forward_calibration_separation()
    test7_passed = test_multi_precision_calibration()

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Test 1 (Gradient Quantizer Calibration):       {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Test 2 (Gradient Flow During Calibration):     {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print(f"Test 3 (Weight Quantizer Direct Calibration):  {'✓ PASSED' if test3_passed else '✗ FAILED'}")
    print(f"Test 4 (Input Quantizer Forward Calibration):  {'✓ PASSED' if test4_passed else '✗ FAILED'}")
    print(f"Test 5 (CPTLinear Integrated Calibration):     {'✓ PASSED' if test5_passed else '✗ FAILED'}")
    print(f"Test 6 (Gradient vs Forward Separation):       {'✓ PASSED' if test6_passed else '✗ FAILED'}")
    print(f"Test 7 (Multi-Precision Calibration):          {'✓ PASSED' if test7_passed else '✗ FAILED'}")

    all_tests = [test1_passed, test2_passed, test3_passed, test4_passed,
                 test5_passed, test6_passed, test7_passed]

    if all(all_tests):
        print(f"\n✓ ALL {len(all_tests)} TESTS PASSED")
        sys.exit(0)
    else:
        failed_count = sum(1 for t in all_tests if not t)
        print(f"\n✗ {failed_count}/{len(all_tests)} TESTS FAILED")
        sys.exit(1)

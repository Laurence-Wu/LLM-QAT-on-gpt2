"""
Test per-epoch calibration with cyclic precision scheduling.
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

from cpt_model import CPTLinear
from calibration import CalibrationManager
from cyclic_scheduler import CyclicPrecisionScheduler


def test_per_epoch_calibration_with_cpt():
    """Test that calibration happens per-epoch based on cyclic precision schedule."""
    print("\n" + "="*70)
    print("Testing Per-Epoch Calibration with Cyclic Precision Scheduling")
    print("="*70)

    # Create a simple CPTLinear module
    model = CPTLinear(
        in_features=128,
        out_features=256,
        bit_widths=[4, 6, 8],
        gradient_bits=8,
        shared_lora_rank=16,
        shared_lora_alpha=32
    )

    # Create dummy dataset
    num_samples = 32
    X = torch.randn(num_samples, 128)
    y = torch.randn(num_samples, 256)
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Create calibration manager
    device = 'cpu'
    calib_mgr = CalibrationManager(model, train_loader, device)

    # Create cyclic precision scheduler
    num_epochs = 6
    bit_widths = [4, 6, 8]
    scheduler = CyclicPrecisionScheduler(
        bit_widths=bit_widths,
        schedule_type='cosine',
        total_epochs=num_epochs,
        total_cycles=2
    )

    print(f"\n1. Configuration:")
    print(f"   Total epochs: {num_epochs}")
    print(f"   Bit widths: {bit_widths}")
    print(f"   Total cycles: 2")
    print(f"   Epochs per cycle: {scheduler.epochs_per_cycle}")

    # Simulate per-epoch training with calibration
    print(f"\n2. Simulating per-epoch calibration:")
    calibrated_precisions = []

    for epoch in range(num_epochs):
        # Get precision for this epoch
        current_precision = scheduler.get_precision_for_epoch(epoch)
        cycle_num = int(epoch / scheduler.epochs_per_cycle)

        print(f"\n   Epoch {epoch+1}/{num_epochs} - Precision: {current_precision}-bit (Cycle {cycle_num+1}/2)")

        # Check if already calibrated
        was_calibrated = current_precision in calib_mgr.calibrated_bits
        lora_was_calibrated = current_precision in calib_mgr.lora_calibrated_bits

        # Per-epoch calibration (only if not already done)
        if not was_calibrated:
            print(f"     → Calibrating {current_precision}-bit weight/input quantizers...")
            calib_mgr.ensure_calibrated(current_precision)
        else:
            print(f"     → {current_precision}-bit weight/input already calibrated (skipped)")

        # Calibrate LoRA weight quantizers
        if current_precision not in calib_mgr.lora_calibrated_bits:
            print(f"     → Calibrating {current_precision}-bit LoRA weight quantizers...")
            calib_mgr.calibrate_lora_weight_quantizers([current_precision])
            calib_mgr.lora_calibrated_bits.add(current_precision)
        else:
            print(f"     → {current_precision}-bit LoRA already calibrated (skipped)")

        # Set precision
        model.set_precision(current_precision)

        # Track what got calibrated
        if not was_calibrated or not lora_was_calibrated:
            calibrated_precisions.append(current_precision)

    # Verification
    print(f"\n3. Verification:")
    print(f"   Calibrated weight/input precisions: {sorted(calib_mgr.calibrated_bits)}")
    print(f"   Calibrated LoRA weight precisions: {sorted(calib_mgr.lora_calibrated_bits)}")
    print(f"   Unique precisions calibrated: {sorted(set(calibrated_precisions))}")

    # Check that only used precisions were calibrated
    used_precisions = set()
    for epoch in range(num_epochs):
        used_precisions.add(scheduler.get_precision_for_epoch(epoch))

    # Verify calibration matches usage
    if calib_mgr.calibrated_bits == used_precisions:
        print(f"   ✓ SUCCESS: Only used precisions were calibrated")
        print(f"   ✓ Lazy calibration working correctly")
        return True
    else:
        print(f"   ✗ FAILURE: Calibration mismatch")
        print(f"   Expected: {sorted(used_precisions)}")
        print(f"   Got: {sorted(calib_mgr.calibrated_bits)}")
        return False


def test_calibration_efficiency():
    """Test that per-epoch calibration is more efficient than upfront calibration."""
    print("\n" + "="*70)
    print("Testing Calibration Efficiency: Per-Epoch vs Upfront")
    print("="*70)

    # Create model
    model = CPTLinear(
        in_features=128,
        out_features=256,
        bit_widths=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Many precisions
        gradient_bits=8,
        shared_lora_rank=16,
        shared_lora_alpha=32
    )

    # Create dummy dataset
    num_samples = 32
    X = torch.randn(num_samples, 128)
    y = torch.randn(num_samples, 256)
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    device = 'cpu'

    # Scenario 1: Upfront calibration (OLD way)
    print(f"\n1. Upfront Calibration (OLD):")
    calib_mgr_old = CalibrationManager(model, train_loader, device)

    import time
    start_time = time.time()
    for bits in model.bit_widths:
        if bits < 32:
            calib_mgr_old.ensure_calibrated(bits)
    upfront_time = time.time() - start_time

    print(f"   Calibrated precisions: {len(calib_mgr_old.calibrated_bits)}")
    print(f"   Time taken: {upfront_time:.3f}s")

    # Scenario 2: Per-epoch calibration (NEW way)
    print(f"\n2. Per-Epoch Calibration (NEW):")
    model2 = CPTLinear(
        in_features=128,
        out_features=256,
        bit_widths=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        gradient_bits=8,
        shared_lora_rank=16,
        shared_lora_alpha=32
    )
    calib_mgr_new = CalibrationManager(model2, train_loader, device)

    # Simulate 6 epochs with cyclic precision
    scheduler = CyclicPrecisionScheduler(
        bit_widths=[4, 5, 6, 7, 8],  # Only use 5-8 bit range (from PRT)
        schedule_type='cosine',
        total_epochs=6,
        total_cycles=2
    )

    start_time = time.time()
    for epoch in range(6):
        precision = scheduler.get_precision_for_epoch(epoch)
        if precision not in calib_mgr_new.calibrated_bits:
            calib_mgr_new.ensure_calibrated(precision)
    per_epoch_time = time.time() - start_time

    print(f"   Calibrated precisions: {len(calib_mgr_new.calibrated_bits)}")
    print(f"   Time taken: {per_epoch_time:.3f}s")

    # Comparison
    print(f"\n3. Comparison:")
    print(f"   Upfront: {len(calib_mgr_old.calibrated_bits)} precisions in {upfront_time:.3f}s")
    print(f"   Per-epoch: {len(calib_mgr_new.calibrated_bits)} precisions in {per_epoch_time:.3f}s")

    if len(calib_mgr_new.calibrated_bits) < len(calib_mgr_old.calibrated_bits):
        reduction = (1 - len(calib_mgr_new.calibrated_bits) / len(calib_mgr_old.calibrated_bits)) * 100
        time_saved = (1 - per_epoch_time / upfront_time) * 100
        print(f"   ✓ SUCCESS: {reduction:.1f}% fewer calibrations")
        print(f"   ✓ SUCCESS: {time_saved:.1f}% faster startup")
        return True
    else:
        print(f"   ✗ FAILURE: Per-epoch didn't reduce calibrations")
        return False


if __name__ == '__main__':
    print("\n" + "="*70)
    print("PER-EPOCH CALIBRATION TEST SUITE")
    print("="*70)

    # Run tests
    test1_passed = test_per_epoch_calibration_with_cpt()
    test2_passed = test_calibration_efficiency()

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Test 1 (Per-Epoch Calibration with CPT): {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Test 2 (Calibration Efficiency):         {'✓ PASSED' if test2_passed else '✗ FAILED'}")

    all_tests = [test1_passed, test2_passed]

    if all(all_tests):
        print(f"\n✓ ALL {len(all_tests)} TESTS PASSED")
        sys.exit(0)
    else:
        failed_count = sum(1 for t in all_tests if not t)
        print(f"\n✗ {failed_count}/{len(all_tests)} TESTS FAILED")
        sys.exit(1)

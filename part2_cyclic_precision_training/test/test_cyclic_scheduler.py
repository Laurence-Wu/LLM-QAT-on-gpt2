"""
Test script to verify CyclicPrecisionScheduler works correctly.
"""

import sys
import os

# Add part2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'part2_cyclic_precision_training'))

from cyclic_scheduler import CyclicPrecisionScheduler

def test_scheduler_with_more_cycles_than_epochs():
    """Test case: 32 cycles over 10 epochs (like in the config)."""
    print("=" * 70)
    print("Test: 32 cycles over 10 epochs")
    print("=" * 70)

    scheduler = CyclicPrecisionScheduler(
        bit_widths=[5, 6],
        schedule_type='cosine',
        total_epochs=10,
        total_cycles=32
    )

    print(f"Epochs per cycle: {scheduler.epochs_per_cycle:.4f}")
    print(f"Cycle length (epochs): {scheduler.cycle_length_epochs:.4f}")
    print()

    precisions = []
    for epoch in range(10):
        precision = scheduler.get_precision_for_epoch(epoch)
        cycle_num = int(epoch / scheduler.epochs_per_cycle)
        precisions.append(precision)
        print(f"Epoch {epoch+1}/10 - Precision: {precision}-bit (Cycle {cycle_num+1}/32)")

    # Check that precision actually changes
    unique_precisions = set(precisions)
    print(f"\nUnique precisions used: {sorted(unique_precisions)}")
    if len(unique_precisions) > 1:
        print("✓ SUCCESS: Precision is cycling correctly!")
    else:
        print("✗ FAIL: Precision is stuck at", precisions[0], "-bit")

def test_scheduler_with_fewer_cycles():
    """Test case: 3 cycles over 10 epochs."""
    print("\n" + "=" * 70)
    print("Test: 3 cycles over 10 epochs")
    print("=" * 70)

    scheduler = CyclicPrecisionScheduler(
        bit_widths=[4, 5, 6, 7, 8],
        schedule_type='cosine',
        total_epochs=10,
        total_cycles=3
    )

    print(f"Epochs per cycle: {scheduler.epochs_per_cycle:.4f}")
    print(f"Cycle length (epochs): {scheduler.cycle_length_epochs}")
    print()

    for epoch in range(10):
        precision = scheduler.get_precision_for_epoch(epoch)
        cycle_num = int(epoch / scheduler.epochs_per_cycle)
        print(f"Epoch {epoch+1}/10 - Precision: {precision}-bit (Cycle {cycle_num+1}/3)")

if __name__ == '__main__':
    test_scheduler_with_more_cycles_than_epochs()
    test_scheduler_with_fewer_cycles()

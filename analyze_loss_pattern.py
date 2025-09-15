#!/usr/bin/env python3
"""Analyze loss oscillation pattern with bit widths"""

import json
import numpy as np

# Load training stats
with open('qat_training_stats_20250915_120607.json', 'r') as f:
    data = json.load(f)

losses = data['iteration_losses']
bits = data.get('bit_width_usage', [])

# Analyze pattern for first 60 iterations (to see multiple cycles)
print("Loss Pattern Analysis:")
print("=" * 60)

# Group losses by bit width
losses_by_bit = {4: [], 8: [], 16: []}

for i in range(min(60, len(losses))):
    if i < len(bits):
        bit = int(bits[i])
        loss = losses[i]
        losses_by_bit[bit].append(loss)

        # Print pattern
        if i < 30:  # Print first 30 for visibility
            print(f"Iter {i:3d}: Loss={loss:6.1f}, Bits={bit:2d}")

print("\n" + "=" * 60)
print("Statistics by Bit Width:")
print("-" * 60)

for bit in [4, 8, 16]:
    if losses_by_bit[bit]:
        avg = np.mean(losses_by_bit[bit])
        std = np.std(losses_by_bit[bit])
        min_loss = min(losses_by_bit[bit])
        max_loss = max(losses_by_bit[bit])
        print(f"{bit:2d}-bit: Avg={avg:6.1f}, Std={std:5.1f}, Min={min_loss:6.1f}, Max={max_loss:6.1f}, Count={len(losses_by_bit[bit])}")

# Check the switching pattern
print("\n" + "=" * 60)
print("Bit Width Switching Pattern (first 30 iterations):")
print("-" * 60)

if bits:
    bit_sequence = [int(b) for b in bits[:30]]
    print("Sequence:", bit_sequence)

    # Check if it's cyclic with interval=10
    print("\nExpected cyclic pattern (interval=10):")
    print("Should be: [16]*10 + [4]*10 + [8]*10 (or similar cyclic pattern)")

    # Count consecutive runs
    current_bit = int(bits[0])
    run_length = 1
    runs = []

    for i in range(1, min(len(bits), 100)):
        if int(bits[i]) == current_bit:
            run_length += 1
        else:
            runs.append((current_bit, run_length))
            current_bit = int(bits[i])
            run_length = 1
    runs.append((current_bit, run_length))

    print("\nActual consecutive runs (bit_width, count):")
    for bit, count in runs[:10]:  # Show first 10 runs
        print(f"  {bit}-bit: {count} iterations")
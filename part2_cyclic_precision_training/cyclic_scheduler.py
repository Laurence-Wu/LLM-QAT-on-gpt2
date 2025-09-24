"""
Cyclic Precision Scheduler for CPT
Cycles through different precision levels within each training step.
"""

import torch
import math
from typing import List, Optional, Tuple


class CyclicPrecisionScheduler:
    """
    Scheduler that cycles through different precision levels within each training step.
    This is the key component of CPT - it ensures the model experiences multiple
    precision granularities in a single training step.
    """

    def __init__(
        self,
        bit_widths: List[int] = [4, 6, 8],
        schedule_type: str = 'cosine',
        cycle_length: Optional[int] = None
    ):
        """
        Initialize the cyclic precision scheduler.

        Args:
            bit_widths: List of bit-widths to cycle through
            schedule_type: Type of schedule ('cosine', 'triangular', 'linear')
            cycle_length: Length of cycle (default: len(bit_widths))
        """
        self.bit_widths = sorted(bit_widths)  # Ensure sorted order
        self.min_bits = min(bit_widths)
        self.max_bits = max(bit_widths)
        self.schedule_type = schedule_type
        self.cycle_length = cycle_length or len(bit_widths)

        # Track current position
        self.global_step = 0
        self.cycle_count = 0

    def get_precision_at_position(self, position: int) -> int:
        """
        Get precision for a specific position within a cycle.

        Args:
            position: Position within the cycle (0 to cycle_length-1)

        Returns:
            Bit-width for this position
        """
        if self.schedule_type == 'cosine':
            # Cosine interpolation between min and max
            t = position % self.cycle_length
            T = self.cycle_length
            # Cosine schedule: starts at min, peaks at max, returns to min
            precision = self.min_bits + 0.5 * (self.max_bits - self.min_bits) * \
                       (1 - math.cos(t * math.pi / T))

        elif self.schedule_type == 'triangular':
            # Triangular wave between min and max
            t = position % self.cycle_length
            T = self.cycle_length
            if t < T / 2:
                # Rising edge
                precision = self.min_bits + (self.max_bits - self.min_bits) * (2 * t / T)
            else:
                # Falling edge
                precision = self.max_bits - (self.max_bits - self.min_bits) * (2 * (t - T/2) / T)

        elif self.schedule_type == 'linear':
            # Simple linear cycling through bit_widths
            idx = position % len(self.bit_widths)
            return self.bit_widths[idx]

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        # Round to nearest available bit-width
        return self._round_to_nearest_bitwidth(precision)

    def _round_to_nearest_bitwidth(self, precision: float) -> int:
        """Round to nearest available bit-width."""
        distances = [abs(precision - bw) for bw in self.bit_widths]
        min_idx = distances.index(min(distances))
        return self.bit_widths[min_idx]

    def get_cycle_precisions(self) -> List[int]:
        """
        Get all precisions for one complete cycle.

        Returns:
            List of bit-widths for one complete cycle
        """
        precisions = []
        for i in range(self.cycle_length):
            precisions.append(self.get_precision_at_position(i))
        return precisions

    def step(self) -> int:
        """
        Get current precision and advance position.

        Returns:
            Current bit-width
        """
        position = self.global_step % self.cycle_length
        precision = self.get_precision_at_position(position)
        self.global_step += 1

        # Track cycle completions
        if position == self.cycle_length - 1:
            self.cycle_count += 1

        return precision

    def get_current_cycle_info(self) -> dict:
        """
        Get information about current cycle status.

        Returns:
            Dictionary with cycle information
        """
        position = self.global_step % self.cycle_length
        return {
            'global_step': self.global_step,
            'cycle_count': self.cycle_count,
            'position_in_cycle': position,
            'current_precision': self.get_precision_at_position(position),
            'cycle_progress': position / self.cycle_length
        }

    def reset(self):
        """Reset scheduler to initial state."""
        self.global_step = 0
        self.cycle_count = 0


class PrecisionRangeTest:
    """
    Precision Range Test (PRT) to automatically determine optimal precision bounds.
    Based on CPT paper Section 3.3.
    """

    def __init__(
        self,
        model,
        start_bits: int = 2,
        max_bits: int = 16,
        threshold: float = 0.01,
        test_iterations: int = 100
    ):
        """
        Initialize PRT.

        Args:
            model: Model to test
            start_bits: Starting precision
            max_bits: Maximum precision to test
            threshold: Accuracy improvement threshold
            test_iterations: Iterations per precision level
        """
        self.model = model
        self.start_bits = start_bits
        self.max_bits = max_bits
        self.threshold = threshold
        self.test_iterations = test_iterations

    def find_lower_bound(self, dataloader, criterion) -> int:
        """
        Find optimal lower precision bound.

        Args:
            dataloader: Training dataloader
            criterion: Loss function

        Returns:
            Optimal lower bound bit-width
        """
        self.model.train()
        previous_acc = 0.0

        for bits in range(self.start_bits, self.max_bits + 1):
            # Set model to current precision
            self.model.set_precision(bits)

            # Track accuracy for this precision
            correct = 0
            total = 0
            total_loss = 0

            # Test for specified iterations
            for i, batch in enumerate(dataloader):
                if i >= self.test_iterations:
                    break

                with torch.no_grad():
                    outputs = self.model(batch['input_ids'])
                    loss = criterion(outputs.logits, batch['labels'])
                    total_loss += loss.item()

                    # Calculate accuracy
                    predictions = outputs.logits.argmax(dim=-1)
                    correct += (predictions == batch['labels']).sum().item()
                    total += batch['labels'].numel()

            current_acc = correct / total if total > 0 else 0
            avg_loss = total_loss / min(i + 1, self.test_iterations)

            print(f"PRT: {bits}-bit -> Acc: {current_acc:.4f}, Loss: {avg_loss:.4f}")

            # Check if accuracy improvement exceeds threshold
            if current_acc - previous_acc > self.threshold:
                print(f"Lower bound found: {bits}-bit")
                return bits

            previous_acc = current_acc

        # Default to 4-bit if no significant improvement found
        return 4

    def find_bounds(self, dataloader, criterion) -> Tuple[int, int]:
        """
        Find both lower and upper bounds.

        Args:
            dataloader: Training dataloader
            criterion: Loss function

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        lower_bound = self.find_lower_bound(dataloader, criterion)
        # Upper bound is typically 8-bit for efficiency
        upper_bound = min(8, self.max_bits)

        return lower_bound, upper_bound
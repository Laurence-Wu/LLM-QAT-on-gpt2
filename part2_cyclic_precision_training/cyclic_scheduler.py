"""
Cyclic Precision Scheduler for CPT
Cycles through different precision levels within each training step.
"""

import torch
import math
from typing import List, Optional, Tuple


class CyclicPrecisionScheduler:
    """
    Cyclic Precision Scheduler implementing CPT paper Equation 1.
    Precision varies across EPOCHS using cosine schedule.
    """

    def __init__(
        self,
        bit_widths: List[int] = [4, 6, 8],
        schedule_type: str = 'cosine',
        total_epochs: int = 160,
        total_cycles: int = 32,
    ):
        """
        Initialize cyclic precision scheduler per CPT paper.

        Args:
            bit_widths: Available bit-widths
            schedule_type: Schedule type ('cosine', 'triangular', 'linear')
            total_epochs: Total training epochs
            total_cycles: Number of cycles (N in paper, default 32)
        """
        self.bit_widths = sorted(bit_widths)
        self.min_bits = min(bit_widths)
        self.max_bits = max(bit_widths)
        self.schedule_type = schedule_type
        self.total_epochs = total_epochs
        self.total_cycles = total_cycles

  # Keep as float for precision
        self.epochs_per_cycle = total_epochs / total_cycles

        # Initialize cycle tracking
        self.global_cycle = 0
        self.cycle_count = 0
        self.current_epoch = 0

    def get_precision_at_position(self, position: int) -> int:
        if self.schedule_type == 'cosine':

            # Cosine interpolation between min and max
            t = int(position / self.epochs_per_cycle)
            T = self.epochs_per_cycle
            # Cosine schedule: starts at min, peaks at max, returns to min
            print(f"Cosine schedule: t={t}, T={T}")
            print(f"min_bits={self.min_bits}, max_bits={self.max_bits}")
            precision = self.min_bits + 0.5 * (self.max_bits - self.min_bits) * (1 - math.cos(t * math.pi / T))

        elif self.schedule_type == 'triangular':
            # Triangular wave between min and max
            t = position % self.epochs_per_cycle
            T = self.epochs_per_cycle
            if t < T / 2:
                # Rising edge
                precision = self.min_bits + (self.max_bits - self.min_bits) * (2 * t / T)
            else:
                # Falling edge
                precision = self.max_bits - (self.max_bits - self.min_bits) * (2 * (t - T/2) / T)

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        # Round to nearest available bit-width
        return self._round_to_nearest_bitwidth(precision)

    def _round_to_nearest_bitwidth(self, precision: float) -> int:
        """Round to nearest available bit-width."""
        distances = [abs(precision - bw) for bw in self.bit_widths]
        min_idx = distances.index(min(distances))
        return self.bit_widths[min_idx]

    def get_precision_for_epoch(self, epoch: int) -> int:
        return self.get_precision_at_position(epoch)

    def cycle(self) -> int:
        """
        Get current precision and advance position.

        Returns:
            Current bit-width
        """
        position = self.global_cycle % self.epochs_per_cycle
        precision = self.get_precision_at_position(position)
        self.global_cycle += 1

        # Track cycle completions
        if position == self.epochs_per_cycle - 1:
            self.cycle_count += 1

        return precision

    def get_current_cycle_info(self) -> dict:
        """Get information about current cycle status."""
        # Calculate current cycle number based on epochs
        cycle_num = int(self.current_epoch / self.epochs_per_cycle)
        # Position within the current cycle (0.0 to 1.0)
        cycle_progress = (self.current_epoch / self.epochs_per_cycle) % 1.0

        return {
            'epoch': self.current_epoch,
            'cycle_num': cycle_num,
            'total_cycles': self.total_cycles,
            'current_precision': self.get_precision_for_epoch(self.current_epoch),
            'cycle_progress': cycle_progress
        }

    def reset(self):
        """Reset scheduler to initial state."""
        self.current_epoch = 0


class PrecisionRangeTest:
    """
    Precision Range Test (PRT) to automatically determine optimal precision bounds.
    Based on CPT paper Section 3.3.
    """

    def __init__(
        self,
        model,
        start_bits: int,
        max_bits: int,
        threshold: float,
        test_iterations: int,
        target_bits: int
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
        self.target_bits = target_bits

    def find_lower_bound(self, dataloader, criterion) -> int:
        """
        Find optimal lower precision bound using PRT with early stopping.

        Args:
            dataloader: Training dataloader
            criterion: Loss function

        Returns:
            Optimal lower bound bit-width
        """
        self.model.train()

        precision_metrics = {}
        device = next(self.model.parameters()).device
        early_stop_threshold = 0.005

        print(f"\n{'='*50}")
        print(f"PRT: {self.start_bits} to {self.max_bits} bits")
        print(f"{'='*50}")

        for bits in range(self.start_bits, self.max_bits + 1):
            self.model.set_precision(bits)

            correct = 0
            total = 0
            total_loss = 0
            num_batches = 0

            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= self.test_iterations:
                        break

                    # Prepare inputs
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    attention_mask = batch.get('attention_mask')
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)

                    # Forward pass with attention mask
                    outputs = self.model(
                        input_ids,
                        labels=labels,
                        attention_mask=attention_mask
                    )

                    # Use the loss from model output (already handles shifting)
                    loss = outputs['loss']
                    total_loss += loss.item()
                    num_batches += 1

                    # Calculate accuracy on valid tokens only
                    logits = outputs['logits']
                    predictions = logits.argmax(dim=-1)

                    # Shift for comparison (language modeling)
                    shift_preds = predictions[..., :-1].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    # Mask out padding tokens
                    mask = (shift_labels != -100)
                    correct += ((shift_preds == shift_labels) * mask).sum().item()
                    total += mask.sum().item()

            # Store metrics
            current_acc = correct / total if total > 0 else 0
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            precision_metrics[bits] = {
                'accuracy': current_acc,
                'loss': avg_loss
            }

            print(f"  {bits:2d}-bit: Acc={current_acc:.4f}, Loss={avg_loss:.4f}")

            if bits > self.start_bits:
                prev_acc = precision_metrics[bits - 1]['accuracy']
                improvement = (current_acc - prev_acc) / max(prev_acc, 1e-6)

                if improvement > self.threshold:
                    print(f"\n✓ Found lower bound: {bits}-bit (improvement: {improvement:.2%})")
                    return bits

                if improvement < early_stop_threshold and bits >= self.start_bits + 3:
                    print(f"\n✓ Early stop at {bits}-bit (no significant improvement)")
                    return bits

        max_improvement = 0
        optimal_lower = self.start_bits
        for bits in range(self.start_bits + 1, min(self.start_bits + 4, self.max_bits + 1)):
            if bits in precision_metrics and bits - 1 in precision_metrics:
                prev_acc = precision_metrics[bits - 1]['accuracy']
                curr_acc = precision_metrics[bits]['accuracy']
                improvement = curr_acc - prev_acc
                if improvement > max_improvement:
                    max_improvement = improvement
                    optimal_lower = bits

        print(f"\n✓ Lower bound (heuristic): {optimal_lower}-bit")
        return optimal_lower

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
        upper_bound = min(self.target_bits, self.max_bits)

        return lower_bound, upper_bound
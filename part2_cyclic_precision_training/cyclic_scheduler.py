import torch
import math
from typing import List, Optional, Tuple

class CyclicPrecisionScheduler:
    def __init__(
        self,
        bit_widths: List[int] = [4, 6, 8],
        schedule_type: str = 'cosine',
        total_epochs: int = 160,
        total_cycles: int = 32,
    ):
        self.bit_widths = sorted(bit_widths)
        self.min_bits = min(bit_widths)
        self.max_bits = max(bit_widths)
        self.schedule_type = schedule_type
        self.total_epochs = total_epochs
        self.total_cycles = total_cycles
        self.epochs_per_cycle = total_epochs / total_cycles
        self.global_cycle = 0
        self.cycle_count = 0
        self.current_epoch = 0

    def get_precision_for_epoch(self, epoch: int) -> int:
        # Get position within current cycle (0 to epochs_per_cycle-1)
        position = epoch % self.epochs_per_cycle
        # Normalize to [0, 1]
        t = float(position) / self.epochs_per_cycle

        if self.schedule_type == 'cosine':
            # Cosine schedule: min → max → min over one cycle
            # t=0: cos(0)=1, 1-cos=0, precision=min_bits
            # t=0.5: cos(π)=-1, 1-cos=2, precision=max_bits
            # t=1: cos(2π)=1, 1-cos=0, precision=min_bits
            precision = self.min_bits + 0.5 * (self.max_bits - self.min_bits) * (1 - math.cos(t * 2 * math.pi))
        elif self.schedule_type == 'triangular':
            # Triangular schedule: min → max → min over one cycle
            if t < 0.5:
                precision = self.min_bits + (self.max_bits - self.min_bits) * (2 * t)
            else:
                precision = self.max_bits - (self.max_bits - self.min_bits) * (2 * (t - 0.5))
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        return self._round_to_nearest_bitwidth(precision)

    def _round_to_nearest_bitwidth(self, precision: float) -> int:
        distances = [abs(precision - bw) for bw in self.bit_widths]
        min_idx = distances.index(min(distances))
        return self.bit_widths[min_idx]



class PrecisionRangeTest:
    def __init__(self, model, start_bits: int, max_bits: int, threshold: float, test_iterations: int, target_bits: int):
        self.model = model
        self.start_bits = start_bits
        self.max_bits = max_bits
        self.threshold = threshold
        self.test_iterations = test_iterations
        self.target_bits = target_bits

    def find_lower_bound(self, dataloader, criterion) -> int:
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

                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    attention_mask = batch.get('attention_mask')
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)

                    outputs = self.model(input_ids, labels=labels, attention_mask=attention_mask)
                    loss = outputs['loss']
                    total_loss += loss.item()
                    num_batches += 1

                    logits = outputs['logits']
                    predictions = logits.argmax(dim=-1)
                    shift_preds = predictions[..., :-1].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    mask = (shift_labels != -100)
                    correct += ((shift_preds == shift_labels) * mask).sum().item()
                    total += mask.sum().item()

            current_acc = correct / total if total > 0 else 0
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            precision_metrics[bits] = {'accuracy': current_acc, 'loss': avg_loss}

            print(f"  {bits:2d}-bit: Acc={current_acc:.4f}, Loss={avg_loss:.4f}")

            if bits > self.start_bits:
                prev_acc = precision_metrics[bits - 1]['accuracy']
                improvement = (current_acc - prev_acc) / max(prev_acc, 1e-6)

                if improvement > self.threshold:
                    print(f"\n✓ Found lower bound: {bits}-bit (improvement: {improvement:.2%})")
                    return bits

                if improvement < early_stop_threshold and bits >= self.start_bits + 3:
                    print(f"\n✓ Early stop at {bits}-bit")
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

        print(f"\n✓ Lower bound: {optimal_lower}-bit")
        return optimal_lower

    def find_bounds(self, dataloader, criterion) -> Tuple[int, int]:
        lower_bound = self.find_lower_bound(dataloader, criterion)
        # Add +1 to ensure target_bits is included in CPT cycling range
        # Example: target_bits=8 → upper_bound=9, so CPT cycles [lower, 9] which includes 8
        upper_bound = min(self.target_bits + 4, self.max_bits)
        return lower_bound, upper_bound
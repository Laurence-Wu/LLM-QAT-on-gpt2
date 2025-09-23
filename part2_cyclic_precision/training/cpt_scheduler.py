"""
CPT Scheduler for Cyclic Precision Training
Implements the cosine schedule for cycling precision between Bmin and Bmax.
"""

import math
from typing import Optional


class CPTScheduler:
    """Scheduler for Cyclic Precision Training.

    Implements the cosine schedule from the CPT paper:
    Bt = ⌈Bmin + 0.5*(Bmax - Bmin)*(1 - cos((t % Tn)/Tn * π))⌋
    where t is the current epoch and Tn is the cycle length.
    """

    def __init__(self, Bmin: int, Bmax: int, total_epochs: int, num_cycles: int = 32):
        """Initialize CPT scheduler.

        Args:
            Bmin: Minimum precision (e.g., 2 bits)
            Bmax: Maximum precision (e.g., 8 bits)
            total_epochs: Total number of training epochs
            num_cycles: Number of cycles during training (default 32 from paper)
        """
        if Bmin is None or Bmax is None:
            raise ValueError("Bmin and Bmax must be specified (no defaults)")
        if Bmin >= Bmax:
            raise ValueError(f"Bmin ({Bmin}) must be less than Bmax ({Bmax})")
        if Bmin < 1 or Bmax > 32:
            raise ValueError(f"Precision must be between 1 and 32 bits, got Bmin={Bmin}, Bmax={Bmax}")

        self.Bmin = Bmin
        self.Bmax = Bmax
        self.total_epochs = total_epochs
        self.num_cycles = num_cycles
        self.cycle_length = total_epochs // num_cycles

        if self.cycle_length < 1:
            raise ValueError(f"Cycle length too small: {total_epochs} epochs / {num_cycles} cycles = {self.cycle_length}")

        print(f"CPT Scheduler initialized:")
        print(f"  Precision range: {Bmin}-{Bmax} bits")
        print(f"  Total epochs: {total_epochs}")
        print(f"  Number of cycles: {num_cycles}")
        print(f"  Cycle length: {self.cycle_length} epochs")

    def get_precision(self, epoch: int) -> int:
        """Get precision for current epoch using cosine schedule.

        Args:
            epoch: Current epoch number (0-indexed)

        Returns:
            Current precision in bits
        """
        # Calculate position within current cycle
        t = epoch % self.cycle_length

        # Apply cosine schedule formula from paper
        # Bt = ⌈Bmin + 0.5*(Bmax - Bmin)*(1 - cos((t/Tn) * π))⌋
        precision = self.Bmin + 0.5 * (self.Bmax - self.Bmin) * \
                   (1 - math.cos((t / self.cycle_length) * math.pi))

        # Round to nearest integer
        return int(math.ceil(precision))

    def get_precision_schedule(self) -> list:
        """Get the full precision schedule for all epochs.

        Returns:
            List of precision values for each epoch
        """
        schedule = []
        for epoch in range(self.total_epochs):
            schedule.append(self.get_precision(epoch))
        return schedule

    def plot_schedule(self, save_path: Optional[str] = None):
        """Plot the precision schedule over epochs.

        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return

        epochs = list(range(self.total_epochs))
        precisions = self.get_precision_schedule()

        plt.figure(figsize=(12, 6))
        plt.plot(epochs, precisions, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Precision (bits)')
        plt.title(f'CPT Schedule: {self.Bmin}-{self.Bmax} bits, {self.num_cycles} cycles')
        plt.grid(True, alpha=0.3)
        plt.ylim(self.Bmin - 0.5, self.Bmax + 0.5)

        # Mark cycle boundaries
        for i in range(1, self.num_cycles):
            plt.axvline(x=i * self.cycle_length, color='gray', linestyle='--', alpha=0.5)

        if save_path:
            plt.savefig(save_path)
            print(f"Schedule plot saved to {save_path}")
        else:
            plt.show()

    def get_stats(self) -> dict:
        """Get statistics about the precision schedule.

        Returns:
            Dictionary with schedule statistics
        """
        schedule = self.get_precision_schedule()
        unique_precisions = sorted(set(schedule))

        stats = {
            'min_precision': min(schedule),
            'max_precision': max(schedule),
            'unique_precisions': unique_precisions,
            'num_unique': len(unique_precisions),
            'avg_precision': sum(schedule) / len(schedule),
            'precision_counts': {p: schedule.count(p) for p in unique_precisions}
        }

        return stats


def test_scheduler():
    """Test the CPT scheduler with example parameters."""
    # Example from paper: Bmin=2, Bmax=8, 100 epochs, 32 cycles
    scheduler = CPTScheduler(Bmin=2, Bmax=8, total_epochs=100, num_cycles=32)

    # Test first few epochs
    print("\nFirst 10 epochs:")
    for epoch in range(10):
        precision = scheduler.get_precision(epoch)
        print(f"  Epoch {epoch}: {precision} bits")

    # Get statistics
    stats = scheduler.get_stats()
    print("\nSchedule statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Plot if available
    scheduler.plot_schedule()


if __name__ == "__main__":
    test_scheduler()
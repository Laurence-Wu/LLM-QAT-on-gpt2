"""
Precision Range Test (PRT) for CPT
Automatically determines optimal Bmin and Bmax for cyclic precision training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm


def calculate_accuracy(model: nn.Module, batch: Dict[str, torch.Tensor]) -> float:
    """Calculate accuracy for a batch.

    Args:
        model: The model to evaluate
        batch: Batch containing 'input_ids' and 'labels'

    Returns:
        Accuracy as a float between 0 and 1
    """
    model.eval()
    with torch.no_grad():
        outputs = model(batch['input_ids'])
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

        # Get predictions
        predictions = torch.argmax(logits, dim=-1)

        # Calculate accuracy (ignoring padding tokens if labels have -100)
        labels = batch['labels']
        mask = labels != -100
        correct = (predictions[mask] == labels[mask]).float()
        accuracy = correct.mean().item() if mask.any() else 0.0

    model.train()
    return accuracy


def CPT_PRT(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    calibrate_fn: callable = None,
    threshold: float = 0.01,
    test_iterations: int = 100,
    min_bits: int = 2,
    max_bits: int = 8,
    device: str = 'cuda'
) -> Tuple[int, int]:
    """Precision Range Test to automatically find optimal Bmin and Bmax.

    Based on the CPT paper: start from low precision and increase until
    training dynamics show significant improvement.

    Args:
        model: The model to test
        dataloader: DataLoader for training data
        calibrate_fn: Function to calibrate the model after precision change
        threshold: Threshold for detecting significant improvement (default 0.01)
        test_iterations: Number of iterations to test at each precision
        min_bits: Minimum precision to test (default 2)
        max_bits: Maximum precision to test (default 8)
        device: Device to run on

    Returns:
        Tuple of (Bmin, Bmax) optimal precision bounds
    """
    print(f"\nStarting Precision Range Test (PRT)")
    print(f"Testing range: {min_bits}-{max_bits} bits")
    print(f"Test iterations per precision: {test_iterations}")
    print(f"Improvement threshold: {threshold}")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    results = {}
    Bmin_found = None

    # Test each precision level
    for bits in range(min_bits, max_bits + 1):
        print(f"\n{'='*50}")
        print(f"Testing {bits}-bit precision")
        print(f"{'='*50}")

        # Set precision
        model.set_precision(bits, bits)

        # Calibrate if function provided
        if calibrate_fn:
            print(f"Calibrating model at {bits} bits...")
            calibrate_fn(model, dataloader)

        # Track accuracy over iterations
        acc_history = []
        loss_history = []

        # Create iterator for consistent batches
        data_iter = iter(dataloader)

        pbar = tqdm(range(test_iterations), desc=f"{bits}-bit training")
        for i in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                # Compute cross-entropy loss
                logits = outputs
                labels = batch['labels']
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)),
                                            labels.view(-1))

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Calculate accuracy
            acc = calculate_accuracy(model, batch)
            acc_history.append(acc)
            loss_history.append(loss.item())

            # Update progress bar
            if len(acc_history) > 10:
                recent_acc = np.mean(acc_history[-10:])
                pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                 'acc': f'{recent_acc:.4f}'})

        # Analyze training dynamics
        early_acc = np.mean(acc_history[:20]) if len(acc_history) >= 20 else np.mean(acc_history[:len(acc_history)//2])
        late_acc = np.mean(acc_history[-20:]) if len(acc_history) >= 20 else np.mean(acc_history[len(acc_history)//2:])
        improvement = late_acc - early_acc
        avg_acc = np.mean(acc_history)
        avg_loss = np.mean(loss_history)

        results[bits] = {
            'early_acc': early_acc,
            'late_acc': late_acc,
            'improvement': improvement,
            'avg_acc': avg_acc,
            'avg_loss': avg_loss,
            'acc_history': acc_history
        }

        print(f"\nResults for {bits}-bit precision:")
        print(f"  Early accuracy: {early_acc:.4f}")
        print(f"  Late accuracy: {late_acc:.4f}")
        print(f"  Improvement: {improvement:.4f}")
        print(f"  Average accuracy: {avg_acc:.4f}")
        print(f"  Average loss: {avg_loss:.4f}")

        # Check if this is a viable Bmin
        if Bmin_found is None and improvement > threshold:
            Bmin_found = bits
            print(f"\n>>> Found viable Bmin = {bits} (improvement {improvement:.4f} > {threshold})")

        # Early stopping if accuracy is very high
        if avg_acc > 0.95:
            print(f"\n>>> Stopping early due to high accuracy ({avg_acc:.4f})")
            break

    # Determine Bmin and Bmax
    if Bmin_found is None:
        Bmin = min_bits
        print(f"\nNo clear Bmin found, defaulting to {min_bits}")
    else:
        Bmin = Bmin_found

    # Bmax is typically the highest precision tested or where improvement plateaus
    # Find where improvement becomes minimal
    Bmax = max_bits
    for bits in range(Bmin + 1, max_bits):
        if bits in results and bits + 1 in results:
            curr_acc = results[bits]['avg_acc']
            next_acc = results[bits + 1]['avg_acc']
            if (next_acc - curr_acc) / curr_acc < 0.02:  # Less than 2% improvement
                Bmax = bits + 1
                print(f"Improvement plateaus at {Bmax} bits")
                break

    print(f"\n{'='*50}")
    print(f"PRT Results:")
    print(f"  Recommended Bmin: {Bmin}")
    print(f"  Recommended Bmax: {Bmax}")
    print(f"{'='*50}")

    return Bmin, Bmax


def plot_prt_results(results: Dict[int, Dict]) -> None:
    """Plot PRT results for visualization.

    Args:
        results: Dictionary of results from PRT
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    bits = sorted(results.keys())
    avg_accs = [results[b]['avg_acc'] for b in bits]
    improvements = [results[b]['improvement'] for b in bits]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot average accuracy
    ax1.plot(bits, avg_accs, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Precision (bits)')
    ax1.set_ylabel('Average Accuracy')
    ax1.set_title('Accuracy vs Precision')
    ax1.grid(True, alpha=0.3)

    # Plot improvement
    ax2.plot(bits, improvements, 's-', linewidth=2, markersize=8, color='orange')
    ax2.axhline(y=0.01, color='r', linestyle='--', label='Threshold (0.01)')
    ax2.set_xlabel('Precision (bits)')
    ax2.set_ylabel('Improvement (late - early)')
    ax2.set_title('Training Improvement vs Precision')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def test_prt():
    """Test PRT with dummy data."""
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self, vocab_size=1000):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 128)
            self.fc = nn.Linear(128, vocab_size)

        def set_precision(self, bits, activation_bits=None):
            print(f"Setting precision to {bits} bits")

        def forward(self, input_ids, labels=None, **kwargs):
            x = self.embedding(input_ids)
            logits = self.fc(x.mean(dim=1))
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels[:, 0])
                return type('Output', (), {'loss': loss, 'logits': logits})()
            return logits

    # Create dummy data
    dataset = []
    for _ in range(100):
        batch = {
            'input_ids': torch.randint(0, 1000, (4, 32)),
            'labels': torch.randint(0, 1000, (4, 32))
        }
        dataset.append(batch)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    model = DummyModel()

    # Run PRT
    Bmin, Bmax = CPT_PRT(model, dataloader, test_iterations=10, device='cpu')
    print(f"Test complete: Bmin={Bmin}, Bmax={Bmax}")


if __name__ == "__main__":
    test_prt()
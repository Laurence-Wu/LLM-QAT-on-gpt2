"""
CPT Evaluation Script
Evaluate CPT models at different precision levels.
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import json
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import CPT components
from training.cpt_calibrate import calibrate
from models.cpt_model import CPTLMHeadModel


def evaluate_at_precision(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    precision: int,
    calibration_loader: Optional[torch.utils.data.DataLoader] = None,
    calib_batches: int = 10,
    device: str = 'cuda'
) -> Dict[str, float]:
    """Evaluate model at a specific precision.

    Args:
        model: The CPT model to evaluate
        dataloader: DataLoader for evaluation data
        precision: Precision to evaluate at (bits)
        calibration_loader: Optional separate loader for calibration
        calib_batches: Number of batches for calibration
        device: Device to run on

    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\nEvaluating at {precision}-bit precision...")

    model = model.to(device)

    # Set precision
    model.set_precision(precision, precision)

    # Calibrate if loader provided
    if calibration_loader is not None:
        print(f"Calibrating model at {precision} bits...")
        calibrate(model, calibration_loader, calib_batches, device, verbose=False)

    # Evaluation
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval {precision}-bit"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)

            # Get loss
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
                logits = outputs.logits
            else:
                logits = outputs
                labels = batch['labels']
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )

            # Calculate metrics
            total_loss += loss.item() * batch['labels'].numel()
            total_tokens += batch['labels'].numel()

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            labels = batch['labels']
            mask = labels != -100
            if mask.any():
                correct = (predictions[mask] == labels[mask]).sum().item()
                total_correct += correct

            num_batches += 1

    # Calculate final metrics
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    metrics = {
        'precision': precision,
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy,
        'num_batches': num_batches,
        'total_tokens': total_tokens
    }

    return metrics


def evaluate_precision_sweep(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    precision_list: List[int],
    calibration_loader: Optional[torch.utils.data.DataLoader] = None,
    calib_batches: int = 10,
    device: str = 'cuda'
) -> List[Dict[str, float]]:
    """Evaluate model across multiple precision levels.

    Args:
        model: The CPT model to evaluate
        dataloader: DataLoader for evaluation data
        precision_list: List of precisions to evaluate
        calibration_loader: Optional separate loader for calibration
        calib_batches: Number of batches for calibration
        device: Device to run on

    Returns:
        List of metrics dictionaries for each precision
    """
    print(f"\nPrecision Sweep Evaluation")
    print(f"Testing precisions: {precision_list}")
    print("="*60)

    results = []

    for precision in precision_list:
        metrics = evaluate_at_precision(
            model, dataloader, precision,
            calibration_loader, calib_batches, device
        )
        results.append(metrics)

        print(f"\n{precision}-bit Results:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Perplexity: {metrics['perplexity']:.2f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")

    return results


def plot_precision_results(results: List[Dict[str, float]], save_path: Optional[str] = None):
    """Plot evaluation results across precisions.

    Args:
        results: List of metrics dictionaries
        save_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    precisions = [r['precision'] for r in results]
    perplexities = [r['perplexity'] for r in results]
    accuracies = [r['accuracy'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot perplexity
    ax1.plot(precisions, perplexities, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Precision (bits)')
    ax1.set_ylabel('Perplexity')
    ax1.set_title('Perplexity vs Precision')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Lower is better

    # Plot accuracy
    ax2.plot(precisions, accuracies, 's-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Precision (bits)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Precision')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def compare_with_baseline(
    cpt_model: nn.Module,
    baseline_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    precision_list: List[int],
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Compare CPT model with baseline at different precisions.

    Args:
        cpt_model: CPT trained model
        baseline_model: Baseline model (e.g., standard QAT)
        dataloader: Evaluation dataloader
        precision_list: List of precisions to test
        device: Device to run on

    Returns:
        Comparison results dictionary
    """
    print("\nComparing CPT vs Baseline")
    print("="*60)

    # Evaluate CPT model
    cpt_results = evaluate_precision_sweep(
        cpt_model, dataloader, precision_list, device=device
    )

    # Evaluate baseline
    baseline_results = evaluate_precision_sweep(
        baseline_model, dataloader, precision_list, device=device
    )

    # Calculate improvements
    improvements = {}
    for cpt, baseline in zip(cpt_results, baseline_results):
        precision = cpt['precision']
        improvements[precision] = {
            'ppl_improvement': (baseline['perplexity'] - cpt['perplexity']) / baseline['perplexity'] * 100,
            'acc_improvement': (cpt['accuracy'] - baseline['accuracy']) / baseline['accuracy'] * 100
        }

    comparison = {
        'cpt_results': cpt_results,
        'baseline_results': baseline_results,
        'improvements': improvements
    }

    # Print comparison table
    print("\nComparison Results:")
    print(f"{'Precision':<10} {'CPT PPL':<10} {'Base PPL':<10} {'PPL Imp %':<10} {'CPT Acc':<10} {'Base Acc':<10} {'Acc Imp %':<10}")
    print("-"*70)

    for precision in precision_list:
        cpt = next(r for r in cpt_results if r['precision'] == precision)
        base = next(r for r in baseline_results if r['precision'] == precision)
        imp = improvements[precision]

        print(f"{precision:<10} {cpt['perplexity']:<10.2f} {base['perplexity']:<10.2f} "
              f"{imp['ppl_improvement']:<10.2f} {cpt['accuracy']:<10.4f} "
              f"{base['accuracy']:<10.4f} {imp['acc_improvement']:<10.2f}")

    return comparison


def load_checkpoint(checkpoint_path: str, device: str = 'cuda') -> tuple:
    """Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load to

    Returns:
        Tuple of (model, config, history)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config
    config = checkpoint['config']

    # Create model
    from models.cpt_model import CPTLMHeadModel
    from transformers import GPT2Config

    model_config = GPT2Config(**config['model'])
    model = CPTLMHeadModel(model_config)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Get history
    history = checkpoint.get('history', {})

    return model, config, history


def main():
    """Main evaluation entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='CPT Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to evaluation data')
    parser.add_argument('--precisions', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8],
                       help='Precisions to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--save_plot', type=str, default=None,
                       help='Path to save evaluation plot')
    parser.add_argument('--save_results', type=str, default=None,
                       help='Path to save evaluation results JSON')
    args = parser.parse_args()

    # Load model
    model, config, history = load_checkpoint(args.checkpoint, args.device)

    # Create dataloader
    from utils.dataset import create_dataloaders
    _, val_dataloader, _ = create_dataloaders(
        args.data_path,
        batch_size=args.batch_size,
        max_length=config['data'].get('max_length', 1024)
    )

    # Run evaluation
    results = evaluate_precision_sweep(
        model, val_dataloader, args.precisions, device=args.device
    )

    # Save results if requested
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.save_results}")

    # Plot results
    if args.save_plot:
        plot_precision_results(results, args.save_plot)

    # Print summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)

    best_ppl_result = min(results, key=lambda x: x['perplexity'])
    best_acc_result = max(results, key=lambda x: x['accuracy'])

    print(f"Best Perplexity: {best_ppl_result['perplexity']:.2f} at {best_ppl_result['precision']} bits")
    print(f"Best Accuracy: {best_acc_result['accuracy']:.4f} at {best_acc_result['precision']} bits")


if __name__ == "__main__":
    # If no arguments, run test evaluation
    import sys
    if len(sys.argv) == 1:
        print("Running test evaluation...")

        # Create dummy model
        from models.cpt_model import CPTLMHeadModel
        from transformers import GPT2Config

        config = GPT2Config(
            vocab_size=1000,
            n_positions=128,
            n_embd=128,
            n_layer=2,
            n_head=2
        )
        model = CPTLMHeadModel(config)

        # Create dummy data
        data = []
        for _ in range(20):
            batch = {
                'input_ids': torch.randint(0, 1000, (4, 32)),
                'labels': torch.randint(0, 1000, (4, 32))
            }
            data.append(batch)
        dataloader = torch.utils.data.DataLoader(data, batch_size=1)

        # Run evaluation
        results = evaluate_precision_sweep(
            model, dataloader, [2, 4, 8], device='cpu'
        )

        print("\nTest evaluation complete!")
    else:
        main()
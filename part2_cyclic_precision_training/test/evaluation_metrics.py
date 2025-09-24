"""
Evaluation metrics for CPT models.
Includes perplexity, accuracy, and efficiency metrics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
import time


class CPTEvaluator:
    """Evaluator for CPT models with various metrics."""

    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.to(device)

    def calculate_perplexity(self, dataloader, precision: int = 8) -> float:
        """Calculate perplexity at specified precision."""
        self.model.eval()
        self.model.set_precision(precision)

        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Calculating perplexity at {precision}-bit"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                total_loss += loss.item() * labels.numel()
                total_tokens += labels.numel()

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        return perplexity

    def calculate_accuracy(self, dataloader, precision: int = 8) -> float:
        """Calculate token-level accuracy."""
        self.model.eval()
        self.model.set_precision(precision)

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Calculating accuracy at {precision}-bit"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids)
                predictions = outputs.logits.argmax(dim=-1)

                # Shift predictions to align with labels
                predictions = predictions[:, :-1]
                labels = labels[:, 1:]

                mask = labels != -100
                correct += (predictions[mask] == labels[mask]).sum().item()
                total += mask.sum().item()

        accuracy = correct / total if total > 0 else 0
        return accuracy * 100

    def measure_inference_speed(self, dataloader, precision: int = 8, num_batches: int = 100) -> Dict:
        """Measure inference speed at specified precision."""
        self.model.eval()
        self.model.set_precision(precision)

        # Warmup
        for i, batch in enumerate(dataloader):
            if i >= 10:
                break
            input_ids = batch['input_ids'].to(self.device)
            with torch.no_grad():
                _ = self.model(input_ids=input_ids)

        # Measure
        times = []
        tokens_processed = 0

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            input_ids = batch['input_ids'].to(self.device)
            batch_size, seq_len = input_ids.shape

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            with torch.no_grad():
                _ = self.model(input_ids=input_ids)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            times.append(end_time - start_time)
            tokens_processed += batch_size * seq_len

        avg_time = np.mean(times)
        tokens_per_second = tokens_processed / sum(times)

        return {
            'avg_batch_time': avg_time,
            'tokens_per_second': tokens_per_second,
            'total_time': sum(times)
        }

    def calculate_bitops(self, batch_size: int, seq_len: int, precision: int) -> int:
        """
        Calculate BitOPs (bit operations) for given precision.
        This is an approximation based on model architecture.
        """
        model_config = self.model.config['model']
        n_layer = model_config.n_layer
        n_embd = model_config.n_embd
        vocab_size = model_config.vocab_size

        # Embedding lookups
        embedding_ops = batch_size * seq_len * n_embd * precision

        # Attention operations per layer
        # Q, K, V projections + output projection
        attention_ops = 4 * batch_size * seq_len * n_embd * n_embd * precision

        # Attention computation
        attention_compute = batch_size * n_embd * seq_len * seq_len * precision

        # MLP operations per layer
        mlp_ops = 2 * batch_size * seq_len * n_embd * 4 * n_embd * precision

        # Total per layer
        ops_per_layer = attention_ops + attention_compute + mlp_ops

        # Total for all layers
        total_ops = embedding_ops + n_layer * ops_per_layer

        # LM head
        lm_head_ops = batch_size * seq_len * n_embd * vocab_size * precision

        total_bitops = total_ops + lm_head_ops

        return total_bitops

    def evaluate_all_precisions(self, dataloader) -> Dict[int, Dict]:
        """Evaluate model at all configured precisions."""
        results = {}
        bit_widths = self.model.config['model'].bit_widths

        for precision in bit_widths:
            print(f"\nEvaluating at {precision}-bit precision...")

            # Calculate metrics
            perplexity = self.calculate_perplexity(dataloader, precision)
            accuracy = self.calculate_accuracy(dataloader, precision)
            speed_metrics = self.measure_inference_speed(dataloader, precision)

            # Calculate BitOPs for a sample batch
            sample_batch = next(iter(dataloader))
            batch_size, seq_len = sample_batch['input_ids'].shape
            bitops = self.calculate_bitops(batch_size, seq_len, precision)

            results[precision] = {
                'perplexity': perplexity,
                'accuracy': accuracy,
                'tokens_per_second': speed_metrics['tokens_per_second'],
                'avg_batch_time': speed_metrics['avg_batch_time'],
                'bitops': bitops
            }

            print(f"  Perplexity: {perplexity:.2f}")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Speed: {speed_metrics['tokens_per_second']:.2f} tokens/sec")
            print(f"  BitOPs: {bitops / 1e9:.2f} GBitOPs")

        return results


class ComparativeEvaluator:
    """Compare CPT model with baseline models."""

    def __init__(self, cpt_model, baseline_model=None):
        self.cpt_model = cpt_model
        self.baseline_model = baseline_model

    def compare_loss_landscape(self, dataloader, num_points: int = 50) -> Dict:
        """
        Analyze loss landscape sharpness.
        Wider minima indicate better generalization (as shown in CPT paper).
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Get a batch
        batch = next(iter(dataloader))
        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()

        # Get loss at current point
        self.cpt_model.eval()
        with torch.no_grad():
            base_output = self.cpt_model(input_ids=input_ids, labels=labels)
            base_loss = base_output.loss.item()

        # Sample random directions
        losses = []
        model_params = list(self.cpt_model.parameters())

        # Generate random perturbations
        direction1 = [torch.randn_like(p) for p in model_params]
        direction2 = [torch.randn_like(p) for p in model_params]

        # Normalize directions
        norm1 = sum([d.norm() ** 2 for d in direction1]) ** 0.5
        norm2 = sum([d.norm() ** 2 for d in direction2]) ** 0.5
        direction1 = [d / norm1 for d in direction1]
        direction2 = [d / norm2 for d in direction2]

        # Sample points in 2D space
        alphas = np.linspace(-0.5, 0.5, num_points)
        betas = np.linspace(-0.5, 0.5, num_points)

        loss_surface = np.zeros((num_points, num_points))

        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                # Perturb parameters
                with torch.no_grad():
                    for p, d1, d2 in zip(model_params, direction1, direction2):
                        p.data.add_(alpha * d1 + beta * d2)

                    # Calculate loss
                    output = self.cpt_model(input_ids=input_ids, labels=labels)
                    loss_surface[i, j] = output.loss.item()

                    # Restore parameters
                    for p, d1, d2 in zip(model_params, direction1, direction2):
                        p.data.sub_(alpha * d1 + beta * d2)

        return {
            'loss_surface': loss_surface,
            'alphas': alphas,
            'betas': betas,
            'base_loss': base_loss
        }

    def compare_precision_robustness(self, dataloader) -> Dict:
        """Compare model robustness across different precisions."""
        cpt_evaluator = CPTEvaluator(self.cpt_model)

        # Evaluate CPT model
        cpt_results = cpt_evaluator.evaluate_all_precisions(dataloader)

        # Calculate precision sensitivity
        precisions = sorted(cpt_results.keys())
        perplexity_variance = np.var([cpt_results[p]['perplexity'] for p in precisions])
        accuracy_variance = np.var([cpt_results[p]['accuracy'] for p in precisions])

        return {
            'cpt_results': cpt_results,
            'perplexity_variance': perplexity_variance,
            'accuracy_variance': accuracy_variance,
            'precision_robustness': 1.0 / (1.0 + perplexity_variance)  # Higher is better
        }


def print_evaluation_summary(results: Dict):
    """Print formatted evaluation summary."""
    print("\n" + "=" * 60)
    print("CPT Model Evaluation Summary")
    print("=" * 60)

    for precision, metrics in results.items():
        print(f"\n{precision}-bit Precision:")
        print(f"  Perplexity: {metrics['perplexity']:.2f}")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Speed: {metrics['tokens_per_second']:.2f} tokens/sec")
        print(f"  BitOPs: {metrics['bitops'] / 1e9:.2f} GBitOPs")

    # Calculate efficiency improvement
    if 8 in results and 32 in results:
        bitops_reduction = (1 - results[8]['bitops'] / results[32]['bitops']) * 100
        print(f"\n8-bit vs 32-bit BitOPs reduction: {bitops_reduction:.1f}%")

    print("=" * 60)
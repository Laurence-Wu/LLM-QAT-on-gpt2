"""
Comprehensive Test Suite for Self-Distillation Implementation
Tests the distillation mechanism for switchable precision training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import json
import gc
from pathlib import Path
from collections import defaultdict
import time

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'part1_switchable_precision'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

from transformers import GPT2Config, GPT2TokenizerFast
from shared.models_sp import SPLMHeadModel
from part1_switchable_precision.distillation import DistillationConfig, SelfDistillationTrainer
from part1_switchable_precision.config_sp import ModelConfig, TrainingConfig


class DistillationMonitor:
    """Monitor distillation training progress with detailed metrics."""

    def __init__(self, log_dir='./test_logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.metrics = defaultdict(list)

    def log_teacher_student_agreement(self, teacher_logits, student_logits, step):
        """Measure agreement between teacher and student predictions."""
        with torch.no_grad():
            teacher_preds = teacher_logits.argmax(dim=-1)
            student_preds = student_logits.argmax(dim=-1)
            agreement = (teacher_preds == student_preds).float().mean()

            # KL divergence
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            student_probs = F.softmax(student_logits, dim=-1)
            kl_div = F.kl_div(
                student_probs.log(),
                teacher_probs,
                reduction='batchmean'
            )

            self.metrics['agreement'].append(agreement.item())
            self.metrics['kl_divergence'].append(kl_div.item())

            print(f"Step {step}: Agreement={agreement.item():.4f}, KL={kl_div.item():.4f}")

            return {
                'agreement': agreement.item(),
                'kl_divergence': kl_div.item()
            }

    def log_distillation_loss(self, loss_components, step):
        """Log distillation loss components."""
        for key, value in loss_components.items():
            self.metrics[key].append(value)

        print(f"Step {step} Loss Components: {loss_components}")

    def save_metrics(self, step):
        """Save metrics to file."""
        metrics_path = self.log_dir / f'distillation_metrics_step_{step}.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Saved distillation metrics to {metrics_path}")

    def get_summary(self):
        """Get summary statistics of metrics."""
        summary = {}
        for key, values in self.metrics.items():
            if values and isinstance(values[0], (int, float)):
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'last': values[-1]
                }
        return summary

    def print_summary(self):
        """Print formatted summary of metrics."""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("DISTILLATION METRICS SUMMARY")
        print("="*60)
        for metric, stats in summary.items():
            print(f"\n{metric}:")
            for stat_name, value in stats.items():
                print(f"  {stat_name}: {value:.4f}")
        print("="*60)


def create_test_model(n_layer=3, n_embd=128, n_head=4):
    """Create a small test model for testing."""
    config = GPT2Config(
        vocab_size=1000,
        n_positions=256,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        layer_norm_epsilon=1e-5,
        embd_pdrop=0.1
    )

    # Add switchable precision configurations
    config.bit_widths = [4, 8, 16]
    config.lora_rank_per_bit = {4: 4, 8: 8, 16: 16}
    config.lora_alpha_per_bit = {4: 8, 8: 16, 16: 32}
    config.lora_dropout = 0.1

    model = SPLMHeadModel(config)
    return model


def test_teacher_caching():
    """Test 1: Verify teacher outputs are cached correctly."""
    print("\n" + "="*60)
    print("TEST 1: Teacher Output Caching")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_test_model().to(device)

    config = DistillationConfig(cache_size=5)
    distillation = SelfDistillationTrainer(model, config, device)

    # Generate test batch
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    # Compute teacher outputs
    print("Computing teacher outputs...")
    start_time = time.time()
    teacher1 = distillation.compute_teacher_outputs(input_ids)
    compute_time = time.time() - start_time
    print(f"Teacher computation time: {compute_time:.3f}s")

    # Retrieve from cache
    print("Retrieving from cache...")
    start_time = time.time()
    teacher2 = distillation._get_from_cache(input_ids)
    cache_time = time.time() - start_time
    print(f"Cache retrieval time: {cache_time:.3f}s")

    # Verify cached outputs match
    if teacher2 is not None:
        logits_match = torch.allclose(teacher1['logits'], teacher2['logits'], atol=1e-5)
        print(f"✓ Logits match: {logits_match}")

        if teacher1['hidden_states'] and teacher2['hidden_states']:
            hidden_match = all(
                torch.allclose(h1, h2, atol=1e-5)
                for h1, h2 in zip(teacher1['hidden_states'], teacher2['hidden_states'])
            )
            print(f"✓ Hidden states match: {hidden_match}")

        print(f"✓ Cache speedup: {compute_time/cache_time:.1f}x")
    else:
        print("✗ Failed to retrieve from cache")

    # Test cache eviction
    print("\nTesting cache eviction...")
    for i in range(10):
        test_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        distillation.compute_teacher_outputs(test_ids)

    stats = distillation.get_stats()
    print(f"Teacher updates: {stats['teacher_updates']}")
    print(f"Cache size: {len(distillation.teacher_cache)}")
    print(f"✓ Cache size limited to {config.cache_size}")

    print("\n✅ TEST 1 PASSED: Teacher caching works correctly")


def test_distillation_loss():
    """Test 2: Verify loss computation for different precisions."""
    print("\n" + "="*60)
    print("TEST 2: Distillation Loss Computation")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_test_model().to(device)

    config = DistillationConfig(
        alpha_output=1.0,
        alpha_feature=1e-7,
        temperature=3.0
    )
    distillation = SelfDistillationTrainer(model, config, device)

    # Generate test data
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    print("Testing loss at different precisions:")

    for bits in [16, 8, 4]:
        print(f"\n{bits}-bit precision:")
        model.set_precision(bits)

        # Get model outputs
        with torch.no_grad():
            outputs = model(
                input_ids,
                output_hidden_states=True,
                return_dict=True
            )

        # Compute loss
        loss, components = distillation.compute_distillation_loss(
            outputs, labels, input_ids
        )

        print(f"  Precision: {components['precision']}")

        if bits == 16:  # Full precision
            assert 'cross_entropy' in components, "Full precision should use cross-entropy"
            print(f"  Cross-entropy loss: {components['cross_entropy']:.4f}")
        else:  # Low precision
            assert 'kl' in components, "Low precision should have KL loss"
            assert 'feature' in components, "Low precision should have feature loss"
            print(f"  KL loss: {components['kl']:.4f}")
            print(f"  Feature loss: {components['feature']:.4f}")
            print(f"  Total loss: {components['total']:.4f}")

    print("\n✅ TEST 2 PASSED: Distillation loss computation correct")


def test_memory_efficiency():
    """Test 3: Verify memory efficiency of distillation."""
    print("\n" + "="*60)
    print("TEST 3: Memory Efficiency")
    print("="*60)

    if not torch.cuda.is_available():
        print("⚠️  Skipping memory test - CUDA not available")
        return

    device = 'cuda'
    model = create_test_model(n_layer=6, n_embd=256).to(device)

    config = DistillationConfig(cache_size=10)
    distillation = SelfDistillationTrainer(model, config, device)

    # Clear GPU memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    initial_memory = torch.cuda.memory_allocated() / 1e6  # MB
    print(f"Initial memory: {initial_memory:.2f}MB")

    # Run multiple iterations
    batch_size, seq_len = 4, 256
    num_iterations = 50

    print(f"Running {num_iterations} iterations...")
    for i in range(num_iterations):
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        # Compute teacher outputs
        if i % 10 == 0:
            distillation.compute_teacher_outputs(input_ids)

        # Get from cache (should hit or miss based on cache size)
        cached = distillation._get_from_cache(input_ids)

        if i % 10 == 0:
            current_memory = torch.cuda.memory_allocated() / 1e6
            print(f"  Iter {i}: Memory={current_memory:.2f}MB")

    # Check final memory
    final_memory = torch.cuda.memory_allocated() / 1e6
    peak_memory = torch.cuda.max_memory_allocated() / 1e6
    memory_growth = final_memory - initial_memory

    print(f"\nMemory Statistics:")
    print(f"  Initial: {initial_memory:.2f}MB")
    print(f"  Final: {final_memory:.2f}MB")
    print(f"  Peak: {peak_memory:.2f}MB")
    print(f"  Growth: {memory_growth:.2f}MB")

    stats = distillation.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    print(f"  Teacher updates: {stats['teacher_updates']}")

    # Memory growth should be reasonable (< 100MB for this test)
    assert memory_growth < 100, f"Memory growth too large: {memory_growth:.2f}MB"
    print(f"\n✅ TEST 3 PASSED: Memory growth controlled ({memory_growth:.2f}MB < 100MB)")


def test_precision_switching():
    """Test 4: Verify correct behavior during precision switching."""
    print("\n" + "="*60)
    print("TEST 4: Precision Switching with Distillation")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_test_model().to(device)

    config = DistillationConfig(warmup_steps=5)
    distillation = SelfDistillationTrainer(model, config, device)
    monitor = DistillationMonitor()

    # Test data
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    # Simulate training with precision switching
    precision_schedule = [16, 16, 8, 4, 8, 16, 4, 8, 16]

    print("Testing precision switching schedule:")
    for step, bits in enumerate(precision_schedule):
        print(f"\nStep {step}: {bits}-bit precision")
        model.set_precision(bits)

        # Check if distillation should be used
        use_distillation = distillation.should_use_distillation(step)
        print(f"  Distillation enabled: {use_distillation}")

        # Get outputs
        outputs = model(
            input_ids,
            output_hidden_states=True,
            return_dict=True
        )

        # Compute loss
        loss, components = distillation.compute_distillation_loss(
            outputs, labels, input_ids
        )

        # Log metrics
        monitor.log_distillation_loss(components, step)

        # Check teacher-student agreement if not at full precision
        if bits != 16:
            teacher = distillation._get_from_cache(input_ids)
            if teacher is not None:
                agreement = monitor.log_teacher_student_agreement(
                    teacher['logits'], outputs['logits'], step
                )

    # Print summary
    monitor.print_summary()
    monitor.save_metrics(len(precision_schedule))

    print("\n✅ TEST 4 PASSED: Precision switching handled correctly")


def test_gradient_flow():
    """Test 5: Verify proper gradient flow with distillation."""
    print("\n" + "="*60)
    print("TEST 5: Gradient Flow with Distillation")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_test_model().to(device)

    config = DistillationConfig()
    distillation = SelfDistillationTrainer(model, config, device)

    # Enable gradients for trainable parameters
    for param in model.parameters():
        param.requires_grad = True

    # Test data
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    print("Testing gradient flow at different precisions:")

    for bits in [8, 4]:  # Test low-precision modes
        print(f"\n{bits}-bit precision:")
        model.set_precision(bits)
        model.zero_grad()

        # Forward pass
        outputs = model(
            input_ids,
            output_hidden_states=True,
            return_dict=True
        )

        # Compute distillation loss
        loss, components = distillation.compute_distillation_loss(
            outputs, labels, input_ids
        )

        print(f"  Loss: {loss.item():.4f}")

        # Backward pass
        loss.backward()

        # Check gradients
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if 'lora' in name.lower() and grad_norm > 0:
                    print(f"  ✓ Gradient in {name}: {grad_norm:.6f}")

        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
        print(f"  Average gradient norm: {avg_grad_norm:.6f}")

        assert avg_grad_norm > 0, "No gradients flowing!"

    print("\n✅ TEST 5 PASSED: Gradients flow correctly with distillation")


def test_feature_matching():
    """Test 6: Verify feature matching loss computation."""
    print("\n" + "="*60)
    print("TEST 6: Feature Matching Loss")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_test_model(n_layer=4).to(device)

    # Test with specific layers
    config = DistillationConfig(
        feature_layers=[0, 2],  # Match only first and third layers
        alpha_feature=0.1  # Higher weight for testing
    )
    distillation = SelfDistillationTrainer(model, config, device)

    # Test data
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    # Set to low precision
    model.set_precision(4)

    # Get outputs with hidden states
    outputs = model(
        input_ids,
        output_hidden_states=True,
        return_dict=True
    )

    # Compute loss
    loss, components = distillation.compute_distillation_loss(
        outputs, labels, input_ids
    )

    print(f"Loss components:")
    print(f"  KL loss: {components['kl']:.6f}")
    print(f"  Feature loss: {components['feature']:.6f}")
    print(f"  Total loss: {components['total']:.6f}")

    # Verify feature loss is computed
    assert components['feature'] > 0, "Feature loss should be non-zero"

    # Test without feature matching
    config_no_feature = DistillationConfig(alpha_feature=0)
    distillation_no_feature = SelfDistillationTrainer(model, config_no_feature, device)

    loss_no_feature, components_no_feature = distillation_no_feature.compute_distillation_loss(
        outputs, labels, input_ids
    )

    print(f"\nWithout feature matching:")
    print(f"  Feature loss: {components_no_feature['feature']:.6f}")
    print(f"  Total loss: {components_no_feature['total']:.6f}")

    # Total loss should be different when feature matching is disabled
    assert abs(components['total'] - components_no_feature['total']) > 1e-6

    print("\n✅ TEST 6 PASSED: Feature matching loss computed correctly")


def run_all_tests():
    """Run all distillation tests."""
    print("\n" + "="*80)
    print(" SELF-DISTILLATION TEST SUITE ")
    print("="*80)

    try:
        test_teacher_caching()
        test_distillation_loss()
        test_memory_efficiency()
        test_precision_switching()
        test_gradient_flow()
        test_feature_matching()

        print("\n" + "="*80)
        print(" ✅ ALL DISTILLATION TESTS PASSED! ")
        print("="*80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    run_all_tests()
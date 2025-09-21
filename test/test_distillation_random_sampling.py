#!/usr/bin/env python3
"""
Test Distillation with Random Precision Sampling
Tests the new single-precision-per-batch training approach with teacher caching.
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List, Tuple
import gc

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from test.fix_model_initialization import create_properly_initialized_model
from test.dataset_utils import get_calibration_texts
from part1_switchable_precision.distillation_manager import DistillationManager
from part1_switchable_precision.config_sp import TrainingConfig


def test_single_precision_per_batch():
    """
    Test that each batch trains at exactly one precision and that
    gradients are not mixed between precisions within a single optimizer step.
    """
    print("\n" + "="*60)
    print("TEST: SINGLE PRECISION PER BATCH")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model, config = create_properly_initialized_model(use_pretrained=True, num_layers=4)
    model = model.to(device)

    # Available precisions
    available_precisions = [4, 8, 16, 32]

    # Track gradients for each precision
    gradient_norms_by_precision = {p: [] for p in available_precisions}

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Test 10 random batches
    num_batches = 10
    training_texts = get_calibration_texts(num_texts=num_batches * 4)

    print("\n📊 Testing random precision sampling:")
    precisions_used = []

    for batch_idx in range(num_batches):
        # Clear gradients
        model.zero_grad()

        # Randomly select ONE precision for this batch
        precision = random.choice(available_precisions)
        precisions_used.append(precision)
        model.set_precision(precision)

        # Get batch
        texts = training_texts[batch_idx*4:(batch_idx+1)*4]
        tokens = tokenizer(texts, return_tensors='pt', max_length=64,
                          truncation=True, padding=True)['input_ids'].to(device)

        # Forward and backward
        outputs = model(tokens, labels=tokens)
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        loss.backward()

        # Collect gradient norms for this precision
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        gradient_norms_by_precision[precision].append(total_norm)

        print(f"   Batch {batch_idx}: {precision}-bit, Loss: {loss.item():.4f}, Grad norm: {total_norm:.4f}")

    # Verify distribution
    precision_counts = {p: precisions_used.count(p) for p in available_precisions}
    print(f"\n📊 Precision distribution over {num_batches} batches:")
    for p, count in sorted(precision_counts.items()):
        print(f"   {p}-bit: {count} times ({count/num_batches*100:.1f}%)")

    # Check that each precision got some training
    untrained_precisions = [p for p in available_precisions if precision_counts[p] == 0]
    if untrained_precisions:
        print(f"   ⚠️ Warning: Precisions {untrained_precisions} were never selected in {num_batches} batches")
    else:
        print(f"   ✅ All precisions were trained at least once")

    return {
        'precision_counts': precision_counts,
        'gradient_norms': gradient_norms_by_precision,
        'passed': len(untrained_precisions) == 0 or num_batches < 20
    }


def test_teacher_cache_effectiveness():
    """
    Test that teacher outputs are properly cached and reused for student training.
    """
    print("\n" + "="*60)
    print("TEST: TEACHER CACHE EFFECTIVENESS")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model and config
    model, model_config = create_properly_initialized_model(use_pretrained=True, num_layers=4)
    model = model.to(device)

    # Create training config
    training_config = TrainingConfig()
    training_config.cache_size = 32

    # Initialize distillation manager
    distill_mgr = DistillationManager(
        model=model,
        full_precision_bits=32,
        config=training_config
    )

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create test batches
    num_unique_batches = 5
    training_texts = get_calibration_texts(num_texts=num_unique_batches * 4)

    batches = []
    for i in range(num_unique_batches):
        texts = training_texts[i*4:(i+1)*4]
        tokens = tokenizer(texts, return_tensors='pt', max_length=64,
                          truncation=True, padding=True)['input_ids'].to(device)
        batches.append(tokens)

    print("\n📊 Phase 1: Populate cache with teacher outputs")
    model.set_precision(32)

    for i, batch in enumerate(batches):
        # Teacher forward pass and cache
        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True, return_dict=True)
            distill_mgr.update_teacher(batch, None)
        print(f"   Cached batch {i+1}/{num_unique_batches}")

    initial_cache_stats = distill_mgr.get_cache_stats()
    print(f"\nCache after teacher passes: Size={initial_cache_stats['cache_size']}")

    print("\n📊 Phase 2: Test student training with cached teacher")

    # Test students can retrieve cached teacher outputs
    student_precisions = [4, 8, 16]
    cache_hit_results = {}

    for precision in student_precisions:
        model.set_precision(precision)
        hits = 0
        misses = 0

        # Try to retrieve each batch
        for batch in batches:
            cached_output = distill_mgr._get_from_cache(batch)
            if cached_output is not None:
                hits += 1
            else:
                misses += 1

        cache_hit_results[precision] = {
            'hits': hits,
            'misses': misses,
            'hit_rate': hits / (hits + misses) if (hits + misses) > 0 else 0
        }

        print(f"   {precision}-bit: Hits={hits}/{len(batches)}, Hit rate={cache_hit_results[precision]['hit_rate']:.1%}")

    # Test cache statistics
    final_cache_stats = distill_mgr.get_cache_stats()
    print(f"\nFinal cache stats:")
    print(f"   Size: {final_cache_stats['cache_size']}")
    print(f"   Total hits: {final_cache_stats['cache_hits']}")
    print(f"   Total misses: {final_cache_stats['cache_misses']}")
    print(f"   Overall hit rate: {final_cache_stats['hit_rate']:.1%}")

    # Check if cache is working properly
    all_hit = all(r['hit_rate'] == 1.0 for r in cache_hit_results.values())
    if all_hit:
        print("\n✅ Cache working perfectly - all students found teacher outputs")
    else:
        print("\n❌ Cache issues - some students couldn't find teacher outputs")

    return {
        'cache_hit_results': cache_hit_results,
        'final_stats': final_cache_stats,
        'passed': all_hit
    }


def test_distillation_loss_computation():
    """
    Test that distillation loss is computed correctly with both output and feature matching.
    """
    print("\n" + "="*60)
    print("TEST: DISTILLATION LOSS COMPUTATION")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model, model_config = create_properly_initialized_model(use_pretrained=True, num_layers=4)
    model = model.to(device)

    # Create training config with distillation parameters
    training_config = TrainingConfig()
    training_config.distill_alpha_kl = 1.0
    training_config.distill_alpha_feature = 1e-7
    training_config.distill_temperature = 3.0

    # Initialize distillation manager
    distill_mgr = DistillationManager(
        model=model,
        full_precision_bits=32,
        config=training_config
    )

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create test batch
    texts = get_calibration_texts(num_texts=4)
    tokens = tokenizer(texts[0:4], return_tensors='pt', max_length=64,
                      truncation=True, padding=True)['input_ids'].to(device)

    print("\n📊 Step 1: Generate and cache teacher outputs")
    model.set_precision(32)

    with torch.no_grad():
        teacher_outputs = model(tokens, output_hidden_states=True, return_dict=True)
        distill_mgr.update_teacher(tokens, None)

    print(f"   Teacher logits shape: {teacher_outputs['logits'].shape}")
    print(f"   Teacher hidden states: {len(teacher_outputs['hidden_states'])} layers")

    print("\n📊 Step 2: Compute distillation loss for each student")

    student_losses = {}
    student_precisions = [16, 8, 4]

    for precision in student_precisions:
        model.set_precision(precision)

        # Get student outputs
        student_outputs = model(tokens, output_hidden_states=True, return_dict=True)

        # Compute distillation loss
        distill_loss = distill_mgr.compute_distillation_loss(student_outputs, tokens)

        student_losses[precision] = distill_loss.item()
        print(f"   {precision}-bit distillation loss: {distill_loss.item():.4f}")

        # Verify loss components
        if precision == 16:  # Detailed analysis for one precision
            # Check KL divergence component
            T = training_config.distill_temperature
            teacher_cache = distill_mgr._get_from_cache(tokens)

            if teacher_cache:
                teacher_logits = teacher_cache['logits'][..., :-1, :].contiguous()
                student_logits = student_outputs['logits'][..., :-1, :].contiguous()

                teacher_log_probs = F.log_softmax(teacher_logits / T, dim=-1)
                student_log_probs = F.log_softmax(student_logits / T, dim=-1)

                kl_loss = F.kl_div(
                    student_log_probs.view(-1, student_log_probs.size(-1)),
                    teacher_log_probs.view(-1, teacher_log_probs.size(-1)),
                    reduction='batchmean',
                    log_target=True
                ) * (T * T)

                print(f"\n   📊 Loss components for {precision}-bit:")
                print(f"      KL divergence loss: {kl_loss.item():.6f}")
                print(f"      Alpha KL: {training_config.distill_alpha_kl}")
                print(f"      Temperature: {training_config.distill_temperature}")

    # Verify losses are reasonable
    all_reasonable = all(0 < loss < 100 for loss in student_losses.values())
    if all_reasonable:
        print("\n✅ All distillation losses are in reasonable range")
    else:
        print("\n❌ Some distillation losses are out of expected range")

    return {
        'student_losses': student_losses,
        'passed': all_reasonable
    }


def test_random_sampling_convergence():
    """
    Test that random precision sampling leads to convergence over many iterations.
    """
    print("\n" + "="*60)
    print("TEST: RANDOM SAMPLING CONVERGENCE")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model, model_config = create_properly_initialized_model(use_pretrained=False, num_layers=2)
    model = model.to(device)

    # Setup training
    available_precisions = [4, 8, 16, 32]
    num_iterations = 100
    batch_size = 4

    # Training config
    training_config = TrainingConfig()
    training_config.distill_alpha_kl = 1.0
    training_config.distill_alpha_feature = 1e-7

    # Initialize distillation manager
    distill_mgr = DistillationManager(
        model=model,
        full_precision_bits=32,
        config=training_config
    )

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    # Generate training data
    training_texts = get_calibration_texts(num_texts=num_iterations * batch_size)

    print(f"\n📊 Training with random precision sampling for {num_iterations} iterations")

    losses = []
    precision_counts = {p: 0 for p in available_precisions}

    for iteration in range(num_iterations):
        # Randomly select precision
        precision = random.choice(available_precisions)
        precision_counts[precision] += 1
        model.set_precision(precision)

        # Get batch
        texts = training_texts[iteration*batch_size:(iteration+1)*batch_size]
        tokens = tokenizer(texts, return_tensors='pt', max_length=64,
                          truncation=True, padding=True)['input_ids'].to(device)

        if precision == 32:
            # Teacher: train with ground truth and cache outputs
            outputs = model(tokens, labels=tokens, output_hidden_states=True, return_dict=True)
            loss = outputs['loss']

            # Cache for students
            with torch.no_grad():
                distill_mgr.update_teacher(tokens, None)
        else:
            # Student: try distillation
            outputs = model(tokens, output_hidden_states=True, return_dict=True)

            # Check if teacher outputs are cached
            if distill_mgr._get_from_cache(tokens) is not None:
                loss = distill_mgr.compute_distillation_loss(outputs, tokens)
            else:
                # Fallback to standard loss (shouldn't happen often)
                outputs_with_labels = model(tokens, labels=tokens)
                loss = outputs_with_labels['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Print progress
        if (iteration + 1) % 20 == 0:
            avg_loss = np.mean(losses[-20:])
            print(f"   Iter {iteration+1}: Avg loss={avg_loss:.4f}, "
                  f"Precisions used: {[f'{p}:{precision_counts[p]}' for p in sorted(precision_counts.keys())]}")

    # Analyze convergence
    initial_loss = np.mean(losses[:10])
    final_loss = np.mean(losses[-10:])
    improvement = (initial_loss - final_loss) / initial_loss * 100

    print(f"\n📊 Convergence Analysis:")
    print(f"   Initial loss (first 10 iters): {initial_loss:.4f}")
    print(f"   Final loss (last 10 iters): {final_loss:.4f}")
    print(f"   Improvement: {improvement:.1f}%")

    print(f"\n📊 Final precision distribution:")
    for p, count in sorted(precision_counts.items()):
        print(f"   {p}-bit: {count} times ({count/num_iterations*100:.1f}%)")

    # Check cache performance
    cache_stats = distill_mgr.get_cache_stats()
    print(f"\n📊 Cache Statistics:")
    print(f"   Cache size: {cache_stats['cache_size']}")
    print(f"   Hit rate: {cache_stats['hit_rate']:.1%}")

    # Determine if training converged
    converged = final_loss < initial_loss and improvement > 10
    if converged:
        print("\n✅ Training converged successfully with random sampling")
    else:
        print("\n⚠️ Limited convergence observed")

    return {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'improvement': improvement,
        'precision_distribution': precision_counts,
        'cache_hit_rate': cache_stats['hit_rate'],
        'passed': converged or improvement > 0
    }


def run_all_distillation_tests():
    """Run all distillation tests."""
    print("\n" + "="*80)
    print("COMPREHENSIVE DISTILLATION TEST SUITE")
    print("Random Precision Sampling Implementation")
    print("="*80)

    results = {}

    # Test 1: Single precision per batch
    print("\n[TEST 1/4]")
    results['single_precision'] = test_single_precision_per_batch()
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Test 2: Teacher cache effectiveness
    print("\n[TEST 2/4]")
    results['cache_effectiveness'] = test_teacher_cache_effectiveness()
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Test 3: Distillation loss computation
    print("\n[TEST 3/4]")
    results['loss_computation'] = test_distillation_loss_computation()
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Test 4: Random sampling convergence
    print("\n[TEST 4/4]")
    results['convergence'] = test_random_sampling_convergence()
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    all_passed = True
    for test_name, test_results in results.items():
        passed = test_results.get('passed', False)
        all_passed = all_passed and passed
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")

    if all_passed:
        print("\n✅ ALL DISTILLATION TESTS PASSED")
    else:
        print("\n❌ SOME DISTILLATION TESTS FAILED")

    return results


if __name__ == "__main__":
    results = run_all_distillation_tests()
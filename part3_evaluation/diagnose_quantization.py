import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import math

def diagnose_quantization_health(model, tokenizer, device='cuda', test_text="The quick brown fox jumps over the lazy dog"):
    """
    Diagnose quantization health by testing the model at different precisions.

    Returns detailed diagnostics about model behavior at each precision level.
    """
    model.eval()
    model = model.to(device)

    # Tokenize test text
    inputs = tokenizer(test_text, return_tensors='pt', padding=True, truncation=True, max_length=256)
    input_ids = inputs['input_ids'].to(device)

    results = {}
    available_precisions = [32, 16, 8, 6] if hasattr(model, 'set_global_precision') else [8]

    print("\n" + "="*60)
    print("QUANTIZATION HEALTH DIAGNOSTIC")
    print("="*60)

    for precision in available_precisions:
        try:
            if hasattr(model, 'set_global_precision'):
                model.set_global_precision(precision)
            elif hasattr(model, 'set_precision'):
                model.set_precision(precision)
            else:
                print(f"Warning: Cannot set precision to {precision}-bit")
                continue

            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                # Calculate statistics
                logits_np = logits.cpu().numpy()
                stats = {
                    'mean': float(np.mean(logits_np)),
                    'std': float(np.std(logits_np)),
                    'min': float(np.min(logits_np)),
                    'max': float(np.max(logits_np)),
                    'median': float(np.median(logits_np)),
                    'has_nan': bool(np.isnan(logits_np).any()),
                    'has_inf': bool(np.isinf(logits_np).any()),
                    'zero_percentage': float(np.mean(np.abs(logits_np) < 1e-6) * 100)
                }

                # Calculate entropy (measure of randomness)
                probs = torch.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
                stats['entropy'] = entropy

                # Get top predictions
                top_probs, top_indices = torch.topk(probs[0, -1], k=5)
                top_tokens = [tokenizer.decode([idx]) for idx in top_indices.cpu().numpy()]
                stats['top_predictions'] = list(zip(top_tokens, top_probs.cpu().numpy().tolist()))

                results[f'{precision}bit'] = stats

                # Print results
                print(f"\n{precision}-bit Precision:")
                print(f"  Logits Statistics:")
                print(f"    Mean:     {stats['mean']:8.2f}  {'✓' if -20 < stats['mean'] < 0 else '✗ ABNORMAL'}")
                print(f"    Std Dev:  {stats['std']:8.2f}  {'✓' if 5 < stats['std'] < 50 else '✗ ABNORMAL'}")
                print(f"    Min:      {stats['min']:8.2f}")
                print(f"    Max:      {stats['max']:8.2f}")
                print(f"    Median:   {stats['median']:8.2f}")
                print(f"    Entropy:  {stats['entropy']:8.2f}  {'✓' if stats['entropy'] > 2 else '✗ LOW'}")

                if stats['has_nan']:
                    print(f"    ⚠️  Contains NaN values!")
                if stats['has_inf']:
                    print(f"    ⚠️  Contains Inf values!")
                if stats['zero_percentage'] > 10:
                    print(f"    ⚠️  {stats['zero_percentage']:.1f}% of values are near zero!")

                print(f"  Top 3 predictions: {', '.join([f'{tok}({p:.2f})' for tok, p in stats['top_predictions'][:3]])}")

                # Diagnosis
                if stats['mean'] < -50:
                    print(f"  🔴 CRITICAL: Severe quantization degradation detected!")
                    print(f"     Logits are extremely negative, indicating quantization failure.")
                elif stats['mean'] < -30:
                    print(f"  🟡 WARNING: Significant quantization issues detected.")
                elif stats['std'] < 1:
                    print(f"  🟡 WARNING: Very low standard deviation, possible collapse.")
                else:
                    print(f"  🟢 Model appears healthy at {precision}-bit precision.")

        except Exception as e:
            print(f"  ❌ Error testing {precision}-bit: {e}")
            results[f'{precision}bit'] = {'error': str(e)}

    # Compare precisions
    print("\n" + "="*60)
    print("PRECISION COMPARISON")
    print("="*60)

    if len(results) > 1:
        baseline_key = '32bit' if '32bit' in results else '16bit' if '16bit' in results else list(results.keys())[0]
        baseline = results.get(baseline_key, {})

        if 'mean' in baseline:
            for key in results:
                if key != baseline_key and 'mean' in results[key]:
                    mean_diff = results[key]['mean'] - baseline['mean']
                    std_diff = results[key]['std'] - baseline['std']
                    print(f"\n{key} vs {baseline_key}:")
                    print(f"  Mean difference:  {mean_diff:+8.2f}")
                    print(f"  Std difference:   {std_diff:+8.2f}")

                    if abs(mean_diff) > 50:
                        print(f"  🔴 Extreme degradation from baseline!")
                    elif abs(mean_diff) > 20:
                        print(f"  🟡 Significant degradation from baseline.")
                    else:
                        print(f"  🟢 Reasonable quantization loss.")

    # Overall diagnosis
    print("\n" + "="*60)
    print("OVERALL DIAGNOSIS")
    print("="*60)

    critical_issues = []
    warnings = []

    for key, stats in results.items():
        if 'error' in stats:
            critical_issues.append(f"{key}: Error during testing")
        elif 'mean' in stats:
            if stats['mean'] < -50:
                critical_issues.append(f"{key}: Severe degradation (mean={stats['mean']:.1f})")
            elif stats['mean'] < -30:
                warnings.append(f"{key}: Moderate degradation (mean={stats['mean']:.1f})")
            if stats['has_nan']:
                critical_issues.append(f"{key}: Contains NaN values")
            if stats['has_inf']:
                critical_issues.append(f"{key}: Contains Inf values")

    if critical_issues:
        print("🔴 CRITICAL ISSUES FOUND:")
        for issue in critical_issues:
            print(f"  - {issue}")
        print("\nRecommendations:")
        print("  1. Check quantization calibration data")
        print("  2. Verify checkpoint loading correctly")
        print("  3. Test with FP32 baseline first")
        print("  4. Check for numerical overflow in quantization parameters")
        print("  5. Consider re-calibrating or re-training")
    elif warnings:
        print("🟡 WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
        print("\nRecommendations:")
        print("  1. Monitor perplexity scores")
        print("  2. Consider adjusting quantization parameters")
    else:
        print("🟢 Model quantization appears healthy!")

    return results


def test_inference_variance(model, tokenizer, device='cuda', num_runs=5):
    """
    Test if model outputs are consistent across multiple runs.
    """
    model.eval()
    model = model.to(device)

    test_text = "The capital of France is"
    inputs = tokenizer(test_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)

    print("\n" + "="*60)
    print("INFERENCE CONSISTENCY TEST")
    print("="*60)

    all_logits = []
    for i in range(num_runs):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            all_logits.append(logits.cpu().numpy())

    # Check variance
    variance = np.var(np.stack(all_logits), axis=0)
    max_variance = np.max(variance)
    mean_variance = np.mean(variance)

    print(f"Max variance across runs: {max_variance:.6f}")
    print(f"Mean variance: {mean_variance:.6f}")

    if max_variance < 1e-6:
        print("🟢 Model outputs are consistent (deterministic)")
    elif max_variance < 0.01:
        print("🟡 Minor variance in outputs (likely numerical precision)")
    else:
        print("🔴 Significant variance in outputs (possible instability)")

    return max_variance, mean_variance


def compute_simple_perplexity(model, tokenizer, text, device='cuda', max_length=512):
    """
    Compute perplexity using simple method (like test_inference.py).
    This is the baseline method that works correctly.
    """
    model.eval()
    model = model.to(device)

    # Tokenize with truncation
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    input_ids = encodings['input_ids'].to(device)

    if input_ids.size(1) < 2:
        return float('inf'), {}

    with torch.no_grad():
        outputs = model(input_ids)

        # Handle different output formats
        if isinstance(outputs, dict):
            logits = outputs.get('logits', None)
        elif hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

        if logits is None:
            return float('inf'), {}

        # Compute loss manually
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        perplexity = torch.exp(loss).item()

        # Collect statistics
        stats = {
            'loss': loss.item(),
            'perplexity': perplexity,
            'logits_mean': logits.mean().item(),
            'logits_std': logits.std().item(),
            'logits_min': logits.min().item(),
            'logits_max': logits.max().item(),
            'num_tokens': input_ids.size(1)
        }

    return perplexity, stats


def compute_sliding_window_perplexity(model, tokenizer, text, device='cuda', stride=256, max_length=256):
    """
    Compute perplexity using sliding window (like perplexity_eval.py).
    This method seems to cause issues in evaluation.
    """
    model.eval()
    model = model.to(device)

    # Tokenize entire text
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length * 4, padding=False)
    input_ids = encodings['input_ids'].to(device)

    seq_len = input_ids.size(1)
    if seq_len < 10:
        return float('inf'), {}

    all_losses = []
    all_logits_stats = []

    # Sliding window approach
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        if end_loc - begin_loc < 10:
            break

        input_chunk = input_ids[:, begin_loc:end_loc]

        with torch.no_grad():
            outputs = model(input_chunk)

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

            # Calculate loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_chunk[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            all_losses.append(loss.item())
            all_logits_stats.append({
                'mean': logits.mean().item(),
                'std': logits.std().item(),
                'min': logits.min().item(),
                'max': logits.max().item()
            })

    if not all_losses:
        return float('inf'), {}

    avg_loss = np.mean(all_losses)
    perplexity = math.exp(avg_loss)

    # Aggregate statistics
    stats = {
        'loss': avg_loss,
        'perplexity': perplexity,
        'num_windows': len(all_losses),
        'loss_std': np.std(all_losses),
        'logits_mean': np.mean([s['mean'] for s in all_logits_stats]),
        'logits_std': np.mean([s['std'] for s in all_logits_stats]),
        'logits_min': min([s['min'] for s in all_logits_stats]),
        'logits_max': max([s['max'] for s in all_logits_stats])
    }

    return perplexity, stats


def test_simple_vs_sliding_perplexity(model, tokenizer, device='cuda', test_text=None):
    """
    Compare simple perplexity (like test_inference.py) vs sliding window approach.
    This helps identify if the sliding window is causing the degradation.
    """
    print("\n" + "="*60)
    print("SIMPLE VS SLIDING WINDOW PERPLEXITY COMPARISON")
    print("="*60)

    model.eval()
    model = model.to(device)

    # Use a standard test text if none provided
    if test_text is None:
        test_text = """Artificial intelligence and machine learning have revolutionized many industries.
        From healthcare to finance, these technologies are transforming how we work and live.
        Deep learning models can now understand language, recognize images, and even generate
        creative content. The future holds even more exciting possibilities as researchers
        continue to push the boundaries of what's possible with AI.""" * 3  # Repeat to make longer

    print(f"\nTest text length: {len(tokenizer.encode(test_text))} tokens")

    # Test with simple method
    print("\n1. Simple Method (like test_inference.py):")
    simple_ppl, simple_stats = compute_simple_perplexity(model, tokenizer, test_text, device)
    print(f"   Perplexity: {simple_ppl:.2f}")
    print(f"   Loss: {simple_stats.get('loss', 0):.4f}")
    print(f"   Logits mean: {simple_stats.get('logits_mean', 0):.2f}")
    print(f"   Logits std: {simple_stats.get('logits_std', 0):.2f}")

    # Test with sliding window
    print("\n2. Sliding Window Method (like perplexity_eval.py):")
    sliding_ppl, sliding_stats = compute_sliding_window_perplexity(model, tokenizer, test_text, device)
    print(f"   Perplexity: {sliding_ppl:.2f}")
    print(f"   Loss: {sliding_stats.get('loss', 0):.4f}")
    print(f"   Num windows: {sliding_stats.get('num_windows', 0)}")
    print(f"   Logits mean: {sliding_stats.get('logits_mean', 0):.2f}")
    print(f"   Logits std: {sliding_stats.get('logits_std', 0):.2f}")

    # Compare results
    print("\n3. Comparison:")
    if simple_ppl > 0 and sliding_ppl > 0:
        ratio = sliding_ppl / simple_ppl
        print(f"   Perplexity ratio (sliding/simple): {ratio:.2f}x")

        logits_diff = sliding_stats.get('logits_mean', 0) - simple_stats.get('logits_mean', 0)
        print(f"   Logits mean difference: {logits_diff:+.2f}")

        if ratio > 2:
            print("   🔴 WARNING: Sliding window perplexity is significantly worse!")
            print("      This suggests the sliding window approach is causing degradation.")
        elif ratio > 1.5:
            print("   🟡 Sliding window perplexity is moderately worse.")
        else:
            print("   🟢 Both methods give similar results.")

        if abs(logits_diff) > 20:
            print("   🔴 Large logits difference detected - possible numerical issues!")

    return {
        'simple': {'perplexity': simple_ppl, **simple_stats},
        'sliding': {'perplexity': sliding_ppl, **sliding_stats}
    }


def track_batch_degradation(model, tokenizer, texts=None, device='cuda', max_texts=10):
    """
    Track if/when model outputs degrade during batch processing.
    This helps identify if the model state gets corrupted during evaluation.
    """
    print("\n" + "="*60)
    print("BATCH PROCESSING DEGRADATION TRACKING")
    print("="*60)

    model.eval()
    model = model.to(device)

    # Use default texts if none provided
    if texts is None:
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "Machine learning models can understand natural language.",
            "Deep neural networks have many layers of computation.",
            "Python is a popular programming language for data science.",
            "The weather today is sunny and warm.",
            "Technology continues to advance at a rapid pace.",
            "Scientists discover new breakthroughs every day.",
            "The internet has connected people around the world.",
            "Education is the key to a better future."
        ]

    texts = texts[:max_texts]
    batch_stats = []

    print(f"\nProcessing {len(texts)} texts sequentially...")
    print("-" * 50)

    for i, text in enumerate(texts):
        # Tokenize
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        input_ids = inputs['input_ids'].to(device)

        # Get model output
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

            # Calculate statistics
            stats = {
                'text_idx': i,
                'text_preview': text[:50] + "..." if len(text) > 50 else text,
                'logits_mean': logits.mean().item(),
                'logits_std': logits.std().item(),
                'logits_min': logits.min().item(),
                'logits_max': logits.max().item(),
                'is_eval_mode': not model.training
            }

            batch_stats.append(stats)

            # Print stats for each batch
            print(f"\nBatch {i+1}:")
            print(f"  Text: '{stats['text_preview']}'")
            print(f"  Logits mean: {stats['logits_mean']:.2f}")
            print(f"  Logits std: {stats['logits_std']:.2f}")
            print(f"  Model in eval mode: {stats['is_eval_mode']}")

            # Check for degradation
            if i > 0:
                prev_mean = batch_stats[i-1]['logits_mean']
                mean_change = stats['logits_mean'] - prev_mean
                if abs(mean_change) > 10:
                    print(f"  ⚠️ Large change from previous batch: {mean_change:+.2f}")

            if stats['logits_mean'] < -50:
                print(f"  🔴 ABNORMAL: Extremely negative logits detected!")
                print(f"     This indicates model degradation at batch {i+1}")
                break

    # Analyze overall trend
    print("\n" + "-" * 50)
    print("DEGRADATION ANALYSIS:")

    means = [s['logits_mean'] for s in batch_stats]
    if len(means) > 1:
        # Check for trend
        first_mean = means[0]
        last_mean = means[-1]
        total_change = last_mean - first_mean

        print(f"  First batch logits mean: {first_mean:.2f}")
        print(f"  Last batch logits mean: {last_mean:.2f}")
        print(f"  Total change: {total_change:+.2f}")

        # Check for monotonic degradation
        is_degrading = all(means[i] <= means[i-1] + 1 for i in range(1, len(means)))

        if abs(total_change) > 20:
            print(f"  🔴 Significant degradation detected across batches!")
            print(f"     Model state appears to be corrupting during processing.")
        elif abs(total_change) > 10:
            print(f"  🟡 Moderate change detected across batches.")
        else:
            print(f"  🟢 Model outputs remain stable across batches.")

        if is_degrading and total_change < -10:
            print(f"  🔴 Monotonic degradation pattern detected!")

    return batch_stats


def verify_model_consistency(model, tokenizer, device='cuda', num_passes=3):
    """
    Verify model gives consistent outputs across multiple passes with same input.
    This checks if the model state is stable.
    """
    print("\n" + "="*60)
    print("MODEL CONSISTENCY VERIFICATION")
    print("="*60)

    model.eval()
    model = model.to(device)

    test_text = "The future of artificial intelligence is"
    inputs = tokenizer(test_text, return_tensors='pt', truncation=True, max_length=256)
    input_ids = inputs['input_ids'].to(device)

    print(f"\nRunning {num_passes} forward passes with same input...")
    print(f"Test text: '{test_text}'")

    all_outputs = []
    for i in range(num_passes):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            all_outputs.append({
                'logits': logits.cpu().numpy(),
                'mean': logits.mean().item(),
                'std': logits.std().item(),
                'max': logits.max().item(),
                'min': logits.min().item()
            })
            print(f"\nPass {i+1}:")
            print(f"  Logits mean: {all_outputs[-1]['mean']:.4f}")
            print(f"  Logits std: {all_outputs[-1]['std']:.4f}")

    # Check consistency
    print("\n" + "-" * 50)
    print("CONSISTENCY ANALYSIS:")

    # Compare logits directly
    logits_list = [o['logits'] for o in all_outputs]
    max_diff = 0
    for i in range(1, len(logits_list)):
        diff = np.abs(logits_list[i] - logits_list[0]).max()
        max_diff = max(max_diff, diff)
        print(f"  Max difference between pass 1 and pass {i+1}: {diff:.6f}")

    # Check statistics variation
    means = [o['mean'] for o in all_outputs]
    mean_std = np.std(means)
    print(f"\n  Standard deviation of means across passes: {mean_std:.6f}")

    if max_diff < 1e-5:
        print("  🟢 Model outputs are perfectly consistent!")
    elif max_diff < 1e-3:
        print("  🟢 Model outputs are consistent (minor numerical differences)")
    elif max_diff < 0.1:
        print("  🟡 Small variations detected (possible numerical precision issues)")
    else:
        print(f"  🔴 Significant variations detected! Max difference: {max_diff:.4f}")
        print("     This indicates model instability or non-deterministic behavior.")

    return all_outputs


def comprehensive_diagnosis(model, tokenizer, device='cuda'):
    """
    Run comprehensive diagnostics to identify evaluation issues.
    This combines all diagnostic tests to pinpoint the problem.
    """
    print("\n" + "="*80)
    print(" "*20 + "COMPREHENSIVE MODEL DIAGNOSTICS")
    print("="*80)

    results = {}

    # 1. Basic quantization health
    print("\n[1/5] Running quantization health check...")
    quant_results = diagnose_quantization_health(model, tokenizer, device)
    results['quantization'] = quant_results

    # 2. Inference variance
    print("\n[2/5] Testing inference variance...")
    max_var, mean_var = test_inference_variance(model, tokenizer, device)
    results['variance'] = {'max': max_var, 'mean': mean_var}

    # 3. Simple vs sliding window
    print("\n[3/5] Comparing perplexity calculation methods...")
    comparison = test_simple_vs_sliding_perplexity(model, tokenizer, device)
    results['perplexity_comparison'] = comparison

    # 4. Batch degradation
    print("\n[4/5] Tracking batch processing degradation...")
    batch_stats = track_batch_degradation(model, tokenizer, device=device, max_texts=5)
    results['batch_degradation'] = batch_stats

    # 5. Model consistency
    print("\n[5/5] Verifying model consistency...")
    consistency = verify_model_consistency(model, tokenizer, device, num_passes=3)
    results['consistency'] = consistency

    # Final summary
    print("\n" + "="*80)
    print(" "*25 + "DIAGNOSTIC SUMMARY")
    print("="*80)

    issues_found = []

    # Check for quantization issues
    for key in results.get('quantization', {}):
        if 'bit' in key and 'mean' in results['quantization'][key]:
            if results['quantization'][key]['mean'] < -50:
                issues_found.append(f"Severe degradation at {key}")

    # Check variance issues
    if results['variance']['max'] > 0.01:
        issues_found.append("High inference variance detected")

    # Check perplexity method issues
    if 'perplexity_comparison' in results:
        simple_ppl = results['perplexity_comparison']['simple'].get('perplexity', 0)
        sliding_ppl = results['perplexity_comparison']['sliding'].get('perplexity', 0)
        if simple_ppl > 0 and sliding_ppl > 0:
            if sliding_ppl / simple_ppl > 2:
                issues_found.append("Sliding window method causes degradation")

    # Check batch degradation
    if results.get('batch_degradation'):
        means = [s['logits_mean'] for s in results['batch_degradation']]
        if len(means) > 1 and (means[-1] - means[0]) < -20:
            issues_found.append("Model degrades during batch processing")

    if issues_found:
        print("\n🔴 ISSUES DETECTED:")
        for issue in issues_found:
            print(f"  - {issue}")
        print("\nRECOMMENDED ACTIONS:")
        print("  1. Use simple perplexity calculation instead of sliding window")
        print("  2. Verify model remains in eval mode throughout")
        print("  3. Check for accidental model reinitialization")
        print("  4. Ensure quantization parameters are preserved")
        print("  5. Test with smaller batch sizes")
    else:
        print("\n🟢 No major issues detected!")
        print("Model appears to be functioning correctly.")

    return results
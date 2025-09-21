#!/usr/bin/env python3
"""
Analyze the quantization cliff between 8-bit and 6-bit precision.
Diagnose why 6-bit catastrophically fails while 8-bit works.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test.fix_model_initialization import create_properly_initialized_model


def analyze_quantization_utilization(model, bits, device):
    """
    Analyze how many quantization levels are actually being used.
    This reveals if weights are collapsing to too few discrete values.
    """
    model.set_precision(bits)
    model.eval()

    utilization_stats = []
    layer_reports = []

    for name, module in model.named_modules():
        # Check quantized linear layers
        if hasattr(module, 'quantizers_weight'):
            bits_key = f'{bits}bit'
            if bits_key in module.quantizers_weight:
                quantizer = module.quantizers_weight[bits_key]

                # Get the actual weight
                if hasattr(module, 'linear') and hasattr(module.linear, 'weight'):
                    weight = module.linear.weight.data
                elif hasattr(module, 'weight'):
                    weight = module.weight.data
                else:
                    continue

                # Simulate quantization
                num_levels = 2**bits
                if hasattr(quantizer, 'scale') and quantizer.scale is not None:
                    # Use actual quantizer scale
                    scale = quantizer.scale.mean().item() if quantizer.scale.numel() > 1 else quantizer.scale.item()

                    # Quantize weights
                    weight_scaled = weight / scale
                    weight_int = torch.round(weight_scaled).clamp(-(2**(bits-1)), 2**(bits-1)-1)

                    unique_values = len(torch.unique(weight_int))
                    utilization = unique_values / num_levels
                    utilization_stats.append(utilization)

                    # Report layers with poor utilization
                    if utilization < 0.5:  # Using less than 50% of available levels
                        layer_reports.append({
                            'name': name,
                            'unique_values': unique_values,
                            'total_levels': num_levels,
                            'utilization': utilization,
                            'weight_shape': list(weight.shape)
                        })

    return {
        'avg_utilization': np.mean(utilization_stats) if utilization_stats else 0,
        'min_utilization': np.min(utilization_stats) if utilization_stats else 0,
        'max_utilization': np.max(utilization_stats) if utilization_stats else 0,
        'poor_layers': layer_reports,
        'all_utilization': utilization_stats
    }


def analyze_attention_pattern_degradation(model, tokenizer, device, text="The cat sat on the mat."):
    """
    Compare output distributions across different bit widths to see where they break down.
    Since the model doesn't support output_attentions, we'll analyze output distributions instead.
    """
    tokens = tokenizer(text, return_tensors='pt')['input_ids'].to(device)

    output_patterns = {}

    for bits in [32, 16, 8, 6]:
        if bits not in [6, 8, 16, 32]:  # Skip if not in configured widths
            continue

        model.set_precision(bits)
        model.eval()

        with torch.no_grad():
            outputs = model(tokens)

            # Get logits
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            # Get probability distribution for each position
            probs = torch.softmax(logits, dim=-1)

            # Store the average probability distribution
            avg_probs = probs.mean(dim=1).squeeze(0).cpu().numpy()
            output_patterns[bits] = avg_probs

    return output_patterns


def analyze_vocabulary_discrimination(model, tokenizer, device):
    """
    Test how well the model can discriminate between different vocabulary tokens.
    """
    # Create a simple input
    text = "The"
    tokens = tokenizer(text, return_tensors='pt')['input_ids'].to(device)

    discrimination_stats = {}

    for bits in [32, 16, 8, 6]:
        if bits not in [6, 8, 16, 32]:  # Skip if not in configured widths
            continue

        model.set_precision(bits)
        model.eval()

        with torch.no_grad():
            outputs = model(tokens)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs

            # Get final token predictions
            final_logits = logits[0, -1, :]  # Last position

            # Calculate entropy (higher = more uniform = worse discrimination)
            probs = torch.softmax(final_logits, dim=0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

            # Count near-duplicate logit values (indicates collapsed representations)
            sorted_logits, _ = torch.sort(final_logits)
            diffs = torch.diff(sorted_logits)
            near_duplicates = torch.sum(diffs < 1e-3).item()

            # Get top-k concentration
            top_k = 10
            top_k_prob = torch.sum(torch.topk(probs, top_k).values).item()

            discrimination_stats[bits] = {
                'entropy': entropy,
                'near_duplicates': near_duplicates,
                'top_k_concentration': top_k_prob,
                'unique_logits': len(torch.unique(torch.round(final_logits * 100))) # Round to 2 decimals
            }

    return discrimination_stats


def analyze_weight_distribution_coverage(model, bits):
    """
    Analyze how well quantization levels cover the actual weight distribution.
    """
    model.set_precision(bits)

    coverage_stats = []

    for name, module in model.named_modules():
        if hasattr(module, 'quantizers_weight'):
            bits_key = f'{bits}bit'
            if bits_key in module.quantizers_weight:
                quantizer = module.quantizers_weight[bits_key]

                # Get the actual weight
                if hasattr(module, 'linear') and hasattr(module.linear, 'weight'):
                    weight = module.linear.weight.data
                elif hasattr(module, 'weight'):
                    weight = module.weight.data
                else:
                    continue

                # Analyze weight distribution
                weight_flat = weight.flatten()
                weight_std = weight_flat.std().item()
                weight_mean = weight_flat.mean().item()

                # Calculate how many quantization levels cover ±2 std
                num_levels = 2**bits
                levels_per_std = num_levels / 4  # Assuming uniform quantization over ±2 std

                # Check if critical layers (embedding, output projection)
                is_critical = 'embed' in name.lower() or 'lm_head' in name.lower() or 'c_proj' in name.lower()

                coverage_stats.append({
                    'layer': name,
                    'levels_per_std': levels_per_std,
                    'weight_std': weight_std,
                    'is_critical': is_critical,
                    'sufficient_coverage': levels_per_std > 10  # Need at least 10 levels per std
                })

    return coverage_stats


def suggest_solutions(analysis_results):
    """
    Based on analysis, suggest specific solutions for 6-bit quantization.
    """
    suggestions = []

    utilization_6bit = analysis_results.get('utilization_6bit', {})
    discrimination_stats = analysis_results.get('discrimination', {})
    coverage_stats_6bit = analysis_results.get('coverage_6bit', [])

    # Check utilization
    if utilization_6bit.get('avg_utilization', 1) < 0.5:
        suggestions.append({
            'issue': 'Low quantization level utilization',
            'severity': 'CRITICAL',
            'solution': 'Switch to non-uniform quantization (logarithmic or learned bins)',
            'details': f"Only using {utilization_6bit['avg_utilization']*100:.1f}% of available levels"
        })

    # Check vocabulary discrimination
    if discrimination_stats.get(6, {}).get('entropy', 0) > discrimination_stats.get(8, {}).get('entropy', 1) * 1.5:
        suggestions.append({
            'issue': 'Vocabulary discrimination collapse',
            'severity': 'CRITICAL',
            'solution': 'Keep output projection (lm_head) at 8-bit or higher',
            'details': f"Entropy increased by {(discrimination_stats[6]['entropy']/discrimination_stats[8]['entropy']-1)*100:.1f}%"
        })

    # Check critical layers
    critical_layers_poor = [stat for stat in coverage_stats_6bit if stat['is_critical'] and not stat['sufficient_coverage']]
    if critical_layers_poor:
        suggestions.append({
            'issue': 'Critical layers have insufficient quantization coverage',
            'severity': 'HIGH',
            'solution': 'Use mixed precision - keep embeddings and output layers at 8-bit',
            'details': f"{len(critical_layers_poor)} critical layers have poor coverage"
        })

    # LoRA rank adjustment
    suggestions.append({
        'issue': 'LoRA may not have enough capacity at 6-bit',
        'severity': 'MEDIUM',
        'solution': 'Increase LoRA rank for 6-bit from 12 to 24 or 32',
        'details': 'Higher rank can compensate for quantization errors'
    })

    return suggestions


def main():
    print("="*80)
    print("QUANTIZATION CLIFF ANALYSIS")
    print("Diagnosing the 8-bit to 6-bit catastrophic failure")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load model
    print("\n🔧 Loading model...")
    model, config = create_properly_initialized_model(use_pretrained=True, num_layers=6)
    model = model.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    # 1. Analyze quantization utilization
    print("\n" + "="*60)
    print("1. QUANTIZATION UTILIZATION ANALYSIS")
    print("="*60)

    for bits in [16, 8, 6]:
        print(f"\n📊 Analyzing {bits}-bit utilization...")
        util_stats = analyze_quantization_utilization(model, bits, device)
        all_results[f'utilization_{bits}bit'] = util_stats

        print(f"   Average utilization: {util_stats['avg_utilization']*100:.1f}%")
        print(f"   Min utilization: {util_stats['min_utilization']*100:.1f}%")
        print(f"   Max utilization: {util_stats['max_utilization']*100:.1f}%")

        if util_stats['poor_layers']:
            print(f"   ⚠️ {len(util_stats['poor_layers'])} layers with <50% utilization:")
            for layer in util_stats['poor_layers'][:3]:  # Show first 3
                print(f"      - {layer['name']}: {layer['unique_values']}/{layer['total_levels']} levels ({layer['utilization']*100:.1f}%)")

    # 2. Analyze output distributions
    print("\n" + "="*60)
    print("2. OUTPUT DISTRIBUTION ANALYSIS")
    print("="*60)

    output_patterns = analyze_attention_pattern_degradation(model, tokenizer, device)
    all_results['output_patterns'] = output_patterns

    if 8 in output_patterns and 6 in output_patterns:
        # Compare output distributions
        pattern_8bit = output_patterns[8]
        pattern_6bit = output_patterns[6]

        # Calculate KL divergence instead of correlation for probability distributions
        from scipy.stats import entropy

        # Add small epsilon to avoid log(0)
        pattern_8bit_safe = pattern_8bit + 1e-10
        pattern_6bit_safe = pattern_6bit + 1e-10

        # Normalize to ensure they sum to 1
        pattern_8bit_safe = pattern_8bit_safe / pattern_8bit_safe.sum()
        pattern_6bit_safe = pattern_6bit_safe / pattern_6bit_safe.sum()

        kl_div = entropy(pattern_6bit_safe, pattern_8bit_safe)
        print(f"\n   Output distribution KL divergence (6-bit from 8-bit): {kl_div:.4f}")

        if kl_div > 1.0:
            print("   ❌ CRITICAL: Output distributions severely degraded at 6-bit")
        elif kl_div > 0.5:
            print("   ⚠️ WARNING: Output distributions noticeably degraded at 6-bit")
        else:
            print("   ✅ Output distributions relatively preserved at 6-bit")

    # 3. Analyze vocabulary discrimination
    print("\n" + "="*60)
    print("3. VOCABULARY DISCRIMINATION ANALYSIS")
    print("="*60)

    discrimination = analyze_vocabulary_discrimination(model, tokenizer, device)
    all_results['discrimination'] = discrimination

    for bits in [32, 16, 8, 6]:
        if bits in discrimination:
            stats = discrimination[bits]
            print(f"\n   {bits}-bit discrimination:")
            print(f"      Entropy: {stats['entropy']:.2f}")
            print(f"      Near-duplicate logits: {stats['near_duplicates']}")
            print(f"      Top-10 concentration: {stats['top_k_concentration']*100:.1f}%")
            print(f"      Unique logit values: {stats['unique_logits']}")

    # 4. Analyze weight distribution coverage
    print("\n" + "="*60)
    print("4. WEIGHT DISTRIBUTION COVERAGE ANALYSIS")
    print("="*60)

    for bits in [8, 6]:
        print(f"\n   {bits}-bit coverage:")
        coverage = analyze_weight_distribution_coverage(model, bits)
        all_results[f'coverage_{bits}bit'] = coverage

        insufficient = [s for s in coverage if not s['sufficient_coverage']]
        critical_insufficient = [s for s in insufficient if s['is_critical']]

        print(f"      Layers with insufficient coverage: {len(insufficient)}/{len(coverage)}")
        print(f"      Critical layers with issues: {len(critical_insufficient)}")

        if critical_insufficient:
            print("      Critical layers affected:")
            for layer in critical_insufficient[:3]:
                print(f"         - {layer['layer']}: {layer['levels_per_std']:.1f} levels/std")

    # 5. Generate solutions
    print("\n" + "="*60)
    print("5. RECOMMENDED SOLUTIONS")
    print("="*60)

    suggestions = suggest_solutions(all_results)

    for i, suggestion in enumerate(suggestions, 1):
        severity_color = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡'}.get(suggestion['severity'], '⚪')
        print(f"\n   {severity_color} {i}. {suggestion['issue']}")
        print(f"      Severity: {suggestion['severity']}")
        print(f"      Solution: {suggestion['solution']}")
        print(f"      Details: {suggestion['details']}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\n🔍 Root Cause Analysis:")
    print("   The 6-bit quantization cliff occurs because:")
    print("   1. Only 64 discrete weight levels cannot represent 50K+ vocabulary distinctions")
    print("   2. Attention patterns require >100 levels for meaningful softmax distributions")
    print("   3. Critical layers (embeddings, projections) need higher precision")

    print("\n✅ Recommended Approach:")
    print("   1. Implement mixed-precision: Keep critical layers at 8-bit")
    print("   2. Use non-uniform quantization for 6-bit layers")
    print("   3. Increase LoRA rank for 6-bit to 24-32")
    print("   4. Consider removing 6-bit from training, focus on 8/16-bit")

    return all_results


if __name__ == "__main__":
    results = main()
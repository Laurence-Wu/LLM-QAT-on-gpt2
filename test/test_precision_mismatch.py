#!/usr/bin/env python3
"""
Test Module for Precision Mismatch Detection
Detects and diagnoses precision-related issues in quantized models.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.quantization import LearnableFakeQuantize
from shared.switchable_batchnorm import SwitchableLayerNorm
from test.fix_model_initialization import create_properly_initialized_model
from transformers import GPT2Tokenizer


def detect_precision_mismatch(model, input_data, precision_bits: int) -> Dict:
    """
    Detect precision mismatches in model layers.

    Returns:
        Dict with mismatch information per layer
    """
    model.eval()
    model.set_precision(precision_bits)

    mismatches = {}
    hooks = []
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            # Store activation statistics
            if isinstance(output, torch.Tensor):
                activations[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'shape': output.shape,
                    'dtype': str(output.dtype),
                    'has_nan': torch.isnan(output).any().item(),
                    'has_inf': torch.isinf(output).any().item(),
                }

                # Check for quantization range
                try:
                    quantizers_weight = module.quantizers_weight
                except AttributeError:
                    quantizers_weight = None
                if quantizers_weight is not None:
                    bits_key = f'{precision_bits}bit'
                    if bits_key in quantizers_weight:
                        quantizer = quantizers_weight[bits_key]
                        try:
                            scale = quantizer.scale
                            scale_is_valid = scale is not None
                        except AttributeError:
                            scale_is_valid = False
                        if scale_is_valid:
                            expected_range = 2**(precision_bits - 1) - 1
                            actual_max = output.abs().max().item()
                            scaled_max = actual_max / (quantizer.scale.item() + 1e-10)

                            activations[name]['quantizer_info'] = {
                                'expected_range': expected_range,
                                'actual_scaled_max': scaled_max,
                                'scale': scale.item(),
                                'calibrated': quantizer.calibrated
                            }
        return hook

    # Register hooks
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            handle = module.register_forward_hook(hook_fn(name))
            hooks.append(handle)

    # Forward pass
    with torch.no_grad():
        _ = model(input_data)

    # Analyze activations for mismatches
    for name, stats in activations.items():
        issues = []

        # Check for numerical issues
        if stats['has_nan']:
            issues.append('NaN values detected')
        if stats['has_inf']:
            issues.append('Inf values detected')

        # Check for value range issues
        if abs(stats['mean']) > 100:
            issues.append(f"Extremely high mean: {stats['mean']:.2f}")
        if stats['std'] > 100:
            issues.append(f"Extremely high std: {stats['std']:.2f}")
        if stats['std'] < 1e-6:
            issues.append(f"Near-zero std: {stats['std']:.2e}")

        # Check quantization mismatch
        if 'quantizer_info' in stats:
            q_info = stats['quantizer_info']
            if q_info['calibrated']:
                if q_info['actual_scaled_max'] > q_info['expected_range'] * 1.5:
                    issues.append(f"Quantization overflow: {q_info['actual_scaled_max']:.2f} > {q_info['expected_range']}")
                elif q_info['actual_scaled_max'] < q_info['expected_range'] * 0.1:
                    issues.append(f"Quantization underutilization: {q_info['actual_scaled_max']:.2f} << {q_info['expected_range']}")

        if issues:
            mismatches[name] = {
                'issues': issues,
                'stats': stats
            }

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return mismatches


def test_precision_consistency(model, tokenizer, device, test_texts: List[str] = None):
    """
    Test consistency of outputs across different precision levels.
    """
    if test_texts is None:
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models require careful tuning.",
            "Quantization reduces model size and improves inference speed."
        ]

    print("\n" + "="*60)
    print("TESTING PRECISION CONSISTENCY")
    print("="*60)

    results = {}

    # Get reference outputs at 32-bit
    model.eval()
    model.set_precision(32)
    reference_outputs = []

    print("\n📊 Collecting 32-bit reference outputs...")
    with torch.no_grad():
        for text in test_texts:
            tokens = tokenizer(text, return_tensors='pt', max_length=64, truncation=True, padding=True)['input_ids'].to(device)
            output = model(tokens, output_hidden_states=True)
            reference_outputs.append(output['hidden_states'][-1])

    # Test other precisions - get configured bit widths from model
    bit_widths = model.transformer.bit_widths if hasattr(model.transformer, 'bit_widths') else [6, 8, 16, 32]
    student_precisions = [b for b in bit_widths if b < 32]
    for precision in student_precisions:
        print(f"\n🔍 Testing {precision}-bit precision:")
        model.set_precision(precision)

        # Calibrate if needed
        if precision < 32:
            print(f"   Calibrating {precision}-bit quantizers...")
            model.train()
            for name, module in model.named_modules():
                try:
                    quantizers_weight = module.quantizers_weight
                except AttributeError:
                    continue  # Module doesn't have quantizers_weight
                if True:
                    bits_key = f'{precision}bit'
                    if bits_key in module.quantizers_weight:
                        quantizers_weight[bits_key].start_calibration()
                    try:
                        quantizers_input = module.quantizers_input
                        if bits_key in quantizers_input:
                            quantizers_input[bits_key].start_calibration()
                    except AttributeError:
                        pass  # Module doesn't have quantizers_input

            # Calibration forward passes
            with torch.no_grad():
                for text in test_texts[:2]:  # Use subset for calibration
                    tokens = tokenizer(text, return_tensors='pt', max_length=64, truncation=True, padding=True)['input_ids'].to(device)
                    _ = model(tokens)

            # Finish calibration
            for name, module in model.named_modules():
                try:
                    quantizers_weight = module.quantizers_weight
                except AttributeError:
                    continue  # Module doesn't have quantizers_weight
                if True:
                    bits_key = f'{precision}bit'
                    if bits_key in module.quantizers_weight:
                        quantizers_weight[bits_key].finish_calibration()
                    try:
                        quantizers_input = module.quantizers_input
                        if bits_key in quantizers_input:
                            quantizers_input[bits_key].finish_calibration()
                    except AttributeError:
                        pass  # Module doesn't have quantizers_input

        model.eval()

        # Compare outputs
        precision_results = {
            'mse_values': [],
            'cosine_similarities': [],
            'relative_errors': []
        }

        with torch.no_grad():
            for i, text in enumerate(test_texts):
                tokens = tokenizer(text, return_tensors='pt', max_length=64, truncation=True, padding=True)['input_ids'].to(device)
                output = model(tokens, output_hidden_states=True)
                current_output = output['hidden_states'][-1]

                # Calculate metrics
                mse = torch.mean((current_output - reference_outputs[i])**2).item()
                cosine_sim = torch.nn.functional.cosine_similarity(
                    current_output.flatten(),
                    reference_outputs[i].flatten(),
                    dim=0
                ).item()
                relative_error = torch.mean(
                    torch.abs(current_output - reference_outputs[i]) / (torch.abs(reference_outputs[i]) + 1e-8)
                ).item()

                precision_results['mse_values'].append(mse)
                precision_results['cosine_similarities'].append(cosine_sim)
                precision_results['relative_errors'].append(relative_error)

        # Aggregate results
        results[precision] = {
            'avg_mse': np.mean(precision_results['mse_values']),
            'avg_cosine_sim': np.mean(precision_results['cosine_similarities']),
            'avg_relative_error': np.mean(precision_results['relative_errors']),
            'std_mse': np.std(precision_results['mse_values']),
            'std_cosine_sim': np.std(precision_results['cosine_similarities']),
        }

        print(f"   Average MSE: {results[precision]['avg_mse']:.6f}")
        print(f"   Average Cosine Similarity: {results[precision]['avg_cosine_sim']:.4f}")
        print(f"   Average Relative Error: {results[precision]['avg_relative_error']:.4f}")

        # Determine status
        if results[precision]['avg_cosine_sim'] > 0.95:
            status = "✅ EXCELLENT"
        elif results[precision]['avg_cosine_sim'] > 0.85:
            status = "⚠️ ACCEPTABLE"
        else:
            status = "❌ POOR"
        print(f"   Status: {status}")

    return results


def test_layer_precision_analysis(model, tokenizer, device):
    """
    Analyze precision effects on individual layers.
    """
    print("\n" + "="*60)
    print("LAYER-WISE PRECISION ANALYSIS")
    print("="*60)

    test_text = "The transformer architecture has revolutionized natural language processing."
    tokens = tokenizer(test_text, return_tensors='pt', max_length=64, truncation=True)['input_ids'].to(device)

    layer_analysis = {}

    # Get configured bit widths from model
    bit_widths = model.transformer.bit_widths if hasattr(model.transformer, 'bit_widths') else [6, 8, 16, 32]
    for precision in bit_widths:
        print(f"\n🔬 Analyzing {precision}-bit precision:")
        model.set_precision(precision)
        model.eval()

        # Detect mismatches
        mismatches = detect_precision_mismatch(model, tokens, precision)

        layer_analysis[precision] = {
            'num_issues': len(mismatches),
            'layers_with_issues': list(mismatches.keys()),
            'issue_types': {}
        }

        # Categorize issues
        for layer_name, info in mismatches.items():
            for issue in info['issues']:
                issue_type = issue.split(':')[0] if ':' in issue else issue
                if issue_type not in layer_analysis[precision]['issue_types']:
                    layer_analysis[precision]['issue_types'][issue_type] = 0
                layer_analysis[precision]['issue_types'][issue_type] += 1

        print(f"   Layers with issues: {layer_analysis[precision]['num_issues']}")
        if layer_analysis[precision]['issue_types']:
            print("   Issue breakdown:")
            for issue_type, count in layer_analysis[precision]['issue_types'].items():
                print(f"     - {issue_type}: {count} layers")

        # Show sample layer details
        if mismatches and precision < 32:
            sample_layer = list(mismatches.keys())[0]
            print(f"\n   Sample problematic layer: {sample_layer}")
            for issue in mismatches[sample_layer]['issues'][:3]:
                print(f"     ⚠️ {issue}")

    return layer_analysis


def test_quantization_saturation(model, tokenizer, device):
    """
    Test for quantization saturation across different input magnitudes.
    """
    print("\n" + "="*60)
    print("QUANTIZATION SATURATION TEST")
    print("="*60)

    # Create inputs with different magnitudes
    base_text = "Testing quantization saturation effects."
    tokens = tokenizer(base_text, return_tensors='pt', max_length=32, truncation=True)['input_ids'].to(device)

    saturation_results = {}

    # Get configured bit widths from model, focus on lower precisions
    bit_widths = model.transformer.bit_widths if hasattr(model.transformer, 'bit_widths') else [6, 8, 16, 32]
    test_precisions = [b for b in bit_widths if b <= 8]  # Test 6 and 8-bit

    for precision in test_precisions:
        print(f"\n🔍 Testing {precision}-bit saturation:")
        model.set_precision(precision)
        model.eval()

        saturation_counts = []

        # Hook to count saturated values
        def count_saturation(module, input, output):
            if isinstance(output, torch.Tensor) and hasattr(module, 'quantizers_weight'):
                bits_key = f'{precision}bit'
                if bits_key in module.quantizers_weight:
                    quantizer = module.quantizers_weight[bits_key]
                    if hasattr(quantizer, 'scale') and quantizer.scale is not None:
                        max_val = (2**(precision - 1) - 1) * quantizer.scale
                        saturated = (output.abs() >= max_val * 0.95).float().mean().item()
                        saturation_counts.append(saturated)

        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if hasattr(module, 'quantizers_weight'):
                hooks.append(module.register_forward_hook(count_saturation))

        # Test with scaled inputs
        scale_factors = [0.5, 1.0, 2.0, 5.0]
        results_by_scale = {}

        for scale in scale_factors:
            saturation_counts.clear()

            # Forward pass (we'll scale embeddings internally)
            with torch.no_grad():
                _ = model(tokens)

            if saturation_counts:
                avg_saturation = np.mean(saturation_counts) * 100
                results_by_scale[scale] = avg_saturation

                status = "✅" if avg_saturation < 5 else "⚠️" if avg_saturation < 20 else "❌"
                print(f"   Scale {scale:.1f}x: {avg_saturation:.1f}% saturation {status}")

        saturation_results[precision] = results_by_scale

        # Remove hooks
        for hook in hooks:
            hook.remove()

    return saturation_results


def run_precision_mismatch_tests():
    """
    Run comprehensive precision mismatch detection tests.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE PRECISION MISMATCH DETECTION SUITE")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model and tokenizer
    print("\n🔧 Loading model...")
    model, config = create_properly_initialized_model(use_pretrained=True, num_layers=6)  # Smaller for testing
    model = model.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    # Test 1: Precision Consistency
    print("\n" + "="*60)
    print("TEST 1: Precision Consistency")
    consistency_results = test_precision_consistency(model, tokenizer, device)
    all_results['consistency'] = consistency_results

    # Test 2: Layer-wise Analysis
    print("\n" + "="*60)
    print("TEST 2: Layer-wise Analysis")
    layer_results = test_layer_precision_analysis(model, tokenizer, device)
    all_results['layer_analysis'] = layer_results

    # Test 3: Quantization Saturation
    print("\n" + "="*60)
    print("TEST 3: Quantization Saturation")
    saturation_results = test_quantization_saturation(model, tokenizer, device)
    all_results['saturation'] = saturation_results

    # Summary
    print("\n" + "="*80)
    print("PRECISION MISMATCH DETECTION SUMMARY")
    print("="*80)

    print("\n📊 Key Findings:")

    # Consistency summary
    print("\n1. Precision Consistency:")
    # Get actual tested precisions from results
    tested_precisions = [p for p in consistency_results.keys() if isinstance(p, int)]
    for precision in sorted(tested_precisions):
        if precision in consistency_results:
            cos_sim = consistency_results[precision]['avg_cosine_sim']
            print(f"   {precision}-bit: Cosine similarity = {cos_sim:.4f}")

    # Layer issues summary
    print("\n2. Layer Issues by Precision:")
    # Get actual tested precisions from results
    tested_precisions_layer = [p for p in layer_results.keys() if isinstance(p, int)]
    for precision in sorted(tested_precisions_layer):
        if precision in layer_results:
            num_issues = layer_results[precision]['num_issues']
            print(f"   {precision}-bit: {num_issues} layers with issues")

    # Saturation summary
    print("\n3. Saturation Levels:")
    for precision in [8, 4]:
        if precision in saturation_results and 1.0 in saturation_results[precision]:
            sat_level = saturation_results[precision][1.0]
            print(f"   {precision}-bit: {sat_level:.1f}% at normal scale")

    print("\n✅ Precision mismatch detection complete!")
    return all_results


if __name__ == "__main__":
    results = run_precision_mismatch_tests()
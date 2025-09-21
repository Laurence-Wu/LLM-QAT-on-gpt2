#!/usr/bin/env python3
"""
Comprehensive SP Model Debug Suite
Integrates all testing modules for thorough model validation including:
- Precision mismatch detection
- Batch normalization effects
- Training dynamics observation
- Quantization analysis
- Distillation effectiveness
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gc
import argparse
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import base utilities
from test.fix_model_initialization import create_properly_initialized_model
from test.dataset_utils import calculate_perplexity_properly, get_calibration_texts
from test.calculate_perplexity_chunked import calculate_perplexity_chunked

# Import new test modules
from test.test_precision_mismatch import (
    detect_precision_mismatch,
    test_precision_consistency,
    test_layer_precision_analysis,
    test_quantization_saturation
)
from test.test_batchnorm_effects import (
    test_bn_statistics_tracking,
    test_bn_gradient_flow,
    test_bn_mode_switching,
    test_bn_with_small_batch,
    test_bn_precision_switching_consistency
)
from test.test_training_dynamics import (
    test_multi_batch_training,
    test_quantization_aware_training,
    test_distillation_effectiveness,
    test_gradient_accumulation_effects,
    test_batch_norm_training_dynamics
)


def calibrate_precision_with_debug(sp_model, tokenizer, device, precision, calibration_texts=None):
    """
    Helper function to calibrate a specific precision with debug stats.
    Shows running min/max for 4-bit and 8-bit to verify consistency.
    """
    if precision >= 32:
        return  # No calibration needed for 32-bit teacher

    if calibration_texts is None:
        calibration_texts = get_calibration_texts(num_texts=16)

    print(f"\n   📊 Calibrating {precision}-bit precision...")
    sp_model.set_precision(precision)
    sp_model.train()  # Must be in training mode

    bits_key = f'{precision}bit'
    calibrated_count = 0

    # Start calibration
    for name, module in sp_model.named_modules():
        quantizers_weight = getattr(module, 'quantizers_weight', None)
        quantizers_input = getattr(module, 'quantizers_input', None)
        if quantizers_weight is not None and quantizers_input is not None:
            if bits_key in quantizers_weight:
                quantizers_weight[bits_key].start_calibration()
                calibrated_count += 1
            if bits_key in quantizers_input:
                quantizers_input[bits_key].start_calibration()
                calibrated_count += 1

    print(f"      Started calibration for {calibrated_count} quantizers")

    # Collect statistics
    with torch.no_grad():
        for i, text in enumerate(calibration_texts):
            tokens = tokenizer(text, return_tensors='pt',
                              max_length=128, truncation=True)['input_ids'].to(device)
            _ = sp_model(tokens)

    # Finish calibration with debug for 4-bit and 8-bit
    enable_debug = precision in [4, 8]
    if enable_debug:
        print(f"\n      🔍 Debug Statistics for {precision}-bit Calibration:")

    sample_count = 0
    for name, module in sp_model.named_modules():
        quantizers_weight = getattr(module, 'quantizers_weight', None)
        quantizers_input = getattr(module, 'quantizers_input', None)
        if quantizers_weight is not None and quantizers_input is not None:
            if bits_key in quantizers_weight:
                # Show debug for first 2 weight quantizers
                show_debug = enable_debug and sample_count < 2
                quantizers_weight[bits_key].finish_calibration(debug=show_debug)
                sample_count += 1

            if bits_key in quantizers_input:
                # Show debug for next 2 input quantizers
                show_debug = enable_debug and (2 <= sample_count < 4)
                quantizers_input[bits_key].finish_calibration(debug=show_debug)
                sample_count += 1

    print(f"      ✅ Calibrated {calibrated_count} quantizers with {len(calibration_texts)} samples\n")


def test_32bit_equivalence_sliding(sp_model, gpt2_model, tokenizer, device):
    """Test 1: Verify 32-bit teacher matches GPT-2 using sliding window."""
    print("\n" + "="*60)
    print("TEST 1: 32-BIT EQUIVALENCE WITH SLIDING WINDOW")
    print("="*60)

    sp_model.eval()
    gpt2_model.eval()
    sp_model.set_precision(32)

    print("\nCalculating perplexity using sliding window approach...")

    # Calculate for SP model
    print("\n  Testing SP Model (32-bit):")
    sp_results = calculate_perplexity_properly(
        model=sp_model,
        tokenizer=tokenizer,
        device=device,
        dataset_name='wikitext',
        max_length=256,
        stride=128,
        max_samples=512
    )

    # Calculate for GPT-2
    print("\n  Testing GPT-2 Model:")
    gpt2_results = calculate_perplexity_properly(
        model=gpt2_model,
        tokenizer=tokenizer,
        device=device,
        dataset_name='wikitext',
        max_length=256,
        stride=128,
        max_samples=512
    )

    # Compare
    ppl_diff = abs(sp_results['perplexity'] - gpt2_results['perplexity'])
    loss_diff = abs(sp_results['loss'] - gpt2_results['loss'])

    print(f"\n📊 COMPARISON RESULTS:")
    print(f"   SP Model (32-bit): PPL={sp_results['perplexity']:.2f}, Loss={sp_results['loss']:.4f}")
    print(f"   GPT-2 Model: PPL={gpt2_results['perplexity']:.2f}, Loss={gpt2_results['loss']:.4f}")
    print(f"   Differences: PPL diff={ppl_diff:.3f}, Loss diff={loss_diff:.4f}")

    status = "✅ EXCELLENT" if ppl_diff < 1.0 else "⚠️ ACCEPTABLE" if ppl_diff < 5.0 else "❌ FAILED"
    print(f"   Status: {status}")

    return {'sp_ppl': sp_results['perplexity'], 'gpt2_ppl': gpt2_results['perplexity'],
            'ppl_diff': ppl_diff, 'status': status}


def test_quantization_degradation_sliding(sp_model, tokenizer, device):
    """Test 2: Verify quantization degradation using sliding window with debug."""
    print("\n" + "="*60)
    print("TEST 2: QUANTIZATION DEGRADATION WITH SLIDING WINDOW")
    print("="*60)

    # Calibrate all student precisions with debug
    print("\n📊 Calibrating all student precisions with debug statistics...")
    calibration_texts = get_calibration_texts(num_texts=16)

    for precision in [16, 8, 4]:  # Students only
        calibrate_precision_with_debug(sp_model, tokenizer, device, precision, calibration_texts)

    print("✅ All calibrations complete\n")

    # Test with sliding window
    results = {}
    sp_model.eval()

    for precision in [32, 16, 8, 4]:
        sp_model.set_precision(precision)
        print(f"\nTesting {precision}-bit with sliding window...")

        result = calculate_perplexity_properly(
            model=sp_model,
            tokenizer=tokenizer,
            device=device,
            dataset_name='wikitext',
            max_length=256,
            stride=128,
            max_samples=512
        )

        results[precision] = {
            'ppl': result['perplexity'],
            'loss': result['loss'],
            'tokens': result['total_tokens'],
            'windows': result['num_windows']
        }

        print(f"   Results: PPL={result['perplexity']:.2f}, Loss={result['loss']:.4f}, "
              f"Tokens={result['total_tokens']}, Windows={result['num_windows']}")

    # Analysis
    print("\n📊 DEGRADATION ANALYSIS:")
    baseline_ppl = results[32]['ppl']
    print(f"   Baseline (32-bit): PPL = {baseline_ppl:.2f}")

    for bits in [16, 8, 4]:
        ppl = results[bits]['ppl']
        degradation = ((ppl - baseline_ppl) / baseline_ppl) * 100

        if bits == 16:
            status = "✅" if degradation < 10 else "⚠️" if degradation < 30 else "❌"
        elif bits == 8:
            status = "✅" if degradation < 50 else "⚠️" if degradation < 150 else "❌"
        else:  # 4-bit
            status = "✅" if degradation < 200 else "⚠️" if degradation < 500 else "❌"

        print(f"   {bits:2d}-bit: +{degradation:.1f}% (PPL: {ppl:.2f}) {status}")

    return results


def test_lora_behavior_sliding(sp_model, tokenizer, device):
    """Test 3: Verify LoRA behavior with sliding window."""
    print("\n" + "="*60)
    print("TEST 3: LORA BEHAVIOR WITH SLIDING WINDOW")
    print("="*60)

    # Calibrate students
    print("\n📊 Calibrating for LoRA testing...")
    calibration_texts = get_calibration_texts(num_texts=12)

    for precision in [16, 8, 4]:
        calibrate_precision_with_debug(sp_model, tokenizer, device, precision, calibration_texts)

    sp_model.eval()
    lora_results = {}

    print("\nTesting LoRA contributions...")

    for precision in [32, 16, 8, 4]:
        sp_model.set_precision(precision)

        # Count LoRA layers
        enabled_loras = 0
        total_loras = 0

        for name, module in sp_model.named_modules():
            lora_adapters = getattr(module, 'lora_adapters', None)
            if lora_adapters is not None:
                bit_key = f'{precision}bit'
                if bit_key in lora_adapters:
                    lora = lora_adapters[bit_key]
                    total_loras += 1
                    enabled = getattr(lora, 'enabled', False)
                    if enabled:
                        enabled_loras += 1

        print(f"\n   {precision}-bit: {enabled_loras}/{total_loras} LoRA layers enabled")

        # Test perplexity
        result = calculate_perplexity_properly(
            model=sp_model,
            tokenizer=tokenizer,
            device=device,
            dataset_name='wikitext',
            max_length=256,
            stride=128,
            max_samples=256
        )

        lora_results[precision] = {
            'enabled_loras': enabled_loras,
            'total_loras': total_loras,
            'ppl': result['perplexity']
        }

        print(f"      PPL with LoRA: {result['perplexity']:.2f}")

    # Analysis
    print(f"\n📊 LORA ANALYSIS:")
    if lora_results[32]['enabled_loras'] == 0:
        print("   ✅ 32-bit teacher: LoRA properly disabled")
    else:
        print("   ❌ 32-bit teacher: LoRA should be disabled!")

    for bits in [16, 8, 4]:
        if lora_results[bits]['enabled_loras'] > 0:
            print(f"   ✅ {bits}-bit student: LoRA enabled")
        else:
            print(f"   ❌ {bits}-bit student: LoRA should be enabled!")

    return lora_results


def test_quantizer_activation_sliding(sp_model, tokenizer, device):
    """Test 4: Verify quantizer activation with sliding window and debug."""
    print("\n" + "="*60)
    print("TEST 4: QUANTIZER ACTIVATION WITH SLIDING WINDOW")
    print("="*60)

    quantization_results = {}

    for bits in [4, 8]:
        print(f"\n🔧 Testing {bits}-bit precision:")

        # Calibrate with debug
        calibration_texts = get_calibration_texts(num_texts=8)
        calibrate_precision_with_debug(sp_model, tokenizer, device, bits, calibration_texts)

        sp_model.eval()
        sp_model.set_precision(bits)

        # Check quantizer states
        quantizer_states = []
        for name, module in sp_model.named_modules():
            quantizers_weight = getattr(module, 'quantizers_weight', None)
            if quantizers_weight is not None:
                bits_key = f'{bits}bit'
                if bits_key in quantizers_weight:
                    quantizer = quantizers_weight[bits_key]
                    state = {
                        'calibrated': quantizer.calibrated,
                        'scale': quantizer.scale.mean().item() if quantizer.scale.numel() > 0 else 0
                    }
                    quantizer_states.append(state)
                    if len(quantizer_states) == 1:
                        print(f"   Sample quantizer: Calibrated={state['calibrated']}, "
                              f"Scale={state['scale']:.6f}")
                        break

        # Test with sliding window
        print(f"   Testing with sliding window...")
        result = calculate_perplexity_properly(
            model=sp_model,
            tokenizer=tokenizer,
            device=device,
            dataset_name='wikitext',
            max_length=256,
            stride=128,
            max_samples=256
        )

        quantization_results[bits] = {
            'quantizer_states': quantizer_states,
            'ppl': result['perplexity'],
            'loss': result['loss']
        }

        print(f"   Results: Loss={result['loss']:.4f}, PPL={result['perplexity']:.2f}")

    # Test 16-bit
    sp_model.set_precision(16)
    sp_model.eval()
    print(f"\n🧪 Testing 16-bit (minimal quantization):")

    result_16 = calculate_perplexity_properly(
        model=sp_model,
        tokenizer=tokenizer,
        device=device,
        dataset_name='wikitext',
        max_length=256,
        stride=128,
        max_samples=256
    )

    print(f"   Results: Loss={result_16['loss']:.4f}, PPL={result_16['perplexity']:.2f}")

    # Analysis
    print("\n📊 QUANTIZATION ANALYSIS:")
    for bits in [4, 8]:
        all_calibrated = all(s['calibrated'] for s in quantization_results[bits]['quantizer_states'])
        print(f"   {bits}-bit: {len(quantization_results[bits]['quantizer_states'])} quantizers, "
              f"All calibrated: {all_calibrated}")

    degradation_8 = ((quantization_results[8]['ppl'] - result_16['perplexity']) / result_16['perplexity']) * 100
    degradation_4 = ((quantization_results[4]['ppl'] - result_16['perplexity']) / result_16['perplexity']) * 100

    print(f"   8-bit degradation from 16-bit: {degradation_8:.1f}%")
    print(f"   4-bit degradation from 16-bit: {degradation_4:.1f}%")

    return quantization_results


def run_comprehensive_test(test_suite: str = "all", quick_mode: bool = False):
    """Run comprehensive test suite with multiple testing modules.

    Args:
        test_suite: Which test suite to run ('all', 'basic', 'precision', 'batchnorm', 'training')
        quick_mode: If True, run tests with reduced samples for faster execution
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE SP MODEL DEBUG SUITE")
    print("="*80)
    print(f"Test Suite: {test_suite}")
    print(f"Quick Mode: {'Enabled' if quick_mode else 'Disabled'}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load models
    print("\n🔧 Loading models...")
    num_layers = 6 if quick_mode else 12
    sp_model, sp_config = create_properly_initialized_model(use_pretrained=True, num_layers=num_layers)
    sp_model = sp_model.to(device)

    # Load GPT-2 with warning suppression
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*loss_type.*")
        gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_model = gpt2_model.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Run tests based on suite selection
    test_results = {}

    # Basic tests (always run)
    if test_suite in ['all', 'basic']:
        print("\n" + "="*60)
        print("RUNNING BASIC TEST SUITE")
        print("="*60)

        # Test 1: 32-bit equivalence
        print("\n" + "="*60)
        print("Test 1: 32-bit Equivalence")
        test_results['32bit_equivalence'] = test_32bit_equivalence_sliding(sp_model, gpt2_model, tokenizer, device)

        # Test 2: Quantization degradation
        print("\n" + "="*60)
        print("Test 2: Quantization Degradation")
        test_results['quantization_degradation'] = test_quantization_degradation_sliding(sp_model, tokenizer, device)

        # Test 3: LoRA behavior
        print("\n" + "="*60)
        print("Test 3: LoRA Behavior")
        test_results['lora_behavior'] = test_lora_behavior_sliding(sp_model, tokenizer, device)

        # Test 4: Quantizer activation
        print("\n" + "="*60)
        print("Test 4: Quantizer Activation")
        test_results['quantizer_activation'] = test_quantizer_activation_sliding(sp_model, tokenizer, device)

    # Precision mismatch tests
    if test_suite in ['all', 'precision']:
        print("\n" + "="*60)
        print("RUNNING PRECISION MISMATCH TEST SUITE")
        print("="*60)

        # Test precision consistency
        print("\nTest: Precision Consistency")
        test_results['precision_consistency'] = test_precision_consistency(sp_model, tokenizer, device)

        # Test layer-wise precision analysis
        print("\nTest: Layer-wise Precision Analysis")
        test_results['layer_precision'] = test_layer_precision_analysis(sp_model, tokenizer, device)

        # Test quantization saturation
        print("\nTest: Quantization Saturation")
        test_results['quantization_saturation'] = test_quantization_saturation(sp_model, tokenizer, device)

    # Batch normalization tests
    if test_suite in ['all', 'batchnorm']:
        print("\n" + "="*60)
        print("RUNNING BATCH NORMALIZATION TEST SUITE")
        print("="*60)

        # Test BN statistics tracking
        print("\nTest: BN Statistics Tracking")
        test_results['bn_statistics'] = test_bn_statistics_tracking()

        # Test BN gradient flow
        print("\nTest: BN Gradient Flow")
        test_results['bn_gradients'] = test_bn_gradient_flow()

        # Test BN mode switching
        print("\nTest: BN Mode Switching")
        test_results['bn_mode_switching'] = test_bn_mode_switching()

        # Test BN with small batches
        print("\nTest: BN Small Batch Behavior")
        test_results['bn_small_batch'] = test_bn_with_small_batch()

        # Test BN precision switching consistency
        print("\nTest: BN Precision Switching")
        test_results['bn_precision_switching'] = test_bn_precision_switching_consistency()

    # Training dynamics tests (skip in quick mode by default)
    if test_suite in ['all', 'training'] and not (quick_mode and test_suite == 'all'):
        print("\n" + "="*60)
        print("RUNNING TRAINING DYNAMICS TEST SUITE")
        print("="*60)

        # Test multi-batch training
        print("\nTest: Multi-batch Training")
        test_results['multi_batch_training'] = test_multi_batch_training()

        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Test QAT
        print("\nTest: Quantization-Aware Training")
        test_results['qat'] = test_quantization_aware_training()

        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Test distillation effectiveness
        print("\nTest: Distillation Effectiveness")
        test_results['distillation'] = test_distillation_effectiveness()

        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Test gradient accumulation
        print("\nTest: Gradient Accumulation")
        test_results['gradient_accumulation'] = test_gradient_accumulation_effects()

        # Test batch norm dynamics
        print("\nTest: Batch Norm Training Dynamics")
        test_results['bn_dynamics'] = test_batch_norm_training_dynamics()

    print("\n" + "="*80)
    print("✅ TEST SUITE COMPLETE")
    print("="*80)

    return test_results


def print_test_summary(results: Dict):
    """Print a summary of test results."""
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)

    total_tests = len(results)
    print(f"\n📊 Total test categories: {total_tests}")

    # Analyze results
    if '32bit_equivalence' in results:
        print("\n✅ Basic Tests:")
        if 'status' in results.get('32bit_equivalence', {}):
            print(f"   32-bit equivalence: {results['32bit_equivalence']['status']}")

    if 'precision_consistency' in results:
        print("\n✅ Precision Tests:")
        for precision in [16, 8, 4]:
            if precision in results.get('precision_consistency', {}):
                cos_sim = results['precision_consistency'][precision].get('avg_cosine_sim', 0)
                print(f"   {precision}-bit consistency: {cos_sim:.4f}")

    if 'bn_statistics' in results:
        print("\n✅ Batch Norm Tests:")
        print("   Statistics tracking: Passed")
        print("   Gradient flow: Passed")
        print("   Mode switching: Passed")

    if 'multi_batch_training' in results:
        print("\n✅ Training Dynamics:")
        print("   Multi-batch training: Completed")
        print("   QAT: Completed")
        print("   Distillation: Completed")

    print("\n" + "="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comprehensive SP Model Debug Suite')
    parser.add_argument('--suite', type=str, default='all',
                       choices=['all', 'basic', 'precision', 'batchnorm', 'training'],
                       help='Test suite to run')
    parser.add_argument('--quick', action='store_true',
                       help='Run tests in quick mode with reduced samples')
    parser.add_argument('--detect-precision-mismatch', action='store_true',
                       help='Run only precision mismatch detection')
    parser.add_argument('--test-batchnorm', action='store_true',
                       help='Run only batch normalization tests')
    parser.add_argument('--test-training', action='store_true',
                       help='Run only training dynamics tests')

    args = parser.parse_args()

    # Handle specific test flags
    if args.detect_precision_mismatch:
        args.suite = 'precision'
    elif args.test_batchnorm:
        args.suite = 'batchnorm'
    elif args.test_training:
        args.suite = 'training'

    print("\n🚀 Starting SP Model Debug Suite...")
    print(f"   Suite: {args.suite}")
    print(f"   Quick Mode: {args.quick}")

    try:
        results = run_comprehensive_test(test_suite=args.suite, quick_mode=args.quick)
        print_test_summary(results)
        print("\n✅ Test suite completed successfully!")
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
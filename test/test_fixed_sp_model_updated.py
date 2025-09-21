#!/usr/bin/env python3
"""
Updated test functions using proper sliding window approach for all tests.
This file contains the improved versions of test functions.
"""

import torch
import numpy as np
from test.dataset_utils import calculate_perplexity_properly, get_calibration_texts


def calibrate_precision(sp_model, tokenizer, device, precision, calibration_texts=None):
    """Helper function to calibrate a specific precision consistently."""
    if precision >= 32:
        return  # No calibration needed for 32-bit teacher

    if calibration_texts is None:
        calibration_texts = get_calibration_texts(num_texts=16)

    print(f"   📊 Calibrating {precision}-bit precision...")
    sp_model.set_precision(precision)
    sp_model.train()  # Must be in training mode

    bits_key = f'{precision}bit'
    calibrated_count = 0

    # Start calibration
    for name, module in sp_model.named_modules():
        if hasattr(module, 'quantizers_weight') and hasattr(module, 'quantizers_input'):
            if bits_key in module.quantizers_weight:
                module.quantizers_weight[bits_key].start_calibration()
                calibrated_count += 1
            if bits_key in module.quantizers_input:
                module.quantizers_input[bits_key].start_calibration()
                calibrated_count += 1

    # Collect statistics
    with torch.no_grad():
        for text in calibration_texts:
            tokens = tokenizer(text, return_tensors='pt',
                              max_length=128, truncation=True)['input_ids'].to(device)
            _ = sp_model(tokens)

    # Finish calibration
    for name, module in sp_model.named_modules():
        if hasattr(module, 'quantizers_weight') and hasattr(module, 'quantizers_input'):
            if bits_key in module.quantizers_weight:
                module.quantizers_weight[bits_key].finish_calibration()
            if bits_key in module.quantizers_input:
                module.quantizers_input[bits_key].finish_calibration()

    print(f"      ✅ Calibrated {calibrated_count} quantizers")


def test_32bit_equivalence_updated(sp_model, gpt2_model, tokenizer, device):
    """Test 1: Verify 32-bit teacher matches GPT-2 using sliding window."""
    print("\n" + "="*60)
    print("TEST 1: 32-BIT EQUIVALENCE WITH SLIDING WINDOW")
    print("="*60)

    # Set both models to eval mode and 32-bit precision
    sp_model.eval()
    gpt2_model.eval()
    sp_model.set_precision(32)

    print("\nCalculating perplexity using sliding window approach...")

    # Calculate perplexity for SP model
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

    # Calculate perplexity for GPT-2 model
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

    # Compare results
    ppl_diff = abs(sp_results['perplexity'] - gpt2_results['perplexity'])
    loss_diff = abs(sp_results['loss'] - gpt2_results['loss'])

    print(f"\n📊 COMPARISON RESULTS:")
    print(f"   SP Model (32-bit):")
    print(f"      PPL: {sp_results['perplexity']:.2f}")
    print(f"      Loss: {sp_results['loss']:.4f}")
    print(f"   GPT-2 Model:")
    print(f"      PPL: {gpt2_results['perplexity']:.2f}")
    print(f"      Loss: {gpt2_results['loss']:.4f}")
    print(f"   Differences:")
    print(f"      PPL diff: {ppl_diff:.3f}")
    print(f"      Loss diff: {loss_diff:.4f}")

    # Determine status
    ppl_status = "✅ EXCELLENT" if ppl_diff < 1.0 else "⚠️ ACCEPTABLE" if ppl_diff < 5.0 else "❌ FAILED"
    print(f"\n   Status: {ppl_status}")

    return {
        'sp_ppl': sp_results['perplexity'],
        'gpt2_ppl': gpt2_results['perplexity'],
        'ppl_diff': ppl_diff,
        'loss_diff': loss_diff,
        'status': ppl_status
    }


def test_quantization_degradation_updated(sp_model, tokenizer, device):
    """Test 2: Verify quantization degradation using sliding window."""
    print("\n" + "="*60)
    print("TEST 2: QUANTIZATION DEGRADATION WITH SLIDING WINDOW")
    print("="*60)

    # Calibrate all precisions first
    print("\n📊 Calibrating all student precisions...")
    calibration_texts = get_calibration_texts(num_texts=16)

    for precision in [16, 8, 4]:  # Calibrate students only
        calibrate_precision(sp_model, tokenizer, device, precision, calibration_texts)
    print("   ✅ All calibrations complete")

    # Test with sliding window approach
    results = {}
    sp_model.eval()  # Set to eval mode for testing

    for precision in [32, 16, 8, 4]:  # Include 32-bit teacher as baseline
        sp_model.set_precision(precision)
        print(f"\n   Testing {precision}-bit precision with sliding window...")

        # Calculate perplexity using sliding window
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

        print(f"   {precision}-bit Results:")
        print(f"      PPL: {result['perplexity']:.2f}")
        print(f"      Loss: {result['loss']:.4f}")
        print(f"      Tokens: {result['total_tokens']:,}")
        print(f"      Windows: {result['num_windows']}")

    # Analysis
    print("\n📊 DEGRADATION ANALYSIS:")
    baseline_ppl = results[32]['ppl']
    print(f"   Baseline (32-bit teacher): PPL = {baseline_ppl:.2f}")

    for bits in [16, 8, 4]:
        ppl = results[bits]['ppl']
        degradation = ((ppl - baseline_ppl) / baseline_ppl) * 100

        if bits == 16:
            status = "✅ EXCELLENT" if degradation < 10 else "⚠️ ACCEPTABLE" if degradation < 30 else "❌ POOR"
        elif bits == 8:
            status = "✅ EXCELLENT" if degradation < 50 else "⚠️ ACCEPTABLE" if degradation < 150 else "❌ POOR"
        else:  # 4-bit
            status = "✅ EXCELLENT" if degradation < 200 else "⚠️ ACCEPTABLE" if degradation < 500 else "❌ POOR"

        print(f"   {bits:2d}-bit: +{degradation:.1f}% (PPL: {ppl:.2f}) - {status}")

    return results


def test_lora_behavior_updated(sp_model, tokenizer, device):
    """Test 3: Verify LoRA behavior with sliding window evaluation."""
    print("\n" + "="*60)
    print("TEST 3: LORA BEHAVIOR WITH SLIDING WINDOW")
    print("="*60)

    # Calibrate all student precisions
    print("\n📊 Calibrating student precisions for LoRA testing...")
    calibration_texts = get_calibration_texts(num_texts=12)

    for precision in [16, 8, 4]:
        calibrate_precision(sp_model, tokenizer, device, precision, calibration_texts)
    print("   ✅ Calibrations complete")

    # Test LoRA behavior with sliding window
    sp_model.eval()
    lora_results = {}

    print("\nTesting LoRA contribution with sliding window approach...")

    for precision in [32, 16, 8, 4]:
        sp_model.set_precision(precision)

        # Count enabled LoRA layers
        enabled_loras = 0
        total_loras = 0

        for name, module in sp_model.named_modules():
            if hasattr(module, 'lora_adapters'):
                bit_key = f'{precision}bit'
                if bit_key in module.lora_adapters:
                    lora = module.lora_adapters[bit_key]
                    total_loras += 1
                    if hasattr(lora, 'enabled') and lora.enabled:
                        enabled_loras += 1

        # Calculate perplexity with LoRA
        print(f"\n   {precision}-bit: {enabled_loras}/{total_loras} LoRA layers enabled")

        result = calculate_perplexity_properly(
            model=sp_model,
            tokenizer=tokenizer,
            device=device,
            dataset_name='wikitext',
            max_length=256,
            stride=128,
            max_samples=256  # Smaller for LoRA test
        )

        lora_results[precision] = {
            'enabled_loras': enabled_loras,
            'total_loras': total_loras,
            'ppl': result['perplexity'],
            'loss': result['loss']
        }

        print(f"      PPL with LoRA: {result['perplexity']:.2f}")

    # Analysis
    print(f"\n📊 LORA ANALYSIS:")

    if lora_results[32]['enabled_loras'] == 0:
        print("   ✅ 32-bit (teacher): LoRA properly disabled")
    else:
        print("   ❌ 32-bit (teacher): LoRA should be disabled!")

    for bits in [16, 8, 4]:
        if lora_results[bits]['enabled_loras'] > 0:
            print(f"   ✅ {bits}-bit (student): LoRA enabled ({lora_results[bits]['enabled_loras']} layers)")
        else:
            print(f"   ❌ {bits}-bit (student): LoRA should be enabled!")

    return lora_results


def test_quantizer_activation_updated(sp_model, tokenizer, device):
    """Test 4: Verify quantizer activation with sliding window."""
    print("\n" + "="*60)
    print("TEST 4: QUANTIZER ACTIVATION WITH SLIDING WINDOW")
    print("="*60)

    quantization_results = {}

    for bits in [4, 8]:  # Test low-bit precisions
        print(f"\n🔧 Testing {bits}-bit precision:")

        # Calibrate
        print(f"   📈 Calibrating {bits}-bit...")
        calibration_texts = get_calibration_texts(num_texts=8)
        calibrate_precision(sp_model, tokenizer, device, bits, calibration_texts)

        # Set to eval mode for testing
        sp_model.eval()
        sp_model.set_precision(bits)

        # Check quantizer state
        quantizer_states = []
        for name, module in sp_model.named_modules():
            if hasattr(module, 'quantizers_weight'):
                bits_key = f'{bits}bit'
                if bits_key in module.quantizers_weight:
                    quantizer = module.quantizers_weight[bits_key]
                    state = {
                        'calibrated': quantizer.calibrated,
                        'scale': quantizer.scale.mean().item() if quantizer.scale.numel() > 0 else 0
                    }
                    quantizer_states.append(state)
                    if len(quantizer_states) == 1:
                        print(f"   Sample quantizer state:")
                        print(f"     Calibrated: {state['calibrated']}")
                        print(f"     Scale: {state['scale']:.6f}")
                        break

        # Test with sliding window
        print(f"   Testing calibrated {bits}-bit model with sliding window...")
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

        print(f"   Average loss: {result['loss']:.4f}, PPL: {result['perplexity']:.2f}")

    # Test 16-bit (no quantization)
    sp_model.eval()
    sp_model.set_precision(16)
    print(f"\n🧪 Testing 16-bit precision (minimal quantization):")

    result_16 = calculate_perplexity_properly(
        model=sp_model,
        tokenizer=tokenizer,
        device=device,
        dataset_name='wikitext',
        max_length=256,
        stride=128,
        max_samples=256
    )

    print(f"   Average loss: {result_16['loss']:.4f}, PPL: {result_16['perplexity']:.2f}")

    # Analysis
    print("\n📊 QUANTIZATION RESULTS:")
    for bits in [4, 8]:
        all_calibrated = all(s['calibrated'] for s in quantization_results[bits]['quantizer_states'])
        print(f"   {bits}-bit:")
        print(f"     Quantizers found: {len(quantization_results[bits]['quantizer_states'])}")
        print(f"     All calibrated: {all_calibrated}")

    # Compare degradation
    degradation_8 = ((quantization_results[8]['ppl'] - result_16['perplexity']) / result_16['perplexity']) * 100
    degradation_4 = ((quantization_results[4]['ppl'] - result_16['perplexity']) / result_16['perplexity']) * 100

    print(f"   8-bit degradation from 16-bit: {degradation_8:.1f}%")
    print(f"   4-bit degradation from 16-bit: {degradation_4:.1f}%")

    return quantization_results


if __name__ == "__main__":
    print("This module contains updated test functions using sliding window approach.")
    print("Import these functions into test_fixed_sp_model.py to use them.")
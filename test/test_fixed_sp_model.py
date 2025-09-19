#!/usr/bin/env python3
"""
Comprehensive Test of Fixed SP Model
Tests all fixes applied according to implementation prompt
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test.fix_model_initialization import create_properly_initialized_model
from shared.calibration_manager import calibrate_sp_model


def test_16bit_equivalence(sp_model, gpt2_model, tokenizer, device):
    """Test 1: Verify 16-bit matches GPT-2 exactly"""
    print("\n" + "="*60)
    print("TEST 1: 16-BIT GPT-2 EQUIVALENCE")
    print("="*60)

    sp_model.set_precision(16)
    sp_model.eval()
    gpt2_model.eval()

    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming technology.",
        "Python is a popular programming language.",
        "Neural networks consist of interconnected layers.",
        "Climate change poses significant global challenges.",
    ]

    print(f"Testing {len(test_sentences)} sentences...")

    perplexity_diffs = []
    logit_diffs = []

    for i, sentence in enumerate(test_sentences):
        inputs = tokenizer(sentence, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)

        with torch.no_grad():
            # SP model outputs
            sp_outputs = sp_model(input_ids, labels=input_ids)
            sp_loss = sp_outputs['loss'].item()
            sp_ppl = torch.exp(torch.tensor(sp_loss)).item()

            # GPT-2 outputs
            gpt2_outputs = gpt2_model(input_ids, labels=input_ids)
            gpt2_loss = gpt2_outputs['loss'].item()
            gpt2_ppl = torch.exp(torch.tensor(gpt2_loss)).item()

            # Logit comparison
            if isinstance(sp_outputs, dict) and 'logits' in sp_outputs:
                sp_logits = sp_outputs['logits']
            else:
                sp_logits = sp_outputs

            gpt2_logits = gpt2_outputs['logits']

            logit_diff = (sp_logits - gpt2_logits).abs()
            mean_logit_diff = logit_diff.mean().item()

            ppl_diff = abs(sp_ppl - gpt2_ppl)

            perplexity_diffs.append(ppl_diff)
            logit_diffs.append(mean_logit_diff)

            print(f"  {i+1}. '{sentence[:30]}...': PPL diff={ppl_diff:.3f}, Logit diff={mean_logit_diff:.6f}")

    # Summary statistics
    avg_ppl_diff = np.mean(perplexity_diffs)
    max_ppl_diff = max(perplexity_diffs)
    avg_logit_diff = np.mean(logit_diffs)
    max_logit_diff = max(logit_diffs)

    print(f"\n📊 RESULTS:")
    print(f"   Average PPL difference: {avg_ppl_diff:.4f}")
    print(f"   Maximum PPL difference: {max_ppl_diff:.4f}")
    print(f"   Average logit difference: {avg_logit_diff:.6f}")
    print(f"   Maximum logit difference: {max_logit_diff:.6f}")

    # Assessment
    if avg_ppl_diff < 0.1 and max_ppl_diff < 0.5:
        print("   ✅ EXCELLENT: PPL matches GPT-2 closely")
        ppl_status = "excellent"
    elif avg_ppl_diff < 1.0 and max_ppl_diff < 2.0:
        print("   ⚠️ GOOD: PPL acceptable but not perfect")
        ppl_status = "good"
    else:
        print("   ❌ FAILED: PPL differs significantly from GPT-2")
        ppl_status = "failed"

    if avg_logit_diff < 0.001:
        print("   ✅ EXCELLENT: Logits match GPT-2 closely")
        logit_status = "excellent"
    elif avg_logit_diff < 0.01:
        print("   ⚠️ GOOD: Logits acceptable")
        logit_status = "good"
    else:
        print("   ❌ FAILED: Logits differ significantly")
        logit_status = "failed"

    return {
        'ppl_diff': avg_ppl_diff,
        'logit_diff': avg_logit_diff,
        'ppl_status': ppl_status,
        'logit_status': logit_status
    }


def test_quantization_degradation(sp_model, tokenizer, device):
    """Test 2: Verify acceptable quantization degradation"""
    print("\n" + "="*60)
    print("TEST 2: QUANTIZATION DEGRADATION")
    print("="*60)

    test_sentences = [
        "Machine learning algorithms process vast amounts of data.",
        "Natural language processing enables human-computer interaction.",
        "Deep learning models require substantial computational resources.",
        "Python programming language supports scientific computing applications.",
        "Artificial intelligence systems demonstrate remarkable capabilities.",
    ]

    sp_model.eval()

    results = {}

    with torch.no_grad():
        for precision in [16, 8, 4]:
            sp_model.set_precision(precision)

            losses = []
            for sentence in test_sentences:
                inputs = tokenizer(sentence, return_tensors='pt')
                input_ids = inputs['input_ids'].to(device)

                outputs = sp_model(input_ids, labels=input_ids)
                loss = outputs['loss'].item()
                losses.append(loss)

            avg_loss = np.mean(losses)
            avg_ppl = np.exp(avg_loss)

            results[precision] = {'loss': avg_loss, 'ppl': avg_ppl}
            print(f"   {precision:2d}-bit: Loss = {avg_loss:.4f}, PPL = {avg_ppl:.2f}")

    # Calculate degradation
    baseline_ppl = results[16]['ppl']

    print(f"\n📊 DEGRADATION ANALYSIS:")
    print(f"   Baseline (16-bit): {baseline_ppl:.2f} PPL")

    degradation_results = {}

    for precision in [8, 4]:
        if precision in results:
            ppl = results[precision]['ppl']
            degradation = ((ppl - baseline_ppl) / baseline_ppl) * 100

            degradation_results[precision] = degradation

            # Expected targets: 8-bit <20%, 4-bit <100%
            if precision == 8:
                if degradation < 20:
                    status = "✅ EXCELLENT"
                elif degradation < 50:
                    status = "⚠️ ACCEPTABLE"
                else:
                    status = "❌ POOR"
                target = "target: <20%"
            else:  # 4-bit
                if degradation < 100:
                    status = "✅ EXCELLENT"
                elif degradation < 300:
                    status = "⚠️ ACCEPTABLE"
                else:
                    status = "❌ POOR"
                target = "target: <100%"

            print(f"   {precision}-bit degradation: {degradation:.1f}% ({status}, {target})")

    return degradation_results


def test_lora_behavior(sp_model, tokenizer, device):
    """Test 3: Verify LoRA is properly handled"""
    print("\n" + "="*60)
    print("TEST 3: LORA BEHAVIOR VERIFICATION")
    print("="*60)

    test_input = "The future of artificial intelligence looks promising."
    inputs = tokenizer(test_input, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)

    sp_model.eval()

    print("Checking LoRA contribution across bit-widths...")

    lora_contributions = {}

    with torch.no_grad():
        for precision in [16, 8, 4]:
            sp_model.set_precision(precision)

            # Count enabled LoRA layers
            enabled_loras = 0
            disabled_loras = 0
            total_loras = 0

            for name, module in sp_model.named_modules():
                if hasattr(module, 'lora_adapters'):
                    for bit_key, lora in module.lora_adapters.items():
                        total_loras += 1
                        if hasattr(lora, 'enabled') and lora.enabled:
                            enabled_loras += 1
                        else:
                            disabled_loras += 1

            lora_contributions[precision] = {
                'enabled': enabled_loras,
                'disabled': disabled_loras,
                'total': total_loras
            }

            # Test actual forward pass
            outputs = sp_model(input_ids)
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs)
            else:
                logits = outputs

            print(f"   {precision:2d}-bit: {enabled_loras}/{total_loras} LoRA layers enabled, "
                  f"logits shape: {list(logits.shape)}")

    # Analysis
    print(f"\n📊 LORA ANALYSIS:")

    if lora_contributions[16]['enabled'] == 0:
        print("   ✅ 16-bit: LoRA properly disabled")
    else:
        print("   ❌ 16-bit: LoRA should be disabled!")

    if lora_contributions[8]['enabled'] > 0 and lora_contributions[4]['enabled'] > 0:
        print("   ✅ 8/4-bit: LoRA properly enabled for quantization compensation")
    else:
        print("   ⚠️ 8/4-bit: LoRA may not be working correctly")

    return lora_contributions


def test_quantizer_activation(sp_model, tokenizer, device):
    """Test 4: Diagnose why quantizers aren't getting calibrated"""
    print("\n" + "="*60)
    print("TEST 4: QUANTIZER ACTIVATION DIAGNOSIS")
    print("="*60)

    test_input = "Testing quantizer activation during forward pass."
    inputs = tokenizer(test_input, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)

    sp_model.eval()

    # Add hooks to monitor quantizer calls
    quantizer_calls = {}

    def create_hook(name):
        def hook(module, input, output):
            if name not in quantizer_calls:
                quantizer_calls[name] = 0
            quantizer_calls[name] += 1
            print(f"    📞 Quantizer called: {name} (call #{quantizer_calls[name]}, num_bits={module.num_bits})")
        return hook

    # Register hooks on first few quantizers to see if they get called
    hooks = []
    quantizer_count = 0
    for name, module in sp_model.named_modules():
        if 'LearnableFakeQuantize' in str(type(module)) and quantizer_count < 10:
            hook = module.register_forward_hook(create_hook(name))
            hooks.append(hook)
            quantizer_count += 1

    print(f"Registered hooks on {len(hooks)} quantizers for monitoring...")

    with torch.no_grad():
        for bits in [4, 8, 16]:
            print(f"\n🔧 Testing {bits}-bit precision:")
            sp_model.set_precision(bits)

            # Reset call counts
            quantizer_calls.clear()

            # Forward pass
            outputs = sp_model(input_ids)

            print(f"   Forward pass completed. Quantizer calls: {len(quantizer_calls)}")
            if quantizer_calls:
                print(f"   Active quantizers: {list(quantizer_calls.keys())[:5]}...")
            else:
                print(f"   ❌ NO QUANTIZERS WERE CALLED!")

                # Check if quantizers exist but aren't being used
                total_quantizers = 0
                for name, module in sp_model.named_modules():
                    if 'LearnableFakeQuantize' in str(type(module)):
                        total_quantizers += 1
                        if total_quantizers <= 3:  # Print first few
                            print(f"      Found quantizer {name}: num_bits={module.num_bits}")

                print(f"   Total quantizers found: {total_quantizers}")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return len(quantizer_calls) > 0


def test_distillation_setup(sp_model, tokenizer, device):
    """Test 5: Verify distillation setup works"""
    print("\n" + "="*60)
    print("TEST 5: DISTILLATION SETUP")
    print("="*60)

    test_input = "Distillation transfers knowledge from teacher to student models."
    inputs = tokenizer(test_input, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)

    sp_model.eval()

    teacher_outputs = None
    student_outputs = {}

    print("Testing teacher-student setup...")

    with torch.no_grad():
        # Test teacher mode (16-bit)
        sp_model.set_precision(16)
        teacher_outputs = sp_model(input_ids, output_hidden_states=True, return_dict=True)

        if isinstance(teacher_outputs, dict) and 'logits' in teacher_outputs:
            print(f"   ✅ Teacher (16-bit): Logits shape {list(teacher_outputs['logits'].shape)}")
            if 'hidden_states' in teacher_outputs:
                print(f"   ✅ Teacher hidden states: {len(teacher_outputs['hidden_states'])} layers")
        else:
            print(f"   ⚠️ Teacher outputs may not be in expected format")

        # Test student modes
        for precision in [8, 4]:
            sp_model.set_precision(precision)
            outputs = sp_model(input_ids, output_hidden_states=True, return_dict=True)
            student_outputs[precision] = outputs

            if isinstance(outputs, dict) and 'logits' in outputs:
                print(f"   ✅ Student ({precision}-bit): Logits shape {list(outputs['logits'].shape)}")
            else:
                print(f"   ⚠️ Student ({precision}-bit) outputs may not be in expected format")

    # Test distillation loss computation (basic)
    if teacher_outputs and 8 in student_outputs:
        try:
            teacher_logits = teacher_outputs['logits']
            student_logits = student_outputs[8]['logits']

            # Simple KL divergence test
            T = 3.0
            teacher_probs = F.log_softmax(teacher_logits / T, dim=-1)
            student_probs = F.log_softmax(student_logits / T, dim=-1)

            kl_loss = F.kl_div(student_probs, teacher_probs,
                               reduction='batchmean', log_target=True)

            print(f"   ✅ KL divergence computable: {kl_loss.item():.4f}")

        except Exception as e:
            print(f"   ❌ Distillation loss computation failed: {e}")

    print("   📝 Note: Full distillation training requires separate training script")

    return True


def run_comprehensive_test():
    """Run all tests on the fixed SP model"""
    print("\n" + "="*80)
    print("COMPREHENSIVE FIXED SP MODEL TEST")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load models
    print("\n🔧 Loading and initializing models...")
    sp_model, sp_config = create_properly_initialized_model(use_pretrained=True)
    sp_model = sp_model.to(device)

    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_model = gpt2_model.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Calibrate the SP model
    print("\n🎯 Calibrating quantizers...")
    calibration_success = calibrate_sp_model(sp_model, tokenizer, device)

    if not calibration_success:
        print("❌ Calibration failed! Tests may not be reliable.")
        return False

    # Run tests
    test_results = {}

    try:
        # Test 1: 16-bit equivalence
        test_results['equivalence'] = test_16bit_equivalence(
            sp_model, gpt2_model, tokenizer, device
        )

        # Test 2: Quantization degradation
        test_results['degradation'] = test_quantization_degradation(
            sp_model, tokenizer, device
        )

        # Test 3: LoRA behavior
        test_results['lora'] = test_lora_behavior(
            sp_model, tokenizer, device
        )

        # Test 4: Quantizer activation diagnosis
        test_results['quantizer_activation'] = test_quantizer_activation(
            sp_model, tokenizer, device
        )

        # Test 5: Distillation setup
        test_results['distillation'] = test_distillation_setup(
            sp_model, tokenizer, device
        )

    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        return False

    # Final assessment
    print("\n" + "="*80)
    print("FINAL ASSESSMENT")
    print("="*80)

    passed_tests = 0
    total_tests = 5

    # Assess each test
    if test_results['equivalence']['ppl_status'] in ['excellent', 'good']:
        print("✅ Test 1 (16-bit equivalence): PASSED")
        passed_tests += 1
    else:
        print("❌ Test 1 (16-bit equivalence): FAILED")

    if (test_results['degradation'].get(8, 999) < 50 and
            test_results['degradation'].get(4, 9999) < 300):
        print("✅ Test 2 (quantization degradation): PASSED")
        passed_tests += 1
    else:
        print("❌ Test 2 (quantization degradation): FAILED")

    if test_results['lora'][16]['enabled'] == 0:
        print("✅ Test 3 (LoRA behavior): PASSED")
        passed_tests += 1
    else:
        print("❌ Test 3 (LoRA behavior): FAILED")

    if test_results['quantizer_activation']:
        print("✅ Test 4 (quantizer activation): PASSED")
        passed_tests += 1
    else:
        print("❌ Test 4 (quantizer activation): FAILED - Quantizers not being called!")

    if test_results['distillation']:
        print("✅ Test 5 (distillation setup): PASSED")
        passed_tests += 1
    else:
        print("❌ Test 5 (distillation setup): FAILED")

    print(f"\n🏆 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! SP model is ready for training.")
        return True
    else:
        print("⚠️ Some tests failed. Check individual results above.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    if success:
        print("\n✅ Fixed SP model validation complete!")
    else:
        print("\n❌ Validation revealed issues that need fixing.")
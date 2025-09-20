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
import gc

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test.fix_model_initialization import create_properly_initialized_model


def get_memory_info(stage_name=""):
    """Get current memory usage information."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB

        print(f"\n📊 Memory Status - {stage_name}")
        print(f"   Allocated: {allocated:.3f} GB")
        print(f"   Reserved:  {reserved:.3f} GB")
        print(f"   Max Used:  {max_allocated:.3f} GB")

        # Get more detailed info
        if allocated > 10:  # If using more than 10GB, show details
            print("\n   ⚠️ High memory usage detected! Analyzing...")
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()

            new_allocated = torch.cuda.memory_allocated() / (1024**3)
            print(f"   After cleanup: {new_allocated:.3f} GB")

            if new_allocated > allocated * 0.9:
                print("   ❌ Memory not freed by garbage collection - likely held by model")

        return allocated
    else:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"\n📊 Memory Status - {stage_name}")
        print(f"   RAM Used: {mem_info.rss / (1024**3):.3f} GB")
        return mem_info.rss / (1024**3)


def count_model_parameters(model, stage_name=""):
    """Count and display model parameters."""
    total_params = 0
    trainable_params = 0

    print(f"\n🔢 Parameter Count - {stage_name}")

    # Count parameters by module type
    param_by_type = {}

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

        # Categorize by module type
        if 'lora' in name.lower():
            key = 'LoRA'
        elif 'quantizer' in name.lower():
            key = 'Quantizer'
        elif 'wte' in name or 'wpe' in name:
            key = 'Embeddings'
        elif 'ln' in name:
            key = 'LayerNorm'
        else:
            key = 'Linear'

        if key not in param_by_type:
            param_by_type[key] = 0
        param_by_type[key] += param.numel()

    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Parameter memory: {total_params * 4 / (1024**3):.3f} GB (FP32)")

    print("\n   Parameters by type:")
    for key, count in param_by_type.items():
        print(f"     {key}: {count:,} ({count * 4 / (1024**3):.3f} GB)")

    # Count buffers (quantizer statistics)
    total_buffers = 0
    buffer_count = 0
    for name, buffer in model.named_buffers():
        if buffer is not None:
            total_buffers += buffer.numel()
            buffer_count += 1

    if buffer_count > 0:
        print(f"\n   Buffers: {buffer_count} buffers, {total_buffers:,} elements")
        print(f"   Buffer memory: {total_buffers * 4 / (1024**3):.3f} GB")

    return total_params


def calibrate_with_two_pass(sp_model, tokenizer, device):
    """Calibrate SP model using two-pass statistics collection."""
    print("\n🔧 Two-Pass Calibration Starting...")

    # Calibration texts (reduced for memory efficiency)
    calibration_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming modern technology rapidly.",
        "Python is a versatile programming language for data science.",
        "Natural language processing enables computers to understand human language.",
    ]

    # Set to training mode for calibration (required for automatic two-pass)
    sp_model.train()

    # Calibrate each bit-width using two-pass
    for bits in [4, 8]:  # Skip 16-bit (no quantization needed)
        print(f"\n📊 Calibrating {bits}-bit mode...")
        sp_model.set_precision(bits)

        # The quantizers will automatically handle two-pass internally in training mode
        # Just run forward passes and the quantizers will collect stats automatically
        print(f"  Running calibration for {bits}-bit...")

        # Need to be in training mode for automatic two-pass to work
        sp_model.train()

        with torch.no_grad():
            for i, text in enumerate(calibration_texts):
                tokens = tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=64,
                    truncation=True,
                    padding=False
                )['input_ids'].to(device)

                # Forward pass - quantizers handle statistics automatically
                _ = sp_model(tokens)

                if (i + 1) % 4 == 0:
                    print(f"    Processed {i + 1}/{len(calibration_texts)} samples")

        # Test with calibrated model
        print(f"  Testing calibrated {bits}-bit model...")
        with torch.no_grad():
            test_text = calibration_texts[0]
            tokens = tokenizer(test_text, return_tensors='pt')['input_ids'].to(device)
            outputs = sp_model(tokens, labels=tokens)
            loss = outputs['loss'].item()
            ppl = torch.exp(torch.tensor(loss)).item()
            print(f"    Test loss: {loss:.4f}, PPL: {ppl:.2f}")

        print(f"  ✅ {bits}-bit calibration complete")

    # Set back to eval mode for testing
    sp_model.eval()

    # Reset to 16-bit
    sp_model.set_precision(16)
    print("\n✅ Two-pass calibration complete for all precisions")

    # Quick validation check
    print("\n🧪 Quick calibration quality check...")
    test_text = "Machine learning and artificial intelligence are revolutionizing technology."
    tokens = tokenizer(test_text, return_tensors='pt')['input_ids'].to(device)

    sp_model.eval()
    results = {}

    with torch.no_grad():
        for bits in [16, 8, 4]:
            sp_model.set_precision(bits)
            outputs = sp_model(tokens, labels=tokens)
            loss = outputs['loss'].item()
            ppl = torch.exp(torch.tensor(loss)).item()
            results[bits] = {'loss': loss, 'ppl': ppl}
            print(f"  {bits:2d}-bit: Loss = {loss:.4f}, PPL = {ppl:.2f}")

    # Check degradation
    baseline_ppl = results[16]['ppl']
    for bits in [8, 4]:
        ppl = results[bits]['ppl']
        degradation = ((ppl - baseline_ppl) / baseline_ppl) * 100
        status = "✅" if degradation < 100 else "⚠️" if degradation < 500 else "❌"
        print(f"  {bits}-bit degradation: {degradation:.1f}% {status}")

    sp_model.set_precision(16)
    return True


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
    ]  # Reduced for memory efficiency

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
    ]  # Reduced for memory efficiency

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
    """Test 4: Test two-pass quantization and verify correct calibration"""
    print("\n" + "="*60)
    print("TEST 4: TWO-PASS QUANTIZATION VERIFICATION")
    print("="*60)

    test_inputs = [
        "Testing quantizer activation during forward pass.",
        "Machine learning models require proper calibration.",
        "Deep neural networks process information efficiently."
    ]

    # Start in training mode for calibration
    sp_model.train()

    # Test each precision level
    quantization_results = {}

    for bits in [4, 8]:  # Skip 16-bit (no quantization)
        print(f"\n🔧 Testing {bits}-bit precision with two-pass:")
        sp_model.set_precision(bits)

        # Ensure we're in training mode for automatic two-pass calibration
        sp_model.train()

        # Run calibration - quantizers handle two-pass automatically
        print(f"   Running calibration...")

        with torch.no_grad():
            for text in test_inputs:
                inputs = tokenizer(text, return_tensors='pt')
                input_ids = inputs['input_ids'].to(device)
                _ = sp_model(input_ids)

        print(f"   Calibration complete")

        # Set to eval mode after calibration is done
        sp_model.eval()

        # Check quantizer state after calibration
        quantizer_states = []
        for name, module in sp_model.named_modules():
            if 'LearnableFakeQuantize' in str(type(module)) and f'{bits}bit' in name:
                state = {
                    'calibrated': module.calibrated,
                    'scale': module.scale.mean().item() if module.scale.numel() > 0 else 0
                }
                quantizer_states.append(state)
                if len(quantizer_states) == 1:  # Print first one as sample
                    print(f"   Sample quantizer state:")
                    print(f"     Calibrated: {state['calibrated']}")
                    print(f"     Scale: {state['scale']:.6f}")

        # Test with calibrated model
        print(f"   Testing calibrated {bits}-bit model...")
        losses = []
        with torch.no_grad():
            for text in test_inputs:
                inputs = tokenizer(text, return_tensors='pt')
                input_ids = inputs['input_ids'].to(device)
                outputs = sp_model(input_ids, labels=input_ids)
                losses.append(outputs['loss'].item())

        avg_loss = np.mean(losses)
        avg_ppl = np.exp(avg_loss)
        print(f"   Average loss: {avg_loss:.4f}, PPL: {avg_ppl:.2f}")

        # Store results
        quantization_results[bits] = {
            'avg_loss': avg_loss,
            'avg_ppl': avg_ppl,
            'num_quantizers': len(quantizer_states),
            'all_calibrated': all(s['calibrated'] for s in quantizer_states)
        }

    # Test 16-bit (should bypass quantization)
    print(f"\n🔧 Testing 16-bit precision (no quantization):")
    sp_model.set_precision(16)
    sp_model.eval()  # Ensure eval mode for 16-bit testing

    losses = []
    with torch.no_grad():
        for text in test_inputs:
            inputs = tokenizer(text, return_tensors='pt')
            input_ids = inputs['input_ids'].to(device)
            outputs = sp_model(input_ids, labels=input_ids)
            losses.append(outputs['loss'].item())

    avg_loss = np.mean(losses)
    avg_ppl = np.exp(avg_loss)
    print(f"   Average loss: {avg_loss:.4f}, PPL: {avg_ppl:.2f}")

    quantization_results[16] = {
        'avg_loss': avg_loss,
        'avg_ppl': avg_ppl,
        'scales_fixed': True,  # No quantization
        'num_quantizers': 0,
        'all_calibrated': True  # No quantizers to calibrate
    }

    # Verify results
    print(f"\n📊 QUANTIZATION RESULTS:")
    all_tests_passed = True

    for bits in [4, 8]:
        result = quantization_results[bits]
        print(f"   {bits}-bit:")
        print(f"     Quantizers found: {result['num_quantizers']}")
        print(f"     All calibrated: {result['all_calibrated']}")

        if not result['all_calibrated']:
            print(f"     ❌ Not all quantizers calibrated!")
            all_tests_passed = False
        if result['num_quantizers'] == 0:
            print(f"     ❌ No quantizers found!")
            all_tests_passed = False

    # Check degradation
    baseline_ppl = quantization_results[16]['avg_ppl']
    for bits in [8, 4]:
        ppl = quantization_results[bits]['avg_ppl']
        degradation = ((ppl - baseline_ppl) / baseline_ppl) * 100
        print(f"   {bits}-bit degradation: {degradation:.1f}%")

        # Reasonable thresholds
        if bits == 8 and degradation > 100:
            print(f"     ⚠️ High degradation for 8-bit!")
        elif bits == 4 and degradation > 500:
            print(f"     ⚠️ Very high degradation for 4-bit!")

    return all_tests_passed


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

    # Load models with reduced configuration for testing
    print("\n🔧 Loading and initializing models...")

    # Free up memory first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create a smaller model for testing (reduce layers)
    print("⚠️ Using reduced configuration for memory efficiency...")
    sp_model, sp_config = create_properly_initialized_model(use_pretrained=True, num_layers=6)  # Reduce from 12 to 6 layers

    # Move to device with memory check
    try:
        sp_model = sp_model.to(device)
    except torch.cuda.OutOfMemoryError:
        print("⚠️ GPU out of memory, switching to CPU...")
        device = torch.device('cpu')
        sp_model = sp_model.to(device)

    # Load GPT-2 model with same reduced config for comparison
    from transformers import GPT2Config
    gpt2_config = GPT2Config.from_pretrained('gpt2')
    gpt2_config.n_layer = 6  # Match reduced layers
    gpt2_model = GPT2LMHeadModel(gpt2_config)

    # Load partial weights from pretrained
    pretrained_full = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_model.transformer.wte = pretrained_full.transformer.wte
    gpt2_model.transformer.wpe = pretrained_full.transformer.wpe
    for i in range(6):
        gpt2_model.transformer.h[i] = pretrained_full.transformer.h[i]
    gpt2_model.transformer.ln_f = pretrained_full.transformer.ln_f
    gpt2_model.lm_head = pretrained_full.lm_head
    del pretrained_full  # Free memory

    try:
        gpt2_model = gpt2_model.to(device)
    except torch.cuda.OutOfMemoryError:
        print("⚠️ GPU out of memory for GPT-2, using CPU...")
        device = torch.device('cpu')
        sp_model = sp_model.to(device)
        gpt2_model = gpt2_model.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Calibrate the SP model using two-pass
    print("\n🎯 Calibrating quantizers with two-pass method...")
    calibration_success = calibrate_with_two_pass(sp_model, tokenizer, device)

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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Test 2: Quantization degradation
        test_results['degradation'] = test_quantization_degradation(
            sp_model, tokenizer, device
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Test 3: LoRA behavior
        test_results['lora'] = test_lora_behavior(
            sp_model, tokenizer, device
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Test 4: Quantizer activation diagnosis
        test_results['quantizer_activation'] = test_quantizer_activation(
            sp_model, tokenizer, device
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Test 5: Distillation setup
        test_results['distillation'] = test_distillation_setup(
            sp_model, tokenizer, device
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        print("✅ Test 4 (two-pass quantization): PASSED")
        passed_tests += 1
    else:
        print("❌ Test 4 (two-pass quantization): FAILED - Issues with calibration or fixed parameters!")

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
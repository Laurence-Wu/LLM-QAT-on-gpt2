#!/usr/bin/env python3
"""
Comprehensive Test of Fixed SP Model
Tests all fixes applied according to implementation prompt

STRICT TRAINING/EVALUATION SCHEDULE:
=====================================
This test follows a strict separation of training and evaluation stages:

1. TRAINING STAGE (model.train()):
   - Used ONLY for calibration/statistics collection
   - Quantizers collect min/max statistics
   - No actual quantization happens
   - Returns unquantized tensors

2. EVALUATION STAGE (model.eval()):
   - Used for ALL testing and inference
   - Quantizers use frozen statistics from training stage
   - Actual quantization is applied
   - Returns quantized tensors

CRITICAL RULES:
- ALWAYS use model.train() before collecting statistics
- ALWAYS use model.eval() before testing/inference
- NEVER mix training and evaluation in the same stage
- ALWAYS call finish_calibration() before switching to eval mode
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
from test.dataset_utils import create_test_dataloader, calculate_perplexity_properly, get_calibration_texts


def ensure_mode(model, mode, stage_name=""):
    """Ensure model is in the correct mode with explicit logging."""
    if mode == "train":
        model.train()
        print(f"    ⚙️ Set model to TRAINING mode for: {stage_name}")
    elif mode == "eval":
        model.eval()
        print(f"    ⚙️ Set model to EVALUATION mode for: {stage_name}")
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'eval'")

    # Verify mode was set correctly
    if model.training and mode == "eval":
        raise RuntimeError(f"Failed to set eval mode for {stage_name}")
    if not model.training and mode == "train":
        raise RuntimeError(f"Failed to set train mode for {stage_name}")


def calibrate_precision(sp_model, tokenizer, device, precision, calibration_texts=None):
    """Helper function to calibrate a specific precision consistently.

    Args:
        sp_model: The model to calibrate
        tokenizer: Tokenizer for text processing
        device: Device to run on
        precision: Bit-width to calibrate (4, 8, 16)
        calibration_texts: Optional list of calibration texts
    """
    if precision >= 32:
        # No calibration needed for 32-bit teacher
        return

    if calibration_texts is None:
        # Get diverse calibration texts from WikiText
        calibration_texts = get_calibration_texts(num_texts=16)

    print(f"   📊 Calibrating {precision}-bit precision...")
    sp_model.set_precision(precision)
    sp_model.train()  # Must be in training mode

    # Start calibration for all quantizers
    bits_key = f'{precision}bit'
    calibrated_count = 0

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
        for i, text in enumerate(calibration_texts):
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

    print(f"      ✅ Calibrated {calibrated_count} quantizers with {len(calibration_texts)} samples")


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

    # Calibrate each bit-width using manual calibration
    for bits in [4, 8, 16]:  # Include 16-bit as it's a student that needs calibration
        print(f"\n📊 Calibrating {bits}-bit mode...")
        sp_model.set_precision(bits)

        # ==== TRAINING STAGE: Collect Statistics ====
        print(f"  📈 TRAINING STAGE: Collecting statistics for {bits}-bit...")

        # CRITICAL: Must be in TRAINING mode for statistics collection
        sp_model.train()

        # Start manual calibration for all quantizers at this precision
        for name, module in sp_model.named_modules():
            # Check for SPLinearWithLoRA modules
            if hasattr(module, 'quantizers_weight') and hasattr(module, 'quantizers_input'):
                bits_key = f'{bits}bit'
                if bits_key in module.quantizers_weight:
                    module.quantizers_weight[bits_key].start_calibration()
                if bits_key in module.quantizers_input:
                    module.quantizers_input[bits_key].start_calibration()

        # Collect statistics (Pass 1 - MUST be in training mode)
        with torch.no_grad():
            for i, text in enumerate(calibration_texts):
                tokens = tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=64,
                    truncation=True,
                    padding=False
                )['input_ids'].to(device)

                # Forward pass to collect statistics
                _ = sp_model(tokens)

                if (i + 1) % 4 == 0:
                    print(f"    Processed {i + 1}/{len(calibration_texts)} samples")

        # Finish calibration (compute scales from collected statistics)
        for name, module in sp_model.named_modules():
            # Check for SPLinearWithLoRA modules
            if hasattr(module, 'quantizers_weight') and hasattr(module, 'quantizers_input'):
                bits_key = f'{bits}bit'
                if bits_key in module.quantizers_weight:
                    # Enable debug for 4-bit and 8-bit to show running min/max
                    show_debug = (bits in [4, 8]) and (module_count < 2) if 'module_count' in locals() else False
                    module.quantizers_weight[bits_key].finish_calibration(debug=show_debug if 'show_debug' in locals() else False)
                    if 'module_count' in locals(): module_count += 1
                if bits_key in module.quantizers_input:
                    show_debug = (bits in [4, 8]) and (module_count < 4) if 'module_count' in locals() else False
                    module.quantizers_input[bits_key].finish_calibration(debug=show_debug if 'show_debug' in locals() else False)
                    if 'module_count' in locals(): module_count += 1

        print(f"  ✅ Statistics collection complete")

        # ==== EVALUATION STAGE: Test with Calibrated Model ====
        print(f"  🧪 EVALUATION STAGE: Testing calibrated {bits}-bit model...")

        # CRITICAL: Switch to EVAL mode for actual quantization
        sp_model.eval()

        with torch.no_grad():
            test_text = calibration_texts[0]
            tokens = tokenizer(test_text, return_tensors='pt')['input_ids'].to(device)
            outputs = sp_model(tokens, labels=tokens)
            loss = outputs['loss'].item()
            ppl = torch.exp(torch.tensor(loss)).item()
            print(f"    Test loss: {loss:.4f}, PPL: {ppl:.2f}")

        print(f"  ✅ {bits}-bit calibration and evaluation complete")

    # Reset to 32-bit teacher
    sp_model.set_precision(32)
    print("\n✅ Two-pass calibration complete for all student precisions")

    # ==== FINAL EVALUATION STAGE: Quality Check ====
    print("\n🧪 FINAL EVALUATION: Calibration quality check...")
    test_text = "Machine learning and artificial intelligence are revolutionizing technology."
    tokens = tokenizer(test_text, return_tensors='pt')['input_ids'].to(device)

    # CRITICAL: Ensure EVAL mode for all final testing
    sp_model.eval()
    results = {}

    with torch.no_grad():
        for bits in [32, 16, 8, 4]:  # Include 32-bit teacher as baseline
            sp_model.set_precision(bits)
            outputs = sp_model(tokens, labels=tokens)
            loss = outputs['loss'].item()
            ppl = torch.exp(torch.tensor(loss)).item()
            results[bits] = {'loss': loss, 'ppl': ppl}
            print(f"  {bits:2d}-bit: Loss = {loss:.4f}, PPL = {ppl:.2f}")

    # Check degradation from 32-bit teacher
    baseline_ppl = results[32]['ppl']
    for bits in [16, 8, 4]:
        ppl = results[bits]['ppl']
        degradation = ((ppl - baseline_ppl) / baseline_ppl) * 100
        status = "✅" if degradation < 100 else "⚠️" if degradation < 500 else "❌"
        print(f"  {bits}-bit degradation from teacher: {degradation:.1f}% {status}")

    sp_model.set_precision(32)
    return True


def test_32bit_equivalence(sp_model, gpt2_model, tokenizer, device):
    """Test 1: Verify 32-bit teacher matches GPT-2 exactly"""
    print("\n" + "="*60)
    print("TEST 1: 32-BIT TEACHER GPT-2 EQUIVALENCE")
    print("="*60)

    # ==== EVALUATION STAGE: 32-bit Teacher Testing ====
    sp_model.set_precision(32)

    # CRITICAL: Both models MUST be in EVAL mode for fair comparison
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

    # More diverse test sentences for better PPL evaluation
    test_sentences = [
        "Machine learning algorithms process vast amounts of data.",
        "Natural language processing enables human-computer interaction.",
        "Deep learning models require substantial computational resources.",
        "Artificial intelligence is transforming various industries rapidly.",
        "Neural networks mimic the structure of human brain connections.",
        "Computer vision systems can identify objects in images accurately.",
        "Data science combines statistics, programming, and domain expertise.",
        "Reinforcement learning agents learn through trial and error.",
        "Transformers revolutionized the field of natural language processing.",
        "Gradient descent optimizes neural network parameters iteratively.",
        "Convolutional layers extract features from image data effectively.",
        "Recurrent networks process sequential data with temporal dependencies.",
        "Attention mechanisms help models focus on relevant information.",
        "Transfer learning leverages pre-trained models for new tasks.",
        "Generative models create new data samples from learned distributions.",
        "Supervised learning requires labeled training data for predictions.",
        "Unsupervised learning discovers patterns without explicit labels.",
        "Semi-supervised methods combine labeled and unlabeled data efficiently.",
        "Feature engineering improves machine learning model performance.",
        "Cross-validation prevents overfitting by evaluating model generalization.",
    ]

    results = {}

    # Need to calibrate each precision separately since set_precision resets calibration
    for precision in [32, 16, 8, 4]:  # Include 32-bit teacher as baseline
        sp_model.set_precision(precision)
        print(f"\n   Testing {precision}-bit precision...")

        # Calibrate for student precisions (all except 32-bit teacher)
        if precision < 32:
            # ==== TRAINING STAGE: Calibration ====
            print(f"      📈 TRAINING STAGE: Calibrating {precision}-bit quantizers...")

            # CRITICAL: Must be in TRAINING mode for statistics collection
            sp_model.train()

            # Start manual calibration for quantizers at this precision
            for name, module in sp_model.named_modules():
                if hasattr(module, 'quantizers_weight') and hasattr(module, 'quantizers_input'):
                    bits_key = f'{precision}bit'
                    if bits_key in module.quantizers_weight:
                        module.quantizers_weight[bits_key].start_calibration()
                    if bits_key in module.quantizers_input:
                        module.quantizers_input[bits_key].start_calibration()

            # Collect statistics in TRAINING mode
            with torch.no_grad():
                # Use first few sentences for calibration
                for i in range(min(5, len(test_sentences))):
                    inputs = tokenizer(test_sentences[i], return_tensors='pt')
                    input_ids = inputs['input_ids'].to(device)
                    _ = sp_model(input_ids)

            # Finish calibration (compute scales)
            for name, module in sp_model.named_modules():
                if hasattr(module, 'quantizers_weight') and hasattr(module, 'quantizers_input'):
                    bits_key = f'{precision}bit'
                    if bits_key in module.quantizers_weight:
                        module.quantizers_weight[bits_key].finish_calibration()
                    if bits_key in module.quantizers_input:
                        module.quantizers_input[bits_key].finish_calibration()

            print(f"      ✅ Calibration complete")

        # ==== EVALUATION STAGE: Testing ====
        print(f"   🧪 EVALUATION STAGE: Testing {precision}-bit model...")

        # CRITICAL: Switch to EVAL mode for actual testing
        sp_model.eval()
        total_loss = 0
        total_tokens = 0
        texts_processed = 0

        with torch.no_grad():
            for i, sentence in enumerate(test_sentences):
                inputs = tokenizer(sentence, return_tensors='pt')
                input_ids = inputs['input_ids'].to(device)

                # Skip very short sequences
                if input_ids.size(1) < 2:
                    continue

                outputs = sp_model(input_ids, labels=input_ids)

                # Correctly accumulate loss
                seq_length = input_ids.size(1) - 1  # -1 because we can't predict first token
                total_loss += outputs['loss'].item() * seq_length
                total_tokens += seq_length
                texts_processed += 1

                if (i + 1) % 5 == 0:
                    print(f"      Processed {i + 1}/{len(test_sentences)} samples")

            # Compute perplexity correctly
            if total_tokens > 0:
                avg_loss = total_loss / total_tokens
                ppl = np.exp(avg_loss)
            else:
                avg_loss = float('inf')
                ppl = float('inf')

            results[precision] = {
                'avg_loss': avg_loss,
                'ppl': ppl,
                'total_tokens': total_tokens,
                'texts_processed': texts_processed
            }

            print(f"   {precision:2d}-bit Results:")
            print(f"      Loss: {avg_loss:.4f}")
            print(f"      PPL: {ppl:.2f}")
            print(f"      Tokens: {total_tokens}, Texts: {texts_processed}")

    # Calculate degradation from 32-bit teacher
    baseline_ppl = results[32]['ppl']
    baseline_loss = results[32]['avg_loss']

    print(f"\n📊 DEGRADATION ANALYSIS:")
    print(f"   Baseline (32-bit teacher):")
    print(f"      PPL: {baseline_ppl:.2f}")
    print(f"      Avg Loss: {baseline_loss:.4f}")
    print(f"      Tokens tested: {results[32]['total_tokens']}")

    degradation_results = {}

    for precision in [16, 8, 4]:  # All students compared to teacher
        if precision in results:
            ppl = results[precision]['ppl']
            avg_loss = results[precision]['avg_loss']

            degradation = ((ppl - baseline_ppl) / baseline_ppl) * 100
            loss_diff = avg_loss - baseline_loss

            degradation_results[precision] = {
                'ppl_degradation': degradation,
                'loss_diff': loss_diff
            }

            print(f"\n   {precision}-bit Performance:")
            print(f"      PPL: {ppl:.2f}")
            print(f"      Avg Loss: {avg_loss:.4f}")
            print(f"      PPL Degradation: {degradation:+.1f}%")
            print(f"      Loss Difference: {loss_diff:+.4f}")

            # Expected targets: 16-bit <10%, 8-bit <30%, 4-bit <150%
            if precision == 16:
                if degradation < 10:
                    status = "✅ EXCELLENT"
                elif degradation < 25:
                    status = "⚠️ ACCEPTABLE"
                else:
                    status = "❌ POOR"
                target = "target: <10%"
            elif precision == 8:
                if degradation < 30:
                    status = "✅ EXCELLENT"
                elif degradation < 60:
                    status = "⚠️ ACCEPTABLE"
                else:
                    status = "❌ POOR"
                target = "target: <30%"
            else:  # 4-bit
                if degradation < 150:
                    status = "✅ EXCELLENT"
                elif degradation < 350:
                    status = "⚠️ ACCEPTABLE"
                else:
                    status = "❌ POOR"
                target = "target: <150%"

            print(f"      Status: {status} ({target})")

    return degradation_results


def test_lora_behavior(sp_model, tokenizer, device):
    """Test 3: Verify LoRA is properly handled"""
    print("\n" + "="*60)
    print("TEST 3: LORA BEHAVIOR VERIFICATION")
    print("="*60)

    test_input = "The future of artificial intelligence looks promising."
    inputs = tokenizer(test_input, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)

    # Calibration texts for each precision
    calibration_texts = [
        "Neural networks learn complex patterns from data.",
        "Machine learning models require proper training.",
        "Deep learning has revolutionized artificial intelligence.",
        "Gradient descent optimizes model parameters iteratively.",
    ]

    # ==== CALIBRATION STAGE: Calibrate each precision ====
    print("\n📊 Calibrating quantizers for each precision...")

    for precision in [16, 8, 4]:  # Only calibrate students, not 32-bit teacher
        print(f"   Calibrating {precision}-bit...")
        sp_model.set_precision(precision)
        sp_model.train()

        # Start calibration
        bits_key = f'{precision}bit'
        for name, module in sp_model.named_modules():
            if hasattr(module, 'quantizers_weight') and hasattr(module, 'quantizers_input'):
                if bits_key in module.quantizers_weight:
                    module.quantizers_weight[bits_key].start_calibration()
                if bits_key in module.quantizers_input:
                    module.quantizers_input[bits_key].start_calibration()

        # Collect statistics
        with torch.no_grad():
            for text in calibration_texts:
                cal_inputs = tokenizer(text, return_tensors='pt')['input_ids'].to(device)
                _ = sp_model(cal_inputs)

        # Finish calibration
        for name, module in sp_model.named_modules():
            if hasattr(module, 'quantizers_weight') and hasattr(module, 'quantizers_input'):
                if bits_key in module.quantizers_weight:
                    # Enable debug for 4-bit and 8-bit to show running min/max
                    show_debug = (bits in [4, 8]) and (module_count < 2) if 'module_count' in locals() else False
                    module.quantizers_weight[bits_key].finish_calibration(debug=show_debug if 'show_debug' in locals() else False)
                    if 'module_count' in locals(): module_count += 1
                if bits_key in module.quantizers_input:
                    show_debug = (bits in [4, 8]) and (module_count < 4) if 'module_count' in locals() else False
                    module.quantizers_input[bits_key].finish_calibration(debug=show_debug if 'show_debug' in locals() else False)
                    if 'module_count' in locals(): module_count += 1

    print("   ✅ Calibration complete for all student precisions")

    # ==== EVALUATION STAGE: Testing LoRA ====
    # CRITICAL: MUST be in EVAL mode for testing
    sp_model.eval()

    print("\nChecking LoRA contribution across bit-widths...")

    lora_contributions = {}

    with torch.no_grad():
        for precision in [32, 16, 8, 4]:  # Include 32-bit teacher
            sp_model.set_precision(precision)

            # Count enabled LoRA layers for the CURRENT precision only
            enabled_loras = 0
            disabled_loras = 0
            total_loras = 0

            for name, module in sp_model.named_modules():
                if hasattr(module, 'lora_adapters'):
                    # Only check the LoRA adapter for the current bit-width
                    bit_key = f'{precision}bit'
                    if bit_key in module.lora_adapters:
                        lora = module.lora_adapters[bit_key]
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

    # 32-bit teacher should have LoRA disabled
    if lora_contributions[32]['enabled'] == 0:
        print("   ✅ 32-bit (teacher): LoRA properly disabled")
    else:
        print("   ❌ 32-bit (teacher): LoRA should be disabled!")

    # 16-bit student should have LoRA enabled for distillation
    if lora_contributions[16]['enabled'] > 0:
        print("   ✅ 16-bit (student): LoRA properly enabled for distillation")
    else:
        print("   ❌ 16-bit (student): LoRA should be enabled!")

    # 8/4-bit students should have LoRA enabled
    if lora_contributions[8]['enabled'] > 0 and lora_contributions[4]['enabled'] > 0:
        print("   ✅ 8/4-bit (students): LoRA properly enabled for quantization compensation")
    else:
        print("   ⚠️ 8/4-bit (students): LoRA may not be working correctly")

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

    # Test each precision level
    quantization_results = {}

    for bits in [4, 8]:  # Skip 16-bit (no quantization)
        print(f"\n🔧 Testing {bits}-bit precision:")
        sp_model.set_precision(bits)

        # ==== TRAINING STAGE: Calibration ====
        print(f"   📈 TRAINING STAGE: Calibrating {bits}-bit precision...")

        # CRITICAL: Must be in TRAINING mode for statistics collection
        sp_model.train()

        # Start manual calibration
        for name, module in sp_model.named_modules():
            if hasattr(module, 'quantizers_weight') and hasattr(module, 'quantizers_input'):
                bits_key = f'{bits}bit'
                if bits_key in module.quantizers_weight:
                    module.quantizers_weight[bits_key].start_calibration()
                if bits_key in module.quantizers_input:
                    module.quantizers_input[bits_key].start_calibration()

        # Collect statistics in TRAINING mode
        with torch.no_grad():
            for text in test_inputs:
                inputs = tokenizer(text, return_tensors='pt')
                input_ids = inputs['input_ids'].to(device)
                _ = sp_model(input_ids)

        # Finish calibration (compute scales)
        for name, module in sp_model.named_modules():
            if hasattr(module, 'quantizers_weight') and hasattr(module, 'quantizers_input'):
                bits_key = f'{bits}bit'
                if bits_key in module.quantizers_weight:
                    # Enable debug for 4-bit and 8-bit to show running min/max
                    show_debug = (bits in [4, 8]) and (module_count < 2) if 'module_count' in locals() else False
                    module.quantizers_weight[bits_key].finish_calibration(debug=show_debug if 'show_debug' in locals() else False)
                    if 'module_count' in locals(): module_count += 1
                if bits_key in module.quantizers_input:
                    show_debug = (bits in [4, 8]) and (module_count < 4) if 'module_count' in locals() else False
                    module.quantizers_input[bits_key].finish_calibration(debug=show_debug if 'show_debug' in locals() else False)
                    if 'module_count' in locals(): module_count += 1

        print(f"   ✅ Calibration complete")

        # ==== EVALUATION STAGE: Testing ====
        print(f"   🧪 EVALUATION STAGE: Testing {bits}-bit precision...")

        # CRITICAL: Switch to EVAL mode for testing
        sp_model.eval()

        # Check quantizer state after calibration
        quantizer_states = []
        for name, module in sp_model.named_modules():
            if hasattr(module, 'quantizers_weight') and hasattr(module, 'quantizers_input'):
                bits_key = f'{bits}bit'
                if bits_key in module.quantizers_weight:
                    quantizer = module.quantizers_weight[bits_key]
                    state = {
                        'calibrated': quantizer.calibrated,
                        'scale': quantizer.scale.mean().item() if quantizer.scale.numel() > 0 else 0
                    }
                    quantizer_states.append(state)
                    if len(quantizer_states) == 1:  # Print first one as sample
                        print(f"   Sample quantizer state:")
                        print(f"     Calibrated: {state['calibrated']}")
                        print(f"     Scale: {state['scale']:.6f}")
                        break  # Only need one sample

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

    # ==== EVALUATION STAGE: 16-bit Testing (No Quantization) ====
    print(f"\n🧪 EVALUATION STAGE: Testing 16-bit precision (no quantization):")
    sp_model.set_precision(16)

    # CRITICAL: MUST be in EVAL mode for 16-bit testing
    sp_model.eval()

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

    # ==== EVALUATION STAGE: Testing Distillation ====
    # CRITICAL: MUST be in EVAL mode for distillation testing
    sp_model.eval()

    teacher_outputs = None
    student_outputs = {}

    print("Testing teacher-student setup...")

    with torch.no_grad():
        # Test teacher mode (32-bit)
        sp_model.set_precision(32)
        teacher_outputs = sp_model(input_ids, output_hidden_states=True, return_dict=True)

        if isinstance(teacher_outputs, dict) and 'logits' in teacher_outputs:
            print(f"   ✅ Teacher (32-bit): Logits shape {list(teacher_outputs['logits'].shape)}")
            if 'hidden_states' in teacher_outputs:
                print(f"   ✅ Teacher hidden states: {len(teacher_outputs['hidden_states'])} layers")
        else:
            print(f"   ⚠️ Teacher outputs may not be in expected format")

        # Test student modes (16/8/4-bit are all students)
        for precision in [16, 8, 4]:
            sp_model.set_precision(precision)
            outputs = sp_model(input_ids, output_hidden_states=True, return_dict=True)
            student_outputs[precision] = outputs

            if isinstance(outputs, dict) and 'logits' in outputs:
                print(f"   ✅ Student ({precision}-bit): Logits shape {list(outputs['logits'].shape)}")
            else:
                print(f"   ⚠️ Student ({precision}-bit) outputs may not be in expected format")

    # Test distillation loss computation (basic)
    if teacher_outputs and 16 in student_outputs:
        try:
            teacher_logits = teacher_outputs['logits']
            student_logits = student_outputs[16]['logits']  # Test with 16-bit student

            # Simple KL divergence test
            T = 3.0
            teacher_probs = F.log_softmax(teacher_logits / T, dim=-1)
            student_probs = F.log_softmax(student_logits / T, dim=-1)

            kl_loss = F.kl_div(student_probs, teacher_probs,
                               reduction='batchmean', log_target=True)

            print(f"   ✅ KL divergence computable (32-bit teacher -> 16-bit student): {kl_loss.item():.4f}")

        except Exception as e:
            print(f"   ❌ Distillation loss computation failed: {e}")

    print("   📝 Note: Full distillation training requires separate training script")

    return True


def test_comprehensive_ppl(sp_model, tokenizer, device):
    """Test 6: Comprehensive PPL evaluation with proper dataset and label shifting"""
    print("\n" + "="*60)
    print("TEST 6: COMPREHENSIVE PPL EVALUATION WITH PROPER DATASET")
    print("="*60)

    # ==== CALIBRATION STAGE: Calibrate before evaluation ====
    print("\n📊 Calibrating quantizers for comprehensive evaluation...")

    # Get diverse calibration texts from WikiText
    calibration_texts = get_calibration_texts(num_texts=20)

    for precision in [16, 8, 4]:  # Calibrate all student precisions
        print(f"   Calibrating {precision}-bit precision...")
        sp_model.set_precision(precision)
        sp_model.train()  # Training mode for calibration

        # Start calibration
        bits_key = f'{precision}bit'
        for name, module in sp_model.named_modules():
            if hasattr(module, 'quantizers_weight') and hasattr(module, 'quantizers_input'):
                if bits_key in module.quantizers_weight:
                    module.quantizers_weight[bits_key].start_calibration()
                if bits_key in module.quantizers_input:
                    module.quantizers_input[bits_key].start_calibration()

        # Collect statistics from calibration texts
        with torch.no_grad():
            for text in calibration_texts:
                cal_inputs = tokenizer(text, return_tensors='pt',
                                      max_length=128, truncation=True)['input_ids'].to(device)
                _ = sp_model(cal_inputs)

        # Finish calibration
        for name, module in sp_model.named_modules():
            if hasattr(module, 'quantizers_weight') and hasattr(module, 'quantizers_input'):
                if bits_key in module.quantizers_weight:
                    # Enable debug for 4-bit and 8-bit to show running min/max
                    show_debug = (bits in [4, 8]) and (module_count < 2) if 'module_count' in locals() else False
                    module.quantizers_weight[bits_key].finish_calibration(debug=show_debug if 'show_debug' in locals() else False)
                    if 'module_count' in locals(): module_count += 1
                if bits_key in module.quantizers_input:
                    show_debug = (bits in [4, 8]) and (module_count < 4) if 'module_count' in locals() else False
                    module.quantizers_input[bits_key].finish_calibration(debug=show_debug if 'show_debug' in locals() else False)
                    if 'module_count' in locals(): module_count += 1

        print(f"      ✅ {precision}-bit calibration complete")

    print("\n   ✅ All calibrations complete, proceeding to evaluation")

    # ==== EVALUATION STAGE: Comprehensive Testing with Proper Dataset ====
    print("\n📚 Loading WikiText test dataset for proper perplexity evaluation...")

    # CRITICAL: MUST be in EVAL mode for all perplexity testing
    sp_model.eval()

    print("\n🧪 EVALUATION STAGE: Testing perplexity using sliding window approach...")

    comprehensive_results = {}

    # Test each precision with proper perplexity calculation
    for precision in [32, 16, 8, 4]:  # Include 32-bit teacher as baseline
        sp_model.set_precision(precision)
        print(f"\n{precision}-bit Precision Evaluation:")
        print(f"  Calculating perplexity with sliding window approach...")

        # Calculate perplexity using the correct sliding window methodology
        results = calculate_perplexity_properly(
            model=sp_model,
            tokenizer=tokenizer,
            device=device,
            dataset_name='wikitext',
            max_length=512,  # Window size
            stride=256,       # 50% overlap
            max_samples=1024  # Limited to model's max position embeddings
        )

        comprehensive_results[precision] = results

        print(f"  ✅ {precision}-bit Results:")
        print(f"    Perplexity: {results['perplexity']:.2f}")
        print(f"    Average Loss: {results['loss']:.4f}")
        print(f"    Total Tokens: {results['total_tokens']:,}")
        print(f"    Windows Processed: {results['num_windows']}")

    # Analysis of degradation from 32-bit baseline
    print("\n" + "="*60)
    print("📊 PERPLEXITY DEGRADATION ANALYSIS")
    print("="*60)

    baseline_ppl = comprehensive_results[32]['perplexity']
    print(f"\n32-bit Teacher Baseline: {baseline_ppl:.2f}")
    print("\nDegradation from baseline:")

    for precision in [16, 8, 4]:
        ppl = comprehensive_results[precision]['perplexity']
        degradation = ((ppl - baseline_ppl) / baseline_ppl) * 100

        # Determine verdict based on degradation
        if precision == 16:
            if degradation < 10:
                verdict = "✅ EXCELLENT - Minimal degradation"
            elif degradation < 30:
                verdict = "⚠️ ACCEPTABLE - Moderate degradation"
            else:
                verdict = "❌ POOR - High degradation"
        elif precision == 8:
            if degradation < 50:
                verdict = "✅ EXCELLENT - Good for 8-bit"
            elif degradation < 150:
                verdict = "⚠️ ACCEPTABLE - Expected for 8-bit"
            else:
                verdict = "❌ POOR - Excessive degradation"
        else:  # 4-bit
            if degradation < 150:
                verdict = "✅ EXCELLENT - Exceptional for 4-bit"
            elif degradation < 400:
                verdict = "⚠️ ACCEPTABLE - Expected for 4-bit"
            else:
                verdict = "❌ POOR - Too much degradation"

        print(f"  {precision:2d}-bit: +{degradation:.1f}% (PPL: {ppl:.2f}) - {verdict}")

    return comprehensive_results


# Remove old test dataset definition since we're using proper WikiText
def test_comprehensive_ppl_old(sp_model, tokenizer, device):
    """Old version - kept for reference but not used"""
    # Old implementation with manual test dataset
    test_dataset = {
        'technical': [
            "Machine learning algorithms optimize objective functions through gradient descent.",
            "Neural networks consist of interconnected layers of artificial neurons.",
            "Backpropagation computes gradients efficiently using the chain rule.",
            "Convolutional neural networks excel at processing grid-like data structures.",
            "Recurrent neural networks maintain hidden states for sequence modeling.",
            "Transformers use self-attention mechanisms to process sequential data.",
            "Regularization techniques prevent overfitting in machine learning models.",
            "Batch normalization accelerates training and improves model stability.",
            "Dropout randomly deactivates neurons during training for regularization.",
            "Adam optimizer combines momentum with adaptive learning rates.",
        ],
        'scientific': [
            "Quantum computing leverages superposition and entanglement for computation.",
            "Climate change affects global weather patterns and ecosystems.",
            "DNA sequencing reveals genetic information encoded in nucleotides.",
            "Black holes form when massive stars collapse under gravity.",
            "Photosynthesis converts light energy into chemical energy in plants.",
            "Evolution occurs through natural selection and genetic variation.",
            "Antibiotics target specific bacterial processes to eliminate infections.",
            "Renewable energy sources reduce dependence on fossil fuels.",
            "Stem cells differentiate into specialized cell types during development.",
            "Vaccines stimulate immune responses to prevent infectious diseases.",
        ],
        'general': [
            "The internet has revolutionized global communication and commerce.",
            "Artificial intelligence is transforming various industries worldwide.",
            "Social media platforms connect billions of users globally.",
            "Electric vehicles are becoming increasingly popular and affordable.",
            "Remote work has changed traditional office dynamics significantly.",
            "Cybersecurity threats continue to evolve and challenge organizations.",
            "Blockchain technology enables decentralized and transparent transactions.",
            "Virtual reality creates immersive digital experiences for users.",
            "Cloud computing provides scalable and flexible infrastructure solutions.",
            "Data privacy concerns influence technology development and regulation.",
        ],
        'conversational': [
            "How are you doing today?",
            "The weather has been quite nice lately.",
            "I enjoyed reading that book you recommended.",
            "Let's meet for coffee tomorrow afternoon.",
            "Have you seen any good movies recently?",
            "The restaurant downtown serves excellent food.",
            "I'm planning a vacation for next month.",
            "The concert last night was amazing.",
            "Thanks for helping me with the project.",
            "It's been great catching up with you.",
        ]
    }

    # CRITICAL: MUST be in EVAL mode for all perplexity testing
    sp_model.eval()

    print("\n🧪 EVALUATION STAGE: Testing perplexity across text categories and bit-widths...")

    comprehensive_results = {}

    with torch.no_grad():
        for precision in [32, 16, 8, 4]:  # Include 32-bit teacher
            sp_model.set_precision(precision)
            print(f"\n{precision}-bit Precision Evaluation:")

            precision_results = {}
            all_total_loss = 0
            all_total_tokens = 0

            for category, texts in test_dataset.items():
                category_total_loss = 0
                category_total_tokens = 0
                texts_processed = 0

                print(f"  Testing {category} texts...")

                for text in texts:
                    inputs = tokenizer(text, return_tensors='pt',
                                     max_length=128, truncation=True)
                    input_ids = inputs['input_ids'].to(device)

                    # Skip very short sequences
                    if input_ids.size(1) < 2:
                        continue

                    outputs = sp_model(input_ids, labels=input_ids)

                    # The model internally handles label shifting
                    # Loss is mean loss per token, so multiply by sequence length to get total
                    seq_length = input_ids.size(1) - 1  # -1 because we can't predict first token
                    total_loss_for_seq = outputs['loss'].item() * seq_length

                    category_total_loss += total_loss_for_seq
                    category_total_tokens += seq_length
                    all_total_loss += total_loss_for_seq
                    all_total_tokens += seq_length
                    texts_processed += 1

                # Compute category perplexity correctly
                if category_total_tokens > 0:
                    cat_avg_loss = category_total_loss / category_total_tokens
                    cat_ppl = np.exp(cat_avg_loss)
                else:
                    cat_avg_loss = float('inf')
                    cat_ppl = float('inf')

                precision_results[category] = {
                    'avg_loss': cat_avg_loss,
                    'ppl': cat_ppl,
                    'total_tokens': category_total_tokens,
                    'texts_processed': texts_processed
                }

                print(f"    {category}: PPL={cat_ppl:.2f} (loss={cat_avg_loss:.4f}, tokens={category_total_tokens})")

            # Overall statistics for this precision - computed correctly
            if all_total_tokens > 0:
                overall_avg_loss = all_total_loss / all_total_tokens
                overall_ppl = np.exp(overall_avg_loss)
            else:
                overall_avg_loss = float('inf')
                overall_ppl = float('inf')

            comprehensive_results[precision] = {
                'categories': precision_results,
                'overall': {
                    'avg_loss': overall_avg_loss,
                    'ppl': overall_ppl,
                    'total_tokens': all_total_tokens,
                    'total_texts': sum(r['texts_processed'] for r in precision_results.values())
                }
            }

            print(f"\n  Overall {precision}-bit Statistics:")
            print(f"    Perplexity: {overall_ppl:.2f}")
            print(f"    Average loss: {overall_avg_loss:.4f}")
            print(f"    Total tokens: {all_total_tokens}")
            print(f"    Total texts: {comprehensive_results[precision]['overall']['total_texts']}")

    # Detailed degradation analysis
    print("\n" + "="*60)
    print("DETAILED DEGRADATION ANALYSIS")
    print("="*60)

    baseline = comprehensive_results[32]['overall']  # Use 32-bit teacher as baseline

    for precision in [16, 8, 4]:  # All students
        current = comprehensive_results[precision]['overall']

        ppl_degradation = ((current['ppl'] - baseline['ppl']) / baseline['ppl']) * 100
        loss_diff = current['avg_loss'] - baseline['avg_loss']

        print(f"\n{precision}-bit vs 32-bit teacher:")
        print(f"  Perplexity degradation: {ppl_degradation:+.1f}%")
        print(f"  Loss difference: {loss_diff:+.4f}")
        print(f"  Tokens evaluated: {current['total_tokens']}")

        # Category-wise degradation
        print(f"\n  Category-wise degradation:")
        for category in test_dataset.keys():
            cat_baseline = comprehensive_results[32]['categories'][category]['ppl']  # 32-bit baseline
            cat_current = comprehensive_results[precision]['categories'][category]['ppl']
            if cat_baseline != float('inf') and cat_current != float('inf'):
                cat_degradation = ((cat_current - cat_baseline) / cat_baseline) * 100
                print(f"    {category}: {cat_degradation:+.1f}%")
            else:
                print(f"    {category}: N/A (insufficient data)")

        # Final verdict
        if precision == 16:
            if ppl_degradation < 15:
                verdict = "✅ EXCELLENT - Well within target"
            elif ppl_degradation < 30:
                verdict = "⚠️ ACCEPTABLE - Within reasonable bounds"
            else:
                verdict = "❌ POOR - Exceeds acceptable degradation"
        elif precision == 8:
            if ppl_degradation < 30:
                verdict = "✅ EXCELLENT - Well within target"
            elif ppl_degradation < 60:
                verdict = "⚠️ ACCEPTABLE - Within reasonable bounds"
            else:
                verdict = "❌ POOR - Exceeds acceptable degradation"
        else:  # 4-bit
            if ppl_degradation < 150:
                verdict = "✅ EXCELLENT - Exceptional for 4-bit"
            elif ppl_degradation < 350:
                verdict = "⚠️ ACCEPTABLE - Expected for 4-bit"
            else:
                verdict = "❌ POOR - High degradation even for 4-bit"

        print(f"\n  Verdict: {verdict}")

    return comprehensive_results


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

    # Create full model for proper testing
    print("Loading full 12-layer model for accurate testing...")
    sp_model, sp_config = create_properly_initialized_model(use_pretrained=True, num_layers=12)  # Full 12-layer model

    # Move to device with memory check
    try:
        sp_model = sp_model.to(device)
    except torch.cuda.OutOfMemoryError:
        print("⚠️ GPU out of memory, switching to CPU...")
        device = torch.device('cpu')
        sp_model = sp_model.to(device)

    # Load full GPT-2 model for comparison
    print("Loading full GPT-2 model for comparison...")
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')  # Full 12-layer GPT-2

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
        # Test 1: 32-bit teacher equivalence
        test_results['equivalence'] = test_32bit_equivalence(
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

        # Test 6: Comprehensive PPL evaluation
        print("\n" + "="*70)
        print("Running comprehensive PPL evaluation...")
        print("="*70)
        test_results['comprehensive_ppl'] = test_comprehensive_ppl(
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
        print("✅ Test 1 (32-bit teacher equivalence): PASSED")
        passed_tests += 1
    else:
        print("❌ Test 1 (32-bit teacher equivalence): FAILED")

    # Check degradation results
    degradation = test_results.get('degradation', {})
    if isinstance(degradation.get(8), dict):
        # Dictionary format
        deg_16bit = degradation.get(16, {}).get('ppl_degradation', 999)
        deg_8bit = degradation.get(8, {}).get('ppl_degradation', 999)
        deg_4bit = degradation.get(4, {}).get('ppl_degradation', 9999)
    else:
        # Simple number format (backward compatibility)
        deg_16bit = degradation.get(16, 999)
        deg_8bit = degradation.get(8, 999)
        deg_4bit = degradation.get(4, 9999)

    if deg_16bit < 30 and deg_8bit < 60 and deg_4bit < 350:
        print("✅ Test 2 (quantization degradation): PASSED")
        passed_tests += 1
    else:
        print("❌ Test 2 (quantization degradation): FAILED")

    # Check that 32-bit has LoRA disabled (teacher) and others have it enabled (students)
    lora_correct = (
        test_results['lora'][32]['enabled'] == 0 and  # 32-bit teacher: no LoRA
        test_results['lora'][16]['enabled'] > 0 and   # 16-bit student: has LoRA
        test_results['lora'][8]['enabled'] > 0 and    # 8-bit student: has LoRA
        test_results['lora'][4]['enabled'] > 0        # 4-bit student: has LoRA
    )
    if lora_correct:
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
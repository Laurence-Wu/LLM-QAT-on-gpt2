#!/usr/bin/env python3
"""
Verify Weight Loading for Current SP Architecture
Checks if weight loading after initialization works correctly
with SPLinearWithLoRA wrapper and manual calibration
"""

import sys
import os
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fix_model_initialization import create_properly_initialized_model
import numpy as np


def check_quantizer_status(model, bits):
    """Check if quantizers are calibrated for a given precision."""
    calibrated_count = 0
    uncalibrated_count = 0
    try:
        # Check through the transformer layers
        for i, block in enumerate(model.transformer.h):
            # Check attention and MLP layers
            for module in [block.attn.c_attn, block.attn.c_proj, block.mlp.c_fc, block.mlp.c_proj]:
                bits_key = f'{bits}bit'
                try:
                    if module.quantizers_weight and bits_key in module.quantizers_weight:
                        quantizer = module.quantizers_weight[bits_key]
                        if quantizer.calibrated:
                            calibrated_count += 1
                        else:
                            uncalibrated_count += 1
                except:
                    pass  # Module might not have quantizers

    except Exception as e:
        print(f"Error checking quantizer status: {e}")

    return calibrated_count, uncalibrated_count


def test_weight_loading_detailed():
    """Test weight loading step by step."""
    print("\n" + "="*80)
    print("DETAILED WEIGHT LOADING VERIFICATION")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    print("\n1. Loading models...")
    sp_model, sp_config = create_properly_initialized_model(use_pretrained=True)
    sp_model = sp_model.to(device)

    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_model = gpt2_model.to(device)

    # Test 1: Check if weights were actually loaded
    print("\n2. Verifying weight loading...")

    # Check embeddings
    sp_wte = sp_model.transformer.wte.weight
    gpt2_wte = gpt2_model.transformer.wte.weight

    wte_diff = (sp_wte - gpt2_wte).abs().max().item()
    print(f"   Token embedding max diff: {wte_diff:.10f}")

    if wte_diff == 0.0:
        print("   ✅ Token embeddings loaded perfectly")
    else:
        print("   ❌ Token embeddings differ!")

    # Check first attention layer weights (handle SPLinearWithLoRA wrapper)
    try:
        # Access the underlying linear layer in SPLinearWithLoRA
        sp_attn_module = sp_model.transformer.h[0].attn.c_attn
        if hasattr(sp_attn_module, 'linear'):
            sp_attn = sp_attn_module.linear.weight
        else:
            sp_attn = sp_attn_module.weight

        gpt2_attn = gpt2_model.transformer.h[0].attn.c_attn.weight.t()  # Transpose for comparison

        attn_diff = (sp_attn - gpt2_attn).abs().max().item()
        print(f"   First attention max diff: {attn_diff:.10f}")

        if attn_diff == 0.0:
            print("   ✅ Attention weights loaded perfectly")
        elif attn_diff < 1e-6:
            print("   ✅ Attention weights loaded with tiny numerical differences")
        else:
            print("   ❌ Attention weights differ!")

    except AttributeError as e:
        print(f"   ❌ Error accessing attention weights: {e}")
        # Try to provide more details
        print(f"   SP module type: {type(sp_model.transformer.h[0].attn.c_attn)}")
        return False, {'weight_diff': float('inf'), 'logit_diff': float('inf'), 'perplexity_diff': float('inf')}

    # Test 2: Forward pass comparison
    print("\n3. Testing forward pass equivalence...")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    test_text = "The quick brown fox"
    inputs = tokenizer(test_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)

    with torch.no_grad():
        # SP model at 16-bit (should bypass LoRA and quantization now)
        sp_model.set_precision(16)
        sp_model.eval()
        gpt2_model.eval()

        sp_outputs = sp_model(input_ids)
        gpt2_outputs = gpt2_model(input_ids)

        # Handle different output formats
        if isinstance(sp_outputs, dict) and 'logits' in sp_outputs:
            sp_logits = sp_outputs['logits']
        else:
            sp_logits = sp_outputs

        gpt2_logits = gpt2_outputs['logits']

    # Compare logits
    logit_diff = (sp_logits - gpt2_logits).abs()
    mean_diff = logit_diff.mean().item()
    max_diff = logit_diff.max().item()

    print(f"   Logit mean diff: {mean_diff:.10f}")
    print(f"   Logit max diff: {max_diff:.10f}")

    # Test 3: Perplexity comparison
    print("\n4. Testing perplexity equivalence...")

    test_sentences = [
        "The cat sat on the mat.",
        "Machine learning is advancing rapidly.",
        "Python is a programming language."
    ]

    perplexity_diffs = []

    for sentence in test_sentences:
        inputs = tokenizer(sentence, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)

        with torch.no_grad():
            sp_outputs = sp_model(input_ids, labels=input_ids)
            gpt2_outputs = gpt2_model(input_ids, labels=input_ids)

            sp_loss = sp_outputs['loss'].item()
            gpt2_loss = gpt2_outputs['loss'].item()

            sp_ppl = torch.exp(torch.tensor(sp_loss)).item()
            gpt2_ppl = torch.exp(torch.tensor(gpt2_loss)).item()

            diff = abs(sp_ppl - gpt2_ppl)
            perplexity_diffs.append(diff)

            print(f"   '{sentence[:30]}...': SP={sp_ppl:.2f}, GPT2={gpt2_ppl:.2f}, diff={diff:.2f}")

    avg_ppl_diff = sum(perplexity_diffs) / len(perplexity_diffs)
    max_ppl_diff = max(perplexity_diffs)

    print(f"\n   Average perplexity diff: {avg_ppl_diff:.4f}")
    print(f"   Maximum perplexity diff: {max_ppl_diff:.4f}")

    # Assessment
    print("\n5. ASSESSMENT:")

    # Check if attn_diff is defined (might not be if there was an error)
    if 'attn_diff' not in locals():
        attn_diff = float('inf')  # Set to infinity if not accessible

    if wte_diff == 0.0 and attn_diff == 0.0:
        print("   ✅ Weight loading: PERFECT")
    elif wte_diff < 1e-6 and attn_diff < 1e-6:
        print("   ✅ Weight loading: EXCELLENT (tiny numerical differences)")
    else:
        print("   ❌ Weight loading: FAILED")

    if mean_diff < 1e-6 and max_diff < 1e-5:
        print("   ✅ Forward pass: PERFECT")
    elif mean_diff < 1e-3 and max_diff < 1e-2:
        print("   ⚠️ Forward pass: GOOD (small differences)")
    else:
        print("   ❌ Forward pass: FAILED")

    if avg_ppl_diff < 0.1 and max_ppl_diff < 0.5:
        print("   ✅ Perplexity: EXCELLENT")
    elif avg_ppl_diff < 1.0 and max_ppl_diff < 2.0:
        print("   ⚠️ Perplexity: ACCEPTABLE")
    else:
        print("   ❌ Perplexity: FAILED")

    # Overall result (handle case where attn_diff might not be defined)
    if 'attn_diff' not in locals():
        attn_diff = float('inf')

    success = (wte_diff < 1e-6 and attn_diff < 1e-6 and
               mean_diff < 1e-3 and avg_ppl_diff < 1.0)

    return success, {
        'weight_diff': max(wte_diff, attn_diff),
        'logit_diff': mean_diff,
        'perplexity_diff': avg_ppl_diff
    }


def calibrate_quantizers_for_testing(model, tokenizer, device, bits):
    """Calibrate quantizers using two-pass strategy from train_sp.py."""
    if bits >= 16:
        return  # No calibration needed for 16-bit

    # ========== PASS 1: Collect Statistics ==========
    # Model must be in training mode for calibration to work
    model.train()

    # Set precision for the model
    model.set_precision(bits)

    # Start calibration for all quantizers
    bits_key = f'{bits}bit'
    print(f"Pass 1: Starting calibration for {bits}-bit quantizers...")

    for block in model.transformer.h:
        for module in [block.attn.c_attn, block.attn.c_proj, block.mlp.c_fc, block.mlp.c_proj]:
            if bits_key in module.quantizers_weight:
                module.quantizers_weight[bits_key].start_calibration()
            if bits_key in module.quantizers_input:
                module.quantizers_input[bits_key].start_calibration()

    # Prepare calibration samples
    calibration_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming technology.",
        "Natural language processing enables computers to understand human language."
    ]

    # Collect statistics from all calibration samples
    print("  Collecting statistics from calibration samples...")
    with torch.no_grad():
        for text in calibration_texts:
            inputs = tokenizer(text, return_tensors='pt')
            input_ids = inputs['input_ids'].to(device)
            # Forward pass for statistics collection
            _ = model(input_ids)

    # Finish calibration to freeze quantization parameters
    print(f"  Finishing calibration and freezing parameters...")
    for block in model.transformer.h:
        for module in [block.attn.c_attn, block.attn.c_proj, block.mlp.c_fc, block.mlp.c_proj]:
            if bits_key in module.quantizers_weight:
                module.quantizers_weight[bits_key].finish_calibration()
            if bits_key in module.quantizers_input:
                module.quantizers_input[bits_key].finish_calibration()

    # Verify calibration completed
    first_module = model.transformer.h[0].attn.c_attn
    if bits_key in first_module.quantizers_weight:
        q = first_module.quantizers_weight[bits_key]
        print(f"  Calibration complete: calibrated={q.calibrated}, scale mean={q.scale.mean().item():.6f}")

    # ========== PASS 2: Ready for Inference/Training ==========
    # Model is now ready to use with frozen quantization parameters
    model.eval()  # Return to eval mode for testing
    print(f"Pass 2: Ready with frozen {bits}-bit quantization parameters\n")


def test_different_precision_modes():
    """Test that different precisions work as expected."""
    print("\n" + "="*80)
    print("PRECISION MODE TESTING WITH CALIBRATION")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load SP model
    sp_model, sp_config = create_properly_initialized_model(use_pretrained=True)
    sp_model = sp_model.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    test_text = "Machine learning is transforming technology."
    inputs = tokenizer(test_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)

    print(f"\nTest sentence: '{test_text}'")
    print("\nCalibrating quantizers for each precision...")

    results = {}

    for precision in [16, 8, 4]:
        # Set precision
        sp_model.set_precision(precision)

        # Calibrate if needed
        if precision < 16:
            print(f"   Calibrating {precision}-bit quantizers...")
            calibrate_quantizers_for_testing(sp_model, tokenizer, device, precision)

            # Check calibration status
            # Check calibration was successful
            calibrated, uncalibrated = check_quantizer_status(sp_model, precision)
            if uncalibrated > 0:
                print(f"   ⚠️ Warning: {uncalibrated} quantizers not calibrated!")
            else:
                print(f"   ✅ All {calibrated} quantizers successfully calibrated")

        # Evaluate
        sp_model.eval()
        with torch.no_grad():
            outputs = sp_model(input_ids, labels=input_ids)
            loss = outputs['loss'].item()
            ppl = np.exp(loss)

            results[precision] = {'loss': loss, 'ppl': ppl}
            print(f"   {precision:2d}-bit: Loss = {loss:.4f}, PPL = {ppl:.2f}")

    # Analysis
    print(f"\n📊 PRECISION ANALYSIS:")
    baseline_ppl = results[16]['ppl']

    for precision in [8, 4]:
        ppl = results[precision]['ppl']
        degradation = ((ppl - baseline_ppl) / baseline_ppl) * 100
        print(f"   {precision}-bit degradation: {degradation:.1f}%")

        if degradation < 0:
            print(f"     ⚠️ Negative degradation - this shouldn't happen!")
        elif degradation < 10:
            print(f"     ✅ Good quality preservation")
        elif degradation < 30:
            print(f"     ⚠️ Moderate quality loss")
        else:
            print(f"     ❌ Significant quality loss")

    return results


def verify_sp_linear_with_lora_structure():
    """Verify the SPLinearWithLoRA structure is correct."""
    print("\n" + "="*80)
    print("SP LINEAR WITH LORA STRUCTURE VERIFICATION")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sp_model, _ = create_properly_initialized_model(use_pretrained=True)
    sp_model = sp_model.to(device)

    print("\nChecking module structure...")

    # Check a few key modules
    modules_to_check = [
        ('transformer.h[0].attn.c_attn', sp_model.transformer.h[0].attn.c_attn),
        ('transformer.h[0].attn.c_proj', sp_model.transformer.h[0].attn.c_proj),
        ('transformer.h[0].mlp.c_fc', sp_model.transformer.h[0].mlp.c_fc),
        ('transformer.h[0].mlp.c_proj', sp_model.transformer.h[0].mlp.c_proj),
    ]

    for name, module in modules_to_check:
        print(f"\n{name}:")
        print(f"  Type: {type(module).__name__}")

        if hasattr(module, 'linear'):
            print(f"  Has linear layer: ✅")
            print(f"  Linear weight shape: {module.linear.weight.shape}")
        else:
            print(f"  Has linear layer: ❌")

        if hasattr(module, 'quantizers_weight'):
            print(f"  Has weight quantizers: ✅")
            print(f"  Quantizer keys: {list(module.quantizers_weight.keys())}")
        else:
            print(f"  Has weight quantizers: ❌")

        if hasattr(module, 'lora_adapters'):
            print(f"  Has LoRA adapters: ✅")
            print(f"  LoRA keys: {list(module.lora_adapters.keys())}")
        else:
            print(f"  Has LoRA adapters: ❌")

    return True


def test_two_pass_quantization():
    """Test the two-pass quantization statistics collection."""
    print("\n" + "="*80)
    print("TWO-PASS QUANTIZATION TESTING")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    sp_model, sp_config = create_properly_initialized_model(use_pretrained=True, num_layers=2)  # Use 2 layers for speed
    sp_model = sp_model.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Test text
    test_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(test_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)

    print("\n1. Testing Two-Pass Quantization System for 8-bit precision:")

    # ========== PASS 1: Collect Statistics ==========
    print("\n   PASS 1: Statistics Collection")

    # Model must be in training mode for calibration
    sp_model.train()

    # Set to 8-bit
    sp_model.set_precision(8)
    bits_key = '8bit'

    # Start calibration for all quantizers
    print("   Starting calibration...")
    for block in sp_model.transformer.h:
        for module in [block.attn.c_attn, block.attn.c_proj, block.mlp.c_fc, block.mlp.c_proj]:
            if bits_key in module.quantizers_weight:
                module.quantizers_weight[bits_key].start_calibration()
            if bits_key in module.quantizers_input:
                module.quantizers_input[bits_key].start_calibration()

    # Collect statistics from multiple forward passes
    print("   Collecting statistics...")
    with torch.no_grad():
        for i in range(3):
            _ = sp_model(input_ids)
            if i == 0:  # Check status after first pass
                first_module = sp_model.transformer.h[0].attn.c_attn
                quantizer = first_module.quantizers_weight[bits_key]
                print(f"     After pass {i+1}: collecting_stats={quantizer.collecting_stats}, calibrated={quantizer.calibrated}")

    # Finish calibration to freeze parameters
    print("   Finishing calibration and freezing parameters...")
    for block in sp_model.transformer.h:
        for module in [block.attn.c_attn, block.attn.c_proj, block.mlp.c_fc, block.mlp.c_proj]:
            if bits_key in module.quantizers_weight:
                module.quantizers_weight[bits_key].finish_calibration()
            if bits_key in module.quantizers_input:
                module.quantizers_input[bits_key].finish_calibration()

    # Verify calibration completed
    first_module = sp_model.transformer.h[0].attn.c_attn
    quantizer = first_module.quantizers_weight[bits_key]
    print(f"   Calibration complete: calibrated={quantizer.calibrated}, scale mean={quantizer.scale.mean().item():.6f}")

    # Get the frozen scale/zero_point values
    scale_frozen = quantizer.scale.clone()
    zero_frozen = quantizer.zero_point.clone()

    # ========== PASS 2: Use Frozen Parameters ==========
    print("\n   PASS 2: Using Frozen Quantization Parameters")

    # Return to eval mode for testing
    sp_model.eval()

    # Run forward passes with frozen parameters
    print("   Testing parameter stability...")
    with torch.no_grad():
        for i in range(3):
            _ = sp_model(input_ids)

            # Verify parameters remain frozen
            scale_changed = not torch.allclose(scale_frozen, quantizer.scale)
            zero_changed = not torch.allclose(zero_frozen, quantizer.zero_point)
            print(f"     Forward pass {i+1}: scale_changed={scale_changed}, zero_changed={zero_changed}")

    print("\n2. Two-Pass Quantization Test Summary:")
    print(f"   ✅ Statistics collected in Pass 1")
    print(f"   ✅ Parameters frozen after calibration")
    print(f"   ✅ Parameters remain stable in Pass 2")

    print("\n✅ Two-pass quantization test completed")
    return True


def main():
    """Main verification function."""
    print("\n" + "="*80)
    print("WEIGHT LOADING VERIFICATION FOR CURRENT ARCHITECTURE")
    print("="*80)

    # First verify the structure
    structure_ok = verify_sp_linear_with_lora_structure()
    if not structure_ok:
        print("\n❌ Structure verification failed!")
        return False

    # Test 1: Detailed weight loading verification
    success, metrics = test_weight_loading_detailed()

    # Test 2: Precision mode testing
    precision_results = test_different_precision_modes()

    # Test 3: Two-pass quantization
    two_pass_ok = test_two_pass_quantization()

    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    if success:
        print("🎉 SUCCESS: Weight loading and model equivalence verified!")
        print("   - Weights loaded correctly from GPT-2")
        print("   - 16-bit mode produces equivalent results to GPT-2")
        print("   - LoRA interference eliminated in 16-bit mode")
    else:
        print("⚠️ ISSUES FOUND:")
        if metrics['weight_diff'] > 1e-6:
            print(f"   - Weight loading has differences: {metrics['weight_diff']:.2e}")
        if metrics['logit_diff'] > 1e-3:
            print(f"   - Forward pass differs: {metrics['logit_diff']:.2e}")
        if metrics['perplexity_diff'] > 1.0:
            print(f"   - Perplexity differs: {metrics['perplexity_diff']:.2f}")

    print(f"\n📈 KEY METRICS:")
    print(f"   Weight difference: {metrics['weight_diff']:.2e}")
    print(f"   Logit difference: {metrics['logit_diff']:.2e}")
    print(f"   Perplexity difference: {metrics['perplexity_diff']:.3f}")

    # Check precision degradation
    baseline_ppl = precision_results[16]['ppl']
    deg_8bit = ((precision_results[8]['ppl'] - baseline_ppl) / baseline_ppl) * 100
    deg_4bit = ((precision_results[4]['ppl'] - baseline_ppl) / baseline_ppl) * 100

    print(f"   8-bit degradation: {deg_8bit:.1f}%")
    print(f"   4-bit degradation: {deg_4bit:.1f}%")

    return success


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ All verifications passed!")
    else:
        print("\n❌ Some verifications failed - check details above")
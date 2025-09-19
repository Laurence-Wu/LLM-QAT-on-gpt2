#!/usr/bin/env python3
"""
Verify Weight Loading
Checks if weight loading after initialization works correctly
"""

import sys
import os
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test.fix_model_initialization import create_properly_initialized_model


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

    # Check first attention layer weights
    try:
        sp_attn = sp_model.transformer.h[0].attn.c_attn.linear.weight
        gpt2_attn = gpt2_model.transformer.h[0].attn.c_attn.weight.t()  # Transpose for comparison

        attn_diff = (sp_attn - gpt2_attn).abs().max().item()
        print(f"   First attention max diff: {attn_diff:.10f}")

        if attn_diff == 0.0:
            print("   ✅ Attention weights loaded perfectly")
        else:
            print("   ❌ Attention weights differ!")

    except AttributeError as e:
        print(f"   ❌ Error accessing attention weights: {e}")
        return False

    # Test 2: Forward pass comparison
    print("\n3. Testing forward pass equivalence...")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
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

    # Overall result
    success = (wte_diff < 1e-6 and attn_diff < 1e-6 and
               mean_diff < 1e-3 and avg_ppl_diff < 1.0)

    return success, {
        'weight_diff': max(wte_diff, attn_diff),
        'logit_diff': mean_diff,
        'perplexity_diff': avg_ppl_diff
    }


def test_different_precision_modes():
    """Test that different precisions work as expected."""
    print("\n" + "="*80)
    print("PRECISION MODE TESTING")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load SP model
    sp_model, sp_config = create_properly_initialized_model(use_pretrained=True)
    sp_model = sp_model.to(device)
    sp_model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    test_text = "Machine learning is transforming technology."
    inputs = tokenizer(test_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)

    print(f"\nTest sentence: '{test_text}'")

    results = {}

    with torch.no_grad():
        for precision in [16, 8, 4]:
            sp_model.set_precision(precision)

            outputs = sp_model(input_ids, labels=input_ids)
            loss = outputs['loss'].item()
            ppl = torch.exp(torch.tensor(loss)).item()

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


def main():
    """Main verification function."""
    print("\n" + "="*80)
    print("WEIGHT LOADING VERIFICATION")
    print("="*80)

    # Test 1: Detailed weight loading verification
    success, metrics = test_weight_loading_detailed()

    # Test 2: Precision mode testing
    precision_results = test_different_precision_modes()

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
#!/usr/bin/env python3
"""
Debug script to test why 32-bit model has high perplexity
"""

import sys
import os
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test.fix_model_initialization import create_properly_initialized_model


def test_simple_perplexity(model, tokenizer, device, model_name="Model"):
    """Test perplexity on a few simple sentences."""
    model = model.to(device)
    model.eval()

    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
    ]

    total_loss = 0
    total_tokens = 0

    print(f"\nTesting {model_name}:")
    with torch.no_grad():
        for sentence in test_sentences:
            inputs = tokenizer(sentence, return_tensors='pt')
            input_ids = inputs['input_ids'].to(device)

            if input_ids.size(1) < 2:
                continue

            outputs = model(input_ids, labels=input_ids)

            # Get the loss
            loss = outputs['loss' if hasattr(outputs, '__getitem__') else 'loss'].item()
            seq_length = input_ids.size(1) - 1

            total_loss += loss * seq_length
            total_tokens += seq_length

            ppl = np.exp(loss)
            print(f"  '{sentence[:30]}...': Loss={loss:.4f}, PPL={ppl:.2f}")

    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        avg_ppl = np.exp(avg_loss)
        print(f"  Average: Loss={avg_loss:.4f}, PPL={avg_ppl:.2f}")
        return avg_ppl
    return float('inf')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Test 1: Standard GPT-2 (should have low perplexity)
    print("\n" + "="*60)
    print("TEST 1: Standard GPT-2 Model")
    print("="*60)
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_ppl = test_simple_perplexity(gpt2_model, tokenizer, device, "Standard GPT-2")

    # Test 2: SP Model with full 12 layers in 32-bit mode
    print("\n" + "="*60)
    print("TEST 2: SP Model (Full, 12 layers, 32-bit)")
    print("="*60)
    sp_model_full, _ = create_properly_initialized_model(use_pretrained=True, num_layers=12)
    sp_model_full.set_precision(32)
    sp_full_ppl = test_simple_perplexity(sp_model_full, tokenizer, device, "SP Full 32-bit")

    # Test 3: SP Model with reduced 6 layers in 32-bit mode
    print("\n" + "="*60)
    print("TEST 3: SP Model (Reduced, 6 layers, 32-bit)")
    print("="*60)
    sp_model_reduced, _ = create_properly_initialized_model(use_pretrained=True, num_layers=6)
    sp_model_reduced.set_precision(32)
    sp_reduced_ppl = test_simple_perplexity(sp_model_reduced, tokenizer, device, "SP Reduced 32-bit")

    # Test 4: Check if weights are actually loaded
    print("\n" + "="*60)
    print("TEST 4: Weight Verification")
    print("="*60)

    # Check embedding weights
    gpt2_wte = gpt2_model.transformer.wte.weight
    sp_wte = sp_model_full.transformer.wte.weight

    # Compare first few embeddings
    diff = (gpt2_wte[:10, :10] - sp_wte[:10, :10]).abs().max().item()
    print(f"Max embedding weight difference (first 10x10): {diff:.6f}")

    if diff < 1e-5:
        print("✅ Embeddings match!")
    else:
        print("❌ Embeddings don't match!")

    # Check a transformer block weight
    gpt2_attn = gpt2_model.transformer.h[0].attn.c_attn.weight
    # SP model stores weights in linear.weight and transposes them
    sp_attn = sp_model_full.transformer.h[0].attn.c_attn.linear.weight

    # Note: SP model transposes weights, so we need to transpose back
    diff = (gpt2_attn[:10, :10] - sp_attn[:10, :10].t()).abs().max().item()
    print(f"Max attention weight difference (first 10x10): {diff:.6f}")

    if diff < 1e-5:
        print("✅ Attention weights match (accounting for transpose)!")
    else:
        print("❌ Attention weights don't match!")
        print(f"  GPT2 shape: {gpt2_attn.shape}")
        print(f"  SP shape: {sp_attn.shape}")
        print(f"  GPT2 sample: {gpt2_attn[0, :5]}")
        print(f"  SP sample (transposed): {sp_attn[:5, 0]}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Standard GPT-2 PPL: {gpt2_ppl:.2f} (expected: ~20-40)")
    print(f"SP Full 32-bit PPL: {sp_full_ppl:.2f} (should match GPT-2)")
    print(f"SP Reduced 32-bit PPL: {sp_reduced_ppl:.2f} (should be higher but not 1000s)")

    if sp_full_ppl > 100:
        print("\n❌ SP model has abnormally high perplexity!")
        print("Possible issues:")
        print("  1. Weights not properly loaded")
        print("  2. Forward pass not using correct weights in 32-bit mode")
        print("  3. Model architecture mismatch")
    elif abs(sp_full_ppl - gpt2_ppl) > 5:
        print("\n⚠️ SP model perplexity differs from GPT-2")
    else:
        print("\n✅ SP model working correctly!")


if __name__ == "__main__":
    main()
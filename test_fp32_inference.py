#!/usr/bin/env python3
"""
Test script to verify FP32 model inference and compare with standard GPT-2.
"""

import torch
import torch.nn.functional as F
import sys
import os
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(os.path.abspath(__file__))
part1_dir = os.path.join(parent_dir, 'part1_switchable_precision')
if part1_dir not in sys.path:
    sys.path.insert(0, part1_dir)

from part1_switchable_precision.models_sp import SPLMHeadModel
from transformers import GPT2Config


def test_text_generation(model, tokenizer, prompt="The weather today is"):
    """Test text generation capability."""
    print(f"\nGenerating text from prompt: '{prompt}'")

    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    if torch.cuda.is_available():
        model = model.cuda()
        input_ids = input_ids.cuda()

    with torch.no_grad():
        # Generate 20 tokens
        generated = input_ids
        for _ in range(20):
            outputs = model(generated)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            next_token_logits = logits[0, -1, :]

            # Apply temperature and top-k sampling
            next_token_logits = next_token_logits / 0.8  # temperature
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.95  # top-p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")
    return generated_text


def compute_perplexity(model, tokenizer, text):
    """Compute perplexity on a given text."""
    model.eval()
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids

    if torch.cuda.is_available():
        model = model.cuda()
        input_ids = input_ids.cuda()

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss if hasattr(outputs, 'loss') else None

        if loss is None:
            # Compute loss manually
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        perplexity = torch.exp(loss).item()

    return perplexity


def test_fp32_model(checkpoint_path):
    """Test FP32 model performance."""
    print("="*70)
    print("  FP32 MODEL INFERENCE TEST")
    print("="*70)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, dict):
        print("ERROR: Invalid checkpoint format")
        return

    model_config = checkpoint['model_config']
    bit_width = checkpoint.get('bit_width', None)

    print(f"Checkpoint bit width: {bit_width}")

    if bit_width != 32:
        print(f"WARNING: This checkpoint was saved at {bit_width}-bit, not 32-bit FP32!")

    # Create config
    config = GPT2Config(
        vocab_size=model_config.get('vocab_size', 50257),
        n_positions=model_config.get('n_positions', 1024),
        n_embd=model_config.get('n_embd', 768),
        n_layer=model_config.get('n_layer', 12),
        n_head=model_config.get('n_head', 12)
    )

    # Add SP configs
    config.bit_widths = model_config.get('bit_widths', [6, 8, 16, 32])
    config.lora_rank_per_bit = model_config.get('lora_rank_per_bit', {})
    config.lora_alpha_per_bit = model_config.get('lora_alpha_per_bit', {})
    config.activation_bits_per_bit = model_config.get('activation_bits_per_bit', {})
    config.quantizer_per_bit = model_config.get('quantizer_per_bit', {})

    # Convert string keys to int
    for attr in ['lora_rank_per_bit', 'lora_alpha_per_bit']:
        if hasattr(config, attr) and isinstance(getattr(config, attr), dict):
            setattr(config, attr, {int(k) if isinstance(k, str) else k: v
                                  for k, v in getattr(config, attr).items()})

    # Create and load model
    print("\nCreating SP model...")
    sp_model = SPLMHeadModel(config)

    # CRITICAL: Set to FP32 mode BEFORE loading
    sp_model.set_precision(32)
    print("Model set to 32-bit FP32 mode")

    # Load weights
    state_dict = checkpoint['model_state_dict']
    missing_keys, unexpected_keys = sp_model.load_state_dict(state_dict, strict=True)
    print(f"Loaded model weights (missing: {len(missing_keys)}, unexpected: {len(unexpected_keys)})")

    sp_model.eval()

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print("\n" + "="*70)
    print("  COMPARING WITH PRETRAINED GPT-2")
    print("="*70)

    # Load pretrained GPT-2 for comparison
    pretrained_model = GPT2LMHeadModel.from_pretrained('gpt2')
    pretrained_model.eval()

    # Test sentences
    test_sentences = [
        "The capital of France is",
        "Machine learning is",
        "The weather today is",
        "Once upon a time",
        "The quick brown fox jumps over the lazy dog."
    ]

    print("\nPerplexity Comparison:")
    print("-"*50)
    for sentence in test_sentences:
        sp_ppl = compute_perplexity(sp_model, tokenizer, sentence)
        pretrained_ppl = compute_perplexity(pretrained_model, tokenizer, sentence)

        print(f"\n'{sentence[:30]}...'")
        print(f"  SP Model (FP32):     {sp_ppl:.2f}")
        print(f"  Pretrained GPT-2:    {pretrained_ppl:.2f}")
        print(f"  Ratio (SP/Pretrained): {sp_ppl/pretrained_ppl:.2f}x")

        if sp_ppl > pretrained_ppl * 100:
            print("  ⚠️ WARNING: SP model perplexity is >100x worse than pretrained!")

    print("\n" + "="*70)
    print("  TEXT GENERATION TEST")
    print("="*70)

    print("\nSP Model Generation:")
    sp_text = test_text_generation(sp_model, tokenizer)

    print("\nPretrained GPT-2 Generation:")
    pretrained_text = test_text_generation(pretrained_model, tokenizer)

    # Test logits comparison
    print("\n" + "="*70)
    print("  LOGITS COMPARISON")
    print("="*70)

    test_input = tokenizer("Hello world", return_tensors='pt')
    input_ids = test_input['input_ids']

    if torch.cuda.is_available():
        sp_model = sp_model.cuda()
        pretrained_model = pretrained_model.cuda()
        input_ids = input_ids.cuda()

    with torch.no_grad():
        sp_output = sp_model(input_ids)
        sp_logits = sp_output.logits if hasattr(sp_output, 'logits') else sp_output

        pretrained_output = pretrained_model(input_ids)
        pretrained_logits = pretrained_output.logits

        # Compare statistics
        print(f"\nSP Model logits:")
        print(f"  Shape: {sp_logits.shape}")
        print(f"  Mean: {sp_logits.mean().item():.4f}")
        print(f"  Std: {sp_logits.std().item():.4f}")
        print(f"  Min: {sp_logits.min().item():.4f}")
        print(f"  Max: {sp_logits.max().item():.4f}")

        print(f"\nPretrained GPT-2 logits:")
        print(f"  Shape: {pretrained_logits.shape}")
        print(f"  Mean: {pretrained_logits.mean().item():.4f}")
        print(f"  Std: {pretrained_logits.std().item():.4f}")
        print(f"  Min: {pretrained_logits.min().item():.4f}")
        print(f"  Max: {pretrained_logits.max().item():.4f}")

        # Check top-5 predictions
        sp_probs = F.softmax(sp_logits[0, -1, :], dim=-1)
        pretrained_probs = F.softmax(pretrained_logits[0, -1, :], dim=-1)

        sp_top5 = torch.topk(sp_probs, 5)
        pretrained_top5 = torch.topk(pretrained_probs, 5)

        print(f"\nTop-5 predictions for next token:")
        print("SP Model:")
        for i, (prob, idx) in enumerate(zip(sp_top5.values, sp_top5.indices)):
            token = tokenizer.decode([idx.item()])
            print(f"  {i+1}. '{token}' ({prob.item():.4f})")

        print("\nPretrained GPT-2:")
        for i, (prob, idx) in enumerate(zip(pretrained_top5.values, pretrained_top5.indices)):
            token = tokenizer.decode([idx.item()])
            print(f"  {i+1}. '{token}' ({prob.item():.4f})")

    print("\n" + "="*70)
    print("  DIAGNOSIS SUMMARY")
    print("="*70)

    if sp_ppl > pretrained_ppl * 100:
        print("❌ SP model is severely underperforming compared to pretrained GPT-2")
        print("   Possible causes:")
        print("   1. Model was not properly initialized from pretrained weights")
        print("   2. Training corrupted the weights")
        print("   3. Architecture mismatch with standard GPT-2")
    else:
        print("✅ SP model performance is reasonable compared to pretrained GPT-2")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test FP32 model inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to FP32 checkpoint')
    args = parser.parse_args()

    test_fp32_model(args.checkpoint)


if __name__ == "__main__":
    main()
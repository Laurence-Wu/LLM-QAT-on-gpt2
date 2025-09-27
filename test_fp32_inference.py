#!/usr/bin/env python3
"""
Test script to verify FP32 model inference and compare with standard GPT-2.
"""

import torch
import torch.nn as nn
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

            # Handle different output formats
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs)
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            if isinstance(logits, torch.Tensor):
                next_token_logits = logits[0, -1, :]
            else:
                print(f"ERROR: Unexpected output type: {type(logits)}")
                break

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


def compute_perplexity_sliding_window(model, tokenizer, text, stride=512, max_length=1024):
    """Compute perplexity using sliding window approach (proper method)."""
    model.eval()

    # Tokenize the entire text
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    input_ids = encodings.input_ids

    if torch.cuda.is_available():
        model = model.cuda()
        input_ids = input_ids.cuda()

    seq_len = input_ids.size(1)
    if seq_len < 2:
        return float('inf')

    # Calculate perplexity with sliding window
    nlls = []
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # How many tokens we're predicting

        input_ids_chunk = input_ids[:, begin_loc:end_loc]

        with torch.no_grad():
            outputs = model(input_ids_chunk)

            # Handle different output formats
            if isinstance(outputs, dict):
                logits = outputs.get('logits', None)
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            if logits is None:
                return float('inf')

            # Compute negative log likelihood
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids_chunk[..., 1:].contiguous()

            # Only compute loss on the new tokens (not the context)
            if prev_end_loc > 0:
                shift_logits = shift_logits[:, -trg_len:, :]
                shift_labels = shift_labels[:, -trg_len:]

            loss_fct = nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            nlls.append(token_losses)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if len(nlls) == 0:
        return float('inf')

    # Average negative log likelihood across all tokens
    mean_nll = torch.cat(nlls).mean()
    perplexity = torch.exp(mean_nll).item()

    return perplexity


def compute_perplexity_simple(model, tokenizer, text):
    """Compute perplexity on text (simple method without sliding window)."""
    model.eval()

    # Tokenize with truncation for longer texts
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = encodings.input_ids

    if torch.cuda.is_available():
        model = model.cuda()
        input_ids = input_ids.cuda()

    if input_ids.size(1) < 2:
        return float('inf')

    with torch.no_grad():
        outputs = model(input_ids)

        # Handle different output formats
        if isinstance(outputs, dict):
            logits = outputs.get('logits', None)
        elif hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

        if logits is None:
            return float('inf')

        # Compute loss manually
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        perplexity = torch.exp(loss).item()

    return perplexity


def test_fp32_model(checkpoint_path):
    """Test FP32 model performance."""
    print("="*70)
    print("  FP32 MODEL INFERENCE TEST")
    print("="*70)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_config = checkpoint['model_config']
    bit_width = checkpoint.get('bit_width', None)
    print(f"Checkpoint bit width: {bit_width}")

    # Create config with minimal setup
    config = GPT2Config(
        vocab_size=model_config.get('vocab_size', 50257),
        n_positions=model_config.get('n_positions', 1024),
        n_embd=model_config.get('n_embd', 768),
        n_layer=model_config.get('n_layer', 12),
        n_head=model_config.get('n_head', 12)
    )

    # Copy SP-specific configs
    for attr in ['bit_widths', 'lora_rank_per_bit', 'lora_alpha_per_bit',
                 'activation_bits_per_bit', 'quantizer_per_bit']:
        setattr(config, attr, model_config.get(attr, {}))
        # Convert string keys to int for LoRA dicts
        if 'lora' in attr and isinstance(getattr(config, attr), dict):
            setattr(config, attr, {int(k) if isinstance(k, str) else k: v
                                  for k, v in getattr(config, attr).items()})

    # Set per-tensor quantization for evaluation
    config.per_channel_quantization = False

    # Create model and load weights
    print("\nLoading SP model...")
    sp_model = SPLMHeadModel(config)

    # Use bit width from checkpoint, fallback to 16 if not present
    target_precision = bit_width if bit_width is not None else 16
    print(f"Setting model to {target_precision}-bit precision from checkpoint")
    sp_model.set_precision(target_precision)

    sp_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
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

    # Test with longer, more meaningful texts for proper perplexity
    test_texts = [
        # Short text (for quick test)
        "The quick brown fox jumps over the lazy dog. " * 5,

        # Longer coherent text
        """Artificial intelligence and machine learning have revolutionized many industries.
        From healthcare to finance, these technologies are transforming how we work and live.
        Deep learning models can now understand language, recognize images, and even generate
        creative content. The future holds even more exciting possibilities as researchers
        continue to push the boundaries of what's possible with AI.""",
    ]

    print("\nPerplexity Comparison (on longer texts):")
    print("-"*50)

    for i, text in enumerate(test_texts):
        print(f"\nText {i+1} ({len(tokenizer.encode(text))} tokens):")

        # Use simple perplexity for shorter texts
        sp_ppl = compute_perplexity_simple(sp_model, tokenizer, text)
        pretrained_ppl = compute_perplexity_simple(pretrained_model, tokenizer, text)

        print(f"  SP Model (FP32):     {sp_ppl:.2f}")
        print(f"  Pretrained GPT-2:    {pretrained_ppl:.2f}")

        if pretrained_ppl > 0:
            ratio = sp_ppl / pretrained_ppl
            print(f"  Ratio (SP/Pretrained): {ratio:.2f}x")

            if ratio > 10:
                print("  ❌ WARNING: SP model perplexity is >10x worse than pretrained!")
            elif ratio > 3:
                print("  ⚠️ WARNING: SP model perplexity is >3x worse than pretrained!")

    # Also test on a standard benchmark paragraph
    print("\n" + "="*70)
    print("  WIKITEXT-2 STYLE PERPLEXITY TEST")
    print("="*70)

    wikitext_sample = """The city of Paris is the capital and most populous city of France.
    With an estimated population of 2,165,423 residents in 2019 in an area of more than 105 square kilometres,
    Paris is the fifth-most populated city in the European Union. Since the 17th century, Paris has been one of
    Europe's major centres of finance, diplomacy, commerce, fashion, science and arts."""

    print(f"\nWikiText-style paragraph ({len(tokenizer.encode(wikitext_sample))} tokens):")
    sp_wiki_ppl = compute_perplexity_simple(sp_model, tokenizer, wikitext_sample)
    pretrained_wiki_ppl = compute_perplexity_simple(pretrained_model, tokenizer, wikitext_sample)

    print(f"  SP Model (FP32):     {sp_wiki_ppl:.2f}")
    print(f"  Pretrained GPT-2:    {pretrained_wiki_ppl:.2f}")
    if pretrained_wiki_ppl > 0:
        print(f"  Ratio (SP/Pretrained): {sp_wiki_ppl/pretrained_wiki_ppl:.2f}x")

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
        # Handle different output formats
        if isinstance(sp_output, dict):
            sp_logits = sp_output.get('logits', sp_output)
        elif hasattr(sp_output, 'logits'):
            sp_logits = sp_output.logits
        else:
            sp_logits = sp_output

        pretrained_output = pretrained_model(input_ids)
        pretrained_logits = pretrained_output.logits

        # Compare statistics
        print(f"\nSP Model logits:")
        if isinstance(sp_logits, torch.Tensor):
            print(f"  Shape: {sp_logits.shape}")
            print(f"  Mean: {sp_logits.mean().item():.4f}")
            print(f"  Std: {sp_logits.std().item():.4f}")
            print(f"  Min: {sp_logits.min().item():.4f}")
            print(f"  Max: {sp_logits.max().item():.4f}")
        else:
            print(f"  ERROR: Unexpected type: {type(sp_logits)}")

        print(f"\nPretrained GPT-2 logits:")
        print(f"  Shape: {pretrained_logits.shape}")
        print(f"  Mean: {pretrained_logits.mean().item():.4f}")
        print(f"  Std: {pretrained_logits.std().item():.4f}")
        print(f"  Min: {pretrained_logits.min().item():.4f}")
        print(f"  Max: {pretrained_logits.max().item():.4f}")

        # Check top-5 predictions
        if isinstance(sp_logits, torch.Tensor):
            sp_probs = F.softmax(sp_logits[0, -1, :], dim=-1)
            sp_top5 = torch.topk(sp_probs, 5)

            print(f"\nTop-5 predictions for next token:")
            print("SP Model:")
            for i, (prob, idx) in enumerate(zip(sp_top5.values, sp_top5.indices)):
                token = tokenizer.decode([idx.item()])
                print(f"  {i+1}. '{token}' ({prob.item():.4f})")

        pretrained_probs = F.softmax(pretrained_logits[0, -1, :], dim=-1)
        pretrained_top5 = torch.topk(pretrained_probs, 5)

        print("\nPretrained GPT-2:")
        for i, (prob, idx) in enumerate(zip(pretrained_top5.values, pretrained_top5.indices)):
            token = tokenizer.decode([idx.item()])
            print(f"  {i+1}. '{token}' ({prob.item():.4f})")

    print("\n" + "="*70)
    print("  DIAGNOSIS SUMMARY")
    print("="*70)

    # Use the WikiText perplexity for final assessment
    if 'sp_wiki_ppl' in locals() and 'pretrained_wiki_ppl' in locals():
        ratio = sp_wiki_ppl / pretrained_wiki_ppl if pretrained_wiki_ppl > 0 else float('inf')

        if ratio > 10:
            print("❌ SP model is severely underperforming compared to pretrained GPT-2")
            print(f"   Perplexity ratio: {ratio:.2f}x worse")
            print("   Possible causes:")
            print("   1. Model was not properly initialized from pretrained weights")
            print("   2. Training corrupted the weights")
            print("   3. Architecture mismatch with standard GPT-2")
        elif ratio > 3:
            print("⚠️ SP model is moderately underperforming")
            print(f"   Perplexity ratio: {ratio:.2f}x worse")
            print("   This suggests partial weight corruption or training issues")
        else:
            print("✅ SP model performance is reasonable compared to pretrained GPT-2")
            print(f"   Perplexity ratio: {ratio:.2f}x")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test FP32 model inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to FP32 checkpoint')
    args = parser.parse_args()

    test_fp32_model(args.checkpoint)


if __name__ == "__main__":
    main()
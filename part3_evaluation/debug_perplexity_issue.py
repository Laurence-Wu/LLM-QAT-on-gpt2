#!/usr/bin/env python3
"""
Comprehensive debug script to identify why perplexity is high in main_llm_qat_eval.py
Compares loading methods, quantizer states, and logits between different approaches
"""

import torch
import torch.nn as nn
import sys
import os
import numpy as np
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part1_switchable_precision.models_sp import SPLMHeadModel
from transformers import GPT2Config, GPT2Tokenizer
import math


def analyze_quantizer_states(model, name_prefix="Model"):
    """Detailed analysis of all quantizer states in the model"""
    print(f"\n{'='*70}")
    print(f"  {name_prefix} - Quantizer State Analysis")
    print('='*70)

    results = {
        'total_quantizers': 0,
        'calibrated': 0,
        'uncalibrated': 0,
        'scale_stats': [],
        'zero_point_stats': [],
        'problematic': []
    }

    for name, module in model.named_modules():
        if hasattr(module, 'quantizers_input'):
            for bit_key, quantizer in module.quantizers_input.items():
                results['total_quantizers'] += 1

                # Check calibration status
                if quantizer.calibrated:
                    results['calibrated'] += 1
                else:
                    results['uncalibrated'] += 1
                    results['problematic'].append(f"{name}.{bit_key} NOT calibrated")

                # Analyze scale
                if hasattr(quantizer, 'scale'):
                    scale = quantizer.scale
                    scale_mean = scale.mean().item()
                    scale_std = scale.std().item() if scale.numel() > 1 else 0
                    results['scale_stats'].append(scale_mean)

                    # Check for problems
                    if torch.isnan(scale).any():
                        results['problematic'].append(f"{name}.{bit_key} has NaN scale")
                    if torch.isinf(scale).any():
                        results['problematic'].append(f"{name}.{bit_key} has Inf scale")
                    if (scale <= 0).any():
                        results['problematic'].append(f"{name}.{bit_key} has non-positive scale")

                # Analyze zero_point
                if hasattr(quantizer, 'zero_point'):
                    zp = quantizer.zero_point
                    if torch.isnan(zp).any():
                        results['problematic'].append(f"{name}.{bit_key} has NaN zero_point")

    # Print summary
    print(f"Total quantizers: {results['total_quantizers']}")
    print(f"Calibrated: {results['calibrated']}")
    print(f"Uncalibrated: {results['uncalibrated']}")

    if results['scale_stats']:
        print(f"\nScale statistics:")
        print(f"  Mean of all scales: {np.mean(results['scale_stats']):.6f}")
        print(f"  Std of all scales: {np.std(results['scale_stats']):.6f}")
        print(f"  Min scale: {min(results['scale_stats']):.6f}")
        print(f"  Max scale: {max(results['scale_stats']):.6f}")

    if results['problematic']:
        print(f"\n⚠️ Found {len(results['problematic'])} problematic quantizers:")
        for issue in results['problematic'][:5]:  # Show first 5
            print(f"  - {issue}")
    else:
        print("\n✓ No problematic quantizers found")

    return results


def load_model_method_1(checkpoint_path):
    """Method 1: Exactly like test_inference.py"""
    print("\n" + "="*70)
    print("  METHOD 1: test_inference.py style")
    print("  Sequence: CPU load → set_precision → load_state_dict → move to CUDA")
    print("="*70)

    # Load with map_location='cpu' like test_inference
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    bit_width = checkpoint.get('bit_width', 16)
    model_config = checkpoint.get('model_config', {})

    print(f"Checkpoint bit width: {bit_width}")

    # Build config
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
        if attr in model_config:
            setattr(config, attr, model_config[attr])

    # Set per-tensor quantization BEFORE model creation
    config.per_channel_quantization = False

    print("1. Creating model on CPU...")
    model = SPLMHeadModel(config)

    print(f"2. Setting precision to {bit_width}-bit...")
    model.set_precision(bit_width)

    print("3. Loading state dict...")
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    print("4. Moving to CUDA...")
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    print("✓ Model loaded successfully")

    return model, bit_width


def load_model_method_2(checkpoint_path):
    """Method 2: Original main_llm_qat_eval.py style (problematic)"""
    print("\n" + "="*70)
    print("  METHOD 2: Original main_llm_qat_eval.py style")
    print("  Sequence: CUDA load → move to CUDA → load_state_dict → set_precision")
    print("="*70)

    # Load with map_location='cuda' like original
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    bit_width = checkpoint.get('bit_width', 16)
    model_config = checkpoint.get('model_config', {})

    print(f"Checkpoint bit width: {bit_width}")

    # Build config
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
        if attr in model_config:
            setattr(config, attr, model_config[attr])

    config.per_channel_quantization = False

    print("1. Creating model and moving to CUDA immediately...")
    model = SPLMHeadModel(config)
    if torch.cuda.is_available():
        model = model.cuda()

    print("2. Loading state dict...")
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    print(f"3. Setting precision to {bit_width}-bit...")
    model.set_precision(bit_width)

    model.eval()
    print("✓ Model loaded")

    return model, bit_width


def load_model_method_3(checkpoint_path):
    """Method 3: Current main_llm_qat_eval.py with recent fixes"""
    print("\n" + "="*70)
    print("  METHOD 3: Fixed main_llm_qat_eval.py style")
    print("  Sequence: CUDA load → set_precision → load_state_dict → move to CUDA")
    print("="*70)

    # Still using map_location='cuda'
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    bit_width = checkpoint.get('bit_width', 16)
    model_config = checkpoint.get('model_config', {})

    print(f"Checkpoint bit width: {bit_width}")

    # Build config
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
        if attr in model_config:
            setattr(config, attr, model_config[attr])

    config.per_channel_quantization = False

    print("1. Creating model (staying on CPU)...")
    model = SPLMHeadModel(config)

    print(f"2. Setting precision to {bit_width}-bit...")
    model.set_precision(bit_width)

    print("3. Loading state dict...")
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    print("4. Moving to CUDA...")
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    print("✓ Model loaded")

    return model, bit_width


def compare_models_detailed(models, names, tokenizer):
    """Compare multiple models on the same input with detailed analysis"""

    # Test texts of varying lengths
    test_texts = [
        "The quick brown fox",
        "Machine learning is transforming how we interact with technology",
        " = Robert Boulter = \n\n Robert Boulter is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens"
    ]

    for text_idx, text in enumerate(test_texts):
        print(f"\n{'='*70}")
        print(f"  TEST {text_idx + 1}: '{text[:50]}...' ({len(text)} chars)")
        print('='*70)

        # Tokenize
        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        input_ids = tokens['input_ids']
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        print(f"Input shape: {input_ids.shape}")

        all_logits = []
        all_losses = []

        for model, name in zip(models, names):
            print(f"\n{name}:")

            with torch.no_grad():
                # Get outputs
                outputs = model(input_ids)

                # Extract logits
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                all_logits.append(logits)

                # Logits statistics
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits mean: {logits.mean().item():.4f}")
                print(f"  Logits std: {logits.std().item():.4f}")
                print(f"  Logits min: {logits.min().item():.4f}")
                print(f"  Logits max: {logits.max().item():.4f}")

                # Calculate perplexity if possible
                if input_ids.size(1) > 1:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    ppl = torch.exp(loss).item()
                    all_losses.append(loss.item())
                    print(f"  Loss: {loss.item():.4f}")
                    print(f"  Perplexity: {ppl:.2f}")

        # Compare logits between models
        if len(all_logits) > 1:
            print(f"\n📊 Comparison:")

            # Compare each model to Method 1 (test_inference style)
            reference_logits = all_logits[0]
            reference_loss = all_losses[0] if all_losses else None

            for i in range(1, len(all_logits)):
                print(f"\n  {names[i]} vs {names[0]}:")

                # Logits difference
                logits_diff = (all_logits[i] - reference_logits).abs()
                print(f"    Max logits difference: {logits_diff.max().item():.6f}")
                print(f"    Mean logits difference: {logits_diff.mean().item():.6f}")

                # Check if identical
                if torch.allclose(all_logits[i], reference_logits, rtol=1e-4, atol=1e-4):
                    print(f"    ✓ Logits are identical (within tolerance)")
                else:
                    print(f"    ✗ Logits differ significantly!")

                    # Find where they differ most
                    max_diff_idx = logits_diff.argmax()
                    max_diff_flat = max_diff_idx.item()
                    vocab_size = logits_diff.shape[-1]
                    token_idx = max_diff_flat // vocab_size
                    vocab_idx = max_diff_flat % vocab_size
                    print(f"    Maximum difference at token {token_idx}, vocab {vocab_idx}")

                # Loss/PPL difference
                if reference_loss and i < len(all_losses):
                    loss_diff = all_losses[i] - reference_loss
                    ppl_ratio = math.exp(all_losses[i]) / math.exp(reference_loss)
                    print(f"    Loss difference: {loss_diff:.4f}")
                    print(f"    Perplexity ratio: {ppl_ratio:.2f}x")


def check_specific_layers(model, name):
    """Check specific problematic layers"""
    print(f"\n{'='*70}")
    print(f"  {name} - Specific Layer Analysis")
    print('='*70)

    # Check lm_head
    if hasattr(model, 'lm_head'):
        print("\nlm_head analysis:")
        lm_head = model.lm_head

        # Check weight stats
        print(f"  Weight shape: {lm_head.weight.shape}")
        print(f"  Weight mean: {lm_head.weight.mean().item():.6f}")
        print(f"  Weight std: {lm_head.weight.std().item():.6f}")

        # Check if lm_head has quantizers (it shouldn't!)
        if hasattr(lm_head, 'quantizers_input'):
            print("  ⚠️ WARNING: lm_head has input quantizers!")
        else:
            print("  ✓ lm_head has no input quantizers (correct)")

        if hasattr(lm_head, 'quantizers_weight'):
            print("  ⚠️ WARNING: lm_head has weight quantizers!")
        else:
            print("  ✓ lm_head has no weight quantizers (correct)")

    # Check final layer norm
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
        ln_f = model.transformer.ln_f
        print("\nFinal LayerNorm (ln_f) analysis:")
        print(f"  Weight mean: {ln_f.weight.mean().item():.6f}")
        print(f"  Weight std: {ln_f.weight.std().item():.6f}")
        if hasattr(ln_f, 'bias') and ln_f.bias is not None:
            print(f"  Bias mean: {ln_f.bias.mean().item():.6f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--verbose', action='store_true', help='Show more details')
    args = parser.parse_args()

    print("="*70)
    print("  PERPLEXITY DEBUG ANALYSIS")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Load models using different methods
    models = []
    names = []

    # Method 1: test_inference.py style (working)
    model1, bit1 = load_model_method_1(args.checkpoint)
    models.append(model1)
    names.append("Method 1 (test_inference style)")

    # Method 2: Original main_llm_qat_eval.py (broken)
    model2, bit2 = load_model_method_2(args.checkpoint)
    models.append(model2)
    names.append("Method 2 (original main_eval)")

    # Method 3: Fixed main_llm_qat_eval.py
    model3, bit3 = load_model_method_3(args.checkpoint)
    models.append(model3)
    names.append("Method 3 (fixed main_eval)")

    # Analyze quantizer states
    for model, name in zip(models, names):
        analyze_quantizer_states(model, name)
        if args.verbose:
            check_specific_layers(model, name)

    # Compare models on same inputs
    compare_models_detailed(models, names, tokenizer)

    # Final summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)

    print("\n🔍 Key differences to look for:")
    print("1. Logits mean: Should be around -60, not -99")
    print("2. Perplexity: Should be < 30 for good models")
    print("3. Quantizer states: All should be calibrated")
    print("4. Scale values: Should be reasonable (not too small/large)")
    print("\n✅ Method 1 should work best (matches test_inference.py)")
    print("❌ Method 2 likely has issues (original problematic approach)")
    print("❓ Method 3 should be better but may still have issues")


if __name__ == "__main__":
    main()
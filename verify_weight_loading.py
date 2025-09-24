#!/usr/bin/env python3
"""
Script to verify that weights are correctly loaded from checkpoint.
Compares checkpoint weights with loaded model weights.
"""

import torch
import torch.nn as nn
import sys
import os
import numpy as np

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(os.path.abspath(__file__))
part1_dir = os.path.join(parent_dir, 'part1_switchable_precision')
if part1_dir not in sys.path:
    sys.path.insert(0, part1_dir)

from part1_switchable_precision.models_sp import SPLMHeadModel
from transformers import GPT2Config


def compare_weights(checkpoint_path):
    """Compare weights in checkpoint with loaded model weights."""
    print("="*70)
    print("  WEIGHT LOADING VERIFICATION")
    print("="*70)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        print("ERROR: Invalid checkpoint format")
        return

    checkpoint_state = checkpoint['model_state_dict']
    model_config = checkpoint['model_config']
    checkpoint_bit_width = checkpoint.get('bit_width', None)

    print(f"Checkpoint bit width: {checkpoint_bit_width}")
    print(f"Checkpoint has {len(checkpoint_state)} keys")

    # Create model
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
    for attr in ['lora_rank_per_bit', 'lora_alpha_per_bit', 'activation_bits_per_bit', 'quantizer_per_bit']:
        if hasattr(config, attr) and isinstance(getattr(config, attr), dict):
            setattr(config, attr, {int(k) if isinstance(k, str) else k: v
                                  for k, v in getattr(config, attr).items()})

    print(f"\nCreating model...")
    model = SPLMHeadModel(config)

    # Set precision BEFORE loading
    if checkpoint_bit_width:
        print(f"Setting model to {checkpoint_bit_width}-bit precision BEFORE loading")
        model.set_precision(checkpoint_bit_width)

    # Load weights
    print("\nLoading weights...")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state, strict=False)

    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")

    # Get model state dict
    model_state = model.state_dict()

    # Compare weights
    print("\n" + "="*70)
    print("  WEIGHT COMPARISON")
    print("="*70)

    matches = 0
    mismatches = 0
    not_found = 0

    # Critical weights to check
    critical_weights = [
        'transformer.wte.weight',
        'transformer.wpe.weight',
        'lm_head.weight',
        'transformer.ln_f.weight',
        'transformer.ln_f.bias'
    ]

    print("\nCritical Weights Check:")
    for weight_name in critical_weights:
        if weight_name in checkpoint_state and weight_name in model_state:
            checkpoint_weight = checkpoint_state[weight_name]
            model_weight = model_state[weight_name]

            if checkpoint_weight.shape != model_weight.shape:
                print(f"  ❌ {weight_name}: Shape mismatch!")
                print(f"     Checkpoint: {checkpoint_weight.shape}")
                print(f"     Model: {model_weight.shape}")
                mismatches += 1
            elif torch.allclose(checkpoint_weight, model_weight, atol=1e-6):
                print(f"  ✅ {weight_name}: Weights match perfectly")
                matches += 1
            else:
                diff = (checkpoint_weight - model_weight).abs().max().item()
                print(f"  ⚠️ {weight_name}: Weights differ (max diff: {diff:.6f})")
                mismatches += 1
        elif weight_name in checkpoint_state:
            print(f"  ❌ {weight_name}: In checkpoint but NOT in model")
            not_found += 1
        elif weight_name in model_state:
            print(f"  ⚠️ {weight_name}: In model but NOT in checkpoint (using random init)")
            not_found += 1

    # Check transformer blocks
    print("\nTransformer Blocks Check:")
    for i in range(config.n_layer):
        block_prefix = f'transformer.h.{i}'
        block_weights = [k for k in checkpoint_state.keys() if k.startswith(block_prefix)]

        if block_weights:
            # Check attention weights
            attn_weight = f'{block_prefix}.attn.c_attn.linear.weight'
            if attn_weight in checkpoint_state and attn_weight in model_state:
                if torch.allclose(checkpoint_state[attn_weight], model_state[attn_weight], atol=1e-6):
                    matches += 1
                else:
                    mismatches += 1
                    print(f"  ⚠️ Block {i} attention weights differ")

            # Check MLP weights
            mlp_weight = f'{block_prefix}.mlp.c_fc.linear.weight'
            if mlp_weight in checkpoint_state and mlp_weight in model_state:
                if torch.allclose(checkpoint_state[mlp_weight], model_state[mlp_weight], atol=1e-6):
                    matches += 1
                else:
                    mismatches += 1
                    print(f"  ⚠️ Block {i} MLP weights differ")

    # Check LoRA adapters
    print("\nLoRA Adapter Check:")
    lora_keys = [k for k in checkpoint_state.keys() if 'lora' in k.lower()]
    print(f"  Found {len(lora_keys)} LoRA-related keys in checkpoint")

    for bit_width in config.bit_widths:
        if bit_width < 32:  # Skip FP32 teacher
            bit_key = f'{bit_width}bit'
            lora_keys_for_bit = [k for k in lora_keys if bit_key in k]
            print(f"  {bit_width}-bit: {len(lora_keys_for_bit)} LoRA keys")

    # Check quantizer calibration
    print("\nQuantizer Calibration Check:")
    scale_keys = [k for k in checkpoint_state.keys() if '.scale' in k]
    zero_point_keys = [k for k in checkpoint_state.keys() if '.zero_point' in k]
    print(f"  Found {len(scale_keys)} scale parameters")
    print(f"  Found {len(zero_point_keys)} zero_point parameters")

    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    print(f"Weights matched: {matches}")
    print(f"Weights mismatched: {mismatches}")
    print(f"Weights not found: {not_found}")

    if mismatches > 0 or not_found > 0:
        print("\n❌ CRITICAL: Weight loading issues detected!")
        print("   Model will likely underperform due to incorrect weights.")
    else:
        print("\n✅ All critical weights loaded correctly!")

    # Test inference
    print("\n" + "="*70)
    print("  INFERENCE TEST")
    print("="*70)

    model.eval()
    with torch.no_grad():
        # Test with different inputs
        test_cases = [
            torch.tensor([[50256]]),  # EOS token
            torch.randint(0, 50257, (1, 10)),  # Random tokens
            torch.tensor([[464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13]])  # "The quick brown fox jumps over the lazy dog."
        ]

        for i, input_ids in enumerate(test_cases):
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            print(f"\nTest case {i+1}:")
            print(f"  Input shape: {input_ids.shape}")
            print(f"  Output shape: {logits.shape}")
            print(f"  Output stats: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")

            # Check for issues
            if torch.isnan(logits).any():
                print("  ❌ Output contains NaN!")
            if (logits == 0).all():
                print("  ❌ Output is all zeros!")
            if logits.std().item() < 1e-6:
                print("  ⚠️ Very low output variance!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Verify weight loading from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    args = parser.parse_args()

    compare_weights(args.checkpoint)


if __name__ == "__main__":
    main()
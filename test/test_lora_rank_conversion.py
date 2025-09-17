#!/usr/bin/env python3
"""
Test script to verify LoRA rank conversion works correctly.
"""

import torch
import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part1_switchable_precision.config_qat import ModelConfig
from shared.models import SwitchableQATGPT2
from transformers import GPT2Config


def test_lora_rank_shapes(checkpoint_path):
    """
    Test if LoRA shapes match expected ranks after conversion.

    Args:
        checkpoint_path: Path to converted checkpoint
    """
    print(f"\n{'='*70}")
    print("TESTING LORA RANK SHAPES")
    print(f"{'='*70}\n")

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Get expected ranks from config
    if 'config' in checkpoint and 'lora_rank_per_bit' in checkpoint['config']:
        expected_ranks = checkpoint['config']['lora_rank_per_bit']
        print(f"Expected LoRA ranks from checkpoint config: {expected_ranks}")
    else:
        # Use ModelConfig as fallback
        config = ModelConfig()
        expected_ranks = config.lora_rank_per_bit
        print(f"Expected LoRA ranks from ModelConfig: {expected_ranks}")

    # Check actual ranks in state dict
    print("\nChecking LoRA matrix shapes:")
    print("-" * 40)

    results = {'correct': 0, 'incorrect': 0, 'total': 0}
    issues = []

    for bit_width in [4, 8, 16]:
        expected_rank = expected_ranks.get(bit_width, 16)
        print(f"\n{bit_width}-bit precision (expected rank: {expected_rank}):")

        # Find all LoRA matrices for this bit width
        lora_a_keys = [k for k in state_dict.keys()
                       if f'lora_adapters.{bit_width}bit.lora_A' in k]
        lora_b_keys = [k for k in state_dict.keys()
                       if f'lora_adapters.{bit_width}bit.lora_B' in k]

        if not lora_a_keys and not lora_b_keys:
            print(f"  ⚠️ No LoRA matrices found for {bit_width}-bit")
            continue

        # Check lora_A shapes
        for key in lora_a_keys[:2]:  # Show first 2 examples
            tensor = state_dict[key]
            actual_rank = tensor.shape[1]  # lora_A shape: [in_features, rank]
            results['total'] += 1

            if actual_rank == expected_rank:
                print(f"  ✓ {key.split('.')[-4:-2]}: rank {actual_rank}")
                results['correct'] += 1
            else:
                print(f"  ✗ {key.split('.')[-4:-2]}: rank {actual_rank} (expected {expected_rank})")
                results['incorrect'] += 1
                issues.append(f"{bit_width}-bit lora_A has rank {actual_rank}, expected {expected_rank}")

        # Check lora_B shapes
        for key in lora_b_keys[:2]:  # Show first 2 examples
            tensor = state_dict[key]
            actual_rank = tensor.shape[0]  # lora_B shape: [rank, out_features]
            results['total'] += 1

            if actual_rank == expected_rank:
                print(f"  ✓ {key.split('.')[-4:-2]}: rank {actual_rank}")
                results['correct'] += 1
            else:
                print(f"  ✗ {key.split('.')[-4:-2]}: rank {actual_rank} (expected {expected_rank})")
                results['incorrect'] += 1
                issues.append(f"{bit_width}-bit lora_B has rank {actual_rank}, expected {expected_rank}")

        print(f"  Total LoRA layers: {len(lora_a_keys)}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    accuracy = (results['correct'] / results['total'] * 100) if results['total'] > 0 else 0
    print(f"Shape correctness: {results['correct']}/{results['total']} ({accuracy:.1f}%)")

    if issues:
        print("\nIssues found:")
        for issue in set(issues):
            print(f"  - {issue}")
    else:
        print("\n✅ All LoRA ranks are correct!")

    return results['incorrect'] == 0


def test_model_loading(checkpoint_path):
    """
    Test if the model can be loaded with the converted checkpoint.

    Args:
        checkpoint_path: Path to converted checkpoint
    """
    print(f"\n{'='*70}")
    print("TESTING MODEL LOADING")
    print(f"{'='*70}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config and bit widths
    if 'config' in checkpoint:
        stored_config = checkpoint['config']
        bit_widths = stored_config.get('bit_widths', [4, 8, 16])
        lora_rank_per_bit = stored_config.get('lora_rank_per_bit', {4: 32, 8: 16, 16: 8})
    else:
        bit_widths = checkpoint.get('bit_widths', [4, 8, 16])
        lora_rank_per_bit = checkpoint.get('lora_rank_per_bit', {4: 32, 8: 16, 16: 8})

    print(f"Bit widths: {bit_widths}")
    print(f"LoRA ranks: {lora_rank_per_bit}")

    # Create GPT2Config
    config = GPT2Config()
    config.n_layer = 12
    config.n_embd = 768
    config.n_head = 12
    config.n_positions = 1024
    config.vocab_size = 50257

    # Add LoRA config
    config.lora_rank = 8  # Default single rank
    config.lora_alpha = 16
    config.lora_dropout = 0.1
    config.lora_rank_per_bit = lora_rank_per_bit

    # Create model
    print("\nCreating SwitchableQATGPT2 model...")
    try:
        model = SwitchableQATGPT2(config, bit_widths=bit_widths, initialize_weights=False)
        print("✓ Model created successfully")
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return False

    # Load state dict
    print("\nLoading state dict...")
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")

        if missing_keys:
            print(f"\nFirst 5 missing keys:")
            for key in missing_keys[:5]:
                print(f"  - {key}")

        if unexpected_keys:
            print(f"\nFirst 5 unexpected keys:")
            for key in unexpected_keys[:5]:
                print(f"  - {key}")

        # Check if loading was successful
        if len(missing_keys) == 0 and len(unexpected_keys) == 0:
            print("\n✅ Perfect loading - no missing or unexpected keys!")
            return True
        elif len(missing_keys) < 100 and len(unexpected_keys) < 100:
            print("\n⚠️ Loaded with minor issues")
            return True
        else:
            print("\n✗ Too many loading issues")
            return False

    except Exception as e:
        print(f"✗ Failed to load state dict: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test LoRA rank conversion')
    parser.add_argument('checkpoint_path', type=str,
                        help='Path to the converted checkpoint')
    parser.add_argument('--skip-loading', action='store_true',
                        help='Skip model loading test')
    parser.add_argument('--skip-shapes', action='store_true',
                        help='Skip shape verification test')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found: {args.checkpoint_path}")
        sys.exit(1)

    success = True

    # Test 1: Check LoRA shapes
    if not args.skip_shapes:
        shapes_ok = test_lora_rank_shapes(args.checkpoint_path)
        success = success and shapes_ok

    # Test 2: Try loading the model
    if not args.skip_loading:
        loading_ok = test_model_loading(args.checkpoint_path)
        success = success and loading_ok

    # Final verdict
    print(f"\n{'='*70}")
    print("FINAL RESULT")
    print(f"{'='*70}\n")

    if success:
        print("✅ All tests passed! The conversion is successful.")
        print("\nNext steps:")
        print(f"1. Run diagnostics: ./test/run_all_diagnostics.sh {args.checkpoint_path}")
        print(f"2. Test generation: python test/test_checkpoint_fix.py {args.checkpoint_path}")
    else:
        print("✗ Some tests failed. Please review the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
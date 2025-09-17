#!/usr/bin/env python3
"""
Fix checkpoint loading issues by converting single-precision QAT checkpoint
to switchable multi-precision format.

This script converts state dict keys from:
- Single quantizer (quantize_weight) to multiple quantizers (quantizers_weight.4bit/8bit/16bit)
- Single LoRA (lora.) to multiple LoRA adapters (lora_adapters.4bit/8bit/16bit)
"""

import torch
import os
import sys
import argparse
from pathlib import Path
import json
from copy import deepcopy

def convert_single_to_switchable_checkpoint(checkpoint_path, output_path=None):
    """
    Convert a single-precision QAT checkpoint to switchable multi-precision format.

    Args:
        checkpoint_path: Path to the original checkpoint
        output_path: Path for the converted checkpoint (optional)

    Returns:
        Path to the converted checkpoint
    """
    print(f"\n{'='*70}")
    print("CHECKPOINT CONVERSION: Single-Precision → Multi-Precision")
    print(f"{'='*70}\n")

    # Load the original checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract the state dict
    try:
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                old_state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                old_state_dict = checkpoint['model']
            else:
                # Assume the checkpoint is the state dict itself
                old_state_dict = checkpoint
                checkpoint = {'model_state_dict': old_state_dict}
        else:
            raise ValueError(f"Checkpoint has unexpected type: {type(checkpoint)}")
    except Exception as e:
        print(f"Error extracting state dict: {e}")
        raise

    print(f"Original state dict has {len(old_state_dict)} keys")

    # Create the new state dict with converted keys
    new_state_dict = {}
    bit_widths = [4, 8, 16]

    # Statistics
    converted_keys = {
        'quantizers': 0,
        'lora': 0,
        'unchanged': 0
    }

    for old_key, value in old_state_dict.items():
        # Check if this is a quantizer key
        if '.quantize_weight.' in old_key:
            # Convert weight quantizer keys
            base_key = old_key.replace('.quantize_weight.', '.quantizers_weight.')
            for bits in bit_widths:
                new_key = base_key.replace('.quantizers_weight.', f'.quantizers_weight.{bits}bit.')
                new_state_dict[new_key] = value.clone()
            converted_keys['quantizers'] += 1

        elif '.quantize_input.' in old_key:
            # Convert input quantizer keys
            base_key = old_key.replace('.quantize_input.', '.quantizers_input.')
            for bits in bit_widths:
                new_key = base_key.replace('.quantizers_input.', f'.quantizers_input.{bits}bit.')
                new_state_dict[new_key] = value.clone()
            converted_keys['quantizers'] += 1

        elif '.lora.' in old_key and '.lora_adapters.' not in old_key:
            # Convert LoRA adapter keys
            # Handle the pattern: h.X.attn/mlp.c_Y.lora.component
            parts = old_key.split('.')
            lora_idx = parts.index('lora')

            # Build new key with lora_adapters.Xbit structure
            for bits in bit_widths:
                new_parts = parts[:lora_idx] + ['lora_adapters', f'{bits}bit'] + parts[lora_idx+1:]
                new_key = '.'.join(new_parts)
                new_state_dict[new_key] = value.clone()
            converted_keys['lora'] += 1

        else:
            # Keep other keys unchanged (embeddings, layer norms, etc.)
            new_state_dict[old_key] = value
            converted_keys['unchanged'] += 1

    print(f"\nConversion statistics:")
    print(f"  Quantizer keys converted: {converted_keys['quantizers']}")
    print(f"  LoRA keys converted: {converted_keys['lora']}")
    print(f"  Unchanged keys: {converted_keys['unchanged']}")
    print(f"  New state dict has {len(new_state_dict)} keys")

    # Update the checkpoint with new state dict
    checkpoint['model_state_dict'] = new_state_dict

    # Add bit_widths to config
    try:
        if 'config' in checkpoint:
            checkpoint['config']['bit_widths'] = bit_widths
            print(f"Added bit_widths {bit_widths} to config")
        else:
            # Create a minimal config with bit_widths
            checkpoint['bit_widths'] = bit_widths
            print(f"Added bit_widths {bit_widths} to checkpoint")
    except Exception as e:
        print(f"Warning: Could not add bit_widths to checkpoint: {e}")
        # Continue anyway as this is not critical

    # Determine output path
    if output_path is None:
        # Create output path by adding '_switchable' before the extension
        path = Path(checkpoint_path)
        output_path = path.parent / f"{path.stem}_switchable{path.suffix}"

    # Save the converted checkpoint
    print(f"\nSaving converted checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)

    # Verify the conversion
    print("\nVerifying conversion...")
    verify_checkpoint = torch.load(output_path, map_location='cpu')
    verify_state_dict = verify_checkpoint.get('model_state_dict', verify_checkpoint)

    # Check for expected key patterns
    has_quantizers = any('quantizers_weight.8bit' in k for k in verify_state_dict.keys())
    has_lora_adapters = any('lora_adapters.8bit' in k for k in verify_state_dict.keys())

    if has_quantizers and has_lora_adapters:
        print("✓ Conversion successful!")
        print(f"  - Found multi-bit quantizers: {has_quantizers}")
        print(f"  - Found multi-bit LoRA adapters: {has_lora_adapters}")
    else:
        print("⚠️ Warning: Conversion may be incomplete")
        print(f"  - Multi-bit quantizers found: {has_quantizers}")
        print(f"  - Multi-bit LoRA adapters found: {has_lora_adapters}")

    return output_path


def analyze_checkpoint_structure(checkpoint_path):
    """Analyze and print the structure of a checkpoint."""
    print(f"\n{'='*70}")
    print(f"ANALYZING CHECKPOINT: {checkpoint_path}")
    print(f"{'='*70}\n")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Check what's in the checkpoint
    if isinstance(checkpoint, dict):
        print("Checkpoint contents:")
        for key in checkpoint.keys():
            if key not in ['model_state_dict', 'model']:
                if isinstance(checkpoint[key], (int, float, str, list)):
                    print(f"  {key}: {checkpoint[key]}")
                else:
                    print(f"  {key}: {type(checkpoint[key])}")

    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    print(f"\nState dict has {len(state_dict)} keys")

    # Analyze key patterns
    patterns = {
        'embeddings': [],
        'layer_norms': [],
        'quantizers_single': [],
        'quantizers_multi': [],
        'lora_single': [],
        'lora_multi': [],
        'linear_weights': [],
        'other': []
    }

    for key in state_dict.keys():
        if 'wte' in key or 'wpe' in key:
            patterns['embeddings'].append(key)
        elif 'ln_' in key or 'ln_f' in key:
            patterns['layer_norms'].append(key)
        elif '.quantize_weight.' in key or '.quantize_input.' in key:
            patterns['quantizers_single'].append(key)
        elif '.quantizers_weight.' in key or '.quantizers_input.' in key:
            patterns['quantizers_multi'].append(key)
        elif '.lora.' in key and '.lora_adapters.' not in key:
            patterns['lora_single'].append(key)
        elif '.lora_adapters.' in key:
            patterns['lora_multi'].append(key)
        elif '.linear.weight' in key or '.linear.bias' in key:
            patterns['linear_weights'].append(key)
        else:
            patterns['other'].append(key)

    print("\nKey patterns found:")
    for pattern_name, keys in patterns.items():
        if keys:
            print(f"  {pattern_name}: {len(keys)} keys")
            # Show first 2 examples
            for i, key in enumerate(keys[:2]):
                print(f"    - {key}")
            if len(keys) > 2:
                print(f"    ... and {len(keys)-2} more")

    # Determine checkpoint type
    print("\nCheckpoint type:")
    if patterns['quantizers_single'] and not patterns['quantizers_multi']:
        print("  → Single-precision QAT checkpoint (needs conversion)")
    elif patterns['quantizers_multi'] and not patterns['quantizers_single']:
        print("  → Multi-precision switchable checkpoint (ready to use)")
    elif patterns['quantizers_single'] and patterns['quantizers_multi']:
        print("  → Mixed checkpoint (may have issues)")
    else:
        print("  → Standard checkpoint (no quantization)")

    return patterns


def compare_checkpoints(checkpoint1_path, checkpoint2_path):
    """Compare two checkpoints to see the differences."""
    print(f"\n{'='*70}")
    print("COMPARING CHECKPOINTS")
    print(f"{'='*70}\n")

    # Load both checkpoints
    ckpt1 = torch.load(checkpoint1_path, map_location='cpu')
    ckpt2 = torch.load(checkpoint2_path, map_location='cpu')

    # Get state dicts
    state1 = ckpt1.get('model_state_dict', ckpt1)
    state2 = ckpt2.get('model_state_dict', ckpt2)

    keys1 = set(state1.keys())
    keys2 = set(state2.keys())

    print(f"Checkpoint 1: {len(keys1)} keys")
    print(f"Checkpoint 2: {len(keys2)} keys")

    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    common = keys1 & keys2

    print(f"\nKeys only in checkpoint 1: {len(only_in_1)}")
    if only_in_1:
        for key in list(only_in_1)[:5]:
            print(f"  - {key}")
        if len(only_in_1) > 5:
            print(f"  ... and {len(only_in_1)-5} more")

    print(f"\nKeys only in checkpoint 2: {len(only_in_2)}")
    if only_in_2:
        for key in list(only_in_2)[:5]:
            print(f"  - {key}")
        if len(only_in_2) > 5:
            print(f"  ... and {len(only_in_2)-5} more")

    print(f"\nCommon keys: {len(common)}")

    # Check weight differences for common keys
    if common:
        print("\nChecking weight differences for common keys...")
        diffs = []
        for key in list(common)[:10]:  # Check first 10 common keys
            diff = (state1[key] - state2[key]).abs().mean().item()
            diffs.append((key, diff))

        diffs.sort(key=lambda x: x[1], reverse=True)
        print("Top differences:")
        for key, diff in diffs[:5]:
            print(f"  {key}: {diff:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert single-precision QAT checkpoint to multi-precision format'
    )
    parser.add_argument('checkpoint_path', type=str,
                        help='Path to the checkpoint file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for converted checkpoint')
    parser.add_argument('--analyze', action='store_true',
                        help='Only analyze checkpoint structure without conversion')
    parser.add_argument('--compare', type=str, default=None,
                        help='Compare with another checkpoint')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found: {args.checkpoint_path}")
        sys.exit(1)

    if args.analyze:
        # Just analyze the checkpoint structure
        analyze_checkpoint_structure(args.checkpoint_path)
    elif args.compare:
        # Compare two checkpoints
        if not os.path.exists(args.compare):
            print(f"Error: Comparison checkpoint not found: {args.compare}")
            sys.exit(1)
        compare_checkpoints(args.checkpoint_path, args.compare)
    else:
        # Convert the checkpoint
        output_path = convert_single_to_switchable_checkpoint(
            args.checkpoint_path,
            args.output
        )

        print(f"\n{'='*70}")
        print("CONVERSION COMPLETE")
        print(f"{'='*70}")
        print(f"\nConverted checkpoint saved to: {output_path}")
        print("\nNext steps:")
        print("1. Test with diagnostics:")
        print(f"   cd test && ./run_all_diagnostics.sh {output_path}")
        print("2. Run evaluation:")
        print(f"   python part3_evaluation/main_llm_qat_eval.py --model_path {output_path}")


if __name__ == "__main__":
    main()
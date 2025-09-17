#!/usr/bin/env python3
"""
Fix checkpoint loading issues by converting single-precision QAT checkpoint
to switchable multi-precision format with LoRA rank adaptation.

Version 2: Handles LoRA rank mismatch by adapting ranks for different bit-widths.

This script converts:
- Single quantizer to multiple quantizers (4bit/8bit/16bit)
- Single LoRA to multiple LoRA adapters with different ranks
- Adapts LoRA matrices from original rank to target ranks
"""

import torch
import os
import sys
import argparse
from pathlib import Path
import json
from copy import deepcopy
import numpy as np

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the config
try:
    from part1_switchable_precision.config_qat import ModelConfig
    CONFIG_AVAILABLE = True
except ImportError:
    print("Warning: Could not import ModelConfig from config_qat")
    CONFIG_AVAILABLE = False


def adapt_lora_rank(tensor, original_rank, target_rank, matrix_type='A'):
    """
    Adapt a LoRA matrix from original rank to target rank.

    Args:
        tensor: The LoRA weight tensor to adapt
        original_rank: The rank of the original tensor
        target_rank: The desired target rank
        matrix_type: 'A' for lora_A or 'B' for lora_B

    Returns:
        Adapted tensor with target rank
    """
    if original_rank == target_rank:
        return tensor

    if matrix_type == 'A':
        # lora_A shape: [in_features, rank]
        in_features = tensor.shape[0]

        if target_rank > original_rank:
            # Pad with zeros to expand rank
            padding = torch.zeros(in_features, target_rank - original_rank,
                                 dtype=tensor.dtype, device=tensor.device)
            new_tensor = torch.cat([tensor, padding], dim=1)
            print(f"    Expanded lora_A rank from {original_rank} to {target_rank} (padded with zeros)")
        else:
            # Truncate to reduce rank (keep most important components)
            new_tensor = tensor[:, :target_rank]
            print(f"    Truncated lora_A rank from {original_rank} to {target_rank}")

    elif matrix_type == 'B':
        # lora_B shape: [rank, out_features]
        out_features = tensor.shape[1]

        if target_rank > original_rank:
            # Pad with zeros to expand rank
            padding = torch.zeros(target_rank - original_rank, out_features,
                                 dtype=tensor.dtype, device=tensor.device)
            new_tensor = torch.cat([tensor, padding], dim=0)
            print(f"    Expanded lora_B rank from {original_rank} to {target_rank} (padded with zeros)")
        else:
            # Truncate to reduce rank
            new_tensor = tensor[:target_rank, :]
            print(f"    Truncated lora_B rank from {original_rank} to {target_rank}")
    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")

    return new_tensor


def detect_lora_rank(state_dict):
    """
    Detect the LoRA rank used in the checkpoint.

    Args:
        state_dict: The model state dictionary

    Returns:
        The detected LoRA rank
    """
    lora_ranks = []

    for key, tensor in state_dict.items():
        if '.lora.lora_A' in key:
            # lora_A shape is [in_features, rank]
            rank = tensor.shape[1]
            lora_ranks.append(rank)
        elif '.lora.lora_B' in key:
            # lora_B shape is [rank, out_features]
            rank = tensor.shape[0]
            lora_ranks.append(rank)

    if lora_ranks:
        # Check if all ranks are the same
        unique_ranks = list(set(lora_ranks))
        if len(unique_ranks) == 1:
            detected_rank = unique_ranks[0]
            print(f"Detected LoRA rank: {detected_rank}")
            return detected_rank
        else:
            print(f"Warning: Found multiple LoRA ranks: {unique_ranks}")
            print(f"Using most common rank: {max(set(lora_ranks), key=lora_ranks.count)}")
            return max(set(lora_ranks), key=lora_ranks.count)
    else:
        print("No LoRA layers found in checkpoint")
        return None


def get_config_from_checkpoint_or_default(checkpoint):
    """
    Extract configuration from checkpoint or use defaults.

    Args:
        checkpoint: The loaded checkpoint dictionary

    Returns:
        config object or dict with configuration
    """
    config_data = {}

    # First, try to get config from checkpoint
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        stored_config = checkpoint['config']
        print("Found config in checkpoint")

        # Extract relevant config values
        if isinstance(stored_config, dict):
            config_data = stored_config.copy()
        else:
            # If it's an object, try to extract attributes
            try:
                config_data['bit_widths'] = getattr(stored_config, 'bit_widths', [4, 8, 16])
                config_data['lora_rank_per_bit'] = getattr(stored_config, 'lora_rank_per_bit', None)
                config_data['lora_alpha_per_bit'] = getattr(stored_config, 'lora_alpha_per_bit', None)
                config_data['lora_rank'] = getattr(stored_config, 'lora_rank', 8)
                config_data['lora_alpha'] = getattr(stored_config, 'lora_alpha', 16)
            except Exception as e:
                print(f"Warning: Could not extract config attributes: {e}")

    # If config not in checkpoint or incomplete, use ModelConfig if available
    if CONFIG_AVAILABLE and (not config_data or 'lora_rank_per_bit' not in config_data):
        print("Using ModelConfig from config_qat.py")
        model_config = ModelConfig()
        config_data['bit_widths'] = model_config.bit_widths
        config_data['lora_rank_per_bit'] = model_config.lora_rank_per_bit
        config_data['lora_alpha_per_bit'] = model_config.lora_alpha_per_bit
        config_data['lora_rank'] = model_config.lora_rank
        config_data['lora_alpha'] = model_config.lora_alpha
        config_data['activation_bits_per_bit'] = model_config.activation_bits_per_bit
        config_data['kv_cache_bits_per_bit'] = model_config.kv_cache_bits_per_bit

    # Final fallback to hardcoded defaults
    if 'bit_widths' not in config_data:
        config_data['bit_widths'] = [4, 8, 16]
    if 'lora_rank_per_bit' not in config_data:
        # Note: Using the shared/lora.py defaults which seem to be the actual implementation
        config_data['lora_rank_per_bit'] = {4: 32, 8: 16, 16: 8}
        print("Warning: Using fallback LoRA ranks (4:32, 8:16, 16:8)")

    return config_data


def convert_single_to_switchable_checkpoint(checkpoint_path, output_path=None,
                                           lora_rank_per_bit=None, use_config=True):
    """
    Convert a single-precision QAT checkpoint to switchable multi-precision format
    with LoRA rank adaptation.

    Args:
        checkpoint_path: Path to the original checkpoint
        output_path: Path for the converted checkpoint (optional)
        lora_rank_per_bit: Dict mapping bit-width to LoRA rank (optional)
        use_config: Whether to use config from checkpoint/file (default: True)

    Returns:
        Path to the converted checkpoint
    """
    print(f"\n{'='*70}")
    print("CHECKPOINT CONVERSION V2: With LoRA Rank Adaptation")
    print(f"{'='*70}\n")

    # Load the original checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
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

    # Get configuration
    if use_config:
        config_data = get_config_from_checkpoint_or_default(checkpoint)
        bit_widths = config_data['bit_widths']
        if lora_rank_per_bit is None:
            lora_rank_per_bit = config_data['lora_rank_per_bit']
    else:
        bit_widths = [4, 8, 16]
        if lora_rank_per_bit is None:
            lora_rank_per_bit = {4: 32, 8: 16, 16: 8}

    print(f"Using bit widths: {bit_widths}")
    print(f"Target LoRA ranks per bit-width: {lora_rank_per_bit}")

    # Detect original LoRA rank
    original_rank = detect_lora_rank(old_state_dict)
    if original_rank is None:
        # Try to get from config
        if use_config and 'lora_rank' in config_data:
            original_rank = config_data['lora_rank']
            print(f"Using LoRA rank from config: {original_rank}")
        else:
            print("Warning: Could not detect LoRA rank, assuming 8")
            original_rank = 8

    # Create the new state dict with converted keys
    new_state_dict = {}

    # Statistics
    converted_keys = {
        'quantizers': 0,
        'lora': 0,
        'lora_adapted': 0,
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
            # Convert LoRA adapter keys with rank adaptation
            parts = old_key.split('.')
            lora_idx = parts.index('lora')

            # Determine if this is lora_A or lora_B
            component = parts[lora_idx + 1]  # Should be 'lora_A', 'lora_B', or other

            # Build new keys with lora_adapters.Xbit structure
            for bits in bit_widths:
                new_parts = parts[:lora_idx] + ['lora_adapters', f'{bits}bit'] + parts[lora_idx+1:]
                new_key = '.'.join(new_parts)

                # Adapt rank for lora_A and lora_B matrices
                if component == 'lora_A':
                    target_rank = lora_rank_per_bit[bits]
                    adapted_tensor = adapt_lora_rank(value, original_rank, target_rank, 'A')
                    new_state_dict[new_key] = adapted_tensor
                    converted_keys['lora_adapted'] += 1
                elif component == 'lora_B':
                    target_rank = lora_rank_per_bit[bits]
                    adapted_tensor = adapt_lora_rank(value, original_rank, target_rank, 'B')
                    new_state_dict[new_key] = adapted_tensor
                    converted_keys['lora_adapted'] += 1
                else:
                    # Other LoRA parameters (scaling, etc.) - just copy
                    new_state_dict[new_key] = value.clone()
                    converted_keys['lora'] += 1

        else:
            # Keep other keys unchanged (embeddings, layer norms, etc.)
            new_state_dict[old_key] = value
            converted_keys['unchanged'] += 1

    print(f"\nConversion statistics:")
    print(f"  Quantizer keys converted: {converted_keys['quantizers']}")
    print(f"  LoRA keys converted (no adaptation): {converted_keys['lora']}")
    print(f"  LoRA matrices adapted: {converted_keys['lora_adapted']}")
    print(f"  Unchanged keys: {converted_keys['unchanged']}")
    print(f"  New state dict has {len(new_state_dict)} keys")

    # Update the checkpoint with new state dict
    checkpoint['model_state_dict'] = new_state_dict

    # Add complete configuration to checkpoint
    try:
        if use_config:
            # Store the complete config
            if 'config' not in checkpoint:
                checkpoint['config'] = {}

            checkpoint['config']['bit_widths'] = bit_widths
            checkpoint['config']['lora_rank_per_bit'] = lora_rank_per_bit

            # Add other config values if available
            if 'lora_alpha_per_bit' in config_data:
                checkpoint['config']['lora_alpha_per_bit'] = config_data['lora_alpha_per_bit']
            if 'activation_bits_per_bit' in config_data:
                checkpoint['config']['activation_bits_per_bit'] = config_data['activation_bits_per_bit']
            if 'kv_cache_bits_per_bit' in config_data:
                checkpoint['config']['kv_cache_bits_per_bit'] = config_data['kv_cache_bits_per_bit']

            print(f"\nAdded to config:")
            for key, value in checkpoint['config'].items():
                if isinstance(value, dict) or isinstance(value, list):
                    print(f"  {key}: {value}")
        else:
            checkpoint['bit_widths'] = bit_widths
            checkpoint['lora_rank_per_bit'] = lora_rank_per_bit
            print(f"\nAdded to checkpoint:")
            print(f"  bit_widths: {bit_widths}")
            print(f"  lora_rank_per_bit: {lora_rank_per_bit}")
    except Exception as e:
        print(f"Warning: Could not add metadata to checkpoint: {e}")

    # Determine output path
    if output_path is None:
        path = Path(checkpoint_path)
        output_path = path.parent / f"{path.stem}_switchable_v2{path.suffix}"

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

    # Check LoRA matrix shapes
    print("\nVerifying LoRA matrix shapes:")
    for bits in bit_widths:
        target_rank = lora_rank_per_bit[bits]
        # Find a sample lora_A key for this bit width
        sample_keys = [k for k in verify_state_dict.keys()
                      if f'lora_adapters.{bits}bit.lora_A' in k]
        if sample_keys:
            sample_tensor = verify_state_dict[sample_keys[0]]
            actual_rank = sample_tensor.shape[1]
            if actual_rank == target_rank:
                print(f"  ✓ {bits}-bit LoRA rank: {actual_rank} (matches target)")
            else:
                print(f"  ✗ {bits}-bit LoRA rank: {actual_rank} (expected {target_rank})")

    if has_quantizers and has_lora_adapters:
        print("\n✓ Conversion successful!")
        print(f"  - Found multi-bit quantizers")
        print(f"  - Found multi-bit LoRA adapters with adapted ranks")
    else:
        print("\n⚠️ Warning: Conversion may be incomplete")
        print(f"  - Multi-bit quantizers found: {has_quantizers}")
        print(f"  - Multi-bit LoRA adapters found: {has_lora_adapters}")

    return output_path


def analyze_checkpoint_structure(checkpoint_path):
    """Analyze and print the structure of a checkpoint with LoRA rank info."""
    print(f"\n{'='*70}")
    print(f"ANALYZING CHECKPOINT: {checkpoint_path}")
    print(f"{'='*70}\n")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Check what's in the checkpoint
    if isinstance(checkpoint, dict):
        print("Checkpoint contents:")
        for key in checkpoint.keys():
            if key not in ['model_state_dict', 'model']:
                if isinstance(checkpoint[key], (int, float, str, list, dict)):
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

    # Analyze LoRA ranks
    print("\nAnalyzing LoRA ranks:")
    lora_ranks = {}
    for key, tensor in state_dict.items():
        if 'lora_A' in key:
            rank = tensor.shape[1]
            # Extract bit width if present
            if 'bit' in key:
                for part in key.split('.'):
                    if 'bit' in part:
                        bit_width = part
                        if bit_width not in lora_ranks:
                            lora_ranks[bit_width] = []
                        lora_ranks[bit_width].append(rank)
                        break
            else:
                if 'single' not in lora_ranks:
                    lora_ranks['single'] = []
                lora_ranks['single'].append(rank)

    if lora_ranks:
        for bit_width, ranks in lora_ranks.items():
            unique_ranks = list(set(ranks))
            print(f"  {bit_width}: rank {unique_ranks[0]} ({len(ranks)} LoRA layers)")
    else:
        print("  No LoRA layers found")

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

    return patterns, lora_ranks


def main():
    parser = argparse.ArgumentParser(
        description='Convert single-precision QAT checkpoint to multi-precision format with LoRA rank adaptation'
    )
    parser.add_argument('checkpoint_path', type=str,
                        help='Path to the checkpoint file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for converted checkpoint')
    parser.add_argument('--analyze', action='store_true',
                        help='Only analyze checkpoint structure without conversion')
    parser.add_argument('--lora-ranks', type=str, default=None,
                        help='LoRA ranks per bit-width as JSON string, e.g., \'{"4": 32, "8": 16, "16": 8}\'')
    parser.add_argument('--no-config', action='store_true',
                        help='Do not use config from checkpoint or config file')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found: {args.checkpoint_path}")
        sys.exit(1)

    if args.analyze:
        # Just analyze the checkpoint structure
        analyze_checkpoint_structure(args.checkpoint_path)
    else:
        # Parse custom LoRA ranks if provided
        lora_rank_per_bit = None
        if args.lora_ranks:
            try:
                lora_rank_per_bit = json.loads(args.lora_ranks)
                # Convert string keys to int
                lora_rank_per_bit = {int(k): v for k, v in lora_rank_per_bit.items()}
                print(f"Using custom LoRA ranks: {lora_rank_per_bit}")
            except Exception as e:
                print(f"Error parsing LoRA ranks: {e}")
                print("Using default ranks: {4: 32, 8: 16, 16: 8}")

        # Convert the checkpoint
        output_path = convert_single_to_switchable_checkpoint(
            args.checkpoint_path,
            args.output,
            lora_rank_per_bit,
            use_config=not args.no_config
        )

        print(f"\n{'='*70}")
        print("CONVERSION COMPLETE")
        print(f"{'='*70}")
        print(f"\nConverted checkpoint saved to: {output_path}")
        print("\nNext steps:")
        print("1. Test with diagnostics:")
        print(f"   cd test && ./run_all_diagnostics.sh {output_path}")
        print("2. Test checkpoint loading:")
        print(f"   python test/test_checkpoint_fix.py {output_path}")
        print("3. Run evaluation:")
        print(f"   python part3_evaluation/main_llm_qat_eval.py --model_path {output_path}")


if __name__ == "__main__":
    main()
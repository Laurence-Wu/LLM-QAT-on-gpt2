#!/usr/bin/env python3
"""
Check if the SP model was initialized from pretrained GPT-2 weights.
"""

import torch
import sys
import os

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(os.path.abspath(__file__))
part1_dir = os.path.join(parent_dir, 'part1_switchable_precision')
if part1_dir not in sys.path:
    sys.path.insert(0, part1_dir)

from part1_switchable_precision.models_sp import SPLMHeadModel
from transformers import GPT2Config, GPT2LMHeadModel


def check_pretrained_initialization(checkpoint_path):
    """Check if model weights match pretrained GPT-2."""
    print("="*70)
    print("  PRETRAINED INITIALIZATION CHECK")
    print("="*70)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, dict):
        print("ERROR: Invalid checkpoint format")
        return

    state_dict = checkpoint['model_state_dict']

    # Load pretrained GPT-2
    print("\nLoading pretrained GPT-2...")
    pretrained = GPT2LMHeadModel.from_pretrained('gpt2')
    pretrained_state = pretrained.state_dict()

    print(f"Pretrained model has {len(pretrained_state)} parameters")
    print(f"Checkpoint has {len(state_dict)} parameters")

    # Key mappings from GPT-2 to SP model
    key_mappings = {
        # Embeddings
        'transformer.wte.weight': 'transformer.wte.weight',
        'transformer.wpe.weight': 'transformer.wpe.weight',

        # Final layer norm
        'transformer.ln_f.weight': 'transformer.ln_f.weight',
        'transformer.ln_f.bias': 'transformer.ln_f.bias',

        # LM head
        'lm_head.weight': 'lm_head.weight',
    }

    # Add transformer block mappings
    for i in range(12):  # GPT-2 has 12 layers
        # Layer norms
        key_mappings[f'transformer.h.{i}.ln_1.weight'] = f'transformer.h.{i}.ln_1.weight'
        key_mappings[f'transformer.h.{i}.ln_1.bias'] = f'transformer.h.{i}.ln_1.bias'
        key_mappings[f'transformer.h.{i}.ln_2.weight'] = f'transformer.h.{i}.ln_2.weight'
        key_mappings[f'transformer.h.{i}.ln_2.bias'] = f'transformer.h.{i}.ln_2.bias'

        # Attention (GPT-2 uses conv1d, SP uses linear)
        key_mappings[f'transformer.h.{i}.attn.c_attn.weight'] = f'transformer.h.{i}.attn.c_attn.linear.weight'
        key_mappings[f'transformer.h.{i}.attn.c_attn.bias'] = f'transformer.h.{i}.attn.c_attn.linear.bias'
        key_mappings[f'transformer.h.{i}.attn.c_proj.weight'] = f'transformer.h.{i}.attn.c_proj.linear.weight'
        key_mappings[f'transformer.h.{i}.attn.c_proj.bias'] = f'transformer.h.{i}.attn.c_proj.linear.bias'

        # MLP
        key_mappings[f'transformer.h.{i}.mlp.c_fc.weight'] = f'transformer.h.{i}.mlp.c_fc.linear.weight'
        key_mappings[f'transformer.h.{i}.mlp.c_fc.bias'] = f'transformer.h.{i}.mlp.c_fc.linear.bias'
        key_mappings[f'transformer.h.{i}.mlp.c_proj.weight'] = f'transformer.h.{i}.mlp.c_proj.linear.weight'
        key_mappings[f'transformer.h.{i}.mlp.c_proj.bias'] = f'transformer.h.{i}.mlp.c_proj.linear.bias'

    print("\n" + "="*70)
    print("  WEIGHT COMPARISON WITH PRETRAINED GPT-2")
    print("="*70)

    matches = 0
    mismatches = 0
    missing = 0

    critical_weights = ['transformer.wte.weight', 'transformer.wpe.weight', 'lm_head.weight']

    for pretrained_key, sp_key in key_mappings.items():
        if pretrained_key in pretrained_state and sp_key in state_dict:
            pretrained_weight = pretrained_state[pretrained_key]
            sp_weight = state_dict[sp_key]

            # Handle shape differences (GPT-2 uses Conv1D which transposes weights)
            if 'c_attn' in pretrained_key or 'c_proj' in pretrained_key or 'c_fc' in pretrained_key:
                if pretrained_weight.dim() == 1:  # bias
                    weights_match = torch.allclose(pretrained_weight, sp_weight, atol=1e-5)
                else:  # weight - need to transpose
                    pretrained_weight_t = pretrained_weight.t()
                    weights_match = torch.allclose(pretrained_weight_t, sp_weight, atol=1e-5)
            else:
                weights_match = torch.allclose(pretrained_weight, sp_weight, atol=1e-5)

            if weights_match:
                matches += 1
                if pretrained_key in critical_weights:
                    print(f"✅ {pretrained_key}: MATCHES pretrained")
            else:
                mismatches += 1
                if pretrained_key in critical_weights:
                    diff = (pretrained_weight - sp_weight).abs().max().item() if pretrained_weight.shape == sp_weight.shape else float('inf')
                    print(f"❌ {pretrained_key}: DIFFERS from pretrained (max diff: {diff:.6f})")
        else:
            missing += 1
            if sp_key not in state_dict and pretrained_key in critical_weights:
                print(f"⚠️ {sp_key}: MISSING from checkpoint")

    # Sample some attention/MLP weights
    print("\nSample Transformer Block Weights:")
    for i in [0, 6, 11]:  # First, middle, last
        attn_key = f'transformer.h.{i}.attn.c_attn.linear.weight'
        pretrained_attn_key = f'transformer.h.{i}.attn.c_attn.weight'

        if attn_key in state_dict and pretrained_attn_key in pretrained_state:
            sp_weight = state_dict[attn_key]
            pretrained_weight = pretrained_state[pretrained_attn_key].t()  # Transpose for comparison

            if torch.allclose(sp_weight, pretrained_weight, atol=1e-5):
                print(f"  Block {i} attention: ✅ Matches pretrained")
            else:
                diff = (sp_weight - pretrained_weight).abs().max().item()
                print(f"  Block {i} attention: ❌ Differs (max diff: {diff:.6f})")

    print(f"\n" + "="*70)
    print(f"  SUMMARY")
    print(f"="*70)
    print(f"Weights matching pretrained: {matches}")
    print(f"Weights different from pretrained: {mismatches}")
    print(f"Weights missing: {missing}")

    match_percentage = (matches / (matches + mismatches)) * 100 if (matches + mismatches) > 0 else 0

    if match_percentage < 10:
        print(f"\n❌ CRITICAL: Only {match_percentage:.1f}% of weights match pretrained!")
        print("   The model was likely NOT initialized from pretrained GPT-2.")
        print("   This explains the poor performance.")
    elif match_percentage < 50:
        print(f"\n⚠️ WARNING: Only {match_percentage:.1f}% of weights match pretrained.")
        print("   The model may have been heavily modified or corrupted during training.")
    else:
        print(f"\n✅ {match_percentage:.1f}% of weights match pretrained GPT-2.")

    # Check weight statistics
    print(f"\n" + "="*70)
    print(f"  WEIGHT STATISTICS")
    print(f"="*70)

    # Check embedding weights
    if 'transformer.wte.weight' in state_dict:
        wte = state_dict['transformer.wte.weight']
        print(f"\nWord Embeddings (wte):")
        print(f"  Mean: {wte.mean().item():.6f}")
        print(f"  Std: {wte.std().item():.6f}")
        print(f"  Min: {wte.min().item():.6f}")
        print(f"  Max: {wte.max().item():.6f}")

        # Check if it looks like random init (Xavier/He)
        fan_in = wte.shape[1] if wte.dim() > 1 else wte.shape[0]
        expected_std_xavier = (2.0 / (fan_in + wte.shape[0])) ** 0.5
        expected_std_he = (2.0 / fan_in) ** 0.5

        actual_std = wte.std().item()

        if abs(actual_std - expected_std_xavier) < 0.01:
            print(f"  ⚠️ Looks like Xavier initialization (std ≈ {expected_std_xavier:.4f})")
        elif abs(actual_std - expected_std_he) < 0.01:
            print(f"  ⚠️ Looks like He initialization (std ≈ {expected_std_he:.4f})")
        elif actual_std < 0.01:
            print(f"  ⚠️ Very low variance - possible initialization issue")

    # Compare with pretrained
    if 'transformer.wte.weight' in pretrained_state:
        pretrained_wte = pretrained_state['transformer.wte.weight']
        print(f"\nPretrained Word Embeddings:")
        print(f"  Mean: {pretrained_wte.mean().item():.6f}")
        print(f"  Std: {pretrained_wte.std().item():.6f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Check if model was initialized from pretrained weights')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    args = parser.parse_args()

    check_pretrained_initialization(args.checkpoint)


if __name__ == "__main__":
    main()
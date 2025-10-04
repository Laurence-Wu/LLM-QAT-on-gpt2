"""
Test parameter count to verify True CPT implementation.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
part2_dir = os.path.join(parent_dir, 'part2_cyclic_precision_training')
sys.path.insert(0, part2_dir)

import torch
from config_cpt import get_config
from cpt_model import CPTModel

def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def count_lora_parameters(model):
    """Count LoRA parameters specifically."""
    lora_params = 0
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_params += param.numel()
    return lora_params

def main():
    print("\n" + "="*70)
    print("PARAMETER COUNT VERIFICATION")
    print("="*70)

    # Get config
    config = get_config()

    print(f"\nConfiguration:")
    print(f"  Shared LoRA rank: {config['model'].shared_lora_rank}")
    print(f"  Shared LoRA alpha: {config['model'].shared_lora_alpha}")
    print(f"  Bit widths: {config['model'].bit_widths}")
    print(f"  Number of layers: {config['model'].n_layer}")

    # Create model
    print(f"\nCreating CPTModel...")
    model = CPTModel(config)

    # Load pretrained weights and freeze base model
    print(f"\nLoading pretrained weights and freezing base model...")
    sys.path.insert(0, part2_dir)
    from main_cpt import load_pretrained_weights
    load_pretrained_weights(model, config['model'])

    # Count parameters
    total_params, trainable_params = count_parameters(model)
    lora_params = count_lora_parameters(model)

    print(f"\nParameter Statistics:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  LoRA parameters:      {lora_params:,}")

    # Verify architecture
    print(f"\nArchitecture Verification:")

    # Check shared LoRA
    sample_linear = None
    for module in model.modules():
        if hasattr(module, 'shared_lora'):
            sample_linear = module
            break

    if sample_linear:
        print(f"  ✓ Found shared_lora")
        print(f"    lora_A shape: {sample_linear.shared_lora.lora_A.shape}")
        print(f"    lora_B shape: {sample_linear.shared_lora.lora_B.shape}")
        print(f"    Number of LoRA weight quantizers: {len(sample_linear.lora_weight_quantizers)}")

        # Calculate expected LoRA params per layer
        # Each CPTLinear has: lora_A (in_features x rank) + lora_B (out_features x rank)
        # Attention: c_attn (768, 2304) and c_proj (768, 768)
        # MLP: fc_in (768, 3072) and fc_out (3072, 768)
        # lm_head: (768, 50257)

        expected_lora_per_layer = (
            (768 * 16 + 2304 * 16) +  # c_attn
            (768 * 16 + 768 * 16) +    # c_proj
            (768 * 16 + 3072 * 16) +   # fc_in
            (3072 * 16 + 768 * 16)     # fc_out
        )
        expected_lm_head = 768 * 16 + 50257 * 16
        expected_total_lora = expected_lora_per_layer * 12 + expected_lm_head

        print(f"\n  Expected LoRA params: ~{expected_total_lora:,}")
        print(f"  Actual LoRA params:   {lora_params:,}")

        # Verify trainable percentage
        trainable_percentage = (trainable_params / total_params) * 100
        print(f"\n  Trainable percentage: {trainable_percentage:.2f}%")

        # Check if this is True CPT (single LoRA per layer)
        if trainable_params < 5_000_000:  # Less than 5M trainable params
            print(f"\n  ✓ SUCCESS: True CPT implementation")
            print(f"    Single shared LoRA per layer, quantized at multiple precisions")
            print(f"    Parameter reduction: ~{((218_000_000 - trainable_params) / 218_000_000 * 100):.1f}%")
            return True
        else:
            print(f"\n  ⚠️  WARNING: High trainable parameter count")
            print(f"    Expected < 5M for True CPT")
            print(f"    May still be using multi-adapter approach")
            return False
    else:
        print(f"  ✗ ERROR: No shared_lora found")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

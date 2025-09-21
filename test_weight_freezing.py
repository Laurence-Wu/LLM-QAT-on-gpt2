"""Test script to verify weight freezing/unfreezing for different precisions."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from part1_switchable_precision.config_sp import ModelConfig
from part1_switchable_precision.main_sp import initialize_model

def count_trainable_params(model):
    """Count trainable and frozen parameters."""
    trainable = 0
    frozen = 0
    details = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable += param.numel()
            details[name] = "trainable"
        else:
            frozen += param.numel()
            details[name] = "frozen"

    return trainable, frozen, details

def test_precision_switching():
    """Test that weights are properly frozen/unfrozen for different precisions."""
    print("=" * 80)
    print("Testing Weight Freezing/Unfreezing for Different Precisions")
    print("=" * 80)

    # Initialize model
    config = ModelConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = initialize_model(config, device)

    print("\n" + "=" * 80)

    # Test each precision
    for bits in [4, 8, 16, 32]:
        print(f"\n{'='*40}")
        print(f"Testing {bits}-bit precision")
        print(f"{'='*40}")

        model.set_precision(bits)
        trainable, frozen, details = count_trainable_params(model)
        total = trainable + frozen

        print(f"Trainable params: {trainable:,} ({100*trainable/total:.1f}%)")
        print(f"Frozen params: {frozen:,} ({100*frozen/total:.1f}%)")

        # Check key components
        print(f"\nKey component status:")

        # Embeddings (should always be frozen)
        embed_status = details.get('transformer.wte.weight', 'NOT FOUND')
        print(f"  - Embeddings (wte): {embed_status}")
        assert embed_status == "frozen", f"ERROR: Embeddings should be frozen for {bits}-bit"

        # LM Head (should always be frozen since it's tied to embeddings)
        lm_head_status = details.get('lm_head.weight', 'NOT FOUND')
        print(f"  - LM Head: {lm_head_status}")
        assert lm_head_status == "frozen", f"ERROR: LM Head should be frozen for {bits}-bit"

        # Check transformer blocks for 32-bit
        if bits == 32:
            # For 32-bit teacher, transformer weights should be trainable
            attn_weight = details.get('transformer.h.0.attn.c_attn.linear.weight', 'NOT FOUND')
            print(f"  - Attention weights: {attn_weight}")
            assert attn_weight == "trainable", f"ERROR: Attention weights should be trainable for 32-bit teacher"

            mlp_weight = details.get('transformer.h.0.mlp.c_fc.linear.weight', 'NOT FOUND')
            print(f"  - MLP weights: {mlp_weight}")
            assert mlp_weight == "trainable", f"ERROR: MLP weights should be trainable for 32-bit teacher"

            ln_weight = details.get('transformer.h.0.ln_1.weight', 'NOT FOUND')
            print(f"  - LayerNorm weights: {ln_weight}")
            assert ln_weight == "trainable", f"ERROR: LayerNorm weights should be trainable for 32-bit teacher"

            final_ln = details.get('transformer.ln_f.weight', 'NOT FOUND')
            print(f"  - Final LayerNorm: {final_ln}")
            assert final_ln == "trainable", f"ERROR: Final LayerNorm should be trainable for 32-bit teacher"
        else:
            # For student models, only LoRA should be trainable
            attn_weight = details.get('transformer.h.0.attn.c_attn.linear.weight', 'NOT FOUND')
            print(f"  - Attention weights: {attn_weight}")
            assert attn_weight == "frozen", f"ERROR: Attention weights should be frozen for {bits}-bit student"

            # Check if LoRA parameters exist and are trainable
            lora_found = False
            for name, status in details.items():
                if 'lora' in name.lower() and status == "trainable":
                    lora_found = True
                    break

            if bits < 32:  # LoRA should exist for student models
                print(f"  - LoRA adapters: {'found and trainable' if lora_found else 'NOT FOUND or frozen'}")
                assert lora_found, f"ERROR: LoRA adapters should be trainable for {bits}-bit student"

    print("\n" + "=" * 80)
    print("✅ All tests passed! Weight freezing/unfreezing works correctly.")
    print("=" * 80)

if __name__ == "__main__":
    test_precision_switching()
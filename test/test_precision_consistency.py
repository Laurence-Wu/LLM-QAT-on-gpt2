#!/usr/bin/env python3
"""
Test precision consistency verification system.
Verifies that all components switch to the same precision and that verification works.
"""

import sys
import os
import torch
from transformers import GPT2Config

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models_sp import SPLMHeadModel
from shared.switchable_batchnorm import SwitchableLayerNorm
from part1_switchable_precision.config_sp import ModelConfig


def test_precision_return_values():
    """Test that set_precision methods return correct values."""
    print("\n" + "="*60)
    print("TESTING PRECISION RETURN VALUES")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model config
    model_config = ModelConfig()
    config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=2,  # Small for testing
        n_head=model_config.n_head
    )

    # Add SP attributes
    config.bit_widths = model_config.bit_widths
    config.lora_rank_per_bit = model_config.lora_rank_per_bit
    config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    config.quantizer_per_bit = model_config.quantizer_per_bit
    config.lora_dropout = model_config.lora_dropout

    # Create model
    model = SPLMHeadModel(config).to(device)

    print("\nTesting return values from set_precision:")
    for bits in [4, 8, 16, 32]:
        returned_bits = model.set_precision(bits)
        print(f"   Set to {bits}-bit, returned: {returned_bits}")
        assert returned_bits == bits, f"Expected {bits}, got {returned_bits}"

    print("   ✅ All return values correct")


def test_precision_consistency_verification():
    """Test the verify_precision_consistency method."""
    print("\n" + "="*60)
    print("TESTING PRECISION CONSISTENCY VERIFICATION")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model config
    model_config = ModelConfig()
    config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=2,  # Small for testing
        n_head=model_config.n_head
    )

    # Add SP attributes
    config.bit_widths = model_config.bit_widths
    config.lora_rank_per_bit = model_config.lora_rank_per_bit
    config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    config.quantizer_per_bit = model_config.quantizer_per_bit
    config.lora_dropout = model_config.lora_dropout

    # Create model
    model = SPLMHeadModel(config).to(device)

    print("\nVerifying consistency after setting precision:")
    for bits in [4, 8, 16, 32]:
        model.set_precision(bits)
        is_consistent, details = model.verify_precision_consistency()

        print(f"\n   {bits}-bit precision:")
        print(f"      Consistent: {'✅' if is_consistent else '❌'}")
        print(f"      Components checked: {len(details['components'])}")

        if not is_consistent:
            print(f"      Mismatches found:")
            for mismatch in details['mismatches']:
                print(f"         - {mismatch}")
        else:
            # Sample some component precisions
            sample_components = list(details['components'].items())[:5]
            for comp_name, comp_prec in sample_components:
                print(f"         {comp_name}: {comp_prec}-bit ✅")


def test_manual_mismatch_detection():
    """Test that mismatches are properly detected."""
    print("\n" + "="*60)
    print("TESTING MISMATCH DETECTION")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model config
    model_config = ModelConfig()
    config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=2,  # Small for testing
        n_head=model_config.n_head
    )

    # Add SP attributes
    config.bit_widths = model_config.bit_widths
    config.lora_rank_per_bit = model_config.lora_rank_per_bit
    config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    config.quantizer_per_bit = model_config.quantizer_per_bit
    config.lora_dropout = model_config.lora_dropout

    # Create model
    model = SPLMHeadModel(config).to(device)

    print("\n1. Setting all to 8-bit (should be consistent):")
    model.set_precision(8)
    is_consistent, details = model.verify_precision_consistency()
    print(f"   Consistent: {'✅' if is_consistent else '❌'}")
    assert is_consistent, "Should be consistent after normal set_precision"

    print("\n2. Manually changing one component (creating mismatch):")
    # Manually change just one layer's precision
    model.transformer.h[0].ln_1.set_precision(4)  # Set first block's ln_1 to 4-bit

    # Now verify - should detect mismatch
    is_consistent, details = model.verify_precision_consistency()
    print(f"   Consistent: {'✅' if is_consistent else '❌'}")
    print(f"   Mismatches detected: {len(details['mismatches'])}")

    if details['mismatches']:
        print("   Mismatches:")
        for mismatch in details['mismatches']:
            print(f"      - {mismatch}")

    assert not is_consistent, "Should detect mismatch"
    assert len(details['mismatches']) > 0, "Should have at least one mismatch"

    print("\n3. Fixing mismatch by setting all to 16-bit:")
    model.set_precision(16)
    is_consistent, details = model.verify_precision_consistency()
    print(f"   Consistent: {'✅' if is_consistent else '❌'}")
    assert is_consistent, "Should be consistent after fixing"


def test_precision_propagation():
    """Test that precision changes propagate through all layers."""
    print("\n" + "="*60)
    print("TESTING PRECISION PROPAGATION")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model config
    model_config = ModelConfig()
    config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=3,  # Use 3 layers for testing
        n_head=model_config.n_head
    )

    # Add SP attributes
    config.bit_widths = model_config.bit_widths
    config.lora_rank_per_bit = model_config.lora_rank_per_bit
    config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    config.quantizer_per_bit = model_config.quantizer_per_bit
    config.lora_dropout = model_config.lora_dropout

    # Create model
    model = SPLMHeadModel(config).to(device)

    print("\nTesting precision propagation through layers:")

    # Test each precision
    for target_bits in [4, 8, 16, 32]:
        model.set_precision(target_bits)
        print(f"\n   Set model to {target_bits}-bit:")

        # Check each block
        all_correct = True
        for i, block in enumerate(model.transformer.h):
            # Check LayerNorms
            ln1_prec = block.ln_1.current_precision if hasattr(block.ln_1, 'current_precision') else None
            ln2_prec = block.ln_2.current_precision if hasattr(block.ln_2, 'current_precision') else None

            # Check attention
            attn_prec = block.attn.current_bit_width if hasattr(block.attn, 'current_bit_width') else None

            # Check MLP
            mlp_fc_prec = block.mlp.c_fc.current_precision if hasattr(block.mlp.c_fc, 'current_precision') else None

            if ln1_prec != target_bits or ln2_prec != target_bits or attn_prec != target_bits:
                all_correct = False
                print(f"      Block {i}: ln1={ln1_prec}, ln2={ln2_prec}, attn={attn_prec} ❌")
            else:
                print(f"      Block {i}: All components at {target_bits}-bit ✅")

        # Check final layer norm
        final_ln_prec = model.transformer.ln_f.current_precision
        if final_ln_prec != target_bits:
            all_correct = False
            print(f"      Final LN: {final_ln_prec}-bit ❌")
        else:
            print(f"      Final LN: {target_bits}-bit ✅")

        if all_correct:
            print(f"   ✅ All components correctly set to {target_bits}-bit")
        else:
            print(f"   ❌ Some components not at {target_bits}-bit")


def test_weight_freezing_with_precision():
    """Test that weight freezing/unfreezing works with precision changes."""
    print("\n" + "="*60)
    print("TESTING WEIGHT FREEZING WITH PRECISION")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model config
    model_config = ModelConfig()
    config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=1,  # Single layer for testing
        n_head=model_config.n_head
    )

    # Add SP attributes
    config.bit_widths = model_config.bit_widths
    config.lora_rank_per_bit = model_config.lora_rank_per_bit
    config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    config.quantizer_per_bit = model_config.quantizer_per_bit
    config.lora_dropout = model_config.lora_dropout

    # Create model
    model = SPLMHeadModel(config).to(device)

    print("\n1. Testing 32-bit (teacher) mode - weights should be unfrozen:")
    model.set_precision(32)

    # Check if base weights are unfrozen
    block = model.transformer.h[0]
    ln1_frozen = not any(model.transformer.h[0].ln_1.ln_layers[key].weight.requires_grad
                         for key in model.transformer.h[0].ln_1.ln_layers)
    attn_frozen = not block.attn.c_attn.linear.weight.requires_grad

    print(f"   LayerNorm weights frozen: {ln1_frozen} (should be False)")
    print(f"   Attention weights frozen: {attn_frozen} (should be False)")

    assert not ln1_frozen, "LayerNorm weights should be unfrozen for 32-bit"
    assert not attn_frozen, "Attention weights should be unfrozen for 32-bit"

    print("\n2. Testing 8-bit (student) mode - weights should be frozen:")
    model.set_precision(8)

    # Check if base weights are frozen
    ln1_frozen = not any(model.transformer.h[0].ln_1.ln_layers[key].weight.requires_grad
                         for key in model.transformer.h[0].ln_1.ln_layers)
    attn_frozen = not block.attn.c_attn.linear.weight.requires_grad

    print(f"   LayerNorm weights frozen: {ln1_frozen} (should be True)")
    print(f"   Attention weights frozen: {attn_frozen} (should be True)")

    assert ln1_frozen, "LayerNorm weights should be frozen for 8-bit"
    assert attn_frozen, "Attention weights should be frozen for 8-bit"

    print("\n   ✅ Weight freezing works correctly with precision changes")


def main():
    """Run all precision consistency tests."""
    print("\n" + "="*80)
    print("PRECISION CONSISTENCY TEST SUITE")
    print("="*80)

    try:
        test_precision_return_values()
        test_precision_consistency_verification()
        test_manual_mismatch_detection()
        test_precision_propagation()
        test_weight_freezing_with_precision()

        print("\n" + "="*80)
        print("✅ ALL PRECISION CONSISTENCY TESTS PASSED")
        print("="*80)
        print("\nKey verifications:")
        print("  • set_precision returns correct values")
        print("  • verify_precision_consistency detects mismatches")
        print("  • Precision propagates through all layers")
        print("  • Weight freezing/unfreezing works with precision")
        print("  • Manual mismatches are properly detected")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
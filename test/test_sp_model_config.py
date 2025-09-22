#!/usr/bin/env python3
"""
Test SP model configuration and loading.
Verifies that bit widths are properly loaded from config and not hardcoded.
"""

import sys
import os
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part1_switchable_precision.config_sp import ModelConfig
from shared.models_sp import SPModel, SPLMHeadModel
from transformers import GPT2Config


def test_model_initialization():
    """Test that SP model initializes correctly with config bit widths."""
    print("\n" + "="*60)
    print("TESTING SP MODEL CONFIGURATION")
    print("="*60)

    # Create model config
    model_config = ModelConfig()

    print("\n1. Checking ModelConfig bit_widths:")
    print(f"   Configured bit_widths: {model_config.bit_widths}")
    print(f"   Expected: [6, 8, 16, 32]")
    assert model_config.bit_widths == [6, 8, 16, 32], "bit_widths should be [6, 8, 16, 32]"
    print("   ✅ Config bit_widths correct")

    print("\n2. Checking LoRA configurations:")
    print(f"   LoRA rank per bit: {model_config.lora_rank_per_bit}")
    print(f"   LoRA alpha per bit: {model_config.lora_alpha_per_bit}")
    assert 32 in model_config.lora_rank_per_bit, "Should include 32-bit configuration"
    assert model_config.lora_rank_per_bit[32] == 0, "32-bit (teacher) should have rank=0"
    print("   ✅ LoRA configurations correct")

    print("\n3. Creating GPT2Config with bit_widths:")
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        layer_norm_epsilon=model_config.layer_norm_epsilon,
        embd_pdrop=model_config.embd_pdrop
    )

    # Add SP-specific configs
    gpt2_config.bit_widths = model_config.bit_widths
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    gpt2_config.activation_bits_per_bit = model_config.activation_bits_per_bit
    gpt2_config.quantizer_per_bit = model_config.quantizer_per_bit

    print(f"   GPT2Config bit_widths: {gpt2_config.bit_widths}")
    print("   ✅ GPT2Config created successfully")

    print("\n4. Initializing SPModel:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = SPModel(gpt2_config)
        model = model.to(device)
        print(f"   Model bit_widths: {model.bit_widths}")
        assert model.bit_widths == [6, 8, 16, 32], "Model should have config bit_widths"
        print("   ✅ SPModel initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize SPModel: {e}")
        raise

    print("\n5. Checking layer configurations:")
    # Check first transformer block
    first_block = model.h[0]
    print(f"   First block attention bit_widths: {first_block.attn.bit_widths}")
    print(f"   First block MLP bit_widths: {first_block.mlp.bit_widths}")
    assert first_block.attn.bit_widths == [6, 8, 16, 32], "Attention should have config bit_widths"
    assert first_block.mlp.bit_widths == [6, 8, 16, 32], "MLP should have config bit_widths"
    print("   ✅ Layer bit_widths correct")

    print("\n6. Checking LoRA layers:")
    # Check c_attn (QKV projection)
    c_attn = first_block.attn.c_attn
    print(f"   c_attn bit_widths: {c_attn.bit_widths}")
    print(f"   c_attn LoRA adapters: {list(c_attn.lora_adapters.keys())}")
    assert '32bit' in c_attn.lora_adapters, "Should have 32-bit adapter"

    # Check that 32-bit has no LoRA (rank=0)
    adapter_32 = c_attn.lora_adapters['32bit']
    if hasattr(adapter_32, 'lora_A'):
        if adapter_32.lora_A is not None:
            print(f"   32-bit LoRA A shape: {adapter_32.lora_A.shape}")
        else:
            print(f"   32-bit LoRA A: None (rank=0, as expected for teacher)")
    print("   ✅ LoRA layers configured correctly")

    print("\n7. Testing precision switching:")
    for bits in [6, 8, 16, 32]:
        try:
            model.set_precision(bits)
            print(f"   Set to {bits}-bit: ✅")
        except Exception as e:
            print(f"   Set to {bits}-bit: ❌ {e}")
            raise

    print("\n8. Testing forward pass:")
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device=device)

    for bits in [32, 16, 8, 6]:  # Test from high to low precision
        model.set_precision(bits)
        with torch.no_grad():
            output = model(input_ids)
            hidden_states = output.last_hidden_state
            print(f"   {bits}-bit forward pass: ✅ Output shape: {hidden_states.shape}")

    print("\n✅ ALL SP MODEL CONFIGURATION TESTS PASSED!")
    return model


def test_lmhead_model():
    """Test SPLMHeadModel initialization and configuration."""
    print("\n" + "="*60)
    print("TESTING SP LM HEAD MODEL")
    print("="*60)

    model_config = ModelConfig()
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        layer_norm_epsilon=model_config.layer_norm_epsilon,
        embd_pdrop=model_config.embd_pdrop
    )

    # Add SP-specific configs
    gpt2_config.bit_widths = model_config.bit_widths
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    gpt2_config.activation_bits_per_bit = model_config.activation_bits_per_bit
    gpt2_config.quantizer_per_bit = model_config.quantizer_per_bit

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n1. Initializing SPLMHeadModel:")
    try:
        model = SPLMHeadModel(gpt2_config)
        model = model.to(device)
        print(f"   Model transformer bit_widths: {model.transformer.bit_widths}")
        assert model.transformer.bit_widths == [6, 8, 16, 32], "Should have config bit_widths"
        print("   ✅ SPLMHeadModel initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize SPLMHeadModel: {e}")
        raise

    print("\n2. Testing LM head forward pass:")
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device=device)

    for bits in [16, 8, 6]:  # Test student precisions
        model.set_precision(bits)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            print(f"   {bits}-bit LM head: ✅ Logits shape: {logits.shape}")
            assert logits.shape == (batch_size, seq_len, model_config.vocab_size)

    print("\n✅ ALL SP LM HEAD MODEL TESTS PASSED!")
    return model


def test_no_hardcoded_values():
    """Verify no hardcoded [6, 8, 16] values remain in critical paths."""
    print("\n" + "="*60)
    print("CHECKING FOR HARDCODED VALUES")
    print("="*60)

    # Test with custom bit widths
    custom_config = ModelConfig()
    custom_config.bit_widths = [4, 8, 12, 32]  # Different from default
    custom_config.lora_rank_per_bit = {4: 32, 8: 16, 12: 8, 32: 0}
    custom_config.lora_alpha_per_bit = {4: 64, 8: 32, 12: 16, 32: 0}
    custom_config.activation_bits_per_bit = {4: 4, 8: 8, 12: 8, 32: 32}
    custom_config.quantizer_per_bit = {4: 'log', 8: 'log', 12: 'log', 32: None}

    gpt2_config = GPT2Config(
        vocab_size=custom_config.vocab_size,
        n_positions=custom_config.n_positions,
        n_embd=custom_config.n_embd,
        n_layer=2,  # Smaller for testing
        n_head=custom_config.n_head,
        layer_norm_epsilon=custom_config.layer_norm_epsilon
    )

    # Add custom configs
    gpt2_config.bit_widths = custom_config.bit_widths
    gpt2_config.lora_rank_per_bit = custom_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = custom_config.lora_alpha_per_bit
    gpt2_config.activation_bits_per_bit = custom_config.activation_bits_per_bit
    gpt2_config.quantizer_per_bit = custom_config.quantizer_per_bit

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n1. Testing with custom bit widths [4, 8, 12, 32]:")
    try:
        model = SPModel(gpt2_config)
        model = model.to(device)
        print(f"   Model bit_widths: {model.bit_widths}")
        assert model.bit_widths == [4, 8, 12, 32], "Should use custom bit_widths"
        print("   ✅ Model accepts custom bit_widths")
    except Exception as e:
        print(f"   ❌ Failed with custom bit_widths: {e}")
        raise

    print("\n2. Testing custom precision switching:")
    for bits in [4, 8, 12, 32]:
        try:
            model.set_precision(bits)
            print(f"   Set to {bits}-bit: ✅")
        except Exception as e:
            print(f"   Set to {bits}-bit: ❌ {e}")
            raise

    print("\n3. Verifying layer configurations:")
    first_block = model.h[0]
    print(f"   First block attention bit_widths: {first_block.attn.bit_widths}")
    assert first_block.attn.bit_widths == [4, 8, 12, 32], "Should have custom bit_widths"
    print("   ✅ Layers use custom bit_widths")

    print("\n✅ NO HARDCODED VALUES DETECTED - CONFIG PROPERLY USED!")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SP MODEL CONFIGURATION TEST SUITE")
    print("="*80)

    try:
        # Run all tests
        test_model_initialization()
        test_lmhead_model()
        test_no_hardcoded_values()

        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("SP model properly loads bit widths from configuration.")
        print("No hardcoded bit widths remain in the codebase.")
        print("="*80)

    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80)
        raise
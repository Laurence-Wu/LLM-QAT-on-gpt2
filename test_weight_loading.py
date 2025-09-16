#!/usr/bin/env python3
"""
Test script to verify weight loading functions work correctly for QATGPT2 and SwitchableQATGPT2 models.
"""

import os
import sys
import torch
import numpy as np
from transformers import GPT2Model, GPT2Config

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'part1_switchable_precision'))

from models import QATGPT2, SwitchableQATGPT2
from config_qat import ModelConfig


def test_weight_shapes():
    """Test that all weight shapes match between pretrained and QAT models."""
    print("\n" + "="*60)
    print("Testing Weight Shape Compatibility")
    print("="*60)

    # Load pretrained model
    pretrained = GPT2Model.from_pretrained('gpt2')
    pretrained_state = pretrained.state_dict()

    # Create QAT model with same config
    config = ModelConfig()
    gpt2_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        layer_norm_epsilon=config.layer_norm_epsilon,
        embd_pdrop=config.embd_pdrop,
        quantization_bits=8,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout
    )

    # Test regular QATGPT2
    print("\n1. Testing QATGPT2 model:")
    qat_model = QATGPT2(gpt2_config, quantization_bits=8)

    # Check embeddings
    print(f"   - wte shape: pretrained={pretrained.wte.weight.shape}, qat={qat_model.wte.weight.shape}")
    print(f"   - wpe shape: pretrained={pretrained.wpe.weight.shape}, qat={qat_model.wpe.weight.shape}")

    # Check layer norms
    print(f"   - ln_f shape: pretrained={pretrained.ln_f.weight.shape}, qat={qat_model.ln_f.weight.shape}")

    # Test SwitchableQATGPT2
    print("\n2. Testing SwitchableQATGPT2 model:")
    switchable_model = SwitchableQATGPT2(gpt2_config, bit_widths=[4, 8, 16])

    print(f"   - wte shape: pretrained={pretrained.wte.weight.shape}, switchable={switchable_model.wte.weight.shape}")
    print(f"   - wpe shape: pretrained={pretrained.wpe.weight.shape}, switchable={switchable_model.wpe.weight.shape}")
    print(f"   - ln_f shape: pretrained={pretrained.ln_f.weight.shape}, switchable={switchable_model.ln_f.weight.shape}")

    del pretrained, qat_model, switchable_model
    torch.cuda.empty_cache()


def test_weight_loading():
    """Test the actual weight loading function."""
    print("\n" + "="*60)
    print("Testing Weight Loading Function")
    print("="*60)

    # Import the loading function
    from main_qat import load_pretrained_weights

    # Create models
    config = ModelConfig()
    gpt2_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        layer_norm_epsilon=config.layer_norm_epsilon,
        embd_pdrop=config.embd_pdrop,
        quantization_bits=8,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout
    )

    print("\n1. Testing QATGPT2 weight loading:")
    qat_model = QATGPT2(gpt2_config, quantization_bits=8)

    # Get initial random weights
    initial_wte = qat_model.wte.weight.data.clone()
    initial_ln_f = qat_model.ln_f.weight.data.clone()

    # Load pretrained weights
    load_pretrained_weights(qat_model)

    # Check that weights changed
    wte_changed = not torch.allclose(initial_wte, qat_model.wte.weight.data)
    ln_f_changed = not torch.allclose(initial_ln_f, qat_model.ln_f.weight.data)

    print(f"   - wte weights changed: {wte_changed}")
    print(f"   - ln_f weights changed: {ln_f_changed}")

    # Check first layer weights
    if qat_model.h and len(qat_model.h) > 0:
        print(f"   - First block ln_1 weight sum: {qat_model.h[0].ln_1.weight.data.sum().item():.4f}")
        print(f"   - First block ln_2 weight sum: {qat_model.h[0].ln_2.weight.data.sum().item():.4f}")

    print("\n2. Testing SwitchableQATGPT2 weight loading:")
    switchable_model = SwitchableQATGPT2(gpt2_config, bit_widths=[4, 8, 16])

    # Get initial random weights
    initial_wte_s = switchable_model.wte.weight.data.clone()
    initial_ln_f_s = switchable_model.ln_f.weight.data.clone()

    # Load pretrained weights
    load_pretrained_weights(switchable_model)

    # Check that weights changed
    wte_changed_s = not torch.allclose(initial_wte_s, switchable_model.wte.weight.data)
    ln_f_changed_s = not torch.allclose(initial_ln_f_s, switchable_model.ln_f.weight.data)

    print(f"   - wte weights changed: {wte_changed_s}")
    print(f"   - ln_f weights changed: {ln_f_changed_s}")

    # Check first layer weights
    if switchable_model.h and len(switchable_model.h) > 0:
        print(f"   - First block ln_1 weight sum: {switchable_model.h[0].ln_1.weight.data.sum().item():.4f}")
        print(f"   - First block ln_2 weight sum: {switchable_model.h[0].ln_2.weight.data.sum().item():.4f}")

    del qat_model, switchable_model
    torch.cuda.empty_cache()


def test_position_embeddings_size():
    """Test that position embeddings are correctly sized after loading."""
    print("\n" + "="*60)
    print("Testing Position Embeddings Size Handling")
    print("="*60)

    from main_qat import load_pretrained_weights

    # Create model with 256 positions (smaller than GPT-2's 1024)
    config = ModelConfig()
    config.n_positions = 256  # Smaller than pretrained

    gpt2_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        layer_norm_epsilon=config.layer_norm_epsilon,
        embd_pdrop=config.embd_pdrop,
        quantization_bits=8,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout
    )

    model = SwitchableQATGPT2(gpt2_config, bit_widths=[4, 8, 16])

    print(f"Model created with n_positions={config.n_positions}")
    print(f"Initial wpe shape: {model.wpe.weight.shape}")

    # Load pretrained weights
    load_pretrained_weights(model)

    print(f"After loading pretrained weights:")
    print(f"Final wpe shape: {model.wpe.weight.shape}")

    # Verify the shape is still correct (256, not 1024)
    assert model.wpe.weight.shape[0] == config.n_positions, \
        f"Position embeddings size mismatch! Expected {config.n_positions}, got {model.wpe.weight.shape[0]}"

    print("✓ Position embeddings size correctly maintained!")

    del model
    torch.cuda.empty_cache()


def test_attention_bias():
    """Test attention bias handling."""
    print("\n" + "="*60)
    print("Testing Attention Bias Handling")
    print("="*60)

    config = ModelConfig()
    gpt2_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        layer_norm_epsilon=config.layer_norm_epsilon,
        embd_pdrop=config.embd_pdrop,
        quantization_bits=8,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout
    )

    model = SwitchableQATGPT2(gpt2_config, bit_widths=[4, 8, 16])

    # Check attention bias in each layer
    for i, block in enumerate(model.h):
        if hasattr(block.attn, 'bias'):
            bias_shape = block.attn.bias.shape
            print(f"   - Block {i} attention bias shape: {bias_shape}")
            assert bias_shape == (config.n_positions, config.n_positions), \
                f"Attention bias shape mismatch in block {i}!"

    print("✓ All attention biases correctly sized!")

    del model
    torch.cuda.empty_cache()


def test_weight_mapping():
    """Test that weights are correctly mapped from pretrained to QAT structure."""
    print("\n" + "="*60)
    print("Testing Weight Mapping Correctness")
    print("="*60)

    from main_qat import load_pretrained_weights

    # Load pretrained model
    pretrained = GPT2Model.from_pretrained('gpt2')

    # Create QAT model
    config = ModelConfig()
    gpt2_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        layer_norm_epsilon=config.layer_norm_epsilon,
        embd_pdrop=config.embd_pdrop,
        quantization_bits=8,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout
    )

    model = QATGPT2(gpt2_config, quantization_bits=8)
    load_pretrained_weights(model)

    print("\n1. Checking embedding weights:")
    # Check token embeddings
    wte_match = torch.allclose(model.wte.weight.data, pretrained.wte.weight.data, rtol=1e-5)
    print(f"   - Token embeddings match: {wte_match}")

    # Check position embeddings (first 256 positions)
    min_pos = min(model.wpe.weight.shape[0], pretrained.wpe.weight.shape[0])
    wpe_match = torch.allclose(
        model.wpe.weight.data[:min_pos],
        pretrained.wpe.weight.data[:min_pos],
        rtol=1e-5
    )
    print(f"   - Position embeddings match (first {min_pos}): {wpe_match}")

    print("\n2. Checking layer normalizations:")
    # Check final layer norm
    ln_f_weight_match = torch.allclose(model.ln_f.weight.data, pretrained.ln_f.weight.data, rtol=1e-5)
    ln_f_bias_match = torch.allclose(model.ln_f.bias.data, pretrained.ln_f.bias.data, rtol=1e-5)
    print(f"   - Final LN weight match: {ln_f_weight_match}")
    print(f"   - Final LN bias match: {ln_f_bias_match}")

    # Check first block layer norms
    if len(model.h) > 0:
        ln1_weight_match = torch.allclose(
            model.h[0].ln_1.weight.data,
            pretrained.h[0].ln_1.weight.data,
            rtol=1e-5
        )
        ln1_bias_match = torch.allclose(
            model.h[0].ln_1.bias.data,
            pretrained.h[0].ln_1.bias.data,
            rtol=1e-5
        )
        print(f"   - Block 0 LN1 weight match: {ln1_weight_match}")
        print(f"   - Block 0 LN1 bias match: {ln1_bias_match}")

    print("\n3. Summary:")
    all_match = wte_match and wpe_match and ln_f_weight_match and ln_f_bias_match
    if all_match:
        print("✓ All checked weights correctly loaded from pretrained model!")
    else:
        print("✗ Some weights did not load correctly!")

    del pretrained, model
    torch.cuda.empty_cache()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("WEIGHT LOADING TEST SUITE")
    print("="*60)

    try:
        # Test 1: Shape compatibility
        test_weight_shapes()

        # Test 2: Weight loading function
        test_weight_loading()

        # Test 3: Position embeddings size handling
        test_position_embeddings_size()

        # Test 4: Attention bias handling
        test_attention_bias()

        # Test 5: Weight mapping correctness
        test_weight_mapping()

        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
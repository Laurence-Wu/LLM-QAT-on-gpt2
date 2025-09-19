#!/usr/bin/env python3
"""
Test script for Switchable Precision (SP) Model
Tests basic functionalities and the entire workflow
"""

import sys
import os
import torch
import torch.nn as nn
import gc
from transformers import GPT2Config, GPT2TokenizerFast

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models_sp import SPModel, SPLMHeadModel, SPAttention, SPMLP, SPBlock
from shared.lora import SwitchableLinearWithLoRA
from part1_switchable_precision.config_sp import ModelConfig, TrainingConfig
from part1_switchable_precision.train_sp import train_sp, get_next_bitwidth

def test_basic_components():
    """Test basic SP model components."""
    print("\n" + "="*60)
    print("Testing SP Model Basic Components")
    print("="*60)

    # Test configuration
    print("\n1. Testing Configuration Classes...")
    model_config = ModelConfig()
    training_config = TrainingConfig()

    assert hasattr(model_config, 'bit_widths'), "ModelConfig missing bit_widths"
    assert hasattr(model_config, 'lora_rank_per_bit'), "ModelConfig missing lora_rank_per_bit"
    assert hasattr(model_config, 'lora_alpha_per_bit'), "ModelConfig missing lora_alpha_per_bit"
    assert model_config.bit_widths == [4, 8, 16], f"Expected [4, 8, 16], got {model_config.bit_widths}"
    print("✓ Configuration classes initialized correctly")

    # Test SwitchableLinearWithLoRA
    print("\n2. Testing SwitchableLinearWithLoRA...")
    linear = SwitchableLinearWithLoRA(
        in_features=768,
        out_features=768,
        bit_widths=[4, 8, 16],
        lora_rank_per_bit={4: 8, 8: 16, 16: 32},
        lora_alpha_per_bit={4: 16, 8: 32, 16: 64}
    )

    # Test precision switching
    x = torch.randn(2, 10, 768)
    for bits in [4, 8, 16]:
        linear.set_precision(bits)
        assert linear.current_bits == bits, f"Failed to set precision to {bits}"
        output = linear(x)
        assert output.shape == (2, 10, 768), f"Wrong output shape: {output.shape}"
    print("✓ SwitchableLinearWithLoRA works correctly")

    return True

def test_sp_attention():
    """Test SP Attention module."""
    print("\n3. Testing SP Attention Module...")

    config = GPT2Config(
        n_embd=768,
        n_head=12,
        n_positions=1024,
        vocab_size=50257
    )
    config.lora_rank_per_bit = {4: 8, 8: 16, 16: 32}
    config.lora_alpha_per_bit = {4: 16, 8: 32, 16: 64}

    attn = SPAttention(config, bit_widths=[4, 8, 16])

    # Test forward pass with different bit widths
    hidden_states = torch.randn(2, 10, 768)
    for bits in [4, 8, 16]:
        attn.set_precision(bits)
        output = attn(hidden_states)
        assert output.shape == (2, 10, 768), f"Wrong attention output shape: {output.shape}"

    print("✓ SP Attention module works correctly")
    return True

def test_sp_mlp():
    """Test SP MLP module."""
    print("\n4. Testing SP MLP Module...")

    config = GPT2Config(
        n_embd=768,
        n_head=12,
        n_positions=1024,
        vocab_size=50257
    )
    config.lora_rank_per_bit = {4: 8, 8: 16, 16: 32}
    config.lora_alpha_per_bit = {4: 16, 8: 32, 16: 64}

    mlp = SPMLP(config, bit_widths=[4, 8, 16])

    # Test forward pass
    hidden_states = torch.randn(2, 10, 768)
    for bits in [4, 8, 16]:
        mlp.set_precision(bits)
        output = mlp(hidden_states)
        assert output.shape == (2, 10, 768), f"Wrong MLP output shape: {output.shape}"

    print("✓ SP MLP module works correctly")
    return True

def test_sp_block():
    """Test SP Transformer Block."""
    print("\n5. Testing SP Transformer Block...")

    config = GPT2Config(
        n_embd=768,
        n_head=12,
        n_positions=1024,
        vocab_size=50257
    )
    config.lora_rank_per_bit = {4: 8, 8: 16, 16: 32}
    config.lora_alpha_per_bit = {4: 16, 8: 32, 16: 64}

    block = SPBlock(config, bit_widths=[4, 8, 16])

    # Test forward pass
    hidden_states = torch.randn(2, 10, 768)
    for bits in [4, 8, 16]:
        block.set_precision(bits)
        output = block(hidden_states, use_checkpoint=False)
        assert output.shape == (2, 10, 768), f"Wrong block output shape: {output.shape}"

    print("✓ SP Transformer Block works correctly")
    return True

def test_sp_model():
    """Test complete SP Model."""
    print("\n6. Testing Complete SP Model...")

    config = GPT2Config(
        n_embd=768,
        n_head=12,
        n_layer=2,  # Small model for testing
        n_positions=256,
        vocab_size=50257
    )
    config.lora_rank_per_bit = {4: 8, 8: 16, 16: 32}
    config.lora_alpha_per_bit = {4: 16, 8: 32, 16: 64}
    config.bit_widths = [4, 8, 16]

    model = SPModel(config)

    # Test forward pass
    input_ids = torch.randint(0, 50257, (2, 10))

    for bits in [4, 8, 16]:
        model.set_precision(bits)
        assert model.get_current_precision() == bits, f"Failed to get correct precision"
        output = model(input_ids)
        assert output.shape == (2, 10, 768), f"Wrong model output shape: {output.shape}"

    print("✓ SP Model works correctly")
    return True

def test_sp_lm_head_model():
    """Test SP Language Model Head."""
    print("\n7. Testing SP Language Model Head...")

    config = GPT2Config(
        n_embd=768,
        n_head=12,
        n_layer=2,  # Small model for testing
        n_positions=256,
        vocab_size=50257
    )
    config.lora_rank_per_bit = {4: 8, 8: 16, 16: 32}
    config.lora_alpha_per_bit = {4: 16, 8: 32, 16: 64}
    config.bit_widths = [4, 8, 16]

    model = SPLMHeadModel(config)

    # Test forward pass without labels
    input_ids = torch.randint(0, 50257, (2, 10))
    for bits in [4, 8, 16]:
        model.set_precision(bits)
        output = model(input_ids)
        assert output.shape == (2, 10, 50257), f"Wrong logits shape: {output.shape}"

    # Test forward pass with labels (loss computation)
    labels = torch.randint(0, 50257, (2, 10))
    for bits in [4, 8, 16]:
        model.set_precision(bits)
        output = model(input_ids, labels=labels)
        assert 'loss' in output, "Loss not computed"
        assert 'logits' in output, "Logits not returned"
        assert output['loss'].dim() == 0, "Loss should be scalar"
        assert output['logits'].shape == (2, 10, 50257), f"Wrong logits shape with labels"

    print("✓ SP Language Model Head works correctly")
    return True

def test_bit_width_switching():
    """Test bit-width switching strategies."""
    print("\n8. Testing Bit-Width Switching Strategies...")

    class MockConfig:
        bit_widths = [4, 8, 16]
        switch_strategy = 'cyclic'
        switch_interval = 10
        curriculum_schedule = [16, 16, 8, 8, 4]

    config = MockConfig()

    # Test cyclic strategy
    print("   - Testing cyclic strategy...")
    config.switch_strategy = 'cyclic'
    expected_pattern = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4,  # 0-9: 4-bit
                       8, 8, 8, 8, 8, 8, 8, 8, 8, 8,  # 10-19: 8-bit
                       16, 16, 16, 16, 16, 16, 16, 16, 16, 16,  # 20-29: 16-bit
                       4, 4, 4, 4, 4]  # 30-34: back to 4-bit

    for i, expected in enumerate(expected_pattern):
        bits = get_next_bitwidth(i, config)
        assert bits == expected, f"Iteration {i}: expected {expected}, got {bits}"

    # Test curriculum strategy
    print("   - Testing curriculum strategy...")
    config.switch_strategy = 'curriculum'
    bits_0 = get_next_bitwidth(0, config)
    assert bits_0 == 16, f"Curriculum: Expected 16 at start, got {bits_0}"
    bits_100 = get_next_bitwidth(100, config)
    assert bits_100 == 8, f"Curriculum: Expected 8 at iter 100, got {bits_100}"
    bits_200 = get_next_bitwidth(200, config)
    assert bits_200 == 4, f"Curriculum: Expected 4 at iter 200, got {bits_200}"

    # Test random strategy (just check it returns valid values)
    print("   - Testing random strategy...")
    config.switch_strategy = 'random'
    for i in range(20):
        bits = get_next_bitwidth(i, config)
        assert bits in [4, 8, 16], f"Random: Invalid bit width {bits}"

    print("✓ Bit-width switching strategies work correctly")
    return True

def test_training_workflow():
    """Test a minimal training workflow."""
    print("\n9. Testing Training Workflow...")

    if not torch.cuda.is_available():
        print("   ⚠ CUDA not available, skipping training test")
        return True

    # Create small model for testing
    config = GPT2Config(
        n_embd=256,  # Smaller for testing
        n_head=4,
        n_layer=2,
        n_positions=128,
        vocab_size=1000  # Smaller vocab for testing
    )
    config.lora_rank_per_bit = {4: 4, 8: 8, 16: 16}
    config.lora_alpha_per_bit = {4: 8, 8: 16, 16: 32}
    config.bit_widths = [4, 8, 16]

    model = SPLMHeadModel(config).cuda()

    # Create dummy data
    batch_size = 2
    seq_length = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_length)).cuda()
    labels = input_ids.clone()

    # Test forward pass
    model.set_precision(8)
    output = model(input_ids, labels=labels)
    loss = output['loss']

    # Test backward pass
    loss.backward()

    # Check gradients exist
    has_grads = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grads = True
            break

    assert has_grads, "No gradients computed"

    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()

    print("✓ Training workflow works correctly")
    return True

def run_all_tests():
    """Run all SP model tests."""
    print("\n" + "="*70)
    print(" SWITCHABLE PRECISION (SP) MODEL TEST SUITE")
    print("="*70)

    tests = [
        ("Basic Components", test_basic_components),
        ("SP Attention", test_sp_attention),
        ("SP MLP", test_sp_mlp),
        ("SP Block", test_sp_block),
        ("SP Model", test_sp_model),
        ("SP LM Head Model", test_sp_lm_head_model),
        ("Bit-Width Switching", test_bit_width_switching),
        ("Training Workflow", test_training_workflow)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n✗ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print(f" TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*70)

    if failed == 0:
        print("\n✅ All SP model tests passed successfully!")
    else:
        print(f"\n⚠ {failed} test(s) failed. Please review the errors above.")

    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
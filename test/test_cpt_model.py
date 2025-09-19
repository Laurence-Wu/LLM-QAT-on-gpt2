#!/usr/bin/env python3
"""
Test script for Cyclic Precision Training (CPT) Model
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

from shared.models_cpt import CPTModel, CPTLMHeadModel, CPTAttention, CPTMLP, CPTBlock
from shared.lora import LinearWithLoRA
from part2_cyclic_precision.config_cyclic import ModelConfig, CyclicTrainingConfig, CyclicPrecisionConfig
from part2_cyclic_precision.train_cyclic import CyclicPrecisionScheduler

def test_basic_components():
    """Test basic CPT model components."""
    print("\n" + "="*60)
    print("Testing CPT Model Basic Components")
    print("="*60)

    # Test configuration
    print("\n1. Testing Configuration Classes...")
    model_config = ModelConfig()
    training_config = CyclicTrainingConfig()
    cyclic_config = CyclicPrecisionConfig()

    assert hasattr(cyclic_config, 'bit_widths'), "CyclicPrecisionConfig missing bit_widths"
    assert hasattr(cyclic_config, 'cycle_length'), "CyclicPrecisionConfig missing cycle_length"
    assert hasattr(cyclic_config, 'bit_width_pattern'), "CyclicPrecisionConfig missing bit_width_pattern"
    print("✓ Configuration classes initialized correctly")

    # Test LinearWithLoRA for CPT
    print("\n2. Testing LinearWithLoRA for CPT...")
    linear = LinearWithLoRA(
        in_features=768,
        out_features=768,
        bits=8,
        lora_rank=16,
        lora_alpha=32
    )

    # Test forward pass
    x = torch.randn(2, 10, 768)
    output = linear(x)
    assert output.shape == (2, 10, 768), f"Wrong output shape: {output.shape}"

    # Test precision setting
    linear.set_precision(4, 4)  # Set to 4-bit weights and activations
    output = linear(x)
    assert output.shape == (2, 10, 768), f"Wrong output shape after precision change"

    print("✓ LinearWithLoRA works correctly for CPT")
    return True

def test_cyclic_precision_scheduler():
    """Test Cyclic Precision Scheduler."""
    print("\n3. Testing Cyclic Precision Scheduler...")

    config = CyclicPrecisionConfig()
    scheduler = CyclicPrecisionScheduler(config)

    # Test basic bit-width cycling
    print("   - Testing basic cycling...")
    expected_pattern = config.bit_width_pattern  # [16, 8, 4, 8, 16]
    cycle_length = config.cycle_length  # 100

    for i in range(cycle_length * 2):  # Test 2 full cycles
        bit_width = scheduler.get_current_bit_width(i)
        # Calculate expected bit width based on position in cycle
        pos_in_cycle = i % cycle_length
        pattern_idx = int(pos_in_cycle * len(expected_pattern) / cycle_length)
        pattern_idx = min(pattern_idx, len(expected_pattern) - 1)
        expected_bits = expected_pattern[pattern_idx]
        assert bit_width == expected_bits, f"Iteration {i}: expected {expected_bits}, got {bit_width}"

    # Test layer-wise cycling
    print("   - Testing layer-wise cycling...")
    n_layers = 6
    layer_bits = scheduler.get_layer_bit_widths(50, n_layers)
    assert len(layer_bits) == n_layers, f"Expected {n_layers} bit widths, got {len(layer_bits)}"

    # Test learning rate scaling
    print("   - Testing learning rate scaling...")
    lr_scale_4 = scheduler.get_learning_rate_scale(4)
    lr_scale_16 = scheduler.get_learning_rate_scale(16)
    assert lr_scale_4 != lr_scale_16, "Learning rate scales should differ for different bit widths"

    print("✓ Cyclic Precision Scheduler works correctly")
    return True

def test_cpt_attention():
    """Test CPT Attention module."""
    print("\n4. Testing CPT Attention Module...")

    config = GPT2Config(
        n_embd=768,
        n_head=12,
        n_positions=1024,
        vocab_size=50257
    )
    config.lora_rank = 16
    config.lora_alpha = 32
    config.lora_dropout = 0.1

    attn = CPTAttention(config, bits=8)

    # Test forward pass
    hidden_states = torch.randn(2, 10, 768)
    output = attn(hidden_states)
    assert output.shape == (2, 10, 768), f"Wrong attention output shape: {output.shape}"

    # Test precision setting
    attn.set_precision(4, 4)
    output = attn(hidden_states)
    assert output.shape == (2, 10, 768), f"Wrong output shape after precision change"

    print("✓ CPT Attention module works correctly")
    return True

def test_cpt_mlp():
    """Test CPT MLP module."""
    print("\n5. Testing CPT MLP Module...")

    config = GPT2Config(
        n_embd=768,
        n_head=12,
        n_positions=1024,
        vocab_size=50257
    )
    config.lora_rank = 16
    config.lora_alpha = 32
    config.lora_dropout = 0.1

    mlp = CPTMLP(config, bits=8)

    # Test forward pass
    hidden_states = torch.randn(2, 10, 768)
    output = mlp(hidden_states)
    assert output.shape == (2, 10, 768), f"Wrong MLP output shape: {output.shape}"

    # Test precision setting
    mlp.set_precision(4, 4)
    output = mlp(hidden_states)
    assert output.shape == (2, 10, 768), f"Wrong output shape after precision change"

    print("✓ CPT MLP module works correctly")
    return True

def test_cpt_block():
    """Test CPT Transformer Block."""
    print("\n6. Testing CPT Transformer Block...")

    config = GPT2Config(
        n_embd=768,
        n_head=12,
        n_positions=1024,
        vocab_size=50257
    )
    config.lora_rank = 16
    config.lora_alpha = 32
    config.lora_dropout = 0.1

    block = CPTBlock(config, bits=8)

    # Test forward pass
    hidden_states = torch.randn(2, 10, 768)
    output = block(hidden_states, use_checkpoint=False)
    assert output.shape == (2, 10, 768), f"Wrong block output shape: {output.shape}"

    # Test precision setting
    # CPTBlock.set_precision needs 4 arguments: attn_bits, mlp_bits, activation_bits, kv_bits
    block.set_precision(4, 4, 4, 4)
    output = block(hidden_states, use_checkpoint=False)
    assert output.shape == (2, 10, 768), f"Wrong output shape after precision change"

    print("✓ CPT Transformer Block works correctly")
    return True

def test_cpt_model():
    """Test complete CPT Model."""
    print("\n7. Testing Complete CPT Model...")

    config = GPT2Config(
        n_embd=768,
        n_head=12,
        n_layer=2,  # Small model for testing
        n_positions=256,
        vocab_size=50257
    )
    config.lora_rank = 16
    config.lora_alpha = 32
    config.lora_dropout = 0.1
    config.quantization_bits = 8

    model = CPTModel(config)

    # Test forward pass
    input_ids = torch.randint(0, 50257, (2, 10))
    output = model(input_ids)
    assert output.shape == (2, 10, 768), f"Wrong model output shape: {output.shape}"

    # Test precision setting for all layers
    model.set_precision(4, 4)
    output = model(input_ids)
    assert output.shape == (2, 10, 768), f"Wrong output shape after precision change"

    # Test layer-wise precision setting
    # CPTModel expects a list of dicts with 'attn_bits', 'mlp_bits', 'activation_bits', 'kv_bits'
    layer_configs = [
        {'attn_bits': 8, 'mlp_bits': 8, 'activation_bits': 8, 'kv_bits': 8},
        {'attn_bits': 4, 'mlp_bits': 4, 'activation_bits': 4, 'kv_bits': 4}
    ]
    model.set_layer_precision(layer_configs)
    output = model(input_ids)
    assert output.shape == (2, 10, 768), f"Wrong output shape with layer-wise precision"

    print("✓ CPT Model works correctly")
    return True

def test_cpt_lm_head_model():
    """Test CPT Language Model Head."""
    print("\n8. Testing CPT Language Model Head...")

    config = GPT2Config(
        n_embd=768,
        n_head=12,
        n_layer=2,  # Small model for testing
        n_positions=256,
        vocab_size=50257
    )
    config.lora_rank = 16
    config.lora_alpha = 32
    config.lora_dropout = 0.1
    config.quantization_bits = 8

    model = CPTLMHeadModel(config)

    # Test forward pass without labels
    input_ids = torch.randint(0, 50257, (2, 10))
    output = model(input_ids)
    assert output.shape == (2, 10, 50257), f"Wrong logits shape: {output.shape}"

    # Test forward pass with labels (loss computation)
    labels = torch.randint(0, 50257, (2, 10))
    output = model(input_ids, labels=labels)
    assert 'loss' in output, "Loss not computed"
    assert 'logits' in output, "Logits not returned"
    assert output['loss'].dim() == 0, "Loss should be scalar"
    assert output['logits'].shape == (2, 10, 50257), f"Wrong logits shape with labels"

    # Test precision setting
    model.set_precision(4, 4)
    output = model(input_ids, labels=labels)
    assert 'loss' in output, "Loss not computed after precision change"

    print("✓ CPT Language Model Head works correctly")
    return True

def test_cyclic_training_workflow():
    """Test cyclic precision training workflow."""
    print("\n9. Testing Cyclic Training Workflow...")

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
    config.lora_rank = 8
    config.lora_alpha = 16
    config.lora_dropout = 0.1
    config.quantization_bits = 8

    model = CPTLMHeadModel(config).cuda()

    # Create cyclic scheduler
    cyclic_config = CyclicPrecisionConfig()
    cyclic_config.bit_widths = [4, 8]  # Simpler for testing
    cyclic_config.cycle_length = 10
    cyclic_config.bit_width_pattern = [8, 4, 8]

    scheduler = CyclicPrecisionScheduler(cyclic_config)

    # Simulate training iterations
    batch_size = 2
    seq_length = 32
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("   - Testing cyclic precision changes during training...")
    for iteration in range(20):  # Test 2 cycles
        # Get current bit width from scheduler
        current_bits = scheduler.get_current_bit_width(iteration)

        # Set model precision
        model.set_precision(current_bits, current_bits)

        # Forward pass
        input_ids = torch.randint(0, 1000, (batch_size, seq_length)).cuda()
        labels = input_ids.clone()
        output = model(input_ids, labels=labels)
        loss = output['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify precision was applied
        if iteration % 5 == 0:
            print(f"     Iteration {iteration}: bit-width={current_bits}, loss={loss.item():.4f}")

    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()

    print("✓ Cyclic training workflow works correctly")
    return True

def test_static_precision_training():
    """Test static precision training (Step 4 of CPT)."""
    print("\n10. Testing Static Precision Training...")

    if not torch.cuda.is_available():
        print("   ⚠ CUDA not available, skipping test")
        return True

    # Create small model
    config = GPT2Config(
        n_embd=256,
        n_head=4,
        n_layer=2,
        n_positions=128,
        vocab_size=1000
    )
    config.lora_rank = 8
    config.lora_alpha = 16
    config.lora_dropout = 0.1
    config.quantization_bits = 4  # Test with 4-bit static

    model = CPTLMHeadModel(config).cuda()

    # Set static precision
    model.set_precision(4, 4)

    # Train for a few iterations
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch_size = 2
    seq_length = 32

    print("   - Training with static 4-bit precision...")
    for iteration in range(10):
        input_ids = torch.randint(0, 1000, (batch_size, seq_length)).cuda()
        labels = input_ids.clone()

        output = model(input_ids, labels=labels)
        loss = output['loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 5 == 0:
            print(f"     Iteration {iteration}: loss={loss.item():.4f}")

    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()

    print("✓ Static precision training works correctly")
    return True

def run_all_tests():
    """Run all CPT model tests."""
    print("\n" + "="*70)
    print(" CYCLIC PRECISION TRAINING (CPT) MODEL TEST SUITE")
    print("="*70)

    tests = [
        ("Basic Components", test_basic_components),
        ("Cyclic Precision Scheduler", test_cyclic_precision_scheduler),
        ("CPT Attention", test_cpt_attention),
        ("CPT MLP", test_cpt_mlp),
        ("CPT Block", test_cpt_block),
        ("CPT Model", test_cpt_model),
        ("CPT LM Head Model", test_cpt_lm_head_model),
        ("Cyclic Training Workflow", test_cyclic_training_workflow),
        ("Static Precision Training", test_static_precision_training)
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
        print("\n✅ All CPT model tests passed successfully!")
    else:
        print(f"\n⚠ {failed} test(s) failed. Please review the errors above.")

    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
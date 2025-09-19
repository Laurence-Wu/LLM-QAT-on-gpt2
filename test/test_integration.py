#!/usr/bin/env python3
"""
Integration Test Suite
Tests the complete workflow for both SP and CPT models
"""

import sys
import os
import torch
import torch.nn as nn
import gc
import time
from transformers import GPT2Config, GPT2TokenizerFast

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import SP components
from shared.models_sp import SPLMHeadModel
from part1_switchable_precision.config_sp import ModelConfig as SPModelConfig
from part1_switchable_precision.config_sp import TrainingConfig as SPTrainingConfig
from part1_switchable_precision.train_sp import get_next_bitwidth

# Import CPT components
from shared.models_cpt import CPTLMHeadModel, CyclicPrecisionScheduler
from part2_cyclic_precision.config_cyclic import ModelConfig as CPTModelConfig
from part2_cyclic_precision.config_cyclic import CyclicTrainingConfig, CyclicPrecisionConfig

# Import shared components
from shared.dataset import create_dataloaders

def test_sp_full_workflow():
    """Test complete SP model workflow."""
    print("\n" + "="*60)
    print("Testing SP Model Full Workflow")
    print("="*60)

    # 1. Setup configuration
    print("\n1. Setting up SP configuration...")
    model_config = SPModelConfig()
    model_config.n_layer = 2  # Small model for testing
    model_config.n_embd = 256
    model_config.n_head = 4
    model_config.vocab_size = 1000  # Smaller vocab

    training_config = SPTrainingConfig()
    training_config.num_iterations = 10  # Quick test
    training_config.eval_interval = 5
    training_config.batch_size = 2

    # 2. Create model
    print("2. Creating SP model...")
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
    )
    gpt2_config.bit_widths = model_config.bit_widths
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit

    model = SPLMHeadModel(gpt2_config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {param_count:,}")

    # 3. Test precision switching
    print("3. Testing precision switching...")
    for iteration in range(10):
        bits = get_next_bitwidth(iteration, model_config)
        model.set_precision(bits)
        assert model.get_current_precision() == bits

    print("   ✓ Precision switching works")

    # 4. Test forward pass
    print("4. Testing forward pass...")
    input_ids = torch.randint(0, model_config.vocab_size, (2, 32))
    labels = input_ids.clone()

    for bits in [4, 8, 16]:
        model.set_precision(bits)
        output = model(input_ids, labels=labels)
        assert 'loss' in output
        assert 'logits' in output
        print(f"   {bits}-bit loss: {output['loss'].item():.4f}")

    print("✓ SP model workflow complete")
    return True

def test_cpt_full_workflow():
    """Test complete CPT model workflow."""
    print("\n" + "="*60)
    print("Testing CPT Model Full Workflow")
    print("="*60)

    # 1. Setup configuration
    print("\n1. Setting up CPT configuration...")
    model_config = CPTModelConfig()
    model_config.n_layer = 2  # Small model
    model_config.n_embd = 256
    model_config.n_head = 4
    model_config.vocab_size = 1000

    training_config = CyclicTrainingConfig()
    training_config.num_cpt_iterations = 10
    cyclic_config = CyclicPrecisionConfig()

    # 2. Create model
    print("2. Creating CPT model...")
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
    )
    gpt2_config.lora_rank = model_config.lora_rank
    gpt2_config.lora_alpha = model_config.lora_alpha
    gpt2_config.lora_dropout = model_config.lora_dropout
    gpt2_config.quantization_bits = model_config.default_bit_width

    model = CPTLMHeadModel(gpt2_config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {param_count:,}")

    # 3. Test cyclic scheduler
    print("3. Testing cyclic scheduler...")
    scheduler = CyclicPrecisionScheduler(cyclic_config)
    bit_sequence = []
    for i in range(cyclic_config.cycle_length):
        bits = scheduler.get_current_bit_width(i)
        bit_sequence.append(bits)
    print(f"   Bit-width cycle: {bit_sequence[:10]}...")

    # 4. Test precision changes
    print("4. Testing precision changes...")
    input_ids = torch.randint(0, model_config.vocab_size, (2, 32))
    labels = input_ids.clone()

    for bits in [4, 8, 16]:
        model.set_precision(bits, bits)
        output = model(input_ids, labels=labels)
        assert 'loss' in output
        print(f"   {bits}-bit loss: {output['loss'].item():.4f}")

    print("✓ CPT model workflow complete")
    return True

def test_model_comparison():
    """Compare SP and CPT models side by side."""
    print("\n" + "="*60)
    print("Comparing SP and CPT Models")
    print("="*60)

    # Use same base configuration
    vocab_size = 1000
    n_embd = 256
    n_head = 4
    n_layer = 2
    n_positions = 128

    # Create SP model
    print("\n1. Creating SP model...")
    sp_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
    )
    sp_config.bit_widths = [4, 8, 16]
    sp_config.lora_rank_per_bit = {4: 4, 8: 8, 16: 16}
    sp_config.lora_alpha_per_bit = {4: 8, 8: 16, 16: 32}

    sp_model = SPLMHeadModel(sp_config)
    sp_params = sum(p.numel() for p in sp_model.parameters())

    # Create CPT model
    print("2. Creating CPT model...")
    cpt_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
    )
    cpt_config.lora_rank = 8
    cpt_config.lora_alpha = 16
    cpt_config.lora_dropout = 0.1
    cpt_config.quantization_bits = 8

    cpt_model = CPTLMHeadModel(cpt_config)
    cpt_params = sum(p.numel() for p in cpt_model.parameters())

    # Compare parameters
    print("\n3. Model comparison:")
    print(f"   SP model parameters: {sp_params:,}")
    print(f"   CPT model parameters: {cpt_params:,}")
    print(f"   SP has {sp_params - cpt_params:,} more parameters (multiple LoRAs)")

    # Test same input
    print("\n4. Testing with same input...")
    input_ids = torch.randint(0, vocab_size, (2, 32))
    labels = input_ids.clone()

    # SP model at 8-bit
    sp_model.set_precision(8)
    sp_output = sp_model(input_ids, labels=labels)
    sp_loss = sp_output['loss'].item()

    # CPT model at 8-bit
    cpt_model.set_precision(8, 8)
    cpt_output = cpt_model(input_ids, labels=labels)
    cpt_loss = cpt_output['loss'].item()

    print(f"   SP loss (8-bit): {sp_loss:.4f}")
    print(f"   CPT loss (8-bit): {cpt_loss:.4f}")

    print("✓ Model comparison complete")
    return True

def test_training_compatibility():
    """Test that both models work with training pipeline."""
    print("\n" + "="*60)
    print("Testing Training Compatibility")
    print("="*60)

    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping training test")
        return True

    # Common configuration
    vocab_size = 1000
    n_embd = 256
    n_head = 4
    n_layer = 2
    batch_size = 2
    seq_length = 32

    print("\n1. Testing SP model training...")
    sp_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=128,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
    )
    sp_config.bit_widths = [4, 8, 16]
    sp_config.lora_rank_per_bit = {4: 4, 8: 8, 16: 16}
    sp_config.lora_alpha_per_bit = {4: 8, 8: 16, 16: 32}

    sp_model = SPLMHeadModel(sp_config).cuda()
    sp_optimizer = torch.optim.Adam(sp_model.parameters(), lr=1e-4)

    # Train for a few steps
    for i in range(5):
        bits = [4, 8, 16][i % 3]
        sp_model.set_precision(bits)

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).cuda()
        labels = input_ids.clone()

        output = sp_model(input_ids, labels=labels)
        loss = output['loss']

        sp_optimizer.zero_grad()
        loss.backward()
        sp_optimizer.step()

        print(f"   SP step {i}: {bits}-bit, loss={loss.item():.4f}")

    del sp_model
    torch.cuda.empty_cache()

    print("\n2. Testing CPT model training...")
    cpt_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=128,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
    )
    cpt_config.lora_rank = 8
    cpt_config.lora_alpha = 16
    cpt_config.lora_dropout = 0.1
    cpt_config.quantization_bits = 8

    cpt_model = CPTLMHeadModel(cpt_config).cuda()
    cpt_optimizer = torch.optim.Adam(cpt_model.parameters(), lr=1e-4)

    # Train with cyclic precision
    cyclic_config = CyclicPrecisionConfig()
    cyclic_config.cycle_length = 6
    cyclic_config.bit_width_pattern = [8, 4, 8]
    scheduler = CyclicPrecisionScheduler(cyclic_config)

    for i in range(5):
        bits = scheduler.get_current_bit_width(i)
        cpt_model.set_precision(bits, bits)

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).cuda()
        labels = input_ids.clone()

        output = cpt_model(input_ids, labels=labels)
        loss = output['loss']

        cpt_optimizer.zero_grad()
        loss.backward()
        cpt_optimizer.step()

        print(f"   CPT step {i}: {bits}-bit, loss={loss.item():.4f}")

    del cpt_model
    torch.cuda.empty_cache()
    gc.collect()

    print("✓ Both models train successfully")
    return True

def test_memory_footprint():
    """Compare memory footprint of both models."""
    print("\n" + "="*60)
    print("Testing Memory Footprint")
    print("="*60)

    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping memory test")
        return True

    torch.cuda.empty_cache()
    gc.collect()

    # Configuration
    config = GPT2Config(
        vocab_size=1000,
        n_positions=128,
        n_embd=256,
        n_layer=2,
        n_head=4,
    )

    print("\n1. SP model memory usage...")
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()

    sp_config = config
    sp_config.bit_widths = [4, 8, 16]
    sp_config.lora_rank_per_bit = {4: 4, 8: 8, 16: 16}
    sp_config.lora_alpha_per_bit = {4: 8, 8: 16, 16: 32}

    sp_model = SPLMHeadModel(sp_config).cuda()

    # Forward pass
    input_ids = torch.randint(0, 1000, (2, 32)).cuda()
    sp_model.set_precision(8)
    output = sp_model(input_ids)

    sp_memory = torch.cuda.max_memory_allocated() - initial_memory
    print(f"   SP model memory: {sp_memory / 1024**2:.1f} MB")

    del sp_model, output
    torch.cuda.empty_cache()
    gc.collect()

    print("\n2. CPT model memory usage...")
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()

    cpt_config = config
    cpt_config.lora_rank = 8
    cpt_config.lora_alpha = 16
    cpt_config.lora_dropout = 0.1
    cpt_config.quantization_bits = 8

    cpt_model = CPTLMHeadModel(cpt_config).cuda()

    # Forward pass
    cpt_model.set_precision(8, 8)
    output = cpt_model(input_ids)

    cpt_memory = torch.cuda.max_memory_allocated() - initial_memory
    print(f"   CPT model memory: {cpt_memory / 1024**2:.1f} MB")

    print(f"\n   Memory difference: {(sp_memory - cpt_memory) / 1024**2:.1f} MB")
    print("   (SP has multiple LoRA adapters, hence more memory)")

    del cpt_model, output, input_ids
    torch.cuda.empty_cache()
    gc.collect()

    print("✓ Memory footprint analysis complete")
    return True

def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*70)
    print(" INTEGRATION TEST SUITE")
    print("="*70)

    tests = [
        ("SP Full Workflow", test_sp_full_workflow),
        ("CPT Full Workflow", test_cpt_full_workflow),
        ("Model Comparison", test_model_comparison),
        ("Training Compatibility", test_training_compatibility),
        ("Memory Footprint", test_memory_footprint)
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
        print("\n✅ All integration tests passed successfully!")
    else:
        print(f"\n⚠ {failed} test(s) failed. Please review the errors above.")

    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
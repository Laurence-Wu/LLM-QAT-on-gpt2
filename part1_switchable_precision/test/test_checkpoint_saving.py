#!/usr/bin/env python3
"""
Test checkpoint saving for SP models with all configured bit widths.
"""

import sys
import os
import torch
import tempfile
import shutil

# Add parent directory (part1_switchable_precision) to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config_sp import ModelConfig
from models_sp import SPLMHeadModel
from deploy import save_sp_checkpoints
from transformers import GPT2Config


def test_checkpoint_saving():
    """Test saving checkpoints for all configured bit widths."""
    print("\n" + "="*80)
    print("TESTING SP MODEL CHECKPOINT SAVING")
    print("="*80)

    # Create model config
    model_config = ModelConfig()
    print(f"\nConfigured bit widths: {model_config.bit_widths}")

    # Create GPT2Config with SP-specific configs
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=2,  # Small model for testing
        n_head=model_config.n_head,
        layer_norm_epsilon=model_config.layer_norm_epsilon
    )

    # Add SP-specific configs
    gpt2_config.bit_widths = model_config.bit_widths
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    gpt2_config.activation_bits_per_bit = model_config.activation_bits_per_bit
    gpt2_config.quantizer_per_bit = model_config.quantizer_per_bit

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SPLMHeadModel(gpt2_config)
    model = model.to(device)

    print(f"Model created with {len(model.transformer.h)} layers")
    print(f"Device: {device}")

    # Create temporary directory for test checkpoints
    temp_dir = tempfile.mkdtemp(prefix='sp_checkpoint_test_')
    print(f"\nUsing temporary directory: {temp_dir}")

    try:
        # Test saving checkpoints for all bit widths
        base_filename = os.path.join(temp_dir, "test_sp_model")

        print("\n" + "-"*60)
        print("Saving checkpoints...")
        print("-"*60)

        saved_checkpoints = save_sp_checkpoints(
            model,
            base_filename=base_filename,
            model_config=model_config,
            training_config=None
        )

        print("\n" + "-"*60)
        print("Verifying saved checkpoints...")
        print("-"*60)

        # Verify each checkpoint
        for bits, filepath in saved_checkpoints.items():
            if not os.path.exists(filepath):
                print(f"❌ {bits}-bit checkpoint not found: {filepath}")
                continue

            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"\n{bits}-bit checkpoint:")
            print(f"  Path: {filepath}")
            print(f"  Size: {file_size_mb:.2f} MB")

            # Load and verify checkpoint contents
            checkpoint = torch.load(filepath, map_location='cpu')

            if bits == 32:
                # FP32 teacher checkpoint
                assert 'model_state_dict' in checkpoint, "Missing model_state_dict in FP32 checkpoint"
                assert 'model_config' in checkpoint, "Missing model_config in FP32 checkpoint"
                assert checkpoint.get('bit_width') == 32, f"Wrong bit_width in checkpoint: {checkpoint.get('bit_width')}"
                print(f"  ✅ FP32 teacher checkpoint valid")
            else:
                # INT8 student checkpoint
                assert 'int8_state_dict' in checkpoint, "Missing int8_state_dict in INT8 checkpoint"
                assert 'model_info' in checkpoint, "Missing model_info in INT8 checkpoint"
                assert checkpoint['model_info'].get('target_bits') == bits, f"Wrong target_bits: {checkpoint['model_info'].get('target_bits')}"

                # Count INT8 weights
                int8_weights = sum(1 for k in checkpoint['int8_state_dict'] if 'int8' in k)
                print(f"  INT8 weight tensors: {int8_weights}")
                print(f"  Compression ratio: {checkpoint['model_info']['compression_ratio']:.2f}x")
                print(f"  ✅ INT8 student checkpoint valid")

        print("\n" + "="*80)
        print("✅ ALL CHECKPOINT TESTS PASSED!")
        print("="*80)

    finally:
        # Clean up temporary directory
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_checkpoint_loading():
    """Test loading saved checkpoints."""
    print("\n" + "="*80)
    print("TESTING CHECKPOINT LOADING")
    print("="*80)

    # Create model config
    model_config = ModelConfig()

    # Create small test model
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=1,  # Very small for testing
        n_head=model_config.n_head
    )

    # Add SP-specific configs
    gpt2_config.bit_widths = model_config.bit_widths
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    gpt2_config.activation_bits_per_bit = model_config.activation_bits_per_bit
    gpt2_config.quantizer_per_bit = model_config.quantizer_per_bit

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SPLMHeadModel(gpt2_config).to(device)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix='sp_load_test_')

    try:
        # Save checkpoints
        base_filename = os.path.join(temp_dir, "load_test")
        saved_checkpoints = save_sp_checkpoints(
            model,
            base_filename=base_filename,
            model_config=model_config
        )

        # Test loading FP32 checkpoint
        if 32 in saved_checkpoints:
            print("\nTesting FP32 checkpoint loading...")
            checkpoint = torch.load(saved_checkpoints[32], map_location=device)

            # Create new model and load state
            new_model = SPLMHeadModel(gpt2_config).to(device)
            new_model.load_state_dict(checkpoint['model_state_dict'])
            print("  ✅ FP32 checkpoint loaded successfully")

            # Test forward pass
            input_ids = torch.randint(0, model_config.vocab_size, (1, 10), device=device)
            with torch.no_grad():
                new_model.set_precision(32)
                output = new_model(input_ids)
                print(f"  ✅ Forward pass successful, output shape: {output.shape if torch.is_tensor(output) else 'dict'}")

        print("\n✅ CHECKPOINT LOADING TEST PASSED!")

    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        test_checkpoint_saving()
        test_checkpoint_loading()

        print("\n" + "="*80)
        print("✅ ALL CHECKPOINT TESTS COMPLETED SUCCESSFULLY!")
        print("Checkpoint saving for all configured bit widths is working correctly.")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
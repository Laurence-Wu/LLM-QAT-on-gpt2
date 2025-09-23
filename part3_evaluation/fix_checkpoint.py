#!/usr/bin/env python3
"""
Script to diagnose and fix checkpoint saving issues.
Creates a working checkpoint from a trained model or creates a test checkpoint.
"""

import sys
import os
import torch
import json
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_checkpoint_file(filepath):
    """Check if a checkpoint file is valid."""
    print(f"\nChecking checkpoint: {filepath}")
    print("-" * 60)

    if not os.path.exists(filepath):
        print(f"❌ File does not exist")
        return False

    file_size = os.path.getsize(filepath)
    print(f"File size: {file_size / (1024*1024):.2f} MB")

    if file_size == 0:
        print(f"❌ File is empty (0 bytes)")
        return False

    if file_size < 1000:
        print(f"⚠️ File is suspiciously small ({file_size} bytes)")

    # Try to load the checkpoint
    try:
        print("Attempting to load checkpoint...")

        # Try with map_location='cpu' first (safer)
        checkpoint = torch.load(filepath, map_location='cpu')
        print("✅ Checkpoint loaded successfully")

        # Check contents
        if isinstance(checkpoint, dict):
            print(f"Keys in checkpoint: {list(checkpoint.keys())}")

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"✅ Found model_state_dict with {len(state_dict)} entries")
            else:
                print("⚠️ No 'model_state_dict' found")

            if 'model_config' in checkpoint:
                print(f"✅ Found model_config")
            else:
                print("⚠️ No 'model_config' found")

        return True

    except EOFError:
        print(f"❌ EOFError: File is truncated or corrupted")
        return False
    except RuntimeError as e:
        if "PytorchStreamReader failed" in str(e):
            print(f"❌ RuntimeError: Checkpoint file is corrupted")
            print(f"   Error: {e}")
        else:
            print(f"❌ RuntimeError: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return False


def create_minimal_checkpoint(output_path=None):
    """Create a minimal but valid checkpoint for testing."""
    from part1_switchable_precision.config_sp import ModelConfig
    from shared.models_sp import SPLMHeadModel
    from transformers import GPT2Config

    print("\n" + "="*80)
    print("CREATING MINIMAL TEST CHECKPOINT")
    print("="*80)

    # Use minimal configuration for quick testing
    model_config = ModelConfig()
    model_config.n_layer = 2  # Very small
    model_config.n_embd = 128
    model_config.n_head = 2
    model_config.n_positions = 128

    # Create GPT2Config
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        layer_norm_epsilon=1e-5
    )

    # Add SP configurations
    gpt2_config.bit_widths = model_config.bit_widths
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    gpt2_config.activation_bits_per_bit = model_config.activation_bits_per_bit
    gpt2_config.quantizer_per_bit = getattr(model_config, 'quantizer_per_bit', None)

    print(f"Creating tiny model: layers={model_config.n_layer}, embd={model_config.n_embd}")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SPLMHeadModel(gpt2_config).to(device)

    # Set to FP32 precision
    model.set_precision(32)

    # Create timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    if output_path is None:
        output_path = f"test_checkpoint_fp32_{timestamp}.pth"

    # Prepare checkpoint data
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': model_config.vocab_size,
            'n_positions': model_config.n_positions,
            'n_embd': model_config.n_embd,
            'n_layer': model_config.n_layer,
            'n_head': model_config.n_head,
            'layer_norm_epsilon': 1e-5,
            'bit_widths': model_config.bit_widths,
            'lora_rank_per_bit': model_config.lora_rank_per_bit,
            'lora_alpha_per_bit': model_config.lora_alpha_per_bit,
            'activation_bits_per_bit': model_config.activation_bits_per_bit,
            'quantization_bits': 32,
        },
        'training_config': {
            'batch_size': 4,
            'max_seq_length': model_config.n_positions,
            'learning_rate': 1e-4,
            'num_iterations': 100,
        },
        'bit_width': 32,
        'timestamp': timestamp
    }

    # Save with explicit error handling
    print(f"\nSaving checkpoint to: {output_path}")
    try:
        torch.save(checkpoint_data, output_path)

        # Verify the save worked
        file_size = os.path.getsize(output_path)
        print(f"✅ Checkpoint saved successfully")
        print(f"   File size: {file_size / (1024*1024):.2f} MB")

        # Try to reload it immediately to verify
        test_load = torch.load(output_path, map_location='cpu')
        print(f"✅ Verification: Checkpoint can be reloaded")

        return output_path

    except Exception as e:
        print(f"❌ Failed to save checkpoint: {e}")
        return None


def save_checkpoint_safely(model, filepath, model_config=None, training_config=None):
    """Safely save a checkpoint with verification."""
    print(f"\nSafely saving checkpoint to: {filepath}")

    # Prepare checkpoint data
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'bit_width': 32,
        'timestamp': time.strftime('%Y%m%d_%H%M%S')
    }

    if model_config is not None:
        if hasattr(model_config, '__dict__'):
            checkpoint_data['model_config'] = model_config.__dict__
        else:
            checkpoint_data['model_config'] = model_config

    if training_config is not None:
        if hasattr(training_config, '__dict__'):
            checkpoint_data['training_config'] = training_config.__dict__
        else:
            checkpoint_data['training_config'] = training_config

    # Create backup path
    backup_path = filepath + '.backup'

    try:
        # Save to backup first
        print("Saving to backup file first...")
        torch.save(checkpoint_data, backup_path)

        # Verify backup is valid
        print("Verifying backup...")
        test_load = torch.load(backup_path, map_location='cpu')

        # If backup is good, move to final location
        print("Moving to final location...")
        import shutil
        shutil.move(backup_path, filepath)

        # Final verification
        final_test = torch.load(filepath, map_location='cpu')

        file_size = os.path.getsize(filepath)
        print(f"✅ Checkpoint saved and verified successfully")
        print(f"   File size: {file_size / (1024*1024):.2f} MB")

        return True

    except Exception as e:
        print(f"❌ Failed to save checkpoint: {e}")

        # Clean up backup if it exists
        if os.path.exists(backup_path):
            try:
                os.remove(backup_path)
            except:
                pass

        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fix checkpoint issues')
    parser.add_argument('--check', type=str, help='Path to checkpoint to check')
    parser.add_argument('--create-test', action='store_true',
                       help='Create a minimal test checkpoint')
    parser.add_argument('--output', type=str, help='Output path for new checkpoint')

    args = parser.parse_args()

    if args.check:
        # Check existing checkpoint
        is_valid = check_checkpoint_file(args.check)

        if not is_valid:
            print("\n" + "="*60)
            print("CHECKPOINT IS CORRUPTED")
            print("="*60)
            print("\nSuggestions:")
            print("1. Re-run training to create a new checkpoint")
            print("2. Use --create-test to create a test checkpoint")
            print("3. Check if you have backup checkpoints")

    elif args.create_test:
        # Create test checkpoint
        output_path = args.output or None
        created_path = create_minimal_checkpoint(output_path)

        if created_path:
            print("\n" + "="*60)
            print("TEST CHECKPOINT CREATED")
            print("="*60)
            print(f"\nYou can now test evaluation with:")
            print(f"python part3_evaluation/main_llm_qat_eval.py \\")
            print(f"    --model_path {created_path} \\")
            print(f"    --output_dir results \\")
            print(f"    --max_eval_samples 10 \\")
            print(f"    --skip_few_shot")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
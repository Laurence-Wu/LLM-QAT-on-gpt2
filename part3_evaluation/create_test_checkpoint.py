#!/usr/bin/env python3
"""
Create a small test checkpoint for evaluation testing.
This creates a minimal but valid SP model checkpoint that can be used to test evaluation.
"""

import sys
import os
import torch
import json
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part1_switchable_precision.config_sp import ModelConfig
from shared.models_sp import SPLMHeadModel
from transformers import GPT2Config


def create_test_checkpoint(output_dir: str = ".", model_size: str = "tiny"):
    """Create a valid test checkpoint for evaluation.

    Args:
        output_dir: Directory to save checkpoint
        model_size: "tiny", "small", or "medium"
    """
    print("\n" + "="*80)
    print("CREATING TEST CHECKPOINT FOR EVALUATION")
    print("="*80)

    # Configure model size
    sizes = {
        "tiny": {"n_layer": 2, "n_embd": 128, "n_head": 2},
        "small": {"n_layer": 4, "n_embd": 256, "n_head": 4},
        "medium": {"n_layer": 6, "n_embd": 384, "n_head": 6}
    }

    size_config = sizes.get(model_size, sizes["tiny"])
    print(f"Model size: {model_size}")
    print(f"Configuration: {size_config}")

    # Create model config
    model_config = ModelConfig()

    # Override with smaller values for testing
    model_config.n_layer = size_config["n_layer"]
    model_config.n_embd = size_config["n_embd"]
    model_config.n_head = size_config["n_head"]
    model_config.n_positions = 256  # Smaller context for testing

    # Create GPT2Config
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        layer_norm_epsilon=model_config.layer_norm_epsilon
    )

    # Add SP-specific configurations
    gpt2_config.bit_widths = model_config.bit_widths
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    gpt2_config.activation_bits_per_bit = model_config.activation_bits_per_bit
    gpt2_config.quantizer_per_bit = model_config.quantizer_per_bit

    print(f"\nBit widths: {gpt2_config.bit_widths}")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nCreating SPLMHeadModel...")
    model = SPLMHeadModel(gpt2_config)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Initialize with small random weights (for testing)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param, gain=0.1)
            else:
                torch.nn.init.zeros_(param)

    # Create timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # Save FP32 checkpoint (32-bit teacher)
    print("\n" + "-"*60)
    print("Saving FP32 checkpoint...")

    model.set_precision(32)
    fp32_filename = os.path.join(output_dir, f"test_sp_model_32bit_fp32_{timestamp}.pth")

    # Create model config dict
    model_config_dict = {
        'vocab_size': model_config.vocab_size,
        'n_positions': model_config.n_positions,
        'n_embd': model_config.n_embd,
        'n_layer': model_config.n_layer,
        'n_head': model_config.n_head,
        'layer_norm_epsilon': model_config.layer_norm_epsilon,
        'bit_widths': model_config.bit_widths,
        'lora_rank_per_bit': model_config.lora_rank_per_bit,
        'lora_alpha_per_bit': model_config.lora_alpha_per_bit,
        'activation_bits_per_bit': model_config.activation_bits_per_bit,
        'quantization_bits': 32,
        'teacher_bits': 32
    }

    # Create training config dict (minimal)
    training_config_dict = {
        'batch_size': 4,
        'max_seq_length': model_config.n_positions,
        'learning_rate': 1e-4,
        'num_iterations': 100,
        'test_checkpoint': True
    }

    # Save checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config_dict,
        'training_config': training_config_dict,
        'bit_width': 32,
        'timestamp': timestamp,
        'test_checkpoint': True,
        'description': f'Test checkpoint for evaluation ({model_size} model)'
    }

    torch.save(checkpoint, fp32_filename)
    file_size_mb = os.path.getsize(fp32_filename) / (1024 * 1024)
    print(f"✅ Saved: {fp32_filename}")
    print(f"   Size: {file_size_mb:.2f} MB")

    # Also save matching JSON config
    json_filename = os.path.join(output_dir, f"test_training_stats_{timestamp}.json")
    json_config = {
        'model_config': model_config_dict,
        'training_config': training_config_dict,
        'final_metrics': {
            'test_checkpoint': True,
            'description': f'Test configuration for {model_size} model'
        }
    }

    with open(json_filename, 'w') as f:
        json.dump(json_config, f, indent=2)
    print(f"✅ Saved config: {json_filename}")

    # Save INT8 checkpoints for student models
    print("\n" + "-"*60)
    print("Saving INT8 checkpoints for students...")

    student_bits = [b for b in model_config.bit_widths if b < 32]
    for bits in student_bits:
        print(f"\nSaving {bits}-bit INT8 checkpoint...")
        model.set_precision(bits)

        # Import deploy functions
        from shared.deploy import save_int8_checkpoint

        int8_filename = os.path.join(output_dir, f"test_sp_model_{bits}bit_int8_{timestamp}.pth")

        # Create a simple model config class
        class SimpleConfig:
            def __init__(self, config_dict):
                self.__dict__.update(config_dict)

        config_obj = SimpleConfig(model_config_dict)
        training_obj = SimpleConfig(training_config_dict)

        save_int8_checkpoint(
            model,
            int8_filename,
            config_obj,
            training_obj,
            target_bits=bits
        )

        print(f"✅ Saved: {int8_filename}")

    print("\n" + "="*80)
    print("TEST CHECKPOINTS CREATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nMain checkpoint: {fp32_filename}")
    print(f"Config file: {json_filename}")
    print("\nYou can now test evaluation with:")
    print(f"python part3_evaluation/main_llm_qat_eval.py \\")
    print(f"    --model_path {fp32_filename} \\")
    print(f"    --config_path {json_filename} \\")
    print(f"    --output_dir results \\")
    print(f"    --max_eval_samples 10 \\")
    print(f"    --skip_few_shot")

    return fp32_filename, json_filename


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Create test checkpoints for SP model evaluation'
    )
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Directory to save checkpoints')
    parser.add_argument('--model_size', type=str, default='tiny',
                       choices=['tiny', 'small', 'medium'],
                       help='Model size for testing')

    args = parser.parse_args()

    # Create output directory if needed
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Create checkpoints
    create_test_checkpoint(args.output_dir, args.model_size)


if __name__ == "__main__":
    main()
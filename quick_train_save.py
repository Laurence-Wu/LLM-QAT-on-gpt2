#!/usr/bin/env python3
"""
Quick training script to generate a checkpoint for testing
"""

import torch
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared.models import SwitchableQATGPT2
from transformers import GPT2Config

def create_and_save_model():
    """Create a model and save it for testing"""

    # Create config
    config = GPT2Config(
        vocab_size=50257,
        n_positions=256,  # Use 256 to match training config
        n_embd=768,
        n_layer=6,
        n_head=12,
        layer_norm_epsilon=1e-5,
        embd_pdrop=0.1,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1
    )

    # Create model with switchable precision
    bit_widths = [4, 8, 16]
    model = SwitchableQATGPT2(config, bit_widths=bit_widths)

    # Create a simple checkpoint
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': config.vocab_size,
            'n_positions': config.n_positions,
            'n_embd': config.n_embd,
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'layer_norm_epsilon': config.layer_norm_epsilon,
            'embd_pdrop': config.embd_pdrop,
            'lora_rank': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'bit_widths': bit_widths,
            'quantization_bits': 8
        },
        'training_config': {
            'num_iterations': 100,
            'batch_size': 8,
            'learning_rate': 1e-4
        },
        'timestamp': timestamp
    }

    # Save the checkpoint
    filename = f"test_model_switchable_{timestamp}.pth"
    torch.save(checkpoint, filename)

    print(f"Model saved to {filename}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Supported bit-widths: {bit_widths}")

    return filename

if __name__ == "__main__":
    filename = create_and_save_model()
    print(f"\nYou can now test evaluation with:")
    print(f"python part3_evaluation/main_llm_qat_eval.py --model_path {filename}")
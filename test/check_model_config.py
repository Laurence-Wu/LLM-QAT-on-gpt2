#!/usr/bin/env python3
"""
Script to check the configuration of a saved model
"""

import sys
import torch

def check_model_config(model_path):
    """Check what configuration a saved model has"""
    print(f"Loading checkpoint from: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')

    if isinstance(checkpoint, dict):
        print("\nCheckpoint type: Dictionary")
        print(f"Keys in checkpoint: {list(checkpoint.keys())}")

        if 'model_config' in checkpoint:
            print("\n=== Model Configuration ===")
            config = checkpoint['model_config']
            for key, value in config.items():
                print(f"  {key}: {value}")

            if 'bit_widths' in config:
                print(f"\n✓ This model supports bit-widths: {config['bit_widths']}")
            else:
                print("\n✗ No bit_widths specified in config")

        if 'training_config' in checkpoint:
            print("\n=== Training Configuration ===")
            config = checkpoint['training_config']
            print(f"  Batch size: {config.get('batch_size', 'N/A')}")
            print(f"  Learning rate: {config.get('learning_rate', 'N/A')}")
            print(f"  Iterations: {config.get('num_iterations', 'N/A')}")

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"\n=== Model State Dict ===")
            print(f"  Total parameters: {len(state_dict)}")

            # Check what bit-widths are actually in the model
            bit_widths_found = set()
            for key in state_dict.keys():
                if 'quantizers_weight' in key or 'quantizers_input' in key:
                    # Extract bit width from key like "h.0.attn.c_attn.quantizers_weight.4bit.scale"
                    parts = key.split('.')
                    for part in parts:
                        if 'bit' in part:
                            bit_val = part.replace('bit', '')
                            try:
                                bit_widths_found.add(int(bit_val))
                            except:
                                pass

                if 'lora_adapters' in key:
                    # Extract bit width from key like "h.0.attn.c_attn.lora_adapters.4bit.lora_A"
                    parts = key.split('.')
                    for part in parts:
                        if 'bit' in part:
                            bit_val = part.replace('bit', '')
                            try:
                                bit_widths_found.add(int(bit_val))
                            except:
                                pass

            if bit_widths_found:
                print(f"\n✓ Bit-widths found in state_dict: {sorted(list(bit_widths_found))}")
            else:
                print("\n✗ No bit-width specific parameters found in state_dict")
                print("  This might be a non-switchable model")

            # Show a few example keys
            print("\n  Sample keys:")
            for i, key in enumerate(list(state_dict.keys())[:10]):
                print(f"    {key}")
            if len(state_dict) > 10:
                print(f"    ... and {len(state_dict) - 10} more")

    else:
        print("\nCheckpoint type: Direct model")
        print("Cannot extract configuration from direct model save")

    print("\n" + "="*50)
    print("To use this model for evaluation:")
    print(f"python part3_evaluation/main_llm_qat_eval.py --model_path {model_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_model_config.py <model_path>")
        sys.exit(1)

    check_model_config(sys.argv[1])
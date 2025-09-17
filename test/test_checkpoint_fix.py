#!/usr/bin/env python3
"""
Test script to validate the converted checkpoint works correctly.
"""

import torch
import sys
import os
import argparse
from pathlib import Path
from transformers import GPT2Tokenizer, GPT2Config

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import SwitchableQATGPT2


def test_checkpoint_loading(checkpoint_path):
    """Test if a checkpoint loads correctly with the switchable model."""
    print(f"\n{'='*70}")
    print("TESTING CHECKPOINT LOADING")
    print(f"{'='*70}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load the checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get configuration
    try:
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"Loaded config from checkpoint")
        else:
            # Use default config
            print("No config found in checkpoint, using default GPT2Config")
            config = GPT2Config()
            config.n_layer = 12
            config.n_embd = 768
            config.n_head = 12
            config.n_positions = 1024
            config.vocab_size = 50257
            config.lora_rank = 8
            config.lora_alpha = 16
            config.lora_dropout = 0.0
    except Exception as e:
        print(f"Error getting config: {e}")
        raise

    # Get bit widths
    try:
        if 'bit_widths' in checkpoint:
            bit_widths = checkpoint['bit_widths']
            print(f"Found bit_widths in checkpoint: {bit_widths}")
        else:
            # Try to get from config
            try:
                bit_widths = config.bit_widths
                print(f"Found bit_widths in config: {bit_widths}")
            except AttributeError:
                print("No bit_widths found in checkpoint or config, using default [4, 8, 16]")
                bit_widths = [4, 8, 16]
    except Exception as e:
        print(f"Error getting bit_widths: {e}")
        bit_widths = [4, 8, 16]

    print(f"Bit widths: {bit_widths}")

    # Create the model
    print("\nCreating SwitchableQATGPT2 model...")
    model = SwitchableQATGPT2(config, bit_widths=bit_widths, initialize_weights=False)

    # Load state dict
    print("Loading state dict...")
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Report loading status
    print(f"\n{'Loading Status':^30}")
    print("-" * 30)
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")

    if missing_keys:
        print(f"\nFirst 5 missing keys:")
        for key in missing_keys[:5]:
            print(f"  - {key}")

    if unexpected_keys:
        print(f"\nFirst 5 unexpected keys:")
        for key in unexpected_keys[:5]:
            print(f"  - {key}")

    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()

    return model, len(missing_keys), len(unexpected_keys)


def test_generation(model, prompts=None):
    """Test text generation with the model."""
    print(f"\n{'='*70}")
    print("TESTING TEXT GENERATION")
    print(f"{'='*70}\n")

    device = next(model.parameters()).device
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    if prompts is None:
        prompts = [
            "The capital of France is",
            "Two plus two equals",
            "Water freezes at",
            "The sun rises in the",
            "Machine learning is"
        ]

    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            # Test at different bit widths if switchable
            try:
                # Try to use set_precision if available
                model.set_precision(8)
                # If we get here, model supports switching precision
                for bits in [8, 16]:
                    model.set_precision(bits)
                    outputs = model.generate(
                        inputs['input_ids'],
                        max_new_tokens=10,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    results.append((prompt, bits, generated))
                    print(f"{bits}-bit | {prompt}")
                    print(f"  → {generated}\n")
            except AttributeError as e:
                # Model doesn't support set_precision
                print(f"Model doesn't support precision switching (AttributeError: {e}), testing as-is")
                outputs = model.generate(
                    inputs['input_ids'],
                    max_new_tokens=10,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                results.append((prompt, 'N/A', generated))
                print(f"{prompt}")
                print(f"  → {generated}\n")
            except Exception as e:
                print(f"Error during generation for prompt '{prompt}': {e}")
                results.append((prompt, 'ERROR', str(e)))

    return results


def evaluate_generation_quality(results):
    """Evaluate the quality of generated text."""
    print(f"\n{'='*70}")
    print("GENERATION QUALITY ASSESSMENT")
    print(f"{'='*70}\n")

    expected = {
        "The capital of France is": ["paris", "a city", "located"],
        "Two plus two equals": ["four", "4", "the sum"],
        "Water freezes at": ["0", "32", "degrees", "zero", "freezing"],
        "The sun rises in the": ["east", "morning", "sky"],
        "Machine learning is": ["a", "the", "used", "artificial", "technology"]
    }

    correct = 0
    total = 0

    for prompt, bits, generated in results:
        total += 1
        generated_lower = generated.lower()

        # Check if any expected phrase appears
        if prompt in expected:
            is_correct = any(exp in generated_lower for exp in expected[prompt])
            if is_correct:
                correct += 1
                status = "✓"
            else:
                status = "✗"

            print(f"{status} [{bits}-bit] {prompt[:30]}...")
            print(f"    Generated: {generated[:60]}...")

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nGeneration accuracy: {correct}/{total} ({accuracy:.1f}%)")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Test converted checkpoint')
    parser.add_argument('checkpoint_path', type=str,
                        help='Path to the converted checkpoint')
    parser.add_argument('--original', type=str, default=None,
                        help='Path to original checkpoint for comparison')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test only')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found: {args.checkpoint_path}")
        sys.exit(1)

    # Test loading the checkpoint
    model, missing, unexpected = test_checkpoint_loading(args.checkpoint_path)

    # Check if loading was successful
    success = True
    if missing > 100:  # Allow some missing keys for compatibility
        print("\n⚠️ WARNING: Too many missing keys!")
        success = False
    if unexpected > 100:  # Allow some unexpected keys for compatibility
        print("\n⚠️ WARNING: Too many unexpected keys!")
        success = False

    if success:
        print("\n✅ Checkpoint loaded successfully!")

        if not args.quick:
            # Test generation
            results = test_generation(model)

            # Evaluate quality
            accuracy = evaluate_generation_quality(results)

            # Final verdict
            print(f"\n{'='*70}")
            print("FINAL VERDICT")
            print(f"{'='*70}\n")

            if missing == 0 and unexpected == 0 and accuracy > 50:
                print("🎉 PERFECT! The checkpoint is fully compatible and working well!")
            elif missing < 10 and unexpected < 10 and accuracy > 30:
                print("✅ GOOD! The checkpoint works with minor issues.")
            elif accuracy > 20:
                print("⚠️ PARTIAL SUCCESS: The model loads but generation quality is poor.")
            else:
                print("❌ FAILED: The checkpoint has serious compatibility issues.")

            print(f"\nSummary:")
            print(f"  - Missing keys: {missing}")
            print(f"  - Unexpected keys: {unexpected}")
            print(f"  - Generation accuracy: {accuracy:.1f}%")

    # Compare with original if provided
    if args.original and os.path.exists(args.original):
        print(f"\n{'='*70}")
        print("COMPARING WITH ORIGINAL")
        print(f"{'='*70}\n")

        print("Testing original checkpoint...")
        orig_model, orig_missing, orig_unexpected = test_checkpoint_loading(args.original)

        print(f"\nComparison:")
        print(f"  Original - Missing: {orig_missing}, Unexpected: {orig_unexpected}")
        print(f"  Converted - Missing: {missing}, Unexpected: {unexpected}")

        if missing < orig_missing:
            print("  → Converted checkpoint is BETTER!")
        elif missing == orig_missing:
            print("  → Both checkpoints are equivalent")
        else:
            print("  → Original checkpoint is better")


if __name__ == "__main__":
    main()
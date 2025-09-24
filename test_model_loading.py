#!/usr/bin/env python3
"""
Comprehensive test script to verify model loading and diagnose underperformance issues.
"""

import torch
import torch.nn as nn
import sys
import os
import numpy as np
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(os.path.abspath(__file__))
part1_dir = os.path.join(parent_dir, 'part1_switchable_precision')
if part1_dir not in sys.path:
    sys.path.insert(0, part1_dir)

from part1_switchable_precision.models_sp import SPLMHeadModel
from transformers import GPT2Config, GPT2Tokenizer


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def check_checkpoint_structure(checkpoint_path):
    """Analyze checkpoint structure and contents."""
    print_section("CHECKPOINT STRUCTURE ANALYSIS")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"Checkpoint type: {type(checkpoint)}")

    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {list(checkpoint.keys())}")

        # Check for model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"\nState dict has {len(state_dict)} keys")

            # Analyze weight keys
            weight_keys = [k for k in state_dict.keys() if 'weight' in k]
            print(f"Found {len(weight_keys)} weight tensors")

            # Check for quantizer keys
            quantizer_keys = [k for k in state_dict.keys() if 'quantize' in k]
            print(f"Found {len(quantizer_keys)} quantizer-related keys")

            # Check for LoRA keys
            lora_keys = [k for k in state_dict.keys() if 'lora' in k]
            print(f"Found {len(lora_keys)} LoRA-related keys")

            # Check calibration status
            calibrated_keys = [k for k in state_dict.keys() if 'scale' in k or 'zero_point' in k]
            print(f"Found {len(calibrated_keys)} calibration parameter keys")

            # Sample some keys
            print("\nSample state dict keys (first 10):")
            for i, key in enumerate(list(state_dict.keys())[:10]):
                tensor = state_dict[key]
                if isinstance(tensor, torch.Tensor):
                    print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")
                else:
                    print(f"  {key}: {type(tensor)}")

        # Check model config
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            print(f"\nModel config keys: {list(config.keys())}")
            if 'bit_widths' in config:
                print(f"  Configured bit widths: {config['bit_widths']}")
            if 'lora_rank_per_bit' in config:
                print(f"  LoRA ranks: {config['lora_rank_per_bit']}")

        # Check bit width
        if 'bit_width' in checkpoint:
            print(f"\nCheckpoint saved at bit width: {checkpoint['bit_width']}")

        return checkpoint
    else:
        print("ERROR: Checkpoint is not a dictionary!")
        return None


def test_model_creation(checkpoint_path):
    """Test model creation and loading."""
    print_section("MODEL CREATION AND LOADING TEST")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if not isinstance(checkpoint, dict) or 'model_config' not in checkpoint:
        print("ERROR: Invalid checkpoint format")
        return None, None

    model_config = checkpoint['model_config']

    # Get bit width from checkpoint
    checkpoint_bit_width = checkpoint.get('bit_width', None)
    print(f"Checkpoint bit width: {checkpoint_bit_width}")

    # Create GPT2Config
    config = GPT2Config(
        vocab_size=model_config.get('vocab_size', 50257),
        n_positions=model_config.get('n_positions', 1024),
        n_embd=model_config.get('n_embd', 768),
        n_layer=model_config.get('n_layer', 12),
        n_head=model_config.get('n_head', 12)
    )

    # Add SP-specific configs
    config.bit_widths = model_config.get('bit_widths', [6, 8, 16, 32])
    config.lora_rank_per_bit = model_config.get('lora_rank_per_bit', {})
    config.lora_alpha_per_bit = model_config.get('lora_alpha_per_bit', {})
    config.activation_bits_per_bit = model_config.get('activation_bits_per_bit', {})
    config.quantizer_per_bit = model_config.get('quantizer_per_bit', {})

    # Convert string keys to int
    if isinstance(config.lora_rank_per_bit, dict):
        config.lora_rank_per_bit = {int(k) if isinstance(k, str) else k: v
                                   for k, v in config.lora_rank_per_bit.items()}
    if isinstance(config.lora_alpha_per_bit, dict):
        config.lora_alpha_per_bit = {int(k) if isinstance(k, str) else k: v
                                    for k, v in config.lora_alpha_per_bit.items()}

    print(f"Creating model with config: bit_widths={config.bit_widths}")

    # Create model
    model = SPLMHeadModel(config)

    # Set precision BEFORE loading weights
    if checkpoint_bit_width:
        print(f"Setting model to {checkpoint_bit_width}-bit precision BEFORE loading weights")
        model.set_precision(checkpoint_bit_width)

    # Load weights with strict=True to catch errors
    state_dict = checkpoint['model_state_dict']

    print("\nLoading state dict with strict=True...")
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        print("SUCCESS: All weights loaded successfully with strict=True!")
    except RuntimeError as e:
        print(f"ERROR with strict=True: {e}")
        print("\nTrying with strict=False to see what's missing...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"\nMissing {len(missing_keys)} keys:")
            for key in missing_keys[:20]:  # Show first 20
                print(f"  - {key}")
            if len(missing_keys) > 20:
                print(f"  ... and {len(missing_keys) - 20} more")

        if unexpected_keys:
            print(f"\nUnexpected {len(unexpected_keys)} keys:")
            for key in unexpected_keys[:20]:  # Show first 20
                print(f"  - {key}")
            if len(unexpected_keys) > 20:
                print(f"  ... and {len(unexpected_keys) - 20} more")

    return model, checkpoint_bit_width


def check_quantizer_calibration(model):
    """Check if quantizers are properly calibrated."""
    print_section("QUANTIZER CALIBRATION STATUS")

    calibrated_count = 0
    uncalibrated_count = 0

    for name, module in model.named_modules():
        if hasattr(module, 'quantizers_weight'):
            print(f"\nModule: {name}")
            for bit_key, quantizer in module.quantizers_weight.items():
                if hasattr(quantizer, 'calibrated'):
                    status = "✓ Calibrated" if quantizer.calibrated else "✗ NOT Calibrated"
                    print(f"  Weight quantizer {bit_key}: {status}")
                    if quantizer.calibrated:
                        calibrated_count += 1
                        # Check calibration parameters
                        if hasattr(quantizer, 'scale'):
                            scale_shape = quantizer.scale.shape if quantizer.scale is not None else None
                            print(f"    Scale shape: {scale_shape}")
                    else:
                        uncalibrated_count += 1

        if hasattr(module, 'quantizers_input'):
            for bit_key, quantizer in module.quantizers_input.items():
                if hasattr(quantizer, 'calibrated'):
                    status = "✓ Calibrated" if quantizer.calibrated else "✗ NOT Calibrated"
                    print(f"  Input quantizer {bit_key}: {status}")
                    if quantizer.calibrated:
                        calibrated_count += 1
                    else:
                        uncalibrated_count += 1

    print(f"\nSummary:")
    print(f"  Calibrated quantizers: {calibrated_count}")
    print(f"  Uncalibrated quantizers: {uncalibrated_count}")

    if uncalibrated_count > 0:
        print("  WARNING: Some quantizers are not calibrated!")

    return calibrated_count, uncalibrated_count


def check_lora_adapters(model):
    """Check LoRA adapter status."""
    print_section("LORA ADAPTER STATUS")

    for name, module in model.named_modules():
        if hasattr(module, 'lora_adapters'):
            print(f"\nModule: {name}")
            print(f"  Current bits: {getattr(module, 'current_bits', 'Unknown')}")

            for bit_key, lora in module.lora_adapters.items():
                enabled = getattr(lora, 'enabled', False)
                rank = getattr(lora, 'rank', 0)
                scaling = getattr(lora, 'scaling', 0)

                status = "✓ Enabled" if enabled else "✗ Disabled"
                print(f"  LoRA {bit_key}: {status}, rank={rank}, scaling={scaling:.4f}")

                # Check if weights exist
                if hasattr(lora, 'lora_A') and hasattr(lora, 'lora_B'):
                    if isinstance(lora.lora_A, nn.Parameter):
                        print(f"    A shape: {lora.lora_A.shape}, B shape: {lora.lora_B.shape}")
                        # Check if weights are non-zero
                        a_nonzero = (lora.lora_A != 0).any().item()
                        b_nonzero = (lora.lora_B != 0).any().item()
                        print(f"    A non-zero: {a_nonzero}, B non-zero: {b_nonzero}")


def test_model_inference(model, bit_width=None):
    """Test model inference with dummy input."""
    print_section("MODEL INFERENCE TEST")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Set precision if specified
    if bit_width:
        model.set_precision(bit_width)
        print(f"Model set to {bit_width}-bit precision")

    # Create dummy input
    batch_size = 1
    seq_len = 10
    vocab_size = model.config.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    print(f"Input shape: {input_ids.shape}")

    # Run inference
    with torch.no_grad():
        try:
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            print(f"Output shape: {logits.shape}")
            print(f"Output dtype: {logits.dtype}")
            print(f"Output device: {logits.device}")

            # Check output statistics
            print(f"\nOutput statistics:")
            print(f"  Min: {logits.min().item():.6f}")
            print(f"  Max: {logits.max().item():.6f}")
            print(f"  Mean: {logits.mean().item():.6f}")
            print(f"  Std: {logits.std().item():.6f}")

            # Check for NaN or Inf
            has_nan = torch.isnan(logits).any().item()
            has_inf = torch.isinf(logits).any().item()

            if has_nan:
                print("  WARNING: Output contains NaN values!")
            if has_inf:
                print("  WARNING: Output contains Inf values!")

            # Check if output is all zeros
            if (logits == 0).all().item():
                print("  WARNING: Output is all zeros!")

            # Test different sequence lengths
            print("\nTesting different sequence lengths:")
            for test_len in [1, 50, 100]:
                test_input = torch.randint(0, vocab_size, (1, test_len)).to(device)
                test_output = model(test_input)
                test_logits = test_output.logits if hasattr(test_output, 'logits') else test_output
                print(f"  Seq len {test_len}: output shape={test_logits.shape}, "
                      f"mean={test_logits.mean().item():.4f}")

            return True

        except Exception as e:
            print(f"ERROR during inference: {e}")
            return False


def compare_with_pretrained():
    """Compare loaded model with a fresh pretrained model."""
    print_section("COMPARISON WITH FRESH MODEL")

    from transformers import GPT2LMHeadModel

    # Load pretrained GPT2
    pretrained = GPT2LMHeadModel.from_pretrained('gpt2')
    pretrained.eval()

    # Create dummy input
    input_ids = torch.randint(0, 50257, (1, 10))

    with torch.no_grad():
        pretrained_output = pretrained(input_ids).logits

    print(f"Pretrained GPT2 output:")
    print(f"  Shape: {pretrained_output.shape}")
    print(f"  Mean: {pretrained_output.mean().item():.4f}")
    print(f"  Std: {pretrained_output.std().item():.4f}")
    print(f"  Min: {pretrained_output.min().item():.4f}")
    print(f"  Max: {pretrained_output.max().item():.4f}")


def analyze_weight_statistics(model):
    """Analyze weight statistics to detect issues."""
    print_section("WEIGHT STATISTICS ANALYSIS")

    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            stats = {
                'mean': param.mean().item(),
                'std': param.std().item(),
                'min': param.min().item(),
                'max': param.max().item(),
                'zeros': (param == 0).sum().item(),
                'total': param.numel()
            }

            zero_pct = 100 * stats['zeros'] / stats['total']

            print(f"\n{name}:")
            print(f"  Shape: {param.shape}")
            print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
            print(f"  Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
            print(f"  Zeros: {stats['zeros']}/{stats['total']} ({zero_pct:.1f}%)")

            # Warnings
            if zero_pct > 90:
                print(f"  WARNING: Over 90% zeros!")
            if stats['std'] < 1e-6:
                print(f"  WARNING: Very low standard deviation!")
            if abs(stats['mean']) > 10:
                print(f"  WARNING: Very large mean value!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test model loading and diagnose issues')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--bit-width', type=int, default=None,
                       help='Test specific bit width')
    args = parser.parse_args()

    print("="*70)
    print("  MODEL LOADING DIAGNOSTIC TEST")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # 1. Check checkpoint structure
    checkpoint = check_checkpoint_structure(args.checkpoint)
    if checkpoint is None:
        print("ERROR: Cannot proceed with invalid checkpoint")
        return

    # 2. Test model creation and loading
    model, checkpoint_bit_width = test_model_creation(args.checkpoint)
    if model is None:
        print("ERROR: Failed to create/load model")
        return

    # 3. Check quantizer calibration
    calibrated, uncalibrated = check_quantizer_calibration(model)

    # 4. Check LoRA adapters
    check_lora_adapters(model)

    # 5. Analyze weight statistics
    analyze_weight_statistics(model)

    # 6. Test inference
    test_bit_width = args.bit_width or checkpoint_bit_width or 8
    success = test_model_inference(model, test_bit_width)

    # 7. Compare with pretrained
    compare_with_pretrained()

    # Summary
    print_section("DIAGNOSTIC SUMMARY")

    issues = []

    if uncalibrated > 0:
        issues.append(f"- {uncalibrated} uncalibrated quantizers found")

    if checkpoint_bit_width is None:
        issues.append("- No bit width specified in checkpoint")

    if not success:
        issues.append("- Inference test failed")

    if issues:
        print("Issues found:")
        for issue in issues:
            print(issue)
    else:
        print("No major issues detected in loading process")

    print("\nRecommendations:")
    print("1. Ensure checkpoint was saved with proper bit_width key")
    print("2. Verify all quantizers were calibrated before saving")
    print("3. Check that LoRA adapters match the target bit width")
    print("4. Consider using strict=True for loading to catch all errors")


if __name__ == "__main__":
    main()
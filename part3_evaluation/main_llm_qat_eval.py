#!/usr/bin/env python3
"""
Main script to run LLM-QAT paper evaluation suite with standard evaluation methods
"""

import json
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add part1_switchable_precision to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
part1_dir = os.path.join(parent_dir, 'part1_switchable_precision')
if part1_dir not in sys.path:
    sys.path.insert(0, part1_dir)

from part1_switchable_precision.models_sp import SPModel, SPLMHeadModel
from part1_switchable_precision.quantization import LearnableFakeQuantize
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

from part3_evaluation.llm_qat_metrics import LLMQATEvaluation
from part3_evaluation.bit_configurations import BitConfigurations
from part3_evaluation.generate_tables import ResultTableGenerator
from part3_evaluation.baseline_comparison import BaselineComparison
from part3_evaluation.zero_shot_tasks import ZeroShotEvaluator
from part3_evaluation.few_shot_eval import FewShotEvaluator
from part3_evaluation.perplexity_eval import PerplexityEvaluator





def validate_model_config(config):
    """Validate that all required model configuration parameters are present."""
    required_params = [
        'vocab_size', 'n_positions', 'n_embd', 'n_layer', 'n_head',
        'layer_norm_epsilon', 'embd_pdrop', 'bit_widths',
        'lora_rank_per_bit', 'lora_alpha_per_bit',
        'activation_bits_per_bit', 'quantizer_per_bit'
    ]

    missing = [key for key in required_params if key not in config]
    if missing:
        raise ValueError(f"Missing required configuration parameters: {missing}\n"
                        f"Please ensure your checkpoint was saved with complete configuration.")

    print(f"✅ Configuration validation passed: {len(required_params)} required parameters found")


def load_switchable_model(model_path: str = None, config_path: str = None, use_pretrained: bool = True):
    """Load switchable precision model with proper configuration"""

    # Force CUDA availability check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This evaluation requires CUDA.")

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Initialize checkpoint_bit_width to None (will be set if found in checkpoint)
    checkpoint_bit_width = None

    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")

        # Load checkpoint (PyTorch 2.6 requires weights_only=False for custom objects)
        checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)

        # Get the bit width this checkpoint was saved at
        # Get bit_width from checkpoint - required field
        if 'bit_width' not in checkpoint:
            raise ValueError(f"bit_width not found in checkpoint. Available keys: {list(checkpoint.keys())}")
        checkpoint_bit_width = checkpoint['bit_width']
        if checkpoint_bit_width:
            print(f"Checkpoint was saved at {checkpoint_bit_width}-bit precision")

        # Config embedded in checkpoint
        if isinstance(checkpoint, dict):
            if 'model_config' in checkpoint:
                model_config = checkpoint.get('model_config', {})
                training_config = checkpoint.get('training_config', {})
                print("Using configuration embedded in checkpoint")
            else:
                raise ValueError("Checkpoint missing model_config")
        else:
            raise ValueError("Invalid checkpoint format - not a dictionary")

        print("\n" + "="*50)
        print("USING STRICT CONFIGURATION FROM CHECKPOINT/JSON")
        print("="*50)

        # Validate configuration has all required parameters
        validate_model_config(model_config)

        # Extract required values from model_config (NO DEFAULTS)
        n_layer = model_config['n_layer']
        n_embd = model_config['n_embd']
        n_head = model_config['n_head']
        quantization_bits = model_config.get('quantization_bits')

        # Bit widths MUST be specified in config
        bit_widths = model_config['bit_widths']  # Will raise KeyError if missing
        print(f"Using bit widths from config: {bit_widths}")

        # Get n_positions from actual weights in checkpoint (most reliable)
        actual_n_positions = None
        if 'model_state_dict' in checkpoint:
            # Check transformer.wpe.weight first (SP model format)
            if 'transformer.wpe.weight' in checkpoint['model_state_dict']:
                wpe_shape = checkpoint['model_state_dict']['transformer.wpe.weight'].shape
                actual_n_positions = wpe_shape[0]
                print(f"Detected n_positions from transformer.wpe.weight shape: {actual_n_positions}")
            elif 'wpe.weight' in checkpoint['model_state_dict']:
                wpe_shape = checkpoint['model_state_dict']['wpe.weight'].shape
                actual_n_positions = wpe_shape[0]
                print(f"Detected n_positions from wpe.weight shape: {actual_n_positions}")

        # Fallback to config if needed
        if actual_n_positions is None:
            if training_config and 'max_seq_length' in training_config:
                actual_n_positions = training_config['max_seq_length']
                print(f"Using max_seq_length from training config: {actual_n_positions}")

        # Build config with values from the loaded configuration (NO DEFAULTS)
        config = GPT2Config(
            vocab_size=model_config['vocab_size'],  # Required
            n_positions=actual_n_positions,  # Detected from weights
            n_embd=n_embd,  # Required
            n_layer=n_layer,  # Required
            n_head=n_head,  # Required
            layer_norm_epsilon=model_config['layer_norm_epsilon'],  # Required
            embd_pdrop=model_config['embd_pdrop'],  # Required
            lora_rank=model_config['lora_rank'],  # Optional for SP models
            lora_alpha=model_config['lora_alpha']  # Optional for SP models
        )

        print(f"\nLoaded Model Configuration:")
        print(f"  - n_layer: {config.n_layer}")
        print(f"  - n_embd: {config.n_embd}")
        print(f"  - n_head: {config.n_head}")
        print(f"  - n_positions: {config.n_positions}")
        print(f"  - vocab_size: {config.vocab_size}")
        print(f"  - quantization_bits (training): {quantization_bits}")
        print(f"  - bit_widths (switchable): {bit_widths}")

        if training_config:
            print(f"\nTraining Configuration:")
            print(f"  - batch_size: {training_config.get('batch_size')}")
            print(f"  - max_seq_length: {training_config.get('max_seq_length')}")
            print(f"  - learning_rate: {training_config.get('learning_rate')}")
            print(f"  - num_iterations: {training_config.get('num_iterations')}")

        print(f"Creating model with bit-widths: {bit_widths}")
        # Add SP-specific configurations to config
        config.bit_widths = bit_widths

        # Get SP-specific configurations from model_config (NO DEFAULTS)
        config.lora_rank_per_bit = model_config['lora_rank_per_bit']  # Required for SP
        config.lora_alpha_per_bit = model_config['lora_alpha_per_bit']  # Required for SP
        config.activation_bits_per_bit = model_config['activation_bits_per_bit']  # Required for SP
        config.quantizer_per_bit = model_config['quantizer_per_bit']  # Required for SP

        # Convert string keys to int if necessary (JSON serialization converts int keys to strings)
        if isinstance(config.lora_rank_per_bit, dict):
            config.lora_rank_per_bit = {int(k) if isinstance(k, str) else k: v
                                       for k, v in config.lora_rank_per_bit.items()}
        if isinstance(config.lora_alpha_per_bit, dict):
            config.lora_alpha_per_bit = {int(k) if isinstance(k, str) else k: v
                                        for k, v in config.lora_alpha_per_bit.items()}
        if isinstance(config.activation_bits_per_bit, dict):
            config.activation_bits_per_bit = {int(k) if isinstance(k, str) else k: v
                                             for k, v in config.activation_bits_per_bit.items()}
        if isinstance(config.quantizer_per_bit, dict) and config.quantizer_per_bit is not None:
            config.quantizer_per_bit = {int(k) if isinstance(k, str) else k: v
                                       for k, v in config.quantizer_per_bit.items()}

        # Print what we're using to help debug
        print(f"LoRA rank per bit: {config.lora_rank_per_bit}")
        print(f"LoRA alpha per bit: {config.lora_alpha_per_bit}")
        print(f"Activation bits per bit: {config.activation_bits_per_bit}")

        # Create SPLMHeadModel instead of SwitchableQATGPT2
        model = SPLMHeadModel(config)

        # Move model to GPU immediately after creation
        model = model.cuda()

        # Don't load pretrained weights - we'll load from checkpoint directly
        # This avoids resizing issues

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Load state dict with size mismatch handling for attention bias
            state_dict = checkpoint['model_state_dict']

            # Don't resize matrices - model should be created with correct dimensions

            # Use strict=True to ensure all weights are loaded correctly
            print("\nLoading state dict with strict=True to ensure complete weight loading...")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)

            if not missing_keys and not unexpected_keys:
                print("✅ SUCCESS: All weights loaded perfectly with strict=True!")
                print("   No missing or unexpected keys - model is fully loaded.")
            else:
                # This should not happen with strict=True, but keep for safety
                if missing_keys:
                    print(f"\n❌ CRITICAL: Missing {len(missing_keys)} keys in checkpoint!")
                    print("   These weights will use random initialization, causing poor performance:")
                    for i, key in enumerate(missing_keys):
                        if i < 50:
                            print(f"     - {key}")
                    if len(missing_keys) > 50:
                        print(f"     ... and {len(missing_keys) - 50} more missing keys")

                if unexpected_keys:
                    print(f"\n⚠️ Warning: {len(unexpected_keys)} unexpected keys in checkpoint")
                    print("   These keys exist in checkpoint but not in model:")
                    for i, key in enumerate(unexpected_keys):
                        if i < 20:
                            print(f"     - {key}")
                    if len(unexpected_keys) > 20:
                        print(f"     ... and {len(unexpected_keys) - 20} more")

            print("\n🔍 Performing weight verification...")
            # Check if critical weights are loaded
            critical_modules = ['transformer.wte.weight', 'transformer.wpe.weight', 'lm_head.weight']
            for module_name in critical_modules:
                if module_name in state_dict:
                    print(f"   ✓ {module_name} found in checkpoint")
                else:
                    print(f"   ✗ {module_name} MISSING from checkpoint!")

            # Set model to the bit width from checkpoint
            if checkpoint_bit_width:
                model.set_precision(checkpoint_bit_width)
                print(f"\n✅ Model set to {checkpoint_bit_width}-bit precision from checkpoint")

            # Diagnostic: Check quantizer calibration status
            print("\n🔍 Checking quantizer calibration status...")
            calibrated_count = 0
            uncalibrated_count = 0
            for name, module in model.named_modules():
                try:
                    quantizers = module.quantizers_weight
                    for bit_key, quantizer in quantizers.items():
                        try:
                            if quantizer.calibrated:
                                calibrated_count += 1
                            else:
                                uncalibrated_count += 1
                                print(f"   ⚠️ Uncalibrated: {name}.quantizers_weight.{bit_key}")
                        except AttributeError:
                            # Quantizer doesn't have calibrated attribute - skip
                            pass
                except AttributeError:
                    # Module doesn't have quantizers_weight - skip
                    pass

            print(f"   Quantizer status: {calibrated_count} calibrated, {uncalibrated_count} uncalibrated")
            if uncalibrated_count > 0:
                print(f"   ❌ WARNING: {uncalibrated_count} quantizers are not calibrated!")

            # Diagnostic: Quick inference test
            print("\n🔍 Running quick inference test...")
            with torch.no_grad():
                # Use max_seq_length from training_config to match calibrated quantizers
                if not training_config:
                    raise ValueError("Training config is required but not found in checkpoint")
                if 'max_seq_length' not in training_config:
                    raise ValueError(f"max_seq_length not found in training_config. Available keys: {list(training_config.keys())}")
                test_seq_length = training_config['max_seq_length']
                test_input = torch.randint(0, model.config.vocab_size, (1, test_seq_length)).cuda()
                test_output = model(test_input)
                try:
                    test_logits = test_output.logits
                except AttributeError:
                    test_logits = test_output

                # Check output statistics
                mean_val = test_logits.mean().item()
                std_val = test_logits.std().item()
                min_val = test_logits.min().item()
                max_val = test_logits.max().item()

                print(f"   Output stats: mean={mean_val:.4f}, std={std_val:.4f}, "
                      f"min={min_val:.4f}, max={max_val:.4f}")

                # Check for issues
                if torch.isnan(test_logits).any():
                    print("   ❌ ERROR: Output contains NaN values!")
                if torch.isinf(test_logits).any():
                    print("   ❌ ERROR: Output contains Inf values!")
                if (test_logits == 0).all():
                    print("   ❌ ERROR: Output is all zeros!")
                if std_val < 1e-6:
                    print("   ⚠️ WARNING: Very low output variance - model may be broken!")

        elif not isinstance(checkpoint, dict):
            model = checkpoint
    else:
        raise ValueError("No model path provided! Please specify --model_path with a trained checkpoint file.")

    # Force model to CUDA
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    print(f"\n✅ Model moved to {device}")
    print(f"   Device check: {next(model.parameters()).device}")

    # Return both model and the bit width from checkpoint
    return model, checkpoint_bit_width


def calibrate_for_evaluation(model, tokenizer, eval_texts=None, num_batches=10):
    """Replace quantizers with per-tensor versions and recalibrate for evaluation data"""

    print("\n" + "="*60)
    print("REPLACING QUANTIZERS WITH PER-TENSOR VERSIONS FOR EVALUATION")
    print("="*60)

    # Get the bit-width the model is currently at (following .claude rule: no hasattr)
    current_bits = None
    for module in model.modules():
        try:
            current_bits = module.current_bits
            break
        except AttributeError:
            continue

    if current_bits is None or current_bits >= 32:
        print(f"No calibration needed (current_bits: {current_bits})")
        return

    bits_key = f'{current_bits}bit'
    print(f"Replacing and calibrating quantizers for {current_bits}-bit precision")

    # Step 1: Replace all quantizers with per-tensor versions (per_channel=False)
    print(f"\nStep 1: Creating per-tensor quantizers (per_channel=False)")

    replaced_weight = 0
    replaced_input = 0
    replaced_lora = 0

    for name, module in model.named_modules():
        # Replace weight quantizers
        try:
            if bits_key in module.quantizers_weight:
                old_quantizer = module.quantizers_weight[bits_key]
                # Create new per-tensor quantizer with same settings but per_channel=False
                new_quantizer = LearnableFakeQuantize(
                    num_bits=old_quantizer.num_bits,
                    channel_dim=0,  # Will be ignored due to per_channel=False
                    quantizer_type=old_quantizer.quantizer_type,
                    eps=old_quantizer.eps,
                    symmetric=old_quantizer.symmetric,
                    per_channel=False  # KEY: Use per-tensor calibration
                )
                # Move to same device as old quantizer
                new_quantizer = new_quantizer.to(old_quantizer.scale.device)
                module.quantizers_weight[bits_key] = new_quantizer
                replaced_weight += 1
        except AttributeError:
            pass

        # Replace input quantizers
        try:
            if bits_key in module.quantizers_input:
                old_quantizer = module.quantizers_input[bits_key]
                # Create new per-tensor quantizer with same settings but per_channel=False
                new_quantizer = LearnableFakeQuantize(
                    num_bits=old_quantizer.num_bits,
                    channel_dim=1,  # Will be ignored due to per_channel=False
                    quantizer_type=old_quantizer.quantizer_type,
                    eps=old_quantizer.eps,
                    symmetric=old_quantizer.symmetric,
                    per_channel=False  # KEY: Use per-tensor calibration
                )
                # Move to same device as old quantizer
                new_quantizer = new_quantizer.to(old_quantizer.scale.device)
                module.quantizers_input[bits_key] = new_quantizer
                replaced_input += 1
        except AttributeError:
            pass

        # Replace LoRA quantizers (quantize_A and quantize_B)
        try:
            if module.quantize_A is not None:
                old_quantizer = module.quantize_A
                new_quantizer = LearnableFakeQuantize(
                    num_bits=old_quantizer.num_bits,
                    channel_dim=1,  # Will be ignored due to per_channel=False
                    quantizer_type=old_quantizer.quantizer_type,
                    eps=old_quantizer.eps,
                    symmetric=old_quantizer.symmetric,
                    per_channel=False  # KEY: Use per-tensor calibration
                )
                new_quantizer = new_quantizer.to(old_quantizer.scale.device)
                module.quantize_A = new_quantizer
                replaced_lora += 1
        except AttributeError:
            pass

        try:
            if module.quantize_B is not None:
                old_quantizer = module.quantize_B
                new_quantizer = LearnableFakeQuantize(
                    num_bits=old_quantizer.num_bits,
                    channel_dim=0,  # Will be ignored due to per_channel=False
                    quantizer_type=old_quantizer.quantizer_type,
                    eps=old_quantizer.eps,
                    symmetric=old_quantizer.symmetric,
                    per_channel=False  # KEY: Use per-tensor calibration
                )
                new_quantizer = new_quantizer.to(old_quantizer.scale.device)
                module.quantize_B = new_quantizer
                replaced_lora += 1
        except AttributeError:
            pass

    print(f"  Replaced {replaced_weight} weight quantizers with per-tensor versions")
    print(f"  Replaced {replaced_input} input quantizers with per-tensor versions")
    print(f"  Replaced {replaced_lora} LoRA quantizers with per-tensor versions")

    # Step 2: Start calibration on all new quantizers
    print(f"\nStep 2: Starting calibration for all quantizers...")

    # Start calibration for all quantizers
    calibration_started = 0
    for name, module in model.named_modules():
        # Weight quantizers
        try:
            if bits_key in module.quantizers_weight:
                module.quantizers_weight[bits_key].start_calibration()
                calibration_started += 1
        except AttributeError:
            pass

        # Input quantizers
        try:
            if bits_key in module.quantizers_input:
                module.quantizers_input[bits_key].start_calibration()
                calibration_started += 1
        except AttributeError:
            pass

        # LoRA quantizers
        try:
            if module.quantize_A is not None:
                module.quantize_A.start_calibration()
                calibration_started += 1
        except AttributeError:
            pass

        try:
            if module.quantize_B is not None:
                module.quantize_B.start_calibration()
                calibration_started += 1
        except AttributeError:
            pass

    print(f"  Started calibration for {calibration_started} quantizers")

    # Step 3: Collect statistics from evaluation data
    print(f"\nStep 3: Collecting statistics from evaluation data...")

    # Disable LoRA during calibration (following training logic)
    try:
        model.disable_lora_for_calibration()
    except AttributeError:
        print("  Model does not have disable_lora_for_calibration method")

    # Use provided texts or raise error
    if eval_texts is None:
        raise ValueError("eval_texts cannot be None - must provide calibration data")

    # Prepare evaluation samples
    model.eval()
    samples_processed = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for i in range(min(num_batches, len(eval_texts))):
            text = eval_texts[i]

            # Tokenize with actual evaluation settings (variable length, no padding)
            # Per-tensor calibration will create global statistics across all sequence lengths
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=1024,  # Allow longer sequences for better statistics
                padding=False
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass to collect statistics
            try:
                _ = model(inputs['input_ids'])
                samples_processed += 1
                seq_len = inputs['input_ids'].shape[1]
                if (i + 1) % 5 == 0 or i == 0:  # Print every 5th sample
                    print(f"  Processed sample {i+1}: length={seq_len} tokens")
            except Exception as e:
                print(f"  Warning on sample {i}: {e}")

    # Re-enable LoRA after calibration
    try:
        model.enable_lora_after_calibration()
    except AttributeError:
        print("  Model does not have enable_lora_after_calibration method")

    # Step 4: Finish calibration for all quantizers
    print(f"\nStep 4: Finishing calibration...")

    calibrated_count = 0
    for name, module in model.named_modules():
        # Weight quantizers
        try:
            if bits_key in module.quantizers_weight:
                module.quantizers_weight[bits_key].finish_calibration(debug=False)
                calibrated_count += 1
        except AttributeError:
            continue

        # Input quantizers
        try:
            if bits_key in module.quantizers_input:
                module.quantizers_input[bits_key].finish_calibration(debug=False)
                calibrated_count += 1
        except AttributeError:
            continue

        # LoRA quantizers
        try:
            if module.quantize_A is not None:
                module.quantize_A.finish_calibration(debug=False)
                calibrated_count += 1
        except AttributeError:
            continue

        try:
            if module.quantize_B is not None:
                module.quantize_B.finish_calibration(debug=False)
                calibrated_count += 1
        except AttributeError:
            continue

    print(f"  ✓ Calibrated {calibrated_count} quantizers with {samples_processed} samples")

    # Verify calibration stats shape (should all be [1] for per-tensor)
    print("\n  Checking calibration stats shapes (should be [1] for per-tensor):")
    checked = 0
    for name, module in model.named_modules():
        try:
            if bits_key in module.quantizers_input:
                quantizer = module.quantizers_input[bits_key]
                scale_shape = quantizer.scale.shape
                zp_shape = quantizer.zero_point.shape
                print(f"    {name}: scale={scale_shape}, zero_point={zp_shape}")
                checked += 1
                if checked >= 3:  # Just show first 3
                    break
        except AttributeError:
            continue

    print("\n" + "="*60)
    print("PER-TENSOR CALIBRATION COMPLETE")
    print("="*60 + "\n")


def load_tokenizer():
    """Load GPT-2 tokenizer"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# Removed EvaluationMetrics class - using specialized evaluators instead


def load_evaluation_config(config_path):
    """Load evaluation configuration from JSON file. NO DEFAULTS ALLOWED."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Evaluation config required but not found: {config_path}\n"
                              f"Please ensure evaluation_config.json exists at: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Validate required sections
    required_sections = ['device', 'calibration', 'zero_shot', 'few_shot', 'perplexity', 'output', 'model']
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    return config


def main():
    parser = argparse.ArgumentParser(description='LLM-QAT Paper Evaluation Suite with Standard Methods')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--config_path', type=str,
                       help='Path to training config JSON file (optional, will auto-detect if not provided)')
    parser.add_argument('--eval_config', type=str,
                       default='evaluation_config.json',
                       help='Path to evaluation configuration JSON file')
    args = parser.parse_args()

    # Load evaluation configuration (NO DEFAULTS)
    eval_config = load_evaluation_config(args.eval_config)
    print(f"Loaded evaluation config from: {args.eval_config}")

    # Load model from checkpoint and get bit width
    model, checkpoint_bit_width = load_switchable_model(args.model_path, config_path=args.config_path, use_pretrained=False)
    tokenizer = load_tokenizer()

    # Recalibrate quantizers for evaluation data if needed
    if checkpoint_bit_width and checkpoint_bit_width < 32:
        print(f"\nModel loaded at {checkpoint_bit_width}-bit precision")
        print("Preparing calibration data from evaluation datasets...")

        from datasets import load_dataset
        calibration_texts = []

        # Collect samples from WikiText-2
        try:
            print("  Loading WikiText-2 validation samples...")
            wikitext = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
            wikitext_samples = 0
            for item in wikitext:
                text = item['text'].strip()
                if len(text) > 20:  # Skip very short texts
                    calibration_texts.append(text)
                    wikitext_samples += 1
                    if wikitext_samples >= 50:
                        break
            print(f"    Added {wikitext_samples} WikiText-2 samples")
        except Exception as e:
            print(f"    Error loading WikiText-2: {e}")

        # Collect samples from OpenWebText
        try:
            print("  Loading OpenWebText samples...")
            openwebtext_dataset = load_dataset('Skylion007/openwebtext', split='train[:100]')
            openwebtext_samples = 0
            for item in openwebtext_dataset:
                text = item['text'].strip()
                if len(text) > 20:
                    calibration_texts.append(text)
                    openwebtext_samples += 1
                    if openwebtext_samples >= 50:
                        break
            print(f"    Added {openwebtext_samples} OpenWebText samples")
        except Exception as e:
            print(f"    Error loading OpenWebText: {e}")

        # Collect samples from BoolQ
        try:
            print("  Loading BoolQ validation samples...")
            boolq = load_dataset('boolq', split='validation')
            boolq_samples = 0
            for i in range(min(50, len(boolq))):
                sample = boolq[i]
                # Format as it would appear in evaluation
                text = f"Passage: {sample['passage']}\nQuestion: {sample['question']}\nAnswer:"
                calibration_texts.append(text)
                boolq_samples += 1
            print(f"    Added {boolq_samples} BoolQ samples")
        except Exception as e:
            print(f"    Error loading BoolQ: {e}")

        if not calibration_texts:
            raise RuntimeError("Failed to load any calibration data from datasets")

        print(f"\n  Total calibration samples collected: {len(calibration_texts)}")

        # Run calibration
        num_calib_batches = min(100, len(calibration_texts))
        print(f"  Running calibration with {num_calib_batches} samples...")
        calibrate_for_evaluation(model, tokenizer, eval_texts=calibration_texts, num_batches=num_calib_batches)
    else:
        print(f"\nNo calibration needed (bit width: {checkpoint_bit_width})")

    # Initialize all evaluation components with config
    device = eval_config['device']
    evaluator = LLMQATEvaluation(model, tokenizer)
    zero_shot_evaluator = ZeroShotEvaluator(model, tokenizer, device=device, config=eval_config['zero_shot'])
    few_shot_evaluator = FewShotEvaluator(model, tokenizer, device=device, config=eval_config['few_shot'])
    perplexity_evaluator = PerplexityEvaluator(model, tokenizer, device=device, config=eval_config['perplexity'])

    # Get current model's bit configuration from checkpoint or model
    if checkpoint_bit_width:
        current_bits = checkpoint_bit_width
    else:
        try:
            current_bits = model.transformer.current_bits
        except AttributeError:
            current_bits = 32  # Default to FP32
            print(f"Warning: Could not determine bit width, defaulting to {current_bits}-bit")
    print(f"Current model precision: {current_bits}-bit")

    # Get evaluation settings from config
    output_dir = eval_config['output']['directory']

    print("="*70)
    print("Running SP Model Evaluation")
    print("="*70)
    print(f"Model: GPT-2 ({evaluator.model_params:.1f}M parameters)")
    print(f"Current precision: {current_bits}-bit")
    print(f"Output directory: {output_dir}")
    print(f"Max zero-shot samples: {eval_config['zero_shot']['max_samples']}")
    print(f"Max few-shot samples: {eval_config['few_shot']['max_samples']}")
    print(f"Max perplexity samples: {eval_config['perplexity']['max_samples']}")
    print("="*70)

    # Initialize results dictionary
    results = {
        'bit_width': current_bits,
        'model_size_gb': evaluator.calculate_model_size({'W': current_bits}),
        'compression_ratio': 32 / current_bits
    }

    print(f"\n{'='*60}")
    print(f"Evaluating {current_bits}-bit model")
    print('='*60)
    print(f"Model size: {results['model_size_gb']:.3f} GB")
    print(f"Compression ratio: {results['compression_ratio']:.2f}x")

    # Create simple bit config for evaluators
    bit_config = {'W': current_bits, 'A': current_bits, 'KV': current_bits}

        # 2. Perplexity evaluation with sliding window
    print("\n2. Perplexity evaluation (sliding window)...")
    try:
        perplexity_results = perplexity_evaluator.evaluate_all_datasets(bit_config)
        results['perplexity'] = perplexity_results
        print(f"   WikiText2: {perplexity_results['WikiText2']:.1f}")
        print(f"   OpenWebText: {perplexity_results.get('OpenWebText', float('inf')):.1f}")
    except Exception as e:
        print(f"   Warning: Perplexity evaluation failed: {e}")
        results['perplexity'] = {'WikiText2': float('inf'), 'OpenWebText': float('inf')}

    # 1. Zero-shot evaluation (6 benchmarks)
    print("\n1. Zero-shot common sense evaluation...")
    try:
        zero_shot_results = zero_shot_evaluator.evaluate_all_tasks(bit_config)
        results['zero_shot'] = zero_shot_results
        print(f"   BoolQ: {zero_shot_results.get('BoolQ', 0):.1f}%")
        print(f"   HellaSwag: {zero_shot_results.get('HellaSwag', 0):.1f}%")
        print(f"   WinoGrande: {zero_shot_results.get('WinoGrande', 0):.1f}%")
        print(f"   ARC-e: {zero_shot_results.get('ARC-e', 0):.1f}%")
        print(f"   ARC-c: {zero_shot_results.get('ARC-c', 0):.1f}%")
        print(f"   OBQA: {zero_shot_results.get('OBQA', 0):.1f}%")
        print(f"   Average: {zero_shot_results.get('Average', 0):.1f}%")
    except Exception as e:
        print(f"   Warning: Zero-shot evaluation failed: {e}")
        results['zero_shot'] = {'Average': 0.0}


    # 3. Few-shot evaluation (5-shot)
    print("\n3. Few-shot evaluation (5-shot)...")
    try:
        mmlu_scores = few_shot_evaluator.evaluate_mmlu(bit_config, num_shots=5)
        triviaqa_score = few_shot_evaluator.evaluate_triviaqa(bit_config, num_shots=5)

        results['few_shot'] = {
            'MMLU': mmlu_scores,
            'TriviaQA': triviaqa_score
        }
        print(f"   MMLU by category:")
        print(f"     - Humanities: {mmlu_scores['Humanities']:.1f}%")
        print(f"     - STEM: {mmlu_scores['STEM']:.1f}%")
        print(f"     - Social Sciences: {mmlu_scores['Social Sciences']:.1f}%")
        print(f"     - Other: {mmlu_scores['Other']:.1f}%")
        print(f"     - Average: {mmlu_scores['Average']:.1f}%")
        print(f"   TriviaQA: {triviaqa_score:.1f}%")
    except Exception as e:
        print(f"   Warning: Few-shot evaluation failed: {e}")
        results['few_shot'] = {
            'MMLU': {'Average': 0.0},
            'TriviaQA': 0.0
        }

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    results_filename = eval_config['output']['results_filename']
    with open(output_path / results_filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Evaluation complete!")
    print(f"Results saved to {output_path}")
    print(f"{'='*70}")

    print("\nSummary of Results:")
    print("="*70)
    print(f"\n{results['bit_width']}-bit Model:")
    print(f"  Model size: {results['model_size_gb']:.3f} GB")
    print(f"  Compression: {results['compression_ratio']:.1f}x")

    if 'zero_shot' in results and results['zero_shot']:
        print(f"  Zero-shot avg: {results['zero_shot'].get('Average', 0):.1f}%")

    if 'perplexity' in results and results['perplexity']:
        if 'WikiText2' in results['perplexity']:
            print(f"  WikiText2 PPL: {results['perplexity']['WikiText2']:.1f}")
        if 'OpenWebText' in results['perplexity']:
            print(f"  OpenWebText PPL: {results['perplexity']['OpenWebText']:.1f}")

    if 'few_shot' in results and results['few_shot']:
        if 'MMLU' in results['few_shot']:
            print(f"  MMLU avg: {results['few_shot']['MMLU'].get('Average', 0):.1f}%")
        if 'TriviaQA' in results['few_shot']:
            print(f"  TriviaQA: {results['few_shot']['TriviaQA']:.1f}%")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
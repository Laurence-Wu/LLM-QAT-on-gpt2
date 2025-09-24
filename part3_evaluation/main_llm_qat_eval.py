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

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cuda')

        # Get the bit width this checkpoint was saved at
        checkpoint_bit_width = checkpoint.get('bit_width', None)
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
                if hasattr(module, 'quantizers_weight'):
                    for bit_key, quantizer in module.quantizers_weight.items():
                        if hasattr(quantizer, 'calibrated'):
                            if quantizer.calibrated:
                                calibrated_count += 1
                            else:
                                uncalibrated_count += 1
                                print(f"   ⚠️ Uncalibrated: {name}.quantizers_weight.{bit_key}")

            print(f"   Quantizer status: {calibrated_count} calibrated, {uncalibrated_count} uncalibrated")
            if uncalibrated_count > 0:
                print(f"   ❌ WARNING: {uncalibrated_count} quantizers are not calibrated!")

            # Diagnostic: Quick inference test
            print("\n🔍 Running quick inference test...")
            with torch.no_grad():
                test_input = torch.randint(0, model.config.vocab_size, (1, 10)).cuda()
                test_output = model(test_input)
                test_logits = test_output.logits if hasattr(test_output, 'logits') else test_output

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
    device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    print(f"\n✅ Model moved to {device}")
    print(f"   Device check: {next(model.parameters()).device}")

    # Return both model and the bit width from checkpoint
    return model, checkpoint_bit_width


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

    # Initialize all evaluation components with config
    device = eval_config['device']
    evaluator = LLMQATEvaluation(model, tokenizer)
    zero_shot_evaluator = ZeroShotEvaluator(model, tokenizer, device=device, config=eval_config['zero_shot'])
    few_shot_evaluator = FewShotEvaluator(model, tokenizer, device=device, config=eval_config['few_shot'])
    perplexity_evaluator = PerplexityEvaluator(model, tokenizer, device=device, config=eval_config['perplexity'])

    # No calibration needed - using calibration parameters from checkpoint
    print("\nUsing calibration parameters from checkpoint (no recalibration needed)")

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

    # 2. Perplexity evaluation with sliding window
    print("\n2. Perplexity evaluation (sliding window)...")
    try:
        perplexity_results = perplexity_evaluator.evaluate_all_datasets(bit_config)
        results['perplexity'] = perplexity_results
        print(f"   WikiText2: {perplexity_results['WikiText2']:.1f}")
        print(f"   C4: {perplexity_results['C4']:.1f}")
    except Exception as e:
        print(f"   Warning: Perplexity evaluation failed: {e}")
        results['perplexity'] = {'WikiText2': float('inf'), 'C4': float('inf')}

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
        if 'C4' in results['perplexity']:
            print(f"  C4 PPL: {results['perplexity']['C4']:.1f}")

    if 'few_shot' in results and results['few_shot']:
        if 'MMLU' in results['few_shot']:
            print(f"  MMLU avg: {results['few_shot']['MMLU'].get('Average', 0):.1f}%")
        if 'TriviaQA' in results['few_shot']:
            print(f"  TriviaQA: {results['few_shot']['TriviaQA']:.1f}%")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
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
from part3_evaluation.zero_shot_tasks import ZeroShotEvaluator
from part3_evaluation.few_shot_eval import FewShotEvaluator
from part3_evaluation.perplexity_eval import PerplexityEvaluator


def load_switchable_model(model_path: str = None, config_path: str = None, use_pretrained: bool = True):
    """Load switchable precision model with proper configuration"""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This evaluation requires CUDA.")

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    checkpoint_bit_width = None

    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)

        # Get checkpoint configuration
        checkpoint_bit_width = checkpoint.get('bit_width')
        if checkpoint_bit_width:
            print(f"Checkpoint was saved at {checkpoint_bit_width}-bit precision")

        # Extract configs
        model_config = checkpoint.get('model_config', {})
        training_config = checkpoint.get('training_config', {})

        # Extract key parameters
        n_layer = model_config['n_layer']
        n_embd = model_config['n_embd']
        n_head = model_config['n_head']
        bit_widths = model_config['bit_widths']

        # Get n_positions from weights or config
        actual_n_positions = None
        if 'model_state_dict' in checkpoint:
            if 'transformer.wpe.weight' in checkpoint['model_state_dict']:
                wpe_shape = checkpoint['model_state_dict']['transformer.wpe.weight'].shape
                actual_n_positions = wpe_shape[0]
            elif 'wpe.weight' in checkpoint['model_state_dict']:
                wpe_shape = checkpoint['model_state_dict']['wpe.weight'].shape
                actual_n_positions = wpe_shape[0]

        if actual_n_positions is None and training_config:
            actual_n_positions = training_config.get('max_seq_length', 1024)

        # Build GPT2Config
        config = GPT2Config(
            vocab_size=model_config.get('vocab_size', 50257),
            n_positions=actual_n_positions or 1024,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            layer_norm_epsilon=model_config.get('layer_norm_epsilon', 1e-5),
            embd_pdrop=model_config.get('embd_pdrop', 0.0),
            lora_rank=model_config.get('lora_rank', 0),
            lora_alpha=model_config.get('lora_alpha', 0)
        )

        print(f"\nModel Configuration:")
        print(f"  - n_layer: {config.n_layer}")
        print(f"  - n_embd: {config.n_embd}")
        print(f"  - n_positions: {config.n_positions}")
        print(f"  - bit_widths: {bit_widths}")

        # Add SP-specific configurations
        config.bit_widths = bit_widths
        config.lora_rank_per_bit = model_config.get('lora_rank_per_bit', {})
        config.lora_alpha_per_bit = model_config.get('lora_alpha_per_bit', {})
        config.activation_bits_per_bit = model_config.get('activation_bits_per_bit', {})
        config.quantizer_per_bit = model_config.get('quantizer_per_bit', {})
        # CRITICAL: Use per-tensor quantization for evaluation
        config.per_channel_quantization = False

        # Convert string keys to int for dictionaries
        for attr_name in ['lora_rank_per_bit', 'lora_alpha_per_bit', 'activation_bits_per_bit', 'quantizer_per_bit']:
            attr_val = getattr(config, attr_name)
            if isinstance(attr_val, dict):
                setattr(config, attr_name, {int(k) if isinstance(k, str) else k: v for k, v in attr_val.items()})

        # Create model (keep on CPU initially, like test_inference.py)
        model = SPLMHeadModel(config)

        # CRITICAL: Set precision BEFORE loading weights (matches test_inference.py)
        # This ensures quantizers are in the correct state when weights are loaded
        if checkpoint_bit_width:
            model.set_precision(checkpoint_bit_width)
            print(f"✅ Model set to {checkpoint_bit_width}-bit precision")

        # Load weights AFTER setting precision
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print("✅ Model weights loaded successfully")

        # Move to CUDA AFTER loading (matches test_inference.py)
        model = model.cuda()

    else:
        raise ValueError("No model path provided! Please specify --model_path with a trained checkpoint file.")

    # Set to evaluation mode
    model.eval()
    device = torch.device('cuda')
    print(f"✅ Model ready on {device}")

    return model, checkpoint_bit_width


def verify_calibration_status(model):
    """Verify that quantizers are calibrated from checkpoint - no recalibration needed"""

    # Get current bit width
    current_bits = None
    for module in model.modules():
        try:
            current_bits = module.current_bits
            break
        except AttributeError:
            continue

    if current_bits is None or current_bits >= 32:
        print(f"No quantization active (current_bits: {current_bits})")
        return model

    print(f"\nVerifying calibration for {current_bits}-bit precision...")

    # Check calibration status
    bits_key = f'{current_bits}bit'
    calibrated_count = 0
    total_count = 0

    for name, module in model.named_modules():
        if hasattr(module, 'quantizers_input') and bits_key in module.quantizers_input:
            quantizer = module.quantizers_input[bits_key]
            total_count += 1
            if quantizer.calibrated:
                calibrated_count += 1

    if total_count > 0:
        print(f"✓ {calibrated_count}/{total_count} input quantizers calibrated from checkpoint")
        if calibrated_count < total_count:
            print(f"⚠️ Warning: {total_count - calibrated_count} quantizers not calibrated")
    else:
        print("No input quantizers found")

    return model


def load_tokenizer():
    """Load GPT-2 tokenizer"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_evaluation_config(config_path):
    """Load evaluation configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Evaluation config not found: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='LLM-QAT Paper Evaluation Suite')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--config_path', type=str,
                       help='Path to training config JSON file')
    parser.add_argument('--eval_config', type=str,
                       default='evaluation_config.json',
                       help='Path to evaluation configuration JSON file')
    args = parser.parse_args()

    # Load configurations
    eval_config = load_evaluation_config(args.eval_config)
    print(f"Loaded evaluation config from: {args.eval_config}")

    # Load model and tokenizer
    model, checkpoint_bit_width = load_switchable_model(args.model_path, config_path=args.config_path, use_pretrained=False)
    tokenizer = load_tokenizer()

    # Verify calibration status (no recalibration - use training calibration)
    if checkpoint_bit_width and checkpoint_bit_width < 32:
        print(f"\nModel loaded at {checkpoint_bit_width}-bit precision")
        model = verify_calibration_status(model)
    else:
        print(f"No quantization active (bit width: {checkpoint_bit_width})")

    # Initialize evaluators
    device = eval_config.get('device', 'cuda')
    evaluator = LLMQATEvaluation(model, tokenizer)
    zero_shot_evaluator = ZeroShotEvaluator(model, tokenizer, device=device, config=eval_config.get('zero_shot', {}))
    few_shot_evaluator = FewShotEvaluator(model, tokenizer, device=device, config=eval_config.get('few_shot', {}))
    perplexity_evaluator = PerplexityEvaluator(model, tokenizer, device=device, config=eval_config.get('perplexity', {}))

    # Get current bit configuration
    current_bits = checkpoint_bit_width or 32
    print(f"\n{'='*70}")
    print(f"Running Evaluation at {current_bits}-bit precision")
    print(f"{'='*70}")

    # Initialize results
    results = {
        'bit_width': current_bits,
        'model_size_gb': evaluator.calculate_model_size({'W': current_bits}),
        'compression_ratio': 32 / current_bits
    }

    bit_config = {'W': current_bits, 'A': current_bits, 'KV': current_bits}

    # 1. Perplexity evaluation
    print("\n1. Perplexity evaluation...")
    try:
        perplexity_results = perplexity_evaluator.evaluate_all_datasets(bit_config)
        results['perplexity'] = perplexity_results
        for dataset, ppl in perplexity_results.items():
            print(f"   {dataset}: {ppl:.1f}")
    except Exception as e:
        print(f"   Warning: Perplexity evaluation failed: {e}")
        results['perplexity'] = {}

    # 2. Zero-shot evaluation
    print("\n2. Zero-shot evaluation...")
    try:
        zero_shot_results = zero_shot_evaluator.evaluate_all_tasks(bit_config)
        results['zero_shot'] = zero_shot_results
        for task, score in zero_shot_results.items():
            if task != 'Average':
                print(f"   {task}: {score:.1f}%")
        print(f"   Average: {zero_shot_results.get('Average', 0):.1f}%")
    except Exception as e:
        print(f"   Warning: Zero-shot evaluation failed: {e}")
        results['zero_shot'] = {}

    # 3. Few-shot evaluation
    print("\n3. Few-shot evaluation (5-shot)...")
    try:
        mmlu_scores = few_shot_evaluator.evaluate_mmlu(bit_config, num_shots=5)
        triviaqa_score = few_shot_evaluator.evaluate_triviaqa(bit_config, num_shots=5)

        results['few_shot'] = {
            'MMLU': mmlu_scores,
            'TriviaQA': triviaqa_score
        }

        print(f"   MMLU Average: {mmlu_scores.get('Average', 0):.1f}%")
        print(f"   TriviaQA: {triviaqa_score:.1f}%")
    except Exception as e:
        print(f"   Warning: Few-shot evaluation failed: {e}")
        results['few_shot'] = {}

    # Save results
    output_dir = Path(eval_config.get('output', {}).get('directory', 'results'))
    output_dir.mkdir(exist_ok=True, parents=True)

    results_file = output_dir / f"results_{current_bits}bit.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("Evaluation Complete!")
    print(f"Results saved to {results_file}")
    print(f"{'='*70}")

    # Print summary
    print("\nSummary:")
    print(f"  Model: {current_bits}-bit")
    print(f"  Size: {results['model_size_gb']:.3f} GB")
    print(f"  Compression: {results['compression_ratio']:.1f}x")

    if results.get('perplexity'):
        if 'WikiText2' in results['perplexity']:
            print(f"  WikiText2 PPL: {results['perplexity']['WikiText2']:.1f}")

    if results.get('zero_shot'):
        if 'Average' in results['zero_shot']:
            print(f"  Zero-shot Avg: {results['zero_shot']['Average']:.1f}%")

    if results.get('few_shot'):
        if 'MMLU' in results['few_shot']:
            print(f"  MMLU Avg: {results['few_shot']['MMLU'].get('Average', 0):.1f}%")


if __name__ == "__main__":
    main()
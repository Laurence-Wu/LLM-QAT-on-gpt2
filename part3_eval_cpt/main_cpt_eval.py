#!/usr/bin/env python3
"""
Main script to run CPT model evaluation suite with standard evaluation methods
Compatible with Part 3 evaluation infrastructure
"""

import json
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import CPT model loader
from load_cpt_model import load_cpt_model, verify_cpt_quantization_status

# Import standard evaluation components
from transformers import GPT2Tokenizer
from cpt_metrics import CPTEvaluation
from zero_shot_tasks import ZeroShotEvaluator
from perplexity_eval import PerplexityEvaluator


def load_tokenizer():
    """Load GPT-2 tokenizer"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_evaluation_config(config_path):
    """Load evaluation configuration from JSON file"""
    if not os.path.exists(config_path):
        # Try looking in part3_eval_cpt directory
        alt_path = os.path.join(os.path.dirname(__file__), config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Evaluation config not found: {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='CPT Model Evaluation Suite')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained CPT model checkpoint (.pth file)')
    parser.add_argument('--eval_config', type=str,
                       default='evaluation_config.json',
                       help='Path to evaluation configuration JSON file')
    parser.add_argument('--diagnose', action='store_true',
                       help='Run quantization health diagnostic before evaluation')
    args = parser.parse_args()

    # Load configurations
    eval_config = load_evaluation_config(args.eval_config)
    print(f"Loaded evaluation config from: {args.eval_config}")

    # Load CPT model and tokenizer
    model, checkpoint_bit_width, model_config, training_config = load_cpt_model(args.model_path)
    tokenizer = load_tokenizer()

    # Verify quantization status
    if checkpoint_bit_width and checkpoint_bit_width < 32:
        print(f"\nModel loaded at {checkpoint_bit_width}-bit precision")
        model = verify_cpt_quantization_status(model, checkpoint_bit_width)
    else:
        print(f"No quantization active (bit width: {checkpoint_bit_width})")

    # Initialize evaluators (same as SP evaluation)
    device = eval_config['device']

    # Run diagnostic if requested
    if args.diagnose:
        print("\n" + "="*70)
        print("Running Comprehensive Quantization Diagnostics...")
        print("="*70)
        from diagnose_quantization import (
            comprehensive_diagnosis,
            test_sliding_window_perplexity,
            track_batch_degradation
        )

        # Run comprehensive diagnosis
        diagnostic_results = comprehensive_diagnosis(model, tokenizer, device)

        # Check for issues
        if 'perplexity_test' in diagnostic_results:
            ppl = diagnostic_results['perplexity_test']['perplexity']
            logits_mean = diagnostic_results['perplexity_test']['logits_mean']

            if ppl > 100 or logits_mean < -50:
                print("\n⚠️ WARNING: Model shows severe issues!")
                print(f"   Perplexity: {ppl:.2f}")
                print(f"   Logits mean: {logits_mean:.2f}")
                print("   Model may have quantization failure or other critical issues.")

        print("\nDiagnostic complete. Proceeding with evaluation...\n")

    # Initialize evaluation modules
    evaluator = CPTEvaluation(model, tokenizer)

    zero_shot_config = eval_config['zero_shot']
    perplexity_config = eval_config['perplexity']

    zero_shot_evaluator = ZeroShotEvaluator(model, tokenizer, device=device, config=zero_shot_config)
    perplexity_evaluator = PerplexityEvaluator(model, tokenizer, device=device, config=perplexity_config)

    # Get current bit configuration
    current_bits = checkpoint_bit_width or model_config.default_bits
    print(f"\n{'='*70}")
    print(f"Running CPT Evaluation at {current_bits}-bit precision")
    print(f"{'='*70}")

    # Initialize results
    results = {
        'model_type': 'CPT',
        'bit_width': current_bits,
        'model_size_gb': evaluator.calculate_model_size({'W': current_bits}),
        'compression_ratio': 32 / current_bits
    }

    bit_config = {'W': current_bits, 'A': current_bits, 'KV': current_bits}

    # 1. Perplexity evaluation
    print("\n1. Perplexity evaluation...")
    perplexity_results = perplexity_evaluator.evaluate_all_datasets(bit_config)
    results['perplexity'] = perplexity_results
    for dataset, ppl in perplexity_results.items():
        print(f"   {dataset}: {ppl:.1f}")

    # 2. Zero-shot evaluation
    print("\n2. Zero-shot evaluation...")
    zero_shot_results = zero_shot_evaluator.evaluate_all_tasks(bit_config)
    results['zero_shot'] = zero_shot_results
    for task, score in zero_shot_results.items():
        if task != 'Average':
            print(f"   {task}: {score:.1f}%")
    if 'Average' in zero_shot_results:
        print(f"   Average: {zero_shot_results['Average']:.1f}%")

    # Few-shot evaluation removed - focusing on perplexity and zero-shot classification only

    # Save results with CPT-specific naming
    output_config = eval_config['output']
    output_dir_str = output_config['directory']

    output_dir = Path(output_dir_str)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate timestamp in format YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"cpt_results_{current_bits}bit_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("CPT Evaluation Complete!")
    print(f"Results saved to {results_file}")
    print(f"{'='*70}")

    # Print summary
    print("\nSummary:")
    print(f"  Model Type: CPT")
    print(f"  Precision: {current_bits}-bit")
    print(f"  Size: {results['model_size_gb']:.3f} GB")
    print(f"  Compression: {results['compression_ratio']:.1f}x")

    if results['perplexity']:
        if 'WikiText2' in results['perplexity']:
            print(f"  WikiText2 PPL: {results['perplexity']['WikiText2']:.1f}")

    if results['zero_shot']:
        if 'Average' in results['zero_shot']:
            print(f"  Zero-shot Avg: {results['zero_shot']['Average']:.1f}%")



if __name__ == "__main__":
    main()
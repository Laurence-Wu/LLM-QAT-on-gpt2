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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import CPT model loader
from load_cpt_model import load_cpt_model, verify_cpt_quantization_status

# Import standard evaluation components
from transformers import GPT2Tokenizer
from llm_qat_metrics import LLMQATEvaluation
from zero_shot_tasks import ZeroShotEvaluator
from few_shot_eval import FewShotEvaluator
from perplexity_eval import PerplexityEvaluator


def load_tokenizer():
    """Load GPT-2 tokenizer"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_evaluation_config(config_path):
    """Load evaluation configuration from JSON file"""
    if not os.path.exists(config_path):
        # Try looking in part3_evaluation directory
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
    device = eval_config['device'] if 'device' in eval_config else 'cuda'

    # Run diagnostic if requested
    if args.diagnose:
        print("\n" + "="*70)
        print("Running Comprehensive Quantization Diagnostics...")
        print("="*70)
        try:
            from diagnose_quantization import (
                comprehensive_diagnosis,
                test_sliding_window_perplexity,
                track_batch_degradation
            )

            # Run comprehensive diagnosis
            diagnostic_results = comprehensive_diagnosis(model, tokenizer, device)

            # Check for issues
            has_issues = False
            if 'perplexity_test' in diagnostic_results:
                ppl = diagnostic_results['perplexity_test']['perplexity'] if 'perplexity' in diagnostic_results['perplexity_test'] else 0
                logits_mean = diagnostic_results['perplexity_test']['logits_mean'] if 'logits_mean' in diagnostic_results['perplexity_test'] else 0
                if ppl > 100 or logits_mean < -50:
                    has_issues = True
                    print("\n⚠️ WARNING: Model shows severe issues!")
                    print(f"   Perplexity: {ppl:.2f}")
                    print(f"   Logits mean: {logits_mean:.2f}")
                    print("   Model may have quantization failure or other critical issues.")

            print("\nDiagnostic complete. Proceeding with evaluation...\n")
        except Exception as e:
            print(f"Warning: Diagnostic failed with error: {e}")
            print("Continuing with evaluation...\n")

    # Initialize evaluation modules
    evaluator = LLMQATEvaluation(model, tokenizer)
    zero_shot_config = eval_config['zero_shot'] if 'zero_shot' in eval_config else {}
    few_shot_config = eval_config['few_shot'] if 'few_shot' in eval_config else {}
    perplexity_config = eval_config['perplexity'] if 'perplexity' in eval_config else {}

    zero_shot_evaluator = ZeroShotEvaluator(model, tokenizer, device=device, config=zero_shot_config)
    few_shot_evaluator = FewShotEvaluator(model, tokenizer, device=device, config=few_shot_config)
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
        if 'Average' in zero_shot_results:
            print(f"   Average: {zero_shot_results['Average']:.1f}%")
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

        if 'Average' in mmlu_scores:
            print(f"   MMLU Average: {mmlu_scores['Average']:.1f}%")
        print(f"   TriviaQA: {triviaqa_score:.1f}%")
    except Exception as e:
        print(f"   Warning: Few-shot evaluation failed: {e}")
        results['few_shot'] = {}

    # Save results with CPT-specific naming
    output_config = eval_config['output'] if 'output' in eval_config else {}
    output_dir = Path(output_config['directory'] if 'directory' in output_config else 'results')
    output_dir.mkdir(exist_ok=True, parents=True)

    results_file = output_dir / f"cpt_results_{current_bits}bit.json"
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

    if results['few_shot']:
        if 'MMLU' in results['few_shot']:
            if 'Average' in results['few_shot']['MMLU']:
                print(f"  MMLU Avg: {results['few_shot']['MMLU']['Average']:.1f}%")


if __name__ == "__main__":
    main()
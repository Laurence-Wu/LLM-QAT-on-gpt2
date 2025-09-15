#!/usr/bin/env python3
"""
Main script to run LLM-QAT paper evaluation suite
"""

import json
import argparse
from pathlib import Path
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import SwitchableQATGPT2
from transformers import GPT2Config, GPT2Tokenizer

from llm_qat_metrics import LLMQATEvaluation
from bit_configurations import BitConfigurations
from generate_tables import ResultTableGenerator
from baseline_comparison import BaselineComparison


def load_switchable_model(model_path: str = None):
    """Load switchable precision model"""

    # Default bit widths - will be overridden if loading from checkpoint
    default_bit_widths = [4, 8, 16]

    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location='cuda')

        # Extract model configuration from checkpoint if available
        if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            bit_widths = model_config.get('bit_widths', default_bit_widths)

            # Use exact configuration from checkpoint
            config = GPT2Config(
                vocab_size=model_config.get('vocab_size', 50257),
                n_positions=model_config.get('n_positions', 1024),  # Default to 1024 if not specified
                n_embd=model_config.get('n_embd', 768),
                n_layer=model_config.get('n_layer', 6),
                n_head=model_config.get('n_head', 12),
                layer_norm_epsilon=model_config.get('layer_norm_epsilon', 1e-5),
                embd_pdrop=model_config.get('embd_pdrop', 0.1),
                lora_rank=model_config.get('lora_rank', 16),
                lora_alpha=model_config.get('lora_alpha', 32),
                lora_dropout=model_config.get('lora_dropout', 0.1)
            )
            print(f"Model configuration: n_positions={config.n_positions}, n_layer={config.n_layer}, n_embd={config.n_embd}")
        else:
            # Use default configuration
            bit_widths = default_bit_widths
            config = GPT2Config(
                vocab_size=50257,
                n_positions=256,
                n_embd=768,
                n_layer=6,
                n_head=12,
                layer_norm_epsilon=1e-5,
                embd_pdrop=0.1,
                lora_rank=16,
                lora_alpha=32,
                lora_dropout=0.1
            )

        print(f"Creating model with bit-widths: {bit_widths}")
        model = SwitchableQATGPT2(config, bit_widths=bit_widths)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model = checkpoint
    else:
        print("Using randomly initialized model with default bit-widths: [4, 8, 16]")
        config = GPT2Config(
            vocab_size=50257,
            n_positions=256,
            n_embd=768,
            n_layer=6,
            n_head=12,
            layer_norm_epsilon=1e-5,
            embd_pdrop=0.1,
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.1
        )
        model = SwitchableQATGPT2(config, bit_widths=default_bit_widths)

    return model


def load_tokenizer():
    """Load GPT-2 tokenizer"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description='LLM-QAT Paper Evaluation Suite')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='part3_evaluation/results',
                       help='Directory to save results')
    parser.add_argument('--configs', nargs='+',
                       default=['INT4', 'INT8', 'FP16'],
                       help='Configurations to evaluate (e.g., INT4 INT8 FP16)')
    parser.add_argument('--skip_few_shot', action='store_true',
                       help='Skip few-shot evaluation (faster)')
    parser.add_argument('--skip_zero_shot', action='store_true',
                       help='Skip zero-shot evaluation')
    parser.add_argument('--skip_perplexity', action='store_true',
                       help='Skip perplexity evaluation')
    parser.add_argument('--compare_baselines', action='store_true',
                       help='Compare with baseline methods')
    args = parser.parse_args()

    model = load_switchable_model(args.model_path)
    tokenizer = load_tokenizer()

    evaluator = LLMQATEvaluation(model, tokenizer)

    # Automatically determine configurations based on model's supported bit-widths
    if hasattr(model, 'bit_widths'):
        # Model supports switchable precision
        supported_bit_widths = model.bit_widths
        print(f"Model supports bit-widths: {supported_bit_widths}")

        # Map bit-widths to configuration names
        bit_to_config = {
            2: 'INT2',
            4: 'INT4',
            8: 'INT8',
            16: 'FP16'
        }

        # Override args.configs with supported configurations
        if not args.configs or args.configs == ['INT4', 'INT8', 'FP16']:
            # Use default or auto-detect
            args.configs = [bit_to_config.get(b, f'INT{b}') for b in supported_bit_widths if b in bit_to_config]
            print(f"Auto-detected configurations to evaluate: {args.configs}")

    print("="*70)
    print("Running LLM-QAT Paper Evaluation Suite")
    print("="*70)
    print(f"Model: GPT-2 ({evaluator.model_params:.1f}M parameters)")
    print(f"Configurations to evaluate: {args.configs}")
    print(f"Output directory: {args.output_dir}")
    print("="*70)

    results = {}

    for config_name in args.configs:
        print(f"\n{'='*60}")
        print(f"Evaluating configuration: {config_name}")
        print('='*60)

        if config_name not in BitConfigurations.STANDARD_CONFIGS:
            print(f"Warning: Configuration {config_name} not found in standard configs")
            continue

        config = BitConfigurations.STANDARD_CONFIGS[config_name]

        BitConfigurations.apply_config_to_model(model, config)

        results[config_name] = {
            'config_name': config['name'],
            'bits': f"{config['W']}-{config['A']}-{config['KV']}",
            'model_size_gb': evaluator.calculate_model_size(config),
            'description': config.get('description', '')
        }

        print(f"Configuration: {config['name']} ({config['description']})")
        print(f"Model size: {results[config_name]['model_size_gb']} GB")

        if not args.skip_zero_shot:
            print("\n1. Zero-shot common sense evaluation...")
            try:
                zero_shot_results = evaluator.evaluate_zero_shot_common_sense(config)
                results[config_name]['zero_shot'] = zero_shot_results
                print(f"   Average score: {zero_shot_results['Average']:.1f}%")

                for task in ['BoolQ', 'PIQA', 'SIQA', 'HellaSwag', 'WinoGrande', 'ARC-e', 'ARC-c', 'OBQA']:
                    if task in zero_shot_results:
                        print(f"   {task}: {zero_shot_results[task]:.1f}%")
            except Exception as e:
                print(f"   Error in zero-shot evaluation: {e}")
                results[config_name]['zero_shot'] = {}

        if not args.skip_perplexity:
            print("\n2. Perplexity evaluation...")
            try:
                perplexity_results = evaluator.evaluate_perplexity(config)
                results[config_name]['perplexity'] = perplexity_results
                print(f"   WikiText2: {perplexity_results['WikiText2']:.1f}")
                print(f"   C4: {perplexity_results['C4']:.1f}")
            except Exception as e:
                print(f"   Error in perplexity evaluation: {e}")
                results[config_name]['perplexity'] = {}

        if not args.skip_few_shot:
            print("\n3. Few-shot evaluation...")
            try:
                few_shot_results = evaluator.evaluate_few_shot(config)
                results[config_name]['few_shot'] = few_shot_results

                if 'MMLU' in few_shot_results:
                    mmlu = few_shot_results['MMLU']
                    print(f"   MMLU:")
                    for category in ['Humanities', 'STEM', 'Social Sciences', 'Other']:
                        if category in mmlu:
                            print(f"     {category}: {mmlu[category]:.1f}%")
                    print(f"     Average: {mmlu.get('Average', 0):.1f}%")

                if 'TriviaQA' in few_shot_results:
                    print(f"   TriviaQA: {few_shot_results['TriviaQA']:.1f}%")
            except Exception as e:
                print(f"   Error in few-shot evaluation: {e}")
                results[config_name]['few_shot'] = {}

    print("\n" + "="*70)
    print("Generating result tables...")
    print("="*70)

    table_gen = ResultTableGenerator(results)

    if not args.skip_zero_shot:
        table_gen.generate_table_1_zero_shot()

    if not args.skip_perplexity:
        table_gen.generate_table_2_perplexity()

    if not args.skip_few_shot:
        table_gen.generate_table_7_few_shot()

    table_gen.export_to_markdown()
    table_gen.export_to_latex()

    if args.compare_baselines:
        print("\n" + "="*70)
        print("Comparing with baseline methods...")
        print("="*70)

        comparison = BaselineComparison(results)
        comparison.compare_with_baselines()
        comparison.plot_accuracy_vs_bits()
        comparison.calculate_degradation_from_fp16()

    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    with open(output_path / 'llm_qat_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Evaluation complete!")
    print(f"Results saved to {output_path}")
    print(f"{'='*70}")

    print("\nSummary of Results:")
    for config_name, result in results.items():
        print(f"\n{config_name} ({result['bits']}):")
        if 'zero_shot' in result and result['zero_shot']:
            print(f"  Zero-shot avg: {result['zero_shot'].get('Average', 0):.1f}%")
        if 'perplexity' in result and result['perplexity']:
            print(f"  WikiText2 PPL: {result['perplexity'].get('WikiText2', float('inf')):.1f}")
            print(f"  C4 PPL: {result['perplexity'].get('C4', float('inf')):.1f}")


if __name__ == "__main__":
    main()
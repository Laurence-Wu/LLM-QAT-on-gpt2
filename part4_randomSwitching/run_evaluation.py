#!/usr/bin/env python3
"""
Main evaluation pipeline for adversarial robustness with random precision switching.
"""

import argparse
import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simplified_random_switching import (
    load_sp_model_with_bit_config,
    SimplifiedRandomSwitching,
    DefenseEvaluator
)
from adversarial_attacks import TextFoolerAttack, AttackEvaluator
from wikitext_evaluation import prepare_wikitext2_samples, WikiTextEvaluator


def evaluate_fixed_precision_baseline(model, tokenizer, test_samples: List[Dict],
                                     bit_widths: List[int], device: str = 'cuda') -> Dict:
    """
    Evaluate model at fixed precisions to establish baseline.

    Args:
        model: SP model
        tokenizer: Tokenizer
        test_samples: Test samples
        bit_widths: Available bit widths
        device: Device for computation

    Returns:
        Baseline results for each precision
    """
    print("\n" + "="*60)
    print("BASELINE: Fixed Precision Evaluation")
    print("="*60)

    evaluator = DefenseEvaluator(model, tokenizer, bit_widths, device)
    attack_evaluator = AttackEvaluator(model, tokenizer, device)

    results = {}

    for precision in bit_widths:
        print(f"\nEvaluating at fixed {precision}-bit precision...")

        model.set_precision(precision)

        clean_results = evaluator.evaluate_fixed_precision(
            test_samples[:50], precision
        )

        if 'accuracy' in clean_results:
            print(f"  Clean accuracy: {clean_results['accuracy']:.2%}")
        print(f"  Clean perplexity: {clean_results['perplexity']:.2f}")
        print(f"  Clean loss: {clean_results['avg_loss']:.3f}")

        print("  Testing TextFooler attack...")
        textfooler_results = attack_evaluator.evaluate_textfooler(
            test_samples[:30], max_samples=30
        )

        print(f"    Attack success rate: {textfooler_results['attack_success_rate']:.2%}")
        print(f"    Avg original accuracy: {textfooler_results['avg_original_accuracy']:.2%}")
        print(f"    Avg adversarial accuracy: {textfooler_results['avg_adversarial_accuracy']:.2%}")
        print(f"    Avg accuracy drop: {textfooler_results['avg_accuracy_drop']:.2%}")
        print(f"    Avg perturbation ratio: {textfooler_results['avg_perturb_ratio']:.2%}")

        print("  Testing BERT-Attack...")
        bert_attack_results = attack_evaluator.evaluate_bert_attack(
            test_samples[:30], max_samples=30
        )

        print(f"    Attack success rate: {bert_attack_results['attack_success_rate']:.2%}")
        print(f"    Avg original accuracy: {bert_attack_results['avg_original_accuracy']:.2%}")
        print(f"    Avg adversarial accuracy: {bert_attack_results['avg_adversarial_accuracy']:.2%}")
        print(f"    Avg accuracy drop: {bert_attack_results['avg_accuracy_drop']:.2%}")
        print(f"    Avg perturbation ratio: {bert_attack_results['avg_perturb_ratio']:.2%}")

        defense_rate_tf = 1 - textfooler_results['attack_success_rate']
        defense_rate_bert = 1 - bert_attack_results['attack_success_rate']

        results[precision] = {
            'clean_performance': clean_results,
            'textfooler': {
                'attack_success_rate': textfooler_results['attack_success_rate'],
                'defense_rate': defense_rate_tf,
                'avg_accuracy_drop': textfooler_results['avg_accuracy_drop'],
                'avg_original_accuracy': textfooler_results['avg_original_accuracy'],
                'avg_adversarial_accuracy': textfooler_results['avg_adversarial_accuracy'],
                'avg_perturbations': textfooler_results['avg_perturb_ratio']
            },
            'bert_attack': {
                'attack_success_rate': bert_attack_results['attack_success_rate'],
                'defense_rate': defense_rate_bert,
                'avg_accuracy_drop': bert_attack_results['avg_accuracy_drop'],
                'avg_original_accuracy': bert_attack_results['avg_original_accuracy'],
                'avg_adversarial_accuracy': bert_attack_results['avg_adversarial_accuracy'],
                'avg_perturbations': bert_attack_results['avg_perturb_ratio'],
                'avg_perplexity_increase': bert_attack_results['avg_perplexity_increase']
            }
        }

    return results


def evaluate_random_switching_defense(model, tokenizer, test_samples: List[Dict],
                                     bit_widths: List[int],
                                     switch_probabilities: List[float],
                                     device: str = 'cuda') -> Dict:
    """
    Evaluate random switching defense with different probabilities.

    Args:
        model: SP model
        tokenizer: Tokenizer
        test_samples: Test samples
        bit_widths: Available bit widths
        switch_probabilities: List of switching probabilities to test
        device: Device for computation

    Returns:
        Results for each switching probability
    """
    print("\n" + "="*60)
    print("RANDOM SWITCHING DEFENSE EVALUATION")
    print("="*60)

    results = {}

    for switch_prob in switch_probabilities:
        print(f"\nEvaluating with {switch_prob:.0%} switching probability...")

        defender = SimplifiedRandomSwitching(
            model, bit_widths, switch_prob, device
        )

        total_success_tf = 0
        total_success_bert = 0
        total_samples = 0

        # TextFooler defense evaluation
        for i, sample in enumerate(tqdm(test_samples[:30], desc="TextFooler defense")):
            model.set_precision(max(bit_widths))

            text = sample.get('text', '')
            if not text:
                text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)

            textfooler = TextFoolerAttack(model, tokenizer, device)
            attack_result = textfooler.generate_adversarial(text)

            if attack_result['success']:
                adv_text = attack_result['adversarial_text']
                adv_ids = tokenizer.encode(adv_text, return_tensors='pt').to(device)

                outputs_with_switch, precision = defender.forward_with_switching(
                    adv_ids, labels=adv_ids
                )

                # Compute accuracy with switching
                switched_logits = outputs_with_switch['logits']
                switched_predictions = switched_logits[0, :-1, :].argmax(dim=-1)
                switched_labels = adv_ids[0, 1:]
                switched_mask = switched_labels != -100
                switched_correct = (switched_predictions[switched_mask] == switched_labels[switched_mask]).sum().item()
                switched_total = switched_mask.sum().item()
                switched_accuracy = switched_correct / max(switched_total, 1)

                # Defense succeeds if accuracy recovers by >3% absolute
                if 'adversarial_accuracy' in attack_result:
                    accuracy_recovery = switched_accuracy - attack_result['adversarial_accuracy']
                    if accuracy_recovery > 0.03:  # 3% threshold
                        total_success_tf += 1

            total_samples += 1

        # BERT-Attack defense evaluation
        from adversarial_attacks import BERTAttack
        bert_attacker = BERTAttack(model, tokenizer, device)

        for i, sample in enumerate(tqdm(test_samples[:30], desc="BERT-Attack defense")):
            model.set_precision(max(bit_widths))

            text = sample.get('text', '')
            if not text:
                text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)

            attack_result = bert_attacker.generate_adversarial(text)

            if attack_result['success']:
                adv_text = attack_result['adversarial_text']
                adv_ids = tokenizer.encode(adv_text, return_tensors='pt').to(device)

                outputs_with_switch, precision = defender.forward_with_switching(
                    adv_ids, labels=adv_ids
                )

                # Compute accuracy with switching
                switched_logits = outputs_with_switch['logits']
                switched_predictions = switched_logits[0, :-1, :].argmax(dim=-1)
                switched_labels = adv_ids[0, 1:]
                switched_mask = switched_labels != -100
                switched_correct = (switched_predictions[switched_mask] == switched_labels[switched_mask]).sum().item()
                switched_total = switched_mask.sum().item()
                switched_accuracy = switched_correct / max(switched_total, 1)

                # Defense succeeds if accuracy recovers by >3% absolute
                if 'adversarial_accuracy' in attack_result:
                    accuracy_recovery = switched_accuracy - attack_result['adversarial_accuracy']
                    if accuracy_recovery > 0.03:  # 3% threshold
                        total_success_bert += 1

        defense_rate_tf = total_success_tf / max(total_samples, 1)
        defense_rate_bert = total_success_bert / max(total_samples, 1)

        stats = defender.get_statistics()

        print(f"  TextFooler defense rate: {defense_rate_tf:.2%}")
        print(f"  BERT-Attack defense rate: {defense_rate_bert:.2%}")
        print(f"  Actual switch rate: {stats['switch_rate']:.2%}")
        print(f"  Precision distribution: {stats['precision_distribution']}")

        results[switch_prob] = {
            'textfooler_defense_rate': defense_rate_tf,
            'bert_attack_defense_rate': defense_rate_bert,
            'switching_statistics': stats
        }

    return results


def compare_results(fixed_results: Dict, switching_results: Dict) -> Dict:
    """
    Compare fixed precision and random switching results.

    Args:
        fixed_results: Results from fixed precision evaluation
        switching_results: Results from random switching evaluation

    Returns:
        Comparison metrics
    """
    best_fixed_tf = max(
        r['textfooler']['defense_rate'] for r in fixed_results.values()
    )
    best_fixed_bert = max(
        r['bert_attack']['defense_rate'] for r in fixed_results.values()
    )

    best_precision_tf = max(
        fixed_results.keys(),
        key=lambda k: fixed_results[k]['textfooler']['defense_rate']
    )
    best_precision_bert = max(
        fixed_results.keys(),
        key=lambda k: fixed_results[k]['bert_attack']['defense_rate']
    )

    improvements = {}

    for switch_prob, results in switching_results.items():
        tf_improvement = (
            (results['textfooler_defense_rate'] - best_fixed_tf) / best_fixed_tf * 100
            if best_fixed_tf > 0 else 0
        )
        bert_improvement = (
            (results['bert_attack_defense_rate'] - best_fixed_bert) / best_fixed_bert * 100
            if best_fixed_bert > 0 else 0
        )

        improvements[switch_prob] = {
            'textfooler_improvement': tf_improvement,
            'bert_attack_improvement': bert_improvement
        }

    return {
        'best_fixed_textfooler': {
            'precision': best_precision_tf,
            'defense_rate': best_fixed_tf
        },
        'best_fixed_bert_attack': {
            'precision': best_precision_bert,
            'defense_rate': best_fixed_bert
        },
        'improvements': improvements
    }


def generate_report(fixed_results: Dict, switching_results: Dict,
                   comparison: Dict, output_path: Path):
    """
    Generate comprehensive evaluation report.

    Args:
        fixed_results: Fixed precision results
        switching_results: Random switching results
        comparison: Comparison metrics
        output_path: Path to save report
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY REPORT")
    print("="*60)

    print(f"\nBest Fixed Precision Performance:")
    print(f"  TextFooler: {comparison['best_fixed_textfooler']['precision']}-bit "
          f"({comparison['best_fixed_textfooler']['defense_rate']:.2%} defense rate)")
    print(f"  BERT-Attack: {comparison['best_fixed_bert_attack']['precision']}-bit "
          f"({comparison['best_fixed_bert_attack']['defense_rate']:.2%} defense rate)")

    print("\nRandom Switching Improvements:")
    for switch_prob, improvements in comparison['improvements'].items():
        print(f"  Switching probability {switch_prob:.0%}:")
        print(f"    TextFooler: {improvements['textfooler_improvement']:+.1f}%")
        print(f"    BERT-Attack: {improvements['bert_attack_improvement']:+.1f}%")

    # Calculate average improvement across all attacks
    best_switch_prob = max(
        comparison['improvements'].keys(),
        key=lambda k: (comparison['improvements'][k]['textfooler_improvement'] +
                      comparison['improvements'][k]['bert_attack_improvement']) / 2
    )
    print(f"\nOptimal switching probability: {best_switch_prob:.0%}")

    report = {
        'fixed_precision_results': {
            str(k): v for k, v in fixed_results.items()
        },
        'random_switching_results': {
            str(k): v for k, v in switching_results.items()
        },
        'comparison': comparison,
        'recommendation': {
            'optimal_switch_probability': best_switch_prob,
            'expected_improvement': {
                'textfooler': comparison['improvements'][best_switch_prob]['textfooler_improvement'],
                'bert_attack': comparison['improvements'][best_switch_prob]['bert_attack_improvement']
            }
        }
    }

    output_file = output_path / 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return report




def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GPT-2 adversarial robustness with random switching"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained SP model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100)"
    )
    parser.add_argument(
        "--switch_probs",
        type=float,
        nargs='+',
        default=[0.0, 0.3, 0.5, 0.7],
        help="List of switching probabilities to test (default: 0.0 0.3 0.5 0.7)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for results (default: current directory)"
    )
    parser.add_argument(
        "--bit_widths",
        type=int,
        nargs='+',
        default=None,
        help="Bit widths to evaluate (e.g., --bit_widths 3 4 5). Overrides checkpoint bit widths."
    )

    args = parser.parse_args()

    # Ensure CUDA is available and set device
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This evaluation requires GPU.")
        sys.exit(1)

    device = "cuda"

    # Set output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get parameters from arguments
    num_samples = args.num_samples
    switch_probs = args.switch_probs

    print("="*60)
    print("ADVERSARIAL ROBUSTNESS EVALUATION WITH RANDOM SWITCHING")
    print("="*60)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Number of samples: {num_samples}")
    print(f"Switch probabilities: {switch_probs}")
    print(f"Output directory: {output_path}")

    # Load SP model
    model, tokenizer, checkpoint_bit_widths, saved_precision = load_sp_model_with_bit_config(
        args.checkpoint, device
    )

    # Use command-line bit widths if specified, otherwise use checkpoint bit widths
    if args.bit_widths is not None:
        bit_widths = args.bit_widths
        print(f"\nUsing specified bit widths: {bit_widths}")
        print(f"(Checkpoint has bit widths: {checkpoint_bit_widths})")
    else:
        bit_widths = checkpoint_bit_widths
        print(f"\nLoaded SP model with bit widths: {bit_widths}")

    if saved_precision:
        print(f"Model was saved at {saved_precision}-bit precision")

    print(f"\nPreparing WikiText-2 dataset...")
    test_samples = prepare_wikitext2_samples(
        tokenizer, num_samples=num_samples
    )
    print(f"Prepared {len(test_samples)} samples for evaluation")

    fixed_results = evaluate_fixed_precision_baseline(
        model, tokenizer, test_samples, bit_widths, device
    )

    switching_results = evaluate_random_switching_defense(
        model, tokenizer, test_samples, bit_widths,
        switch_probs, device
    )

    comparison = compare_results(fixed_results, switching_results)

    # Add model info and config to report
    report = generate_report(
        fixed_results, switching_results, comparison, output_path
    )

    # Enhance report with model info and evaluation parameters
    report['model_info'] = {
        'type': 'sp',
        'checkpoint': args.checkpoint,
        'bit_widths': bit_widths,
        'saved_precision': saved_precision
    }
    report['evaluation_params'] = {
        'num_samples': num_samples,
        'switch_probabilities': switch_probs,
        'dataset': 'WikiText-2'
    }

    # Save enhanced report
    report_file = output_path / 'evaluation_results_sp.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nEnhanced report saved to: {report_file}")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

    if comparison['improvements']:
        best_improvement = max(
            comparison['improvements'].values(),
            key=lambda x: (x['textfooler_improvement'] + x['bert_attack_improvement']) / 2
        )
        avg_improvement = (best_improvement['textfooler_improvement'] +
                          best_improvement['bert_attack_improvement']) / 2
        print(f"\nKey Finding: Random switching provides up to "
              f"{avg_improvement:.1f}% average improvement "
              f"against adversarial attacks")

    return report


if __name__ == "__main__":
    main()
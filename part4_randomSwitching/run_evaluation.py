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
from adversarial_attacks import TextFoolerAttack, GradientAttack, AttackEvaluator
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

        print(f"  Clean accuracy: {clean_results['accuracy']:.2%}")
        print(f"  Clean loss: {clean_results['avg_loss']:.3f}")

        print("  Testing TextFooler attack...")
        textfooler_results = attack_evaluator.evaluate_textfooler(
            test_samples[:30], max_samples=30
        )

        print(f"    Attack success rate: {textfooler_results['attack_success_rate']:.2%}")
        print(f"    Avg perturbation ratio: {textfooler_results['avg_perturb_ratio']:.2%}")

        print("  Testing Gradient attack (HotFlip)...")
        gradient_results = attack_evaluator.evaluate_gradient(
            test_samples[:30], attack_type='hotflip', max_samples=30
        )

        print(f"    Attack success rate: {gradient_results['attack_success_rate']:.2%}")
        print(f"    Avg changed tokens: {gradient_results['avg_change_ratio']:.2%}")

        defense_rate_tf = 1 - textfooler_results['attack_success_rate']
        defense_rate_grad = 1 - gradient_results['attack_success_rate']

        results[precision] = {
            'clean_performance': clean_results,
            'textfooler': {
                'attack_success_rate': textfooler_results['attack_success_rate'],
                'defense_rate': defense_rate_tf,
                'avg_perturbations': textfooler_results['avg_perturb_ratio']
            },
            'gradient': {
                'attack_success_rate': gradient_results['attack_success_rate'],
                'defense_rate': defense_rate_grad,
                'avg_changes': gradient_results['avg_change_ratio']
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
        total_success_grad = 0
        total_samples = 0

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
                switched_loss = outputs_with_switch['loss'].item()

                if switched_loss < attack_result['adversarial_loss'] * 0.9:
                    total_success_tf += 1

            total_samples += 1

        for i, sample in enumerate(tqdm(test_samples[:30], desc="Gradient defense")):
            input_ids = sample['input_ids'].to(device)
            labels = sample.get('labels', input_ids.clone()).to(device)

            model.set_precision(8)

            gradient_attack = GradientAttack(model, tokenizer, device)
            attack_result = gradient_attack.hotflip_attack(input_ids, labels)

            if attack_result['success']:
                perturbed_ids = attack_result['perturbed_ids']

                outputs_with_switch, precision = defender.forward_with_switching(
                    perturbed_ids, labels=labels
                )
                switched_loss = outputs_with_switch['loss'].item()

                if switched_loss < attack_result['adversarial_loss'] * 0.9:
                    total_success_grad += 1

        defense_rate_tf = total_success_tf / max(total_samples, 1)
        defense_rate_grad = total_success_grad / max(total_samples, 1)

        stats = defender.get_statistics()

        print(f"  TextFooler defense rate: {defense_rate_tf:.2%}")
        print(f"  Gradient defense rate: {defense_rate_grad:.2%}")
        print(f"  Actual switch rate: {stats['switch_rate']:.2%}")
        print(f"  Precision distribution: {stats['precision_distribution']}")

        results[switch_prob] = {
            'textfooler_defense_rate': defense_rate_tf,
            'gradient_defense_rate': defense_rate_grad,
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
    best_fixed_grad = max(
        r['gradient']['defense_rate'] for r in fixed_results.values()
    )

    best_precision_tf = max(
        fixed_results.keys(),
        key=lambda k: fixed_results[k]['textfooler']['defense_rate']
    )
    best_precision_grad = max(
        fixed_results.keys(),
        key=lambda k: fixed_results[k]['gradient']['defense_rate']
    )

    improvements = {}

    for switch_prob, results in switching_results.items():
        tf_improvement = (
            (results['textfooler_defense_rate'] - best_fixed_tf) / best_fixed_tf * 100
            if best_fixed_tf > 0 else 0
        )
        grad_improvement = (
            (results['gradient_defense_rate'] - best_fixed_grad) / best_fixed_grad * 100
            if best_fixed_grad > 0 else 0
        )

        improvements[switch_prob] = {
            'textfooler_improvement': tf_improvement,
            'gradient_improvement': grad_improvement
        }

    return {
        'best_fixed_textfooler': {
            'precision': best_precision_tf,
            'defense_rate': best_fixed_tf
        },
        'best_fixed_gradient': {
            'precision': best_precision_grad,
            'defense_rate': best_fixed_grad
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
    print(f"  Gradient: {comparison['best_fixed_gradient']['precision']}-bit "
          f"({comparison['best_fixed_gradient']['defense_rate']:.2%} defense rate)")

    print("\nRandom Switching Improvements:")
    for switch_prob, improvements in comparison['improvements'].items():
        print(f"  Switching probability {switch_prob:.0%}:")
        print(f"    TextFooler: {improvements['textfooler_improvement']:+.1f}%")
        print(f"    Gradient: {improvements['gradient_improvement']:+.1f}%")

    best_switch_prob = max(
        comparison['improvements'].keys(),
        key=lambda k: comparison['improvements'][k]['gradient_improvement']
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
                'gradient': comparison['improvements'][best_switch_prob]['gradient_improvement']
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
    model, tokenizer, bit_widths, saved_precision = load_sp_model_with_bit_config(
        args.checkpoint, device
    )

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
            key=lambda x: x['gradient_improvement']
        )
        print(f"\nKey Finding: Random switching provides up to "
              f"{best_improvement['gradient_improvement']:.1f}% improvement "
              f"against gradient attacks")

    return report


if __name__ == "__main__":
    main()
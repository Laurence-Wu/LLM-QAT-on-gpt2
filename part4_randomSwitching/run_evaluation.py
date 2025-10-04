
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
from adversarial_attacks import AttackEvaluator
from wikitext_evaluation import prepare_wikitext2_samples

def evaluate_fixed_precision_baseline(model, tokenizer, test_samples: List[Dict],
                                     bit_widths: List[int], device: str = 'cuda') -> Dict:
    print("\n" + "="*60)
    print("BASELINE: Fixed Precision Evaluation")
    print("="*60)

    # Use the highest precision as baseline (typically 32-bit for best quality)
    baseline_precision = max(bit_widths)
    print(f"\nEvaluating baseline at {baseline_precision}-bit precision...")

    evaluator = DefenseEvaluator(model, tokenizer, bit_widths, device)
    attack_evaluator = AttackEvaluator(model, tokenizer, device)

    model.set_precision(baseline_precision)

    clean_results = evaluator.evaluate_fixed_precision(
        test_samples[:50], baseline_precision
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

    results = {
        baseline_precision: {
            'clean_performance': clean_results,
            'textfooler': {
                'attack_success_rate': textfooler_results['attack_success_rate'],
                'avg_accuracy_drop': textfooler_results['avg_accuracy_drop'],
                'avg_original_accuracy': textfooler_results['avg_original_accuracy'],
                'avg_adversarial_accuracy': textfooler_results['avg_adversarial_accuracy'],
                'avg_perturbations': textfooler_results['avg_perturb_ratio'],
                'adversarial_examples': textfooler_results['adversarial_examples']
            },
            'bert_attack': {
                'attack_success_rate': bert_attack_results['attack_success_rate'],
                'avg_accuracy_drop': bert_attack_results['avg_accuracy_drop'],
                'avg_original_accuracy': bert_attack_results['avg_original_accuracy'],
                'avg_adversarial_accuracy': bert_attack_results['avg_adversarial_accuracy'],
                'avg_perturbations': bert_attack_results['avg_perturb_ratio'],
                'avg_perplexity_increase': bert_attack_results['avg_perplexity_increase'],
                'adversarial_examples': bert_attack_results['adversarial_examples']
            }
        }
    }

    return results

def evaluate_random_switching_defense(model, tokenizer,
                                     textfooler_adv_examples: List[Dict],
                                     bert_adv_examples: List[Dict],
                                     bit_widths: List[int],
                                     switch_probabilities: List[float],
                                     device: str = 'cuda') -> Dict:
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

        print(f"\n  Diagnostic samples (every 5th):")
        for idx, adv_example in enumerate(tqdm(textfooler_adv_examples, desc="TextFooler defense")):
            adv_text = adv_example['adversarial_text']
            orig_accuracy = adv_example['original_accuracy']
            adv_accuracy_at_32bit = adv_example['adversarial_accuracy']

            adv_ids = tokenizer.encode(adv_text, return_tensors='pt').to(device)

            outputs_with_switch, precision = defender.forward_with_switching(
                adv_ids, labels=adv_ids
            )

            switched_logits = outputs_with_switch['logits']
            switched_predictions = switched_logits[0, :-1, :].argmax(dim=-1)
            switched_labels = adv_ids[0, 1:]
            switched_mask = switched_labels != -100
            switched_correct = (switched_predictions[switched_mask] == switched_labels[switched_mask]).sum().item()
            switched_total = switched_mask.sum().item()
            switched_accuracy = switched_correct / max(switched_total, 1)

            accuracy_gap = orig_accuracy - adv_accuracy_at_32bit
            recovery_ratio = (switched_accuracy - adv_accuracy_at_32bit) / max(accuracy_gap, 0.01)

            if idx % 5 == 0:
                print(f"    [TF #{idx}] Precision: {precision}-bit | Orig: {orig_accuracy:.3f} | Adv@32: {adv_accuracy_at_32bit:.3f} | Switched: {switched_accuracy:.3f} | Recovery: {recovery_ratio:.2%}")

            if recovery_ratio > 0.15:
                total_success_tf += 1

        for idx, adv_example in enumerate(tqdm(bert_adv_examples, desc="BERT-Attack defense")):
            adv_text = adv_example['adversarial_text']
            orig_accuracy = adv_example['original_accuracy']
            adv_accuracy_at_32bit = adv_example['adversarial_accuracy']

            adv_ids = tokenizer.encode(adv_text, return_tensors='pt').to(device)

            outputs_with_switch, precision = defender.forward_with_switching(
                adv_ids, labels=adv_ids
            )

            switched_logits = outputs_with_switch['logits']
            switched_predictions = switched_logits[0, :-1, :].argmax(dim=-1)
            switched_labels = adv_ids[0, 1:]
            switched_mask = switched_labels != -100
            switched_correct = (switched_predictions[switched_mask] == switched_labels[switched_mask]).sum().item()
            switched_total = switched_mask.sum().item()
            switched_accuracy = switched_correct / max(switched_total, 1)

            accuracy_gap = orig_accuracy - adv_accuracy_at_32bit
            recovery_ratio = (switched_accuracy - adv_accuracy_at_32bit) / max(accuracy_gap, 0.01)

            if idx % 5 == 0:
                print(f"    [BERT #{idx}] Precision: {precision}-bit | Orig: {orig_accuracy:.3f} | Adv@32: {adv_accuracy_at_32bit:.3f} | Switched: {switched_accuracy:.3f} | Recovery: {recovery_ratio:.2%}")

            if recovery_ratio > 0.15:
                total_success_bert += 1

        defense_rate_tf = total_success_tf / max(len(textfooler_adv_examples), 1)
        defense_rate_bert = total_success_bert / max(len(bert_adv_examples), 1)

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
    baseline_defense_tf = 0.0
    baseline_defense_bert = 0.0

    improvements = {}

    for switch_prob, results in switching_results.items():
        tf_absolute_improvement = results['textfooler_defense_rate'] - baseline_defense_tf
        bert_absolute_improvement = results['bert_attack_defense_rate'] - baseline_defense_bert

        improvements[switch_prob] = {
            'textfooler_improvement': tf_absolute_improvement * 100,
            'bert_attack_improvement': bert_absolute_improvement * 100
        }

    return {
        'baseline_defense': {
            'textfooler': baseline_defense_tf,
            'bert_attack': baseline_defense_bert
        },
        'improvements': improvements
    }

def generate_report(fixed_results: Dict, switching_results: Dict,
                   comparison: Dict, output_path: Path):
    print("\n" + "="*60)
    print("EVALUATION SUMMARY REPORT")
    print("="*60)

    print(f"\nBaseline Defense (no switching): 0.00%")

    print("\nRandom Switching Defense Rates and Improvements:")
    for switch_prob, improvements in comparison['improvements'].items():
        print(f"  Switching probability {switch_prob:.0%}:")
        print(f"    TextFooler: {improvements['textfooler_improvement']:+.1f}%")
        print(f"    BERT-Attack: {improvements['bert_attack_improvement']:+.1f}%")

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

    import copy
    report_json = copy.deepcopy(report)

    if 'fixed_precision_results' in report_json:
        for precision_key in report_json['fixed_precision_results']:
            precision_data = report_json['fixed_precision_results'][precision_key]
            if 'textfooler' in precision_data:
                precision_data['textfooler'].pop('adversarial_examples', None)
            if 'bert_attack' in precision_data:
                precision_data['bert_attack'].pop('adversarial_examples', None)

    output_file = output_path / 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(report_json, f, indent=2)

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
        "--switch_probs",
        type=float,
        nargs='+',
        default=[0.0, 0.3, 0.5, 0.7],
        help="List of switching probabilities to test (default: 0.0 0.3 0.5 0.7)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=['sp', 'cpt'],
        default='sp',
        help="Model type: sp (switchable precision) or cpt (cyclic precision training)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save evaluation outputs"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of WikiText-2 samples to use for evaluation (default: 100)"
    )


    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This evaluation requires GPU.")
        sys.exit(1)

    device = "cuda"

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    num_samples = args.num_samples
    switch_probs = args.switch_probs

    print("="*60)
    print("ADVERSARIAL ROBUSTNESS EVALUATION WITH RANDOM SWITCHING")
    print("="*60)
    print(f"\nModel type: {args.model_type.upper()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Number of samples: {num_samples}")
    print(f"Switch probabilities: {switch_probs}")
    print(f"Output directory: {output_path}")

    if args.model_type == 'cpt':
        from simplified_random_switching import load_cpt_model_with_config
        model, tokenizer, checkpoint_bit_widths, saved_precision = load_cpt_model_with_config(
            args.checkpoint, device
        )
    else:
        model, tokenizer, checkpoint_bit_widths, saved_precision = load_sp_model_with_bit_config(
            args.checkpoint, device
        )


    bit_widths = checkpoint_bit_widths


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

    best_precision = max(bit_widths)
    textfooler_adv_examples = fixed_results[best_precision]['textfooler'].get('adversarial_examples', [])
    bert_adv_examples = fixed_results[best_precision]['bert_attack'].get('adversarial_examples', [])

    print(f"\nUsing {len(textfooler_adv_examples)} TextFooler and {len(bert_adv_examples)} BERT-Attack adversarial examples from {best_precision}-bit baseline")

    switching_results = evaluate_random_switching_defense(
        model, tokenizer,
        textfooler_adv_examples, bert_adv_examples,
        bit_widths, switch_probs, device
    )

    comparison = compare_results(fixed_results, switching_results)

    report = generate_report(
        fixed_results, switching_results, comparison, output_path
    )

    report['model_info'] = {
        'type': args.model_type,
        'checkpoint': args.checkpoint,
        'bit_widths': bit_widths,
        'saved_precision': saved_precision
    }
    report['evaluation_params'] = {
        'num_samples': num_samples,
        'switch_probabilities': switch_probs,
        'dataset': 'WikiText-2'
    }

    report_file = output_path / f'evaluation_results_{args.model_type}.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_file}")

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
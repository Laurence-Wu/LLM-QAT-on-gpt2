import torch
import numpy as np
from tqdm import tqdm
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import SwitchableQATGPT2
from part1_switchable_precision.train_qat import train_qat
from part2_cyclic_precision.train_cyclic import train_cyclic_precision
from evaluate_configurations import ConfigurationEvaluator


def compare_training_strategies(train_loader, val_loader, test_loader, model_config, training_config):
    results = {}

    print("Training with joint switchable strategy...")
    model_joint = SwitchableQATGPT2(model_config, bit_widths=model_config.bit_widths)
    model_config.switch_strategy = 'random'
    model_joint = train_qat(model_joint, train_loader, val_loader, training_config, model_config)

    print("Training with cyclic precision strategy...")
    model_cyclic = SwitchableQATGPT2(model_config, bit_widths=model_config.bit_widths)
    model_config.switch_strategy = 'cyclic'
    model_cyclic = train_qat(model_cyclic, train_loader, val_loader, training_config, model_config)

    print("Training with curriculum strategy...")
    model_curriculum = SwitchableQATGPT2(model_config, bit_widths=model_config.bit_widths)
    model_config.switch_strategy = 'curriculum'
    model_curriculum = train_qat(model_curriculum, train_loader, val_loader, training_config, model_config)

    print("Evaluating joint strategy...")
    evaluator_joint = ConfigurationEvaluator(model_joint, test_loader)
    results['joint'] = evaluator_joint.evaluate_all_configurations()

    print("Evaluating cyclic strategy...")
    evaluator_cyclic = ConfigurationEvaluator(model_cyclic, test_loader)
    results['cyclic'] = evaluator_cyclic.evaluate_all_configurations()

    print("Evaluating curriculum strategy...")
    evaluator_curriculum = ConfigurationEvaluator(model_curriculum, test_loader)
    results['curriculum'] = evaluator_curriculum.evaluate_all_configurations()

    results['comparison'] = {
        'joint_best': _find_best_config(results['joint']),
        'cyclic_best': _find_best_config(results['cyclic']),
        'curriculum_best': _find_best_config(results['curriculum']),
    }

    return results


def _find_best_config(results_dict):
    best_accuracy = 0
    best_config = None
    best_name = None

    for name, result in results_dict.items():
        if isinstance(result, dict) and 'accuracy' in result:
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_config = result.get('config', name)
                best_name = name

    return {
        'name': best_name,
        'config': best_config,
        'accuracy': best_accuracy,
        'loss': results_dict[best_name].get('loss', None) if best_name else None
    }


def compare_convergence(stats_joint, stats_cyclic, stats_curriculum):
    convergence_analysis = {}

    for name, stats in [('joint', stats_joint), ('cyclic', stats_cyclic), ('curriculum', stats_curriculum)]:
        if 'iteration_losses' in stats:
            losses = stats['iteration_losses']
            convergence_analysis[name] = {
                'final_loss': losses[-1] if losses else None,
                'convergence_iter': _find_convergence_point(losses),
                'stability': np.std(losses[-100:]) if len(losses) > 100 else np.std(losses)
            }

    return convergence_analysis


def _find_convergence_point(losses, threshold=0.01, window=50):
    if len(losses) < window:
        return len(losses)

    for i in range(window, len(losses)):
        window_std = np.std(losses[i-window:i])
        if window_std < threshold:
            return i

    return len(losses)


def save_comparison_results(results, output_path='part3_evaluation/strategy_comparison.json'):
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results)

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Comparison results saved to {output_path}")


def print_comparison_summary(results):
    print("\n" + "="*50)
    print("Strategy Comparison Summary")
    print("="*50)

    for strategy in ['joint', 'cyclic', 'curriculum']:
        if strategy in results['comparison']:
            best = results['comparison'][f'{strategy}_best']
            print(f"\n{strategy.capitalize()} Strategy:")
            print(f"  Best Config: {best['name']}")
            print(f"  Accuracy: {best['accuracy']:.4f}")
            if best.get('loss'):
                print(f"  Loss: {best['loss']:.4f}")

    print("\n" + "="*50)
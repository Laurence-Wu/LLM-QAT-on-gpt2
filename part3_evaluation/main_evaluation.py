import torch
import json
import sys
import os
from datetime import datetime
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import SwitchableQATGPT2
from shared.dataset import create_dataloaders
from transformers import GPT2Config, GPT2Tokenizer
from .evaluate_configurations import ConfigurationEvaluator
from .compare_strategies import compare_training_strategies, save_comparison_results, print_comparison_summary
from .adversarial_attacks import AdversarialEvaluator, analyze_robustness_results


def load_model(model_path=None, config=None):
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        if 'model_state_dict' in checkpoint:
            model = SwitchableQATGPT2(config, bit_widths=[4, 8, 16])
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model = checkpoint
    else:
        print("Creating new model for evaluation")
        model = SwitchableQATGPT2(config, bit_widths=[4, 8, 16])

    return model


def generate_comprehensive_report(config_results, strategy_comparison, robustness_results, output_path):
    report = {
        'timestamp': datetime.now().isoformat(),
        'configuration_evaluation': config_results,
        'strategy_comparison': strategy_comparison,
        'adversarial_robustness': robustness_results,
        'summary': generate_summary(config_results, strategy_comparison, robustness_results)
    }

    def convert_to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_report = convert_to_serializable(report)

    with open(output_path, 'w') as f:
        json.dump(serializable_report, f, indent=2)

    print(f"\nComprehensive report saved to {output_path}")


def generate_summary(config_results, strategy_comparison, robustness_results):
    summary = {}

    if config_results:
        best_config = max(config_results.items(), key=lambda x: x[1].get('accuracy', 0) if isinstance(x[1], dict) else 0)
        summary['best_configuration'] = {
            'name': best_config[0],
            'accuracy': best_config[1].get('accuracy', 0),
            'config': best_config[1].get('config', None)
        }

    if strategy_comparison and 'comparison' in strategy_comparison:
        strategies = ['joint_best', 'cyclic_best', 'curriculum_best']
        best_strategy = max(
            [s for s in strategies if s in strategy_comparison['comparison']],
            key=lambda x: strategy_comparison['comparison'][x].get('accuracy', 0)
        )
        summary['best_strategy'] = best_strategy.replace('_best', '')

    if robustness_results:
        baseline_8bit = robustness_results.get('fixed_8', {})
        random_switch = robustness_results.get('random_switch', {})

        if baseline_8bit and random_switch:
            baseline_avg = sum(baseline_8bit.values()) / len(baseline_8bit) if baseline_8bit else 0
            switch_avg = sum(random_switch.values()) / len(random_switch) if random_switch else 0
            improvement = ((baseline_avg - switch_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
            summary['robustness_improvement'] = f"{improvement:.1f}%"

    return summary


def main_evaluation_pipeline(args):
    print("="*50)
    print("Starting Comprehensive Evaluation Pipeline")
    print("="*50)

    gpt2_config = GPT2Config(
        vocab_size=50257,
        n_positions=256,
        n_embd=768,
        n_layer=6,
        n_head=12,
        layer_norm_epsilon=1e-5,
        embd_pdrop=0.1
    )

    model_path = args.model_path or 'part1_switchable_precision/best_model.pth'
    model = load_model(model_path, gpt2_config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    print("\nStep 1: Creating data loaders...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader, test_loader = create_dataloaders(
        tokenizer=tokenizer,
        train_split='train[:1000]',
        val_split='validation[:200]',
        test_split='validation[200:400]',
        batch_size=4,
        max_seq_length=128
    )

    print("\nStep 2: Evaluating all configurations...")
    evaluator = ConfigurationEvaluator(model, test_loader)
    config_results = evaluator.evaluate_all_configurations()

    print("\nConfiguration results:")
    for name, result in config_results.items():
        if isinstance(result, dict):
            print(f"  {name}: Accuracy={result.get('accuracy', 0):.4f}, Loss={result.get('loss', 0):.4f}")

    print("\nStep 3: Analyzing trade-offs...")
    pareto_points = evaluator.analyze_tradeoffs(config_results)
    print(f"Found {len(pareto_points)} Pareto optimal points")

    strategy_comparison = None
    if args.compare_strategies:
        print("\nStep 4: Comparing training strategies...")
        from part1_switchable_precision.config_qat import ModelConfig, TrainingConfig
        model_config = ModelConfig()
        training_config = TrainingConfig()
        training_config.num_iterations = 100
        training_config.eval_interval = 50

        strategy_comparison = compare_training_strategies(
            train_loader, val_loader, test_loader,
            model_config, training_config
        )
        print_comparison_summary(strategy_comparison)
        save_comparison_results(strategy_comparison)

    print("\nStep 5: Testing adversarial robustness...")
    adv_evaluator = AdversarialEvaluator(model, tokenizer)
    robustness_results = adv_evaluator.evaluate_dynamic_quantization_defense(test_loader, max_samples=50)
    analyze_robustness_results(robustness_results)

    print("\nStep 6: Generating comprehensive report...")
    generate_comprehensive_report(
        config_results,
        strategy_comparison or {},
        robustness_results,
        output_path='part3_evaluation/final_report.json'
    )

    print("\n" + "="*50)
    print("Key Findings:")
    print("="*50)

    if 'optimal' in config_results:
        print(f"Optimal configuration: {config_results['optimal']['config']}")
        print(f"Optimal accuracy: {config_results['optimal']['accuracy']:.4f}")

    if strategy_comparison and 'comparison' in strategy_comparison:
        best_strat = max(
            ['joint_best', 'cyclic_best', 'curriculum_best'],
            key=lambda x: strategy_comparison['comparison'].get(x, {}).get('accuracy', 0)
        )
        print(f"Best training strategy: {best_strat.replace('_best', '')}")

    if robustness_results:
        baseline = robustness_results.get('fixed_8', {})
        dynamic = robustness_results.get('random_switch', {})
        if baseline and dynamic:
            baseline_avg = sum(baseline.values()) / len(baseline)
            dynamic_avg = sum(dynamic.values()) / len(dynamic)
            improvement = (baseline_avg - dynamic_avg) / baseline_avg * 100
            print(f"Dynamic quantization improves robustness by: {improvement:.1f}%")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation pipeline')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--compare_strategies', action='store_true',
                       help='Compare different training strategies (takes longer)')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Maximum samples for evaluation')

    args = parser.parse_args()
    main_evaluation_pipeline(args)
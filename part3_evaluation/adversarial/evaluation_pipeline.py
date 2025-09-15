import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .attack_methods import AttackMethods
from .dynamic_defense import DynamicQuantizationDefense
from .robustness_metrics import RobustnessMetrics


class AdversarialEvaluationPipeline:
    """
    Comprehensive pipeline for adversarial robustness evaluation
    Combines attacks, defenses, and metrics
    """

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

        self.attacker = AttackMethods(model, tokenizer, device)
        self.defender = DynamicQuantizationDefense(model, tokenizer, device=device)
        self.metrics = RobustnessMetrics(model, tokenizer, device)

        self.results = {
            'attacks': {},
            'defenses': {},
            'metrics': {},
            'comparisons': {}
        }

    def run_comprehensive_evaluation(self, test_data: List[Tuple[torch.Tensor, torch.Tensor]],
                                    bit_configurations: List[Dict] = None,
                                    save_results: bool = True) -> Dict:
        """
        Run complete adversarial evaluation pipeline
        """
        print("="*70)
        print("Starting Comprehensive Adversarial Evaluation")
        print("="*70)

        if bit_configurations is None:
            bit_configurations = [
                {'name': 'FP16', 'bits': 16},
                {'name': 'INT8', 'bits': 8},
                {'name': 'INT4', 'bits': 4}
            ]

        for config in bit_configurations:
            print(f"\n{'='*60}")
            print(f"Evaluating configuration: {config['name']} ({config['bits']}-bit)")
            print('='*60)

            self._apply_bit_configuration(config['bits'])

            config_results = {
                'attacks': self._evaluate_all_attacks(test_data[:10]),
                'defenses': self._evaluate_all_defenses(test_data[:10]),
                'robustness': self._evaluate_robustness(test_data[:10])
            }

            self.results[config['name']] = config_results

            print(f"\nResults for {config['name']}:")
            self._print_summary(config_results)

        comparison_results = self._compare_configurations()
        self.results['comparisons'] = comparison_results

        if save_results:
            self._save_results()
            self._generate_plots()

        return self.results

    def _evaluate_all_attacks(self, test_data: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict:
        """Evaluate all attack methods"""
        print("\n1. Evaluating Attack Methods...")

        attack_results = {}
        attack_methods = [
            ('TextFooler', self.attacker.textfooler_attack),
            ('AutoPrompt', self.attacker.autoprompt_attack),
            ('Gradient-based', self.attacker.gradient_based_token_attack),
            ('Prompt Injection', self.attacker.prompt_injection_attack)
        ]

        for attack_name, attack_fn in attack_methods:
            print(f"   Testing {attack_name}...")
            success_rates = []
            perturbation_sizes = []

            for input_ids, labels in tqdm(test_data[:5], desc=f"   {attack_name}", leave=False):
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                    labels = labels.unsqueeze(0)

                try:
                    if attack_name == 'Prompt Injection':
                        result = attack_fn(input_ids, labels)
                    else:
                        result = attack_fn(input_ids, labels)

                    if 'success_rate' in result:
                        success_rates.append(result['success_rate'])
                    if 'avg_perturbations' in result:
                        perturbation_sizes.append(result['avg_perturbations'])
                except Exception as e:
                    print(f"      Warning: {attack_name} failed: {str(e)[:50]}")
                    continue

            attack_results[attack_name] = {
                'avg_success_rate': np.mean(success_rates) if success_rates else 0,
                'avg_perturbation': np.mean(perturbation_sizes) if perturbation_sizes else 0,
                'samples_tested': len(success_rates)
            }

            print(f"      Success Rate: {attack_results[attack_name]['avg_success_rate']:.2%}")

        universal_result = self._evaluate_universal_attack(test_data[:10])
        attack_results['Universal Trigger'] = universal_result

        return attack_results

    def _evaluate_universal_attack(self, test_data: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict:
        """Evaluate universal trigger attack"""
        print("   Testing Universal Trigger Attack...")
        try:
            result = self.attacker.universal_trigger_attack(test_data)
            return {
                'success_rate': result['success_rate'],
                'trigger_text': result['trigger_text'],
                'avg_loss_increase': result['avg_loss_increase']
            }
        except Exception as e:
            print(f"      Warning: Universal attack failed: {str(e)[:50]}")
            return {'success_rate': 0, 'trigger_text': '', 'avg_loss_increase': 0}

    def _evaluate_all_defenses(self, test_data: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict:
        """Evaluate all defense methods"""
        print("\n2. Evaluating Defense Methods...")

        defense_results = {}
        defense_methods = [
            ('Dynamic Quantization', 'dynamic'),
            ('Gradient Masking', 'gradient'),
            ('Input Transformation', 'transform'),
            ('Ensemble Defense', 'ensemble'),
            ('Detect and Reject', 'detect')
        ]

        for defense_name, defense_type in defense_methods:
            print(f"   Testing {defense_name}...")
            effectiveness = []
            overhead = []

            for input_ids, labels in tqdm(test_data[:5], desc=f"   {defense_name}", leave=False):
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                    labels = labels.unsqueeze(0)

                try:
                    if defense_type == 'dynamic':
                        result = self.defender.defend_with_dynamic_quantization(input_ids, labels)
                        effectiveness.append(1 - result.get('perturbation_score', 0))

                    elif defense_type == 'gradient':
                        result = self.defender.gradient_masking_defense(input_ids, labels)
                        effectiveness.append(result.get('gradient_norm', 0) < 1.0)

                    elif defense_type == 'transform':
                        result = self.defender.input_transformation_defense(input_ids)
                        effectiveness.append(result.get('logit_difference', 0) < 0.5)

                    elif defense_type == 'ensemble':
                        result = self.defender.ensemble_defense(input_ids, labels)
                        effectiveness.append(result.get('prediction_agreement', 0))

                    elif defense_type == 'detect':
                        result = self.defender.detect_and_reject(input_ids)
                        effectiveness.append(result['decision'] == 'ACCEPTED')

                except Exception as e:
                    print(f"      Warning: {defense_name} failed: {str(e)[:50]}")
                    continue

            defense_results[defense_name] = {
                'effectiveness': np.mean(effectiveness) if effectiveness else 0,
                'samples_tested': len(effectiveness)
            }

            print(f"      Effectiveness: {defense_results[defense_name]['effectiveness']:.2%}")

        return defense_results

    def _evaluate_robustness(self, test_data: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict:
        """Evaluate overall robustness metrics"""
        print("\n3. Computing Robustness Metrics...")

        clean_inputs = []
        adversarial_inputs = []
        labels_list = []

        for input_ids, labels in test_data[:5]:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                labels = labels.unsqueeze(0)

            clean_inputs.append(input_ids)
            labels_list.append(labels)

            try:
                attack_result = self.attacker.gradient_based_token_attack(input_ids, labels, epsilon=0.3)
                adversarial_inputs.append(attack_result['perturbed_ids'])
            except:
                adversarial_inputs.append(input_ids)

        robustness_metrics = self.metrics.compute_robustness_score(
            clean_inputs, adversarial_inputs, labels_list
        )

        print(f"   Overall Robustness Score: {robustness_metrics['overall_robustness_score']:.1f}/100")
        print(f"   Robust Accuracy: {robustness_metrics['accuracy_metrics']['robust_accuracy']:.2%}")
        print(f"   Attack Success Rate: {robustness_metrics['accuracy_metrics']['attack_success_rate']:.2%}")

        certified_results = []
        for input_ids, labels in test_data[:3]:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                labels = labels.unsqueeze(0)

            try:
                cert_result = self.metrics.compute_certified_robustness(input_ids, labels, num_samples=20)
                certified_results.append(cert_result['certified_accuracy'])
            except:
                continue

        if certified_results:
            robustness_metrics['certified_accuracy'] = np.mean(certified_results)
            print(f"   Certified Accuracy: {robustness_metrics['certified_accuracy']:.2%}")

        return robustness_metrics

    def _compare_configurations(self) -> Dict:
        """Compare robustness across different bit configurations"""
        print("\n" + "="*60)
        print("Comparing Configurations")
        print("="*60)

        comparison = {
            'robustness_scores': {},
            'attack_success_rates': {},
            'defense_effectiveness': {}
        }

        for config_name, results in self.results.items():
            if config_name == 'comparisons':
                continue

            if 'robustness' in results and 'overall_robustness_score' in results['robustness']:
                comparison['robustness_scores'][config_name] = results['robustness']['overall_robustness_score']

            if 'attacks' in results:
                avg_success = np.mean([
                    attack.get('avg_success_rate', 0)
                    for attack in results['attacks'].values()
                    if isinstance(attack, dict) and 'avg_success_rate' in attack
                ])
                comparison['attack_success_rates'][config_name] = avg_success

            if 'defenses' in results:
                avg_effectiveness = np.mean([
                    defense.get('effectiveness', 0)
                    for defense in results['defenses'].values()
                    if isinstance(defense, dict) and 'effectiveness' in defense
                ])
                comparison['defense_effectiveness'][config_name] = avg_effectiveness

        print("\nRobustness Scores:")
        for config, score in comparison['robustness_scores'].items():
            print(f"  {config}: {score:.1f}/100")

        print("\nAverage Attack Success Rates:")
        for config, rate in comparison['attack_success_rates'].items():
            print(f"  {config}: {rate:.2%}")

        print("\nAverage Defense Effectiveness:")
        for config, effectiveness in comparison['defense_effectiveness'].items():
            print(f"  {config}: {effectiveness:.2%}")

        return comparison

    def _apply_bit_configuration(self, bits: int):
        """Apply bit-width configuration to model"""
        if hasattr(self.model, 'set_global_precision'):
            self.model.set_global_precision(bits)
        elif hasattr(self.model, 'set_precision'):
            self.model.set_precision(bits)

    def _print_summary(self, results: Dict):
        """Print summary of results"""
        if 'attacks' in results:
            print("\nAttack Results:")
            for attack_name, attack_result in results['attacks'].items():
                if isinstance(attack_result, dict) and 'avg_success_rate' in attack_result:
                    print(f"  {attack_name}: {attack_result['avg_success_rate']:.2%} success rate")

        if 'defenses' in results:
            print("\nDefense Results:")
            for defense_name, defense_result in results['defenses'].items():
                if isinstance(defense_result, dict) and 'effectiveness' in defense_result:
                    print(f"  {defense_name}: {defense_result['effectiveness']:.2%} effectiveness")

        if 'robustness' in results and 'overall_robustness_score' in results['robustness']:
            print(f"\nOverall Robustness: {results['robustness']['overall_robustness_score']:.1f}/100")

    def _save_results(self):
        """Save evaluation results to file"""
        output_dir = Path('part3_evaluation/results/adversarial')
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / 'adversarial_evaluation_results.json', 'w') as f:
            json.dump(self._serialize_results(self.results), f, indent=2)

        print(f"\nResults saved to {output_dir / 'adversarial_evaluation_results.json'}")

    def _serialize_results(self, obj):
        """Serialize results for JSON saving"""
        if isinstance(obj, dict):
            return {k: self._serialize_results(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_results(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def _generate_plots(self):
        """Generate visualization plots"""
        output_dir = Path('part3_evaluation/results/adversarial')
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        configs = list(self.results['comparisons']['robustness_scores'].keys())
        robustness_scores = list(self.results['comparisons']['robustness_scores'].values())

        axes[0, 0].bar(configs, robustness_scores, color='blue', alpha=0.7)
        axes[0, 0].set_xlabel('Configuration')
        axes[0, 0].set_ylabel('Robustness Score (/100)')
        axes[0, 0].set_title('Robustness Scores Across Bit Configurations')
        axes[0, 0].grid(True, alpha=0.3)

        attack_rates = list(self.results['comparisons']['attack_success_rates'].values())
        axes[0, 1].bar(configs, attack_rates, color='red', alpha=0.7)
        axes[0, 1].set_xlabel('Configuration')
        axes[0, 1].set_ylabel('Attack Success Rate')
        axes[0, 1].set_title('Vulnerability to Attacks')
        axes[0, 1].grid(True, alpha=0.3)

        defense_effectiveness = list(self.results['comparisons']['defense_effectiveness'].values())
        axes[1, 0].bar(configs, defense_effectiveness, color='green', alpha=0.7)
        axes[1, 0].set_xlabel('Configuration')
        axes[1, 0].set_ylabel('Defense Effectiveness')
        axes[1, 0].set_title('Defense Effectiveness Across Configurations')
        axes[1, 0].grid(True, alpha=0.3)

        attack_names = []
        attack_success = []
        for config_name, results in self.results.items():
            if config_name != 'comparisons' and 'attacks' in results:
                for attack_name, attack_result in results['attacks'].items():
                    if isinstance(attack_result, dict) and 'avg_success_rate' in attack_result:
                        attack_names.append(f"{config_name[:4]}-{attack_name[:8]}")
                        attack_success.append(attack_result['avg_success_rate'])

        if attack_names and attack_success:
            axes[1, 1].barh(attack_names[-10:], attack_success[-10:], color='orange', alpha=0.7)
            axes[1, 1].set_xlabel('Success Rate')
            axes[1, 1].set_ylabel('Config-Attack')
            axes[1, 1].set_title('Attack Success Rates (Top 10)')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'adversarial_evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Plots saved to {output_dir / 'adversarial_evaluation_plots.png'}")
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm


class RobustnessMetrics:
    """
    Comprehensive metrics for evaluating adversarial robustness
    Following recent evaluation practices (2021-2024)
    """

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

    def compute_robustness_score(self, clean_inputs: List[torch.Tensor],
                                adversarial_inputs: List[torch.Tensor],
                                labels: List[torch.Tensor]) -> Dict:
        """
        Compute comprehensive robustness score
        """
        assert len(clean_inputs) == len(adversarial_inputs) == len(labels)

        metrics = {
            'accuracy_metrics': self._compute_accuracy_metrics(clean_inputs, adversarial_inputs, labels),
            'perturbation_metrics': self._compute_perturbation_metrics(clean_inputs, adversarial_inputs),
            'prediction_metrics': self._compute_prediction_metrics(clean_inputs, adversarial_inputs, labels),
            'semantic_metrics': self._compute_semantic_metrics(clean_inputs, adversarial_inputs)
        }

        overall_score = self._compute_overall_robustness(metrics)
        metrics['overall_robustness_score'] = overall_score

        return metrics

    def _compute_accuracy_metrics(self, clean_inputs: List[torch.Tensor],
                                 adversarial_inputs: List[torch.Tensor],
                                 labels: List[torch.Tensor]) -> Dict:
        """Compute accuracy-based robustness metrics"""
        clean_correct = 0
        adv_correct = 0
        robust_correct = 0
        total = 0

        for clean_in, adv_in, label in zip(clean_inputs, adversarial_inputs, labels):
            clean_in = clean_in.to(self.device)
            adv_in = adv_in.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                clean_out = self.model(input_ids=clean_in, labels=label)
                adv_out = self.model(input_ids=adv_in, labels=label)

                clean_pred = torch.argmax(clean_out.logits, dim=-1)
                adv_pred = torch.argmax(adv_out.logits, dim=-1)

                valid_mask = label != -100

                clean_acc = (clean_pred[valid_mask] == label[valid_mask]).float().mean()
                adv_acc = (adv_pred[valid_mask] == label[valid_mask]).float().mean()

                clean_correct += clean_acc.item()
                adv_correct += adv_acc.item()

                if clean_acc > 0 and adv_acc > 0:
                    robust_correct += 1

                total += 1

        return {
            'clean_accuracy': clean_correct / max(total, 1),
            'adversarial_accuracy': adv_correct / max(total, 1),
            'robust_accuracy': robust_correct / max(total, 1),
            'accuracy_drop': (clean_correct - adv_correct) / max(total, 1),
            'attack_success_rate': 1 - (robust_correct / max(total, 1))
        }

    def _compute_perturbation_metrics(self, clean_inputs: List[torch.Tensor],
                                     adversarial_inputs: List[torch.Tensor]) -> Dict:
        """Compute perturbation-based metrics"""
        l0_norms = []
        l2_norms = []
        linf_norms = []
        relative_changes = []

        for clean_in, adv_in in zip(clean_inputs, adversarial_inputs):
            diff = (adv_in - clean_in).float()

            l0 = (diff != 0).float().sum().item()
            l2 = diff.norm(2).item()
            linf = diff.abs().max().item()

            l0_norms.append(l0)
            l2_norms.append(l2)
            linf_norms.append(linf)

            total_tokens = clean_in.numel()
            changed_tokens = (diff != 0).sum().item()
            relative_changes.append(changed_tokens / max(total_tokens, 1))

        return {
            'avg_l0_norm': np.mean(l0_norms),
            'avg_l2_norm': np.mean(l2_norms),
            'avg_linf_norm': np.mean(linf_norms),
            'avg_relative_change': np.mean(relative_changes),
            'max_l0_norm': np.max(l0_norms),
            'max_l2_norm': np.max(l2_norms),
            'max_linf_norm': np.max(linf_norms)
        }

    def _compute_prediction_metrics(self, clean_inputs: List[torch.Tensor],
                                   adversarial_inputs: List[torch.Tensor],
                                   labels: List[torch.Tensor]) -> Dict:
        """Compute prediction stability metrics"""
        kl_divergences = []
        confidence_changes = []
        entropy_changes = []
        loss_increases = []

        for clean_in, adv_in, label in zip(clean_inputs, adversarial_inputs, labels):
            clean_in = clean_in.to(self.device)
            adv_in = adv_in.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                clean_out = self.model(input_ids=clean_in, labels=label)
                adv_out = self.model(input_ids=adv_in, labels=label)

                clean_probs = F.softmax(clean_out.logits, dim=-1)
                adv_probs = F.softmax(adv_out.logits, dim=-1)

                kl_div = F.kl_div(adv_probs.log(), clean_probs, reduction='batchmean').item()
                kl_divergences.append(kl_div)

                clean_conf = clean_probs.max(dim=-1).values.mean().item()
                adv_conf = adv_probs.max(dim=-1).values.mean().item()
                confidence_changes.append(clean_conf - adv_conf)

                clean_entropy = -(clean_probs * torch.log(clean_probs + 1e-8)).sum(dim=-1).mean().item()
                adv_entropy = -(adv_probs * torch.log(adv_probs + 1e-8)).sum(dim=-1).mean().item()
                entropy_changes.append(adv_entropy - clean_entropy)

                loss_increase = (adv_out.loss - clean_out.loss).item()
                loss_increases.append(loss_increase)

        return {
            'avg_kl_divergence': np.mean(kl_divergences),
            'avg_confidence_drop': np.mean(confidence_changes),
            'avg_entropy_increase': np.mean(entropy_changes),
            'avg_loss_increase': np.mean(loss_increases),
            'max_kl_divergence': np.max(kl_divergences),
            'max_confidence_drop': np.max(confidence_changes)
        }

    def _compute_semantic_metrics(self, clean_inputs: List[torch.Tensor],
                                 adversarial_inputs: List[torch.Tensor]) -> Dict:
        """Compute semantic preservation metrics"""
        semantic_similarities = []
        text_similarities = []

        for clean_in, adv_in in zip(clean_inputs, adversarial_inputs):
            clean_text = self.tokenizer.decode(clean_in[0], skip_special_tokens=True)
            adv_text = self.tokenizer.decode(adv_in[0], skip_special_tokens=True)

            clean_words = set(clean_text.lower().split())
            adv_words = set(adv_text.lower().split())

            if clean_words and adv_words:
                jaccard = len(clean_words.intersection(adv_words)) / len(clean_words.union(adv_words))
                text_similarities.append(jaccard)

            clean_emb = self.model.get_input_embeddings()(clean_in.to(self.device))
            adv_emb = self.model.get_input_embeddings()(adv_in.to(self.device))

            cos_sim = F.cosine_similarity(clean_emb.mean(dim=1), adv_emb.mean(dim=1), dim=-1).mean().item()
            semantic_similarities.append(cos_sim)

        return {
            'avg_semantic_similarity': np.mean(semantic_similarities),
            'avg_text_similarity': np.mean(text_similarities) if text_similarities else 0,
            'min_semantic_similarity': np.min(semantic_similarities),
            'min_text_similarity': np.min(text_similarities) if text_similarities else 0
        }

    def _compute_overall_robustness(self, metrics: Dict) -> float:
        """Compute overall robustness score (0-100)"""
        weights = {
            'robust_accuracy': 0.3,
            'accuracy_preservation': 0.2,
            'prediction_stability': 0.2,
            'semantic_preservation': 0.2,
            'perturbation_resistance': 0.1
        }

        scores = {}

        scores['robust_accuracy'] = metrics['accuracy_metrics']['robust_accuracy'] * 100

        acc_drop = metrics['accuracy_metrics']['accuracy_drop']
        scores['accuracy_preservation'] = max(0, (1 - acc_drop) * 100)

        kl_div = metrics['prediction_metrics']['avg_kl_divergence']
        scores['prediction_stability'] = max(0, (1 - min(kl_div / 5, 1)) * 100)

        sem_sim = metrics['semantic_metrics']['avg_semantic_similarity']
        scores['semantic_preservation'] = max(0, sem_sim * 100)

        rel_change = metrics['perturbation_metrics']['avg_relative_change']
        scores['perturbation_resistance'] = max(0, (1 - rel_change) * 100)

        overall = sum(scores[k] * weights[k] for k in weights.keys())

        return round(overall, 2)

    def evaluate_defense_effectiveness(self, attack_results: Dict,
                                      defense_results: Dict) -> Dict:
        """Evaluate how effective a defense is against attacks"""
        metrics = {}

        if 'success_rate' in attack_results and 'success_rate' in defense_results:
            metrics['defense_success_rate'] = 1 - defense_results['success_rate']
            metrics['attack_mitigation'] = (attack_results['success_rate'] -
                                           defense_results['success_rate']) / attack_results['success_rate']

        if 'loss' in attack_results and 'loss' in defense_results:
            metrics['loss_reduction'] = (attack_results['loss'] -
                                        defense_results['loss']) / attack_results['loss']

        if 'perturbation_score' in defense_results:
            metrics['detection_accuracy'] = defense_results['perturbation_score']

        return metrics

    def plot_robustness_curves(self, results: Dict, save_path: Optional[str] = None):
        """Plot various robustness curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        if 'epsilon_values' in results and 'robust_accuracies' in results:
            axes[0, 0].plot(results['epsilon_values'], results['robust_accuracies'], 'b-', marker='o')
            axes[0, 0].set_xlabel('Perturbation Budget (ε)')
            axes[0, 0].set_ylabel('Robust Accuracy')
            axes[0, 0].set_title('Robustness vs Perturbation Strength')
            axes[0, 0].grid(True, alpha=0.3)

        if 'bit_widths' in results and 'robustness_scores' in results:
            axes[0, 1].bar(results['bit_widths'], results['robustness_scores'], color='green', alpha=0.7)
            axes[0, 1].set_xlabel('Quantization Bit-width')
            axes[0, 1].set_ylabel('Robustness Score')
            axes[0, 1].set_title('Robustness across Bit-widths')
            axes[0, 1].grid(True, alpha=0.3)

        if 'attack_types' in results and 'success_rates' in results:
            axes[1, 0].barh(results['attack_types'], results['success_rates'], color='red', alpha=0.7)
            axes[1, 0].set_xlabel('Attack Success Rate')
            axes[1, 0].set_ylabel('Attack Type')
            axes[1, 0].set_title('Vulnerability to Different Attacks')
            axes[1, 0].grid(True, alpha=0.3)

        if 'defense_methods' in results and 'effectiveness' in results:
            axes[1, 1].bar(results['defense_methods'], results['effectiveness'], color='blue', alpha=0.7)
            axes[1, 1].set_xlabel('Defense Method')
            axes[1, 1].set_ylabel('Effectiveness (%)')
            axes[1, 1].set_title('Defense Effectiveness Comparison')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def compute_certified_robustness(self, input_ids: torch.Tensor,
                                    labels: torch.Tensor,
                                    radius: float = 0.1,
                                    num_samples: int = 100) -> Dict:
        """
        Compute certified robustness using randomized smoothing
        """
        self.model.eval()

        predictions = []
        confidences = []

        for _ in range(num_samples):
            noise = torch.randn_like(input_ids.float()) * radius
            noisy_input = input_ids + noise.long()
            noisy_input = torch.clamp(noisy_input, 0, self.tokenizer.vocab_size - 1)

            with torch.no_grad():
                outputs = self.model(input_ids=noisy_input, labels=labels)
                probs = F.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(probs, dim=-1)
                conf = probs.max(dim=-1).values

                predictions.append(pred)
                confidences.append(conf)

        predictions = torch.stack(predictions)
        confidences = torch.stack(confidences)

        base_pred = torch.mode(predictions, dim=0).values
        pred_counts = (predictions == base_pred.unsqueeze(0)).float().mean(dim=0)

        certified_radius = radius * torch.sqrt(2 * torch.log(1 / (1 - pred_counts + 1e-8)))

        return {
            'certified_radius': certified_radius.mean().item(),
            'prediction_stability': pred_counts.mean().item(),
            'confidence_mean': confidences.mean().item(),
            'confidence_std': confidences.std().item(),
            'certified_accuracy': (pred_counts > 0.5).float().mean().item()
        }

    def benchmark_against_baselines(self, model_results: Dict,
                                   baseline_results: Dict) -> Dict:
        """Compare model against baseline robustness benchmarks"""
        comparison = {}

        for metric in ['robust_accuracy', 'attack_success_rate', 'overall_robustness_score']:
            if metric in model_results and metric in baseline_results:
                model_val = model_results[metric]
                baseline_val = baseline_results[metric]

                comparison[f'{metric}_improvement'] = ((model_val - baseline_val) / baseline_val) * 100
                comparison[f'{metric}_ratio'] = model_val / baseline_val

        return comparison
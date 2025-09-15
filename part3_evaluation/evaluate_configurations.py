import torch
import torch.nn.functional as F
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm


class ConfigurationEvaluator:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def evaluate_all_configurations(self):
        results = {}

        for bits in [4, 8, 16]:
            config = [bits] * self.model.n_layer
            results[f'uniform_{bits}'] = self._evaluate_single_config(config)

        mixed_configs = {
            'progressive': [4, 4, 8, 8, 16, 16],
            'hourglass': [16, 8, 4, 4, 8, 16],
            'edges_high': [16, 8, 8, 8, 8, 16],
        }

        for name, config in mixed_configs.items():
            if len(config) != self.model.n_layer:
                config = self._adjust_config_length(config)
            results[name] = self._evaluate_single_config(config)

        optimal_config = self.search_optimal_configuration()
        results['optimal'] = optimal_config

        return results

    def _adjust_config_length(self, config):
        if len(config) < self.model.n_layer:
            config = config + [8] * (self.model.n_layer - len(config))
        elif len(config) > self.model.n_layer:
            config = config[:self.model.n_layer]
        return config

    def search_optimal_configuration(self, max_bits=8.0):
        best_config = None
        best_accuracy = 0

        for config in product([4, 8, 16], repeat=min(self.model.n_layer, 3)):
            extended_config = list(config) + [8] * (self.model.n_layer - len(config))

            avg_bits = np.mean(extended_config)
            if avg_bits <= max_bits:
                result = self._evaluate_single_config(extended_config)
                if result['accuracy'] > best_accuracy:
                    best_accuracy = result['accuracy']
                    best_config = extended_config

        return {'config': best_config, 'accuracy': best_accuracy}

    def _evaluate_single_config(self, config):
        self.model.eval()
        self.model.set_layer_precision(config)

        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        max_batches = 50

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"Evaluating {config}", leave=False):
                if num_batches >= max_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                labels = input_ids

                outputs = self.model(input_ids, labels=labels)
                loss = outputs['loss']

                total_loss += loss.item()
                predictions = outputs['logits'].argmax(dim=-1)

                shift_predictions = predictions[..., :-1].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                correct += (shift_predictions == shift_labels).sum().item()
                total += shift_labels.numel()
                num_batches += 1

        return {
            'loss': total_loss / max(num_batches, 1),
            'accuracy': correct / max(total, 1),
            'effective_bits': np.mean(config),
            'config': config
        }

    def analyze_tradeoffs(self, results):
        accuracies = []
        bits = []
        names = []

        for name, result in results.items():
            if isinstance(result, dict) and 'accuracy' in result:
                accuracies.append(result['accuracy'])
                bits.append(result.get('effective_bits', 8))
                names.append(name)

        plt.figure(figsize=(10, 6))
        plt.scatter(bits, accuracies)

        for i, name in enumerate(names):
            plt.annotate(name, (bits[i], accuracies[i]), fontsize=8)

        plt.xlabel('Average Bits')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Efficiency Trade-off')

        pareto_points = self._find_pareto_points(bits, accuracies)
        if pareto_points:
            plt.scatter([bits[i] for i in pareto_points],
                       [accuracies[i] for i in pareto_points],
                       color='red', marker='*', s=200, label='Pareto Optimal')

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('part3_evaluation/pareto_frontier.png')
        plt.close()

        return pareto_points

    def _find_pareto_points(self, bits, accuracies):
        pareto_points = []
        for i in range(len(bits)):
            is_pareto = True
            for j in range(len(bits)):
                if i != j:
                    if bits[j] <= bits[i] and accuracies[j] > accuracies[i]:
                        is_pareto = False
                        break
            if is_pareto:
                pareto_points.append(i)
        return pareto_points
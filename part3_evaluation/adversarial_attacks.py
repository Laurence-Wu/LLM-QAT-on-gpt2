import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random


class AdversarialEvaluator:
    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attacks = self._initialize_attacks()

    def _initialize_attacks(self):
        return {
            'fgsm': self._fgsm_attack,
            'pgd': self._pgd_attack,
            'hotflip': self._hotflip_attack
        }

    def _fgsm_attack(self, input_ids, labels, epsilon=0.01):
        input_embeds = self.model.wte(input_ids).detach()
        input_embeds.requires_grad = True

        outputs = self.model.forward_from_embeddings(input_embeds, labels=labels)
        loss = outputs['loss']

        self.model.zero_grad()
        loss.backward()

        perturbed_embeds = input_embeds + epsilon * input_embeds.grad.sign()
        perturbed_embeds = perturbed_embeds.detach()

        return perturbed_embeds

    def _pgd_attack(self, input_ids, labels, epsilon=0.01, alpha=0.002, num_iter=10):
        input_embeds = self.model.wte(input_ids).detach()
        ori_embeds = input_embeds.clone()

        for _ in range(num_iter):
            input_embeds = input_embeds.detach()
            input_embeds.requires_grad = True

            outputs = self.model.forward_from_embeddings(input_embeds, labels=labels)
            loss = outputs['loss']

            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                input_embeds = input_embeds + alpha * input_embeds.grad.sign()
                delta = torch.clamp(input_embeds - ori_embeds, -epsilon, epsilon)
                input_embeds = ori_embeds + delta

        return input_embeds.detach()

    def _hotflip_attack(self, input_ids, labels, num_candidates=10):
        batch_size, seq_len = input_ids.shape
        vocab_size = self.model.config.vocab_size

        input_embeds = self.model.wte(input_ids).detach()
        input_embeds.requires_grad = True

        outputs = self.model.forward_from_embeddings(input_embeds, labels=labels)
        loss = outputs['loss']

        self.model.zero_grad()
        loss.backward()

        grad = input_embeds.grad
        grad_norm = grad.norm(dim=-1)

        _, top_positions = grad_norm.view(batch_size, -1).topk(1, dim=-1)

        perturbed_ids = input_ids.clone()
        for b in range(batch_size):
            pos = top_positions[b].item()
            candidates = torch.randint(0, vocab_size, (num_candidates,), device=input_ids.device)
            best_token = candidates[0]
            best_loss = float('inf')

            for candidate in candidates:
                temp_ids = perturbed_ids.clone()
                temp_ids[b, pos] = candidate
                temp_embeds = self.model.wte(temp_ids)

                with torch.no_grad():
                    outputs = self.model.forward_from_embeddings(temp_embeds, labels=labels)
                    if outputs['loss'].item() > best_loss:
                        best_loss = outputs['loss'].item()
                        best_token = candidate

            perturbed_ids[b, pos] = best_token

        return self.model.wte(perturbed_ids)

    def evaluate_dynamic_quantization_defense(self, test_samples, max_samples=100):
        results = {}

        for bits in [4, 8, 16]:
            self.model.set_global_precision(bits)
            results[f'fixed_{bits}'] = self._evaluate_attack_success_rate(test_samples, max_samples)

        results['random_switch'] = self._evaluate_random_switching(test_samples, max_samples)
        results['ensemble'] = self._evaluate_ensemble_defense(test_samples, max_samples)
        results['adaptive'] = self._evaluate_adaptive_precision(test_samples, max_samples)

        return results

    def _evaluate_attack_success_rate(self, test_samples, max_samples=100):
        attack_results = {}

        for attack_name, attack_fn in self.attacks.items():
            success_count = 0
            total_count = 0

            for i, batch in enumerate(tqdm(test_samples, desc=f"Evaluating {attack_name}", leave=False)):
                if i >= max_samples:
                    break

                input_ids = batch['input_ids'].to(self.device)
                labels = input_ids

                with torch.no_grad():
                    clean_outputs = self.model(input_ids, labels=labels)
                    clean_pred = clean_outputs['logits'].argmax(dim=-1)

                if attack_name == 'hotflip':
                    adv_embeds = attack_fn(input_ids, labels)
                else:
                    adv_embeds = attack_fn(input_ids, labels)

                with torch.no_grad():
                    adv_outputs = self.model.forward_from_embeddings(adv_embeds, labels=labels)
                    adv_pred = adv_outputs['logits'].argmax(dim=-1)

                success_count += (clean_pred != adv_pred).float().mean().item()
                total_count += 1

            attack_results[attack_name] = success_count / max(total_count, 1)

        return attack_results

    def _evaluate_random_switching(self, test_samples, max_samples=100):
        attack_results = {}

        for attack_name, attack_fn in self.attacks.items():
            robustness_scores = []

            for i, batch in enumerate(tqdm(test_samples, desc=f"Random switching - {attack_name}", leave=False)):
                if i >= max_samples:
                    break

                input_ids = batch['input_ids'].to(self.device)
                labels = input_ids

                if attack_name == 'hotflip':
                    adv_embeds = attack_fn(input_ids, labels)
                else:
                    adv_embeds = attack_fn(input_ids, labels)

                predictions = []
                for _ in range(10):
                    random_config = [random.choice([4, 8, 16]) for _ in range(self.model.n_layer)]
                    self.model.set_layer_precision(random_config)

                    with torch.no_grad():
                        outputs = self.model.forward_from_embeddings(adv_embeds, labels=labels)
                        predictions.append(outputs['logits'].argmax(dim=-1))

                predictions_stack = torch.stack(predictions)
                majority_vote = torch.mode(predictions_stack, dim=0)[0]

                with torch.no_grad():
                    clean_outputs = self.model(input_ids, labels=labels)
                    clean_pred = clean_outputs['logits'].argmax(dim=-1)

                robustness = (majority_vote == clean_pred).float().mean().item()
                robustness_scores.append(robustness)

            attack_results[attack_name] = 1 - np.mean(robustness_scores)

        return attack_results

    def _evaluate_ensemble_defense(self, test_samples, max_samples=100):
        attack_results = {}

        for attack_name, attack_fn in self.attacks.items():
            robustness_scores = []

            for i, batch in enumerate(tqdm(test_samples, desc=f"Ensemble - {attack_name}", leave=False)):
                if i >= max_samples:
                    break

                input_ids = batch['input_ids'].to(self.device)
                labels = input_ids

                if attack_name == 'hotflip':
                    adv_embeds = attack_fn(input_ids, labels)
                else:
                    adv_embeds = attack_fn(input_ids, labels)

                logits_ensemble = []
                for bits in [4, 8, 16]:
                    self.model.set_global_precision(bits)
                    with torch.no_grad():
                        outputs = self.model.forward_from_embeddings(adv_embeds, labels=labels)
                        logits_ensemble.append(outputs['logits'])

                ensemble_logits = torch.stack(logits_ensemble).mean(0)
                ensemble_pred = ensemble_logits.argmax(dim=-1)

                with torch.no_grad():
                    clean_outputs = self.model(input_ids, labels=labels)
                    clean_pred = clean_outputs['logits'].argmax(dim=-1)

                robustness = (ensemble_pred == clean_pred).float().mean().item()
                robustness_scores.append(robustness)

            attack_results[attack_name] = 1 - np.mean(robustness_scores)

        return attack_results

    def _evaluate_adaptive_precision(self, test_samples, max_samples=100):
        attack_results = {}

        for attack_name, attack_fn in self.attacks.items():
            robustness_scores = []

            for i, batch in enumerate(tqdm(test_samples, desc=f"Adaptive - {attack_name}", leave=False)):
                if i >= max_samples:
                    break

                input_ids = batch['input_ids'].to(self.device)
                labels = input_ids

                uncertainty_scores = []
                for bits in [4, 8, 16]:
                    self.model.set_global_precision(bits)
                    with torch.no_grad():
                        outputs = self.model(input_ids, labels=labels)
                        probs = F.softmax(outputs['logits'], dim=-1)
                        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
                        uncertainty_scores.append(entropy.item())

                if max(uncertainty_scores) > np.mean(uncertainty_scores) * 1.5:
                    selected_bits = 16
                elif min(uncertainty_scores) < np.mean(uncertainty_scores) * 0.5:
                    selected_bits = 4
                else:
                    selected_bits = 8

                self.model.set_global_precision(selected_bits)

                if attack_name == 'hotflip':
                    adv_embeds = attack_fn(input_ids, labels)
                else:
                    adv_embeds = attack_fn(input_ids, labels)

                with torch.no_grad():
                    adv_outputs = self.model.forward_from_embeddings(adv_embeds, labels=labels)
                    adv_pred = adv_outputs['logits'].argmax(dim=-1)

                    clean_outputs = self.model(input_ids, labels=labels)
                    clean_pred = clean_outputs['logits'].argmax(dim=-1)

                robustness = (adv_pred == clean_pred).float().mean().item()
                robustness_scores.append(robustness)

            attack_results[attack_name] = 1 - np.mean(robustness_scores)

        return attack_results


def analyze_robustness_results(results):
    print("\n" + "="*50)
    print("Adversarial Robustness Analysis")
    print("="*50)

    baseline_scores = {}
    for bits in [4, 8, 16]:
        key = f'fixed_{bits}'
        if key in results:
            avg_score = np.mean(list(results[key].values()))
            baseline_scores[bits] = avg_score
            print(f"\nFixed {bits}-bit precision:")
            for attack, score in results[key].items():
                print(f"  {attack}: {score:.3f}")
            print(f"  Average: {avg_score:.3f}")

    defense_methods = ['random_switch', 'ensemble', 'adaptive']
    for method in defense_methods:
        if method in results:
            avg_score = np.mean(list(results[method].values()))
            print(f"\n{method.replace('_', ' ').title()} Defense:")
            for attack, score in results[method].items():
                print(f"  {attack}: {score:.3f}")
            print(f"  Average: {avg_score:.3f}")

            if 8 in baseline_scores:
                improvement = (baseline_scores[8] - avg_score) / baseline_scores[8] * 100
                print(f"  Improvement over 8-bit: {improvement:.1f}%")

    print("\n" + "="*50)
"""
Simplified Adversarial Attack Implementation
Focuses on TextFooler and Gradient-based attacks for evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import random
from nltk.corpus import wordnet
import nltk

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextFoolerAttack:
    """
    TextFooler attack implementation for language models.
    Performs word-level substitutions to create adversarial examples.
    """

    def __init__(self, model, tokenizer, device='cuda'):
        """
        Initialize TextFooler attacker.

        Args:
            model: Target model to attack
            tokenizer: Tokenizer for the model
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    def get_synonyms(self, word: str, pos_tag: Optional[str] = None) -> List[str]:
        """
        Get synonyms for a word using WordNet.

        Args:
            word: Word to find synonyms for
            pos_tag: Part-of-speech tag

        Returns:
            List of synonym words
        """
        synonyms = set()

        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)

        return list(synonyms)[:10]

    def compute_importance_scores(self, input_ids: torch.Tensor,
                                 labels: torch.Tensor) -> torch.Tensor:
        """
        Compute importance scores for each token using gradients.

        Args:
            input_ids: Input token IDs
            labels: Target labels

        Returns:
            Importance scores for each token
        """
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        self.model.zero_grad()

        input_embeds = self.model.transformer.wte(input_ids)
        input_embeds.requires_grad = True

        outputs = self.model(
            inputs_embeds=input_embeds.unsqueeze(0) if input_embeds.dim() == 2 else input_embeds,
            labels=labels.unsqueeze(0) if labels.dim() == 1 else labels
        )

        loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
        loss.backward()

        grad_norm = torch.norm(input_embeds.grad, dim=-1)

        return grad_norm

    def generate_adversarial(self, text: str, target_label: Optional[int] = None,
                           max_perturb_ratio: float = 0.3) -> Dict:
        """
        Generate adversarial example using TextFooler.

        Args:
            text: Input text to perturb
            target_label: Optional target label for targeted attack
            max_perturb_ratio: Maximum ratio of words to perturb

        Returns:
            Dictionary with adversarial example and metrics
        """
        tokens = self.tokenizer.tokenize(text)
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)

        if input_ids.shape[1] < 3:
            return {
                'success': False,
                'original_text': text,
                'adversarial_text': text,
                'num_changes': 0,
                'perturb_ratio': 0.0
            }

        labels = input_ids.clone()

        with torch.no_grad():
            orig_outputs = self.model(input_ids, labels=labels)
            orig_loss = orig_outputs['loss'].item()
            orig_preds = orig_outputs['logits'].argmax(dim=-1)

        importance_scores = self.compute_importance_scores(input_ids[0], labels[0])

        max_changes = int(len(tokens) * max_perturb_ratio)
        num_changes = 0
        perturbed_tokens = tokens.copy()

        important_indices = torch.argsort(importance_scores, descending=True)

        for idx in important_indices[:max_changes]:
            idx = idx.item()

            if idx < 1 or idx >= len(tokens) - 1:
                continue

            original_token = tokens[idx]
            word = self.tokenizer.convert_tokens_to_string([original_token]).strip()

            synonyms = self.get_synonyms(word)

            if not synonyms:
                continue

            best_synonym = None
            best_loss = orig_loss

            for synonym in synonyms:
                temp_tokens = perturbed_tokens.copy()
                temp_tokens[idx] = self.tokenizer.tokenize(synonym)[0] if self.tokenizer.tokenize(synonym) else original_token

                temp_text = self.tokenizer.convert_tokens_to_string(temp_tokens)
                temp_ids = self.tokenizer.encode(temp_text, return_tensors='pt').to(self.device)

                if temp_ids.shape[1] != input_ids.shape[1]:
                    continue

                temp_labels = temp_ids.clone()

                with torch.no_grad():
                    temp_outputs = self.model(temp_ids, labels=temp_labels)
                    temp_loss = temp_outputs['loss'].item()

                if temp_loss > best_loss:
                    best_loss = temp_loss
                    best_synonym = synonym

            if best_synonym:
                perturbed_tokens[idx] = self.tokenizer.tokenize(best_synonym)[0]
                num_changes += 1

                if best_loss > orig_loss * 1.5:
                    break

        adversarial_text = self.tokenizer.convert_tokens_to_string(perturbed_tokens)
        adv_ids = self.tokenizer.encode(adversarial_text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            adv_outputs = self.model(adv_ids, labels=adv_ids)
            adv_loss = adv_outputs['loss'].item()
            adv_preds = adv_outputs['logits'].argmax(dim=-1)

        success = adv_loss > orig_loss * 1.2

        return {
            'success': success,
            'original_text': text,
            'adversarial_text': adversarial_text,
            'num_changes': num_changes,
            'perturb_ratio': num_changes / len(tokens),
            'original_loss': orig_loss,
            'adversarial_loss': adv_loss,
            'loss_increase': (adv_loss - orig_loss) / orig_loss
        }


class GradientAttack:
    """
    Gradient-based attack (similar to HotFlip) for language models.
    Uses gradients to find minimal perturbations in embedding space.
    """

    def __init__(self, model, tokenizer, device='cuda'):
        """
        Initialize gradient attacker.

        Args:
            model: Target model to attack
            tokenizer: Tokenizer for the model
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)

    def hotflip_attack(self, input_ids: torch.Tensor,
                      labels: torch.Tensor,
                      num_iterations: int = 10,
                      epsilon: float = 0.1) -> Dict:
        """
        Perform HotFlip-style gradient attack.

        Args:
            input_ids: Input token IDs
            labels: Target labels
            num_iterations: Number of attack iterations
            epsilon: Perturbation magnitude

        Returns:
            Dictionary with attack results
        """
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        with torch.no_grad():
            orig_outputs = self.model(input_ids, labels=labels)
            orig_loss = orig_outputs['loss'].item()
            orig_logits = orig_outputs['logits'].clone()

        embedding_layer = self.model.transformer.wte
        vocab_size = embedding_layer.weight.shape[0]

        perturbed_ids = input_ids.clone()

        for iteration in range(num_iterations):
            self.model.zero_grad()

            input_embeds = embedding_layer(perturbed_ids)
            input_embeds.requires_grad = True

            outputs = self.model(
                inputs_embeds=input_embeds,
                labels=labels
            )

            loss = outputs['loss']
            loss.backward()

            grad = input_embeds.grad

            with torch.no_grad():
                grad_norm = torch.norm(grad, dim=-1, keepdim=True)
                grad_norm = torch.where(grad_norm > 0, grad_norm, torch.ones_like(grad_norm))
                normalized_grad = grad / grad_norm

                perturbation = epsilon * normalized_grad.sign()
                perturbed_embeds = input_embeds + perturbation

                all_embeddings = embedding_layer.weight
                distances = torch.cdist(
                    perturbed_embeds.view(-1, perturbed_embeds.shape[-1]),
                    all_embeddings
                )
                new_token_ids = distances.argmin(dim=-1)
                perturbed_ids = new_token_ids.view(perturbed_ids.shape)

        with torch.no_grad():
            adv_outputs = self.model(perturbed_ids, labels=labels)
            adv_loss = adv_outputs['loss'].item()
            adv_logits = adv_outputs['logits']

        num_changed = (perturbed_ids != input_ids).sum().item()
        total_tokens = input_ids.numel()

        success = adv_loss > orig_loss * 1.2

        return {
            'success': success,
            'original_ids': input_ids,
            'perturbed_ids': perturbed_ids,
            'num_changed_tokens': num_changed,
            'change_ratio': num_changed / total_tokens,
            'original_loss': orig_loss,
            'adversarial_loss': adv_loss,
            'loss_increase': (adv_loss - orig_loss) / orig_loss,
            'iterations': num_iterations,
            'epsilon': epsilon
        }

    def pgd_attack(self, input_ids: torch.Tensor,
                  labels: torch.Tensor,
                  epsilon: float = 0.3,
                  alpha: float = 0.01,
                  num_iterations: int = 20) -> Dict:
        """
        Projected Gradient Descent attack in embedding space.

        Args:
            input_ids: Input token IDs
            labels: Target labels
            epsilon: Maximum perturbation bound
            alpha: Step size
            num_iterations: Number of PGD iterations

        Returns:
            Dictionary with attack results
        """
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        embedding_layer = self.model.transformer.wte
        original_embeds = embedding_layer(input_ids).detach()

        with torch.no_grad():
            orig_outputs = self.model(input_ids, labels=labels)
            orig_loss = orig_outputs['loss'].item()

        perturbed_embeds = original_embeds.clone()
        perturbed_embeds.requires_grad = True

        for iteration in range(num_iterations):
            self.model.zero_grad()

            outputs = self.model(
                inputs_embeds=perturbed_embeds,
                labels=labels
            )

            loss = -outputs['loss']
            loss.backward()

            with torch.no_grad():
                gradient = perturbed_embeds.grad
                perturbed_embeds = perturbed_embeds - alpha * gradient.sign()

                delta = perturbed_embeds - original_embeds
                delta = torch.clamp(delta, -epsilon, epsilon)
                perturbed_embeds = original_embeds + delta

            perturbed_embeds = perturbed_embeds.detach()
            perturbed_embeds.requires_grad = True

        with torch.no_grad():
            all_embeddings = embedding_layer.weight
            distances = torch.cdist(
                perturbed_embeds.view(-1, perturbed_embeds.shape[-1]),
                all_embeddings
            )
            perturbed_ids = distances.argmin(dim=-1).view(input_ids.shape)

            adv_outputs = self.model(perturbed_ids, labels=labels)
            adv_loss = adv_outputs['loss'].item()

        perturbation_norm = torch.norm(perturbed_embeds - original_embeds, p=2).item()
        num_changed = (perturbed_ids != input_ids).sum().item()

        success = adv_loss > orig_loss * 1.2

        return {
            'success': success,
            'original_ids': input_ids,
            'perturbed_ids': perturbed_ids,
            'perturbation_norm': perturbation_norm,
            'num_changed_tokens': num_changed,
            'change_ratio': num_changed / input_ids.numel(),
            'original_loss': orig_loss,
            'adversarial_loss': adv_loss,
            'loss_increase': (adv_loss - orig_loss) / orig_loss,
            'epsilon': epsilon,
            'alpha': alpha,
            'iterations': num_iterations
        }


class AttackEvaluator:
    """
    Evaluates adversarial attacks against models.
    """

    def __init__(self, model, tokenizer, device='cuda'):
        """
        Initialize attack evaluator.

        Args:
            model: Target model
            tokenizer: Tokenizer
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.textfooler = TextFoolerAttack(model, tokenizer, device)
        self.gradient = GradientAttack(model, tokenizer, device)

    def evaluate_textfooler(self, test_samples: List[Dict],
                           max_samples: int = 50) -> Dict:
        """
        Evaluate TextFooler attack on test samples.

        Args:
            test_samples: List of test samples
            max_samples: Maximum samples to evaluate

        Returns:
            Evaluation results
        """
        results = {
            'total_samples': 0,
            'successful_attacks': 0,
            'avg_num_changes': 0,
            'avg_perturb_ratio': 0,
            'avg_loss_increase': 0,
            'attack_success_rate': 0
        }

        num_samples = min(len(test_samples), max_samples)

        for i, sample in enumerate(test_samples[:num_samples]):
            if 'text' in sample:
                text = sample['text']
            else:
                input_ids = sample['input_ids']
                if input_ids.dim() > 1:
                    input_ids = input_ids[0]
                text = self.tokenizer.decode(input_ids, skip_special_tokens=True)

            attack_result = self.textfooler.generate_adversarial(text)

            results['total_samples'] += 1
            if attack_result['success']:
                results['successful_attacks'] += 1

            results['avg_num_changes'] += attack_result['num_changes']
            results['avg_perturb_ratio'] += attack_result['perturb_ratio']

            if 'loss_increase' in attack_result:
                results['avg_loss_increase'] += attack_result['loss_increase']

        if results['total_samples'] > 0:
            results['avg_num_changes'] /= results['total_samples']
            results['avg_perturb_ratio'] /= results['total_samples']
            results['avg_loss_increase'] /= results['total_samples']
            results['attack_success_rate'] = results['successful_attacks'] / results['total_samples']

        return results

    def evaluate_gradient(self, test_samples: List[Dict],
                         attack_type: str = 'hotflip',
                         max_samples: int = 50) -> Dict:
        """
        Evaluate gradient-based attack on test samples.

        Args:
            test_samples: List of test samples
            attack_type: 'hotflip' or 'pgd'
            max_samples: Maximum samples to evaluate

        Returns:
            Evaluation results
        """
        results = {
            'total_samples': 0,
            'successful_attacks': 0,
            'avg_changed_tokens': 0,
            'avg_change_ratio': 0,
            'avg_loss_increase': 0,
            'avg_perturbation_norm': 0,
            'attack_success_rate': 0
        }

        num_samples = min(len(test_samples), max_samples)

        for sample in test_samples[:num_samples]:
            input_ids = sample['input_ids'].to(self.device)
            labels = sample.get('labels', input_ids.clone())

            if attack_type == 'hotflip':
                attack_result = self.gradient.hotflip_attack(input_ids, labels)
            else:
                attack_result = self.gradient.pgd_attack(input_ids, labels)

            results['total_samples'] += 1
            if attack_result['success']:
                results['successful_attacks'] += 1

            results['avg_changed_tokens'] += attack_result['num_changed_tokens']
            results['avg_change_ratio'] += attack_result['change_ratio']
            results['avg_loss_increase'] += attack_result['loss_increase']

            if 'perturbation_norm' in attack_result:
                results['avg_perturbation_norm'] += attack_result['perturbation_norm']

        if results['total_samples'] > 0:
            results['avg_changed_tokens'] /= results['total_samples']
            results['avg_change_ratio'] /= results['total_samples']
            results['avg_loss_increase'] /= results['total_samples']
            results['avg_perturbation_norm'] /= results['total_samples']
            results['attack_success_rate'] = results['successful_attacks'] / results['total_samples']

        return results
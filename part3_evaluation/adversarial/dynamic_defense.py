import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


class DynamicQuantizationDefense:
    """
    Dynamic quantization-based defense mechanisms
    Adaptively adjusts precision based on input characteristics
    """

    def __init__(self, model, tokenizer, bit_widths=[2, 4, 8, 16], device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.bit_widths = sorted(bit_widths)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

        self.perturbation_detector = PerturbationDetector(model, tokenizer)
        self.adaptive_quantizer = AdaptiveQuantizer(bit_widths)

    def defend_with_dynamic_quantization(self, input_ids: torch.Tensor,
                                        labels: Optional[torch.Tensor] = None) -> Dict:
        """
        Main defense mechanism using dynamic bit-width selection
        """
        perturbation_score = self.perturbation_detector.detect_perturbation(input_ids)

        selected_bits = self.adaptive_quantizer.select_bit_width(perturbation_score)

        self._apply_quantization(selected_bits)

        with torch.no_grad():
            if labels is not None:
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss.item()
            else:
                outputs = self.model(input_ids=input_ids)
                loss = None

            logits = outputs.logits

        uncertainty = self._compute_uncertainty(logits)

        if uncertainty > 0.7:
            refined_bits = max(2, selected_bits - 2)
            self._apply_quantization(refined_bits)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, labels=labels)
                if labels is not None:
                    loss = outputs.loss.item()
                logits = outputs.logits

        return {
            'defended_logits': logits,
            'selected_bits': selected_bits,
            'perturbation_score': perturbation_score,
            'uncertainty': uncertainty,
            'loss': loss,
            'defense_activated': perturbation_score > 0.3
        }

    def gradient_masking_defense(self, input_ids: torch.Tensor, labels: torch.Tensor,
                                mask_ratio: float = 0.5) -> Dict:
        """
        Gradient masking through selective quantization
        Makes gradients less informative for attackers
        """
        self.model.eval()

        random_mask = torch.rand(input_ids.shape[1]) < mask_ratio

        layer_bits = []
        for i, layer in enumerate(self.model.h):
            if i < len(random_mask) and random_mask[i]:
                bits = np.random.choice([2, 4])
            else:
                bits = 16
            layer_bits.append(bits)

        if hasattr(self.model, 'set_layer_precision'):
            self.model.set_layer_precision(layer_bits)

        embeddings = self.model.get_input_embeddings()(input_ids)

        noise_scale = 0.01
        noise = torch.randn_like(embeddings) * noise_scale
        noisy_embeddings = embeddings + noise

        outputs = self.model(inputs_embeds=noisy_embeddings, labels=labels)

        if embeddings.requires_grad and embeddings.grad is not None:
            grad_norm = embeddings.grad.norm()
        else:
            grad_norm = torch.tensor(0.0)

        return {
            'defended_loss': outputs.loss.item(),
            'gradient_norm': grad_norm.item(),
            'masked_layers': sum(random_mask).item(),
            'layer_configurations': layer_bits
        }

    def input_transformation_defense(self, input_ids: torch.Tensor,
                                   transformation_type: str = 'random') -> Dict:
        """
        Input transformation combined with quantization
        """
        transformed_ids = input_ids.clone()

        if transformation_type == 'random':
            num_transforms = np.random.randint(1, 4)
            transforms = np.random.choice(['shuffle', 'dropout', 'synonym'], num_transforms, replace=False)
        else:
            transforms = [transformation_type]

        transformation_log = []

        for transform in transforms:
            if transform == 'shuffle':
                if transformed_ids.shape[1] > 10:
                    shuffle_idx = torch.randperm(5) + 5
                    transformed_ids[:, 5:10] = transformed_ids[:, shuffle_idx]
                    transformation_log.append('shuffle')

            elif transform == 'dropout':
                dropout_mask = torch.rand(transformed_ids.shape) > 0.1
                transformed_ids = transformed_ids * dropout_mask.long().to(self.device)
                transformation_log.append('dropout')

            elif transform == 'synonym':
                for i in range(min(5, transformed_ids.shape[1])):
                    if np.random.random() < 0.3:
                        pos = np.random.randint(1, transformed_ids.shape[1] - 1)
                        vocab_size = self.tokenizer.vocab_size
                        new_token = torch.randint(1000, min(5000, vocab_size), (1,)).item()
                        transformed_ids[:, pos] = new_token
                transformation_log.append('synonym')

        perturbation_score = self.perturbation_detector.detect_perturbation(transformed_ids)
        defense_bits = 4 if perturbation_score > 0.5 else 8

        self._apply_quantization(defense_bits)

        with torch.no_grad():
            original_output = self.model(input_ids=input_ids)
            transformed_output = self.model(input_ids=transformed_ids)

        return {
            'transformed_ids': transformed_ids,
            'transformations_applied': transformation_log,
            'defense_bits': defense_bits,
            'original_logits': original_output.logits,
            'transformed_logits': transformed_output.logits,
            'logit_difference': (original_output.logits - transformed_output.logits).abs().mean().item()
        }

    def ensemble_defense(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                        num_models: int = 3) -> Dict:
        """
        Ensemble defense with different quantization levels
        """
        ensemble_outputs = []
        ensemble_configs = []

        bit_configurations = [
            [4, 4, 4],
            [8, 8, 8],
            [4, 8, 16],
            [16, 16, 16]
        ]

        selected_configs = np.random.choice(len(bit_configurations), num_models, replace=False)

        for config_idx in selected_configs:
            config = bit_configurations[config_idx]

            if hasattr(self.model, 'set_global_precision'):
                self.model.set_global_precision(config[0])

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, labels=labels)
                ensemble_outputs.append(outputs.logits)
                ensemble_configs.append(config)

        ensemble_logits = torch.stack(ensemble_outputs).mean(dim=0)

        variance = torch.stack(ensemble_outputs).var(dim=0).mean().item()

        predictions = torch.argmax(ensemble_logits, dim=-1)
        individual_predictions = [torch.argmax(logits, dim=-1) for logits in ensemble_outputs]
        agreement = sum([(pred == predictions).float().mean().item()
                        for pred in individual_predictions]) / len(individual_predictions)

        return {
            'ensemble_logits': ensemble_logits,
            'ensemble_variance': variance,
            'prediction_agreement': agreement,
            'configurations_used': ensemble_configs,
            'num_models': num_models,
            'high_confidence': agreement > 0.8
        }

    def adversarial_training_defense(self, clean_input: torch.Tensor, clean_labels: torch.Tensor,
                                    attack_strength: float = 0.1) -> Dict:
        """
        Combine adversarial training with quantization
        """
        self.model.eval()

        clean_embeddings = self.model.get_input_embeddings()(clean_input).detach()
        perturbed_embeddings = clean_embeddings.clone()
        perturbed_embeddings.requires_grad = True

        for _ in range(5):
            outputs = self.model(inputs_embeds=perturbed_embeddings, labels=clean_labels)
            loss = -outputs.loss
            loss.backward()

            with torch.no_grad():
                perturbation = attack_strength * perturbed_embeddings.grad.sign()
                perturbed_embeddings = perturbed_embeddings + perturbation
                perturbed_embeddings = clean_embeddings + torch.clamp(
                    perturbed_embeddings - clean_embeddings, -attack_strength, attack_strength
                )
                perturbed_embeddings = perturbed_embeddings.detach()
                perturbed_embeddings.requires_grad = True

        perturbation_norm = (perturbed_embeddings - clean_embeddings).norm().item()

        defense_bits = 4 if perturbation_norm > 1.0 else 8
        self._apply_quantization(defense_bits)

        with torch.no_grad():
            clean_outputs = self.model(inputs_embeds=clean_embeddings, labels=clean_labels)
            robust_outputs = self.model(inputs_embeds=perturbed_embeddings, labels=clean_labels)

        return {
            'clean_loss': clean_outputs.loss.item(),
            'robust_loss': robust_outputs.loss.item(),
            'perturbation_norm': perturbation_norm,
            'defense_bits': defense_bits,
            'robustness_gap': (robust_outputs.loss - clean_outputs.loss).item()
        }

    def detect_and_reject(self, input_ids: torch.Tensor, threshold: float = 0.7) -> Dict:
        """
        Detect adversarial inputs and reject if confidence is low
        """
        perturbation_score = self.perturbation_detector.detect_perturbation(input_ids)

        if perturbation_score > threshold:
            return {
                'decision': 'REJECTED',
                'perturbation_score': perturbation_score,
                'threshold': threshold,
                'confidence': 0.0,
                'output': None
            }

        defense_bits = self.adaptive_quantizer.select_bit_width(perturbation_score)
        self._apply_quantization(defense_bits)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            probs = F.softmax(outputs.logits, dim=-1)
            confidence = probs.max(dim=-1).values.mean().item()

        if confidence < 0.3:
            return {
                'decision': 'LOW_CONFIDENCE',
                'perturbation_score': perturbation_score,
                'threshold': threshold,
                'confidence': confidence,
                'output': outputs.logits
            }

        return {
            'decision': 'ACCEPTED',
            'perturbation_score': perturbation_score,
            'threshold': threshold,
            'confidence': confidence,
            'output': outputs.logits,
            'defense_bits': defense_bits
        }

    def _apply_quantization(self, bit_width: int):
        """Apply quantization to model"""
        if hasattr(self.model, 'set_global_precision'):
            self.model.set_global_precision(bit_width)
        elif hasattr(self.model, 'set_precision'):
            self.model.set_precision(bit_width)

    def _compute_uncertainty(self, logits: torch.Tensor) -> float:
        """Compute prediction uncertainty"""
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        normalized_entropy = entropy / torch.log(torch.tensor(logits.shape[-1], dtype=torch.float32))
        return normalized_entropy.item()


class PerturbationDetector:
    """Detect adversarial perturbations in inputs"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def detect_perturbation(self, input_ids: torch.Tensor) -> float:
        """
        Detect perturbation level in input
        Returns score between 0 (clean) and 1 (highly perturbed)
        """
        scores = []

        freq_score = self._frequency_analysis(input_ids)
        scores.append(freq_score)

        if hasattr(self.model, 'get_input_embeddings'):
            embedding_score = self._embedding_analysis(input_ids)
            scores.append(embedding_score)

        entropy_score = self._entropy_analysis(input_ids)
        scores.append(entropy_score)

        return np.mean(scores)

    def _frequency_analysis(self, input_ids: torch.Tensor) -> float:
        """Analyze token frequency distribution"""
        vocab_size = self.tokenizer.vocab_size
        token_counts = torch.bincount(input_ids.flatten(), minlength=vocab_size)
        token_probs = token_counts.float() / token_counts.sum()

        top_k = 100
        top_probs = torch.topk(token_probs, min(top_k, vocab_size)).values
        concentration = top_probs.sum().item()

        unusual_tokens = (input_ids > vocab_size * 0.9).float().mean().item()

        return min(1.0, unusual_tokens + (1 - concentration))

    def _embedding_analysis(self, input_ids: torch.Tensor) -> float:
        """Analyze embedding space characteristics"""
        embeddings = self.model.get_input_embeddings()(input_ids)

        cosine_sims = F.cosine_similarity(embeddings[:, :-1], embeddings[:, 1:], dim=-1)
        avg_similarity = cosine_sims.mean().item()

        embedding_norms = embeddings.norm(dim=-1)
        norm_variance = embedding_norms.var().item()

        score = max(0, min(1, (1 - avg_similarity) + norm_variance * 0.1))
        return score

    def _entropy_analysis(self, input_ids: torch.Tensor) -> float:
        """Analyze entropy of token distribution"""
        unique_tokens = len(torch.unique(input_ids))
        total_tokens = input_ids.numel()

        diversity_ratio = unique_tokens / total_tokens

        token_counts = torch.bincount(input_ids.flatten())
        token_probs = token_counts.float() / token_counts.sum()
        token_probs = token_probs[token_probs > 0]

        entropy = -(token_probs * torch.log(token_probs)).sum().item()
        max_entropy = torch.log(torch.tensor(unique_tokens, dtype=torch.float32)).item()

        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        score = 1 - normalized_entropy * diversity_ratio
        return max(0, min(1, score))


class AdaptiveQuantizer:
    """Adaptively select quantization bit-width based on input characteristics"""

    def __init__(self, bit_widths: List[int]):
        self.bit_widths = sorted(bit_widths)

    def select_bit_width(self, perturbation_score: float) -> int:
        """
        Select appropriate bit-width based on perturbation score
        Higher perturbation -> Lower bit-width (more aggressive quantization)
        """
        if perturbation_score < 0.2:
            return self.bit_widths[-1]
        elif perturbation_score < 0.4:
            return self.bit_widths[-2] if len(self.bit_widths) > 1 else self.bit_widths[-1]
        elif perturbation_score < 0.6:
            return self.bit_widths[len(self.bit_widths)//2]
        elif perturbation_score < 0.8:
            return self.bit_widths[1] if len(self.bit_widths) > 1 else self.bit_widths[0]
        else:
            return self.bit_widths[0]

    def compute_quantization_schedule(self, input_length: int, perturbation_scores: List[float]) -> List[int]:
        """
        Compute per-position quantization schedule
        """
        schedule = []

        for score in perturbation_scores:
            bits = self.select_bit_width(score)
            schedule.append(bits)

        return schedule
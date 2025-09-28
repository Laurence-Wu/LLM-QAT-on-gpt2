"""
Simplified Random Switching Defense for Adversarial Robustness
"""

import torch
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from part1_switchable_precision.models_sp import SPLMHeadModel
from transformers import GPT2Tokenizer


def load_sp_model_with_bit_config(checkpoint_path: str, device: str = 'cuda'):
    """
    Load SP model and extract bit width configuration from checkpoint.

    Args:
        checkpoint_path: Path to the saved model checkpoint
        device: Device to load model on

    Returns:
        model: Loaded SPLMHeadModel
        tokenizer: GPT2 tokenizer
        bit_widths: List of available bit widths
        saved_precision: Precision the model was saved at
    """
    print(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        model_state = checkpoint.get('model_state_dict', checkpoint)
        model_config = checkpoint.get('config', {})
        bit_widths = checkpoint.get('bit_widths', [4, 8, 16])
        saved_precision = checkpoint.get('current_precision', 8)
    else:
        print("Warning: Checkpoint is not a dictionary, using default bit widths")
        model_state = checkpoint
        bit_widths = [4, 8, 16]
        saved_precision = 8

    from transformers import GPT2Config
    from part1_switchable_precision.config_sp import ModelConfig

    model_cfg = ModelConfig()
    gpt2_config = GPT2Config(
        vocab_size=model_cfg.vocab_size,
        n_positions=model_cfg.n_positions,
        n_embd=model_cfg.n_embd,
        n_layer=model_cfg.n_layer,
        n_head=model_cfg.n_head
    )

    gpt2_config.bit_widths = bit_widths
    gpt2_config.lora_rank_per_bit = model_cfg.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_cfg.lora_alpha_per_bit
    gpt2_config.quantizer_per_bit = model_cfg.quantizer_per_bit

    model = SPLMHeadModel(gpt2_config)

    if isinstance(model_state, dict):
        model.load_state_dict(model_state, strict=False)

    model = model.to(device)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded successfully")
    print(f"  Available bit widths: {bit_widths}")
    print(f"  Model was saved at {saved_precision}-bit precision")

    return model, tokenizer, bit_widths, saved_precision


class SimplifiedRandomSwitching:
    """
    Implements random precision switching for adversarial defense.

    This creates a "moving target" that makes it harder for adversarial
    examples to consistently fool the model.
    """

    def __init__(self, model, bit_widths: List[int],
                 switch_probability: float = 0.3,
                 device: str = 'cuda'):
        """
        Initialize random switching defense.

        Args:
            model: SPLMHeadModel instance
            bit_widths: List of available bit widths
            switch_probability: Probability of switching precision (0-1)
            device: Device for computation
        """
        self.model = model
        self.bit_widths = sorted(bit_widths)
        self.switch_prob = switch_probability
        self.device = device

        self.current_precision = max(self.bit_widths)
        self.model.set_precision(self.current_precision)

        self.precision_history = []
        self.switch_count = 0
        self.total_forwards = 0

    def select_next_precision(self) -> int:
        """
        Randomly decide whether to switch precision and select new one.

        Returns:
            Selected precision for next forward pass
        """
        if random.random() < self.switch_prob:
            new_precision = random.choice(self.bit_widths)
            if new_precision != self.current_precision:
                self.switch_count += 1
            self.current_precision = new_precision

        self.precision_history.append(self.current_precision)
        return self.current_precision

    def forward_with_switching(self, input_ids: torch.Tensor,
                              attention_mask: Optional[torch.Tensor] = None,
                              labels: Optional[torch.Tensor] = None) -> Tuple[Dict, int]:
        """
        Perform forward pass with random precision switching.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation

        Returns:
            outputs: Model outputs dictionary
            precision: Precision used for this forward pass
        """
        precision = self.select_next_precision()
        self.model.set_precision(precision)

        self.total_forwards += 1

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )

        return outputs, precision

    def forward_without_switching(self, input_ids: torch.Tensor,
                                 attention_mask: Optional[torch.Tensor] = None,
                                 labels: Optional[torch.Tensor] = None,
                                 precision: Optional[int] = None) -> Dict:
        """
        Perform forward pass at fixed precision (for baseline evaluation).

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            labels: Optional labels
            precision: Specific precision to use

        Returns:
            Model outputs dictionary
        """
        if precision is not None:
            self.model.set_precision(precision)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )

        return outputs

    def get_statistics(self) -> Dict:
        """
        Get statistics about precision switching behavior.

        Returns:
            Dictionary with switching statistics
        """
        if not self.precision_history:
            return {
                'total_forwards': 0,
                'switch_count': 0,
                'switch_rate': 0.0,
                'precision_distribution': {}
            }

        precision_counts = Counter(self.precision_history)
        precision_dist = {
            bits: count / len(self.precision_history)
            for bits, count in precision_counts.items()
        }

        return {
            'total_forwards': self.total_forwards,
            'switch_count': self.switch_count,
            'switch_rate': self.switch_count / max(self.total_forwards - 1, 1),
            'precision_distribution': precision_dist,
            'precision_counts': dict(precision_counts)
        }

    def reset_statistics(self):
        """Reset all tracking statistics."""
        self.precision_history = []
        self.switch_count = 0
        self.total_forwards = 0


class DefenseEvaluator:
    """
    Evaluates the effectiveness of random switching defense.
    """

    def __init__(self, model, tokenizer, bit_widths: List[int], device: str = 'cuda'):
        """
        Initialize defense evaluator.

        Args:
            model: SPLMHeadModel instance
            tokenizer: Tokenizer instance
            bit_widths: Available bit widths
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.bit_widths = bit_widths
        self.device = device

    def evaluate_fixed_precision(self, test_samples: List[Dict],
                                precision: int) -> Dict:
        """
        Evaluate model at fixed precision (baseline).

        Args:
            test_samples: List of test samples
            precision: Fixed precision to use

        Returns:
            Evaluation results dictionary
        """
        defender = SimplifiedRandomSwitching(
            self.model, [precision], switch_probability=0.0, device=self.device
        )

        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for sample in test_samples:
            input_ids = sample['input_ids'].to(self.device)
            attention_mask = sample.get('attention_mask')
            labels = sample.get('labels')

            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            if labels is not None:
                labels = labels.to(self.device)

            outputs = defender.forward_without_switching(
                input_ids, attention_mask, labels, precision
            )

            if labels is not None and outputs.get('loss') is not None:
                total_loss += outputs['loss'].item()

            predictions = outputs['logits'].argmax(dim=-1)
            if labels is not None:
                mask = labels != -100
                correct_predictions += (predictions[mask] == labels[mask]).sum().item()
                total_predictions += mask.sum().item()

        accuracy = correct_predictions / max(total_predictions, 1)
        avg_loss = total_loss / max(len(test_samples), 1)

        return {
            'precision': precision,
            'accuracy': accuracy,
            'avg_loss': avg_loss,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions
        }

    def evaluate_random_switching(self, test_samples: List[Dict],
                                 switch_probability: float) -> Dict:
        """
        Evaluate model with random switching defense.

        Args:
            test_samples: List of test samples
            switch_probability: Probability of switching

        Returns:
            Evaluation results with switching statistics
        """
        defender = SimplifiedRandomSwitching(
            self.model, self.bit_widths, switch_probability, self.device
        )

        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        precision_at_prediction = []

        for sample in test_samples:
            input_ids = sample['input_ids'].to(self.device)
            attention_mask = sample.get('attention_mask')
            labels = sample.get('labels')

            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            if labels is not None:
                labels = labels.to(self.device)

            outputs, precision = defender.forward_with_switching(
                input_ids, attention_mask, labels
            )
            precision_at_prediction.append(precision)

            if labels is not None and outputs.get('loss') is not None:
                total_loss += outputs['loss'].item()

            predictions = outputs['logits'].argmax(dim=-1)
            if labels is not None:
                mask = labels != -100
                correct_predictions += (predictions[mask] == labels[mask]).sum().item()
                total_predictions += mask.sum().item()

        accuracy = correct_predictions / max(total_predictions, 1)
        avg_loss = total_loss / max(len(test_samples), 1)

        stats = defender.get_statistics()

        return {
            'switch_probability': switch_probability,
            'accuracy': accuracy,
            'avg_loss': avg_loss,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'switching_stats': stats,
            'precision_at_prediction': precision_at_prediction
        }


if __name__ == "__main__":
    print("Testing SimplifiedRandomSwitching module...")

    checkpoint_path = "path/to/model.pth"

    try:
        model, tokenizer, bit_widths, saved_precision = load_sp_model_with_bit_config(
            checkpoint_path
        )

        defender = SimplifiedRandomSwitching(model, bit_widths)

        test_input = tokenizer("This is a test sentence.", return_tensors='pt')
        outputs, precision = defender.forward_with_switching(test_input['input_ids'])

        print(f"Test successful! Used {precision}-bit precision")
        print(f"Output shape: {outputs['logits'].shape}")

    except FileNotFoundError:
        print("Please provide a valid checkpoint path to test")
"""
Self-Distillation Module for Switchable Precision Training
Implements knowledge distillation where full-precision acts as teacher for low-precision students.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import gc


@dataclass
class DistillationConfig:
    """Configuration for self-distillation."""
    use_distillation: bool = True
    alpha_output: float = 1.0  # Weight for KL divergence loss
    alpha_feature: float = 1e-7  # Weight for feature matching loss
    temperature: float = 3.0  # Temperature for KL divergence
    teacher_update_freq: int = 10  # How often to update teacher cache
    feature_layers: Optional[List[int]] = None  # Which layers to match (None = all)
    cache_size: int = 32  # Size of teacher cache
    warmup_steps: int = 100  # Steps before starting distillation


class SelfDistillationTrainer:
    """Manages self-distillation for switchable precision training."""

    def __init__(self, model, config: DistillationConfig, device='cuda'):
        self.model = model
        self.config = config
        self.device = device

        # Get full precision bit width from model
        if hasattr(model, 'transformer'):
            self.full_precision_bits = max(model.transformer.bit_widths)
        else:
            self.full_precision_bits = 16  # Default fallback

        # Teacher cache for storing full-precision outputs
        self.teacher_cache = {}
        self.cache_keys = []
        self.step_count = 0

        # Statistics tracking (no printing here)
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'teacher_updates': 0
        }

    def compute_teacher_outputs(self, input_ids, attention_mask=None):
        """
        Compute and cache teacher (full-precision) outputs.

        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Optional attention mask

        Returns:
            Dictionary with teacher outputs
        """
        # Save current precision
        original_bits = self.model.get_current_precision()

        # Switch to full precision
        self.model.set_precision(self.full_precision_bits)

        # Compute teacher outputs without gradients
        with torch.no_grad():
            teacher_outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            # Cache outputs (detached from computation graph)
            cache_entry = {
                'logits': teacher_outputs['logits'].detach(),
                'hidden_states': [h.detach() for h in teacher_outputs['hidden_states']] if teacher_outputs.get('hidden_states') else []
            }

            # Add to cache with batch hash as key
            batch_key = self._get_batch_key(input_ids)
            self._add_to_cache(batch_key, cache_entry)

            self.stats['teacher_updates'] += 1

        # Restore original precision
        self.model.set_precision(original_bits)

        return cache_entry

    def compute_distillation_loss(self, student_outputs, labels, input_ids, attention_mask=None):
        """
        Compute distillation loss based on current precision.

        Args:
            student_outputs: Current model outputs
            labels: Ground truth labels
            input_ids: Input IDs (for cache lookup)
            attention_mask: Optional attention mask

        Returns:
            Loss tensor and loss components dictionary
        """
        current_bits = self.model.get_current_precision()

        # Full precision: standard cross-entropy loss
        if current_bits == self.full_precision_bits:
            if 'loss' in student_outputs and student_outputs['loss'] is not None:
                loss = student_outputs['loss']
            else:
                # Compute cross-entropy loss
                logits = student_outputs['logits']
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            return loss, {'cross_entropy': loss.item(), 'precision': current_bits}

        # Low precision: distillation losses
        teacher = self._get_from_cache(input_ids)
        if teacher is None:
            # Compute teacher if not in cache
            self.stats['cache_misses'] += 1
            teacher = self.compute_teacher_outputs(input_ids, attention_mask)
        else:
            self.stats['cache_hits'] += 1

        # KL divergence loss for output distribution
        T = self.config.temperature
        teacher_logits = teacher['logits'][..., :-1, :].contiguous()
        student_logits = student_outputs['logits'][..., :-1, :].contiguous()

        # Apply temperature scaling and compute log probabilities
        teacher_log_probs = F.log_softmax(teacher_logits / T, dim=-1)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)

        # KL divergence (note: teacher probs are detached)
        kl_loss = F.kl_div(
            student_log_probs.view(-1, student_log_probs.size(-1)),
            teacher_log_probs.view(-1, teacher_log_probs.size(-1)),
            reduction='batchmean',
            log_target=True
        ) * (T * T)

        # Feature matching loss
        feature_loss = torch.tensor(0.0, device=self.device)
        if student_outputs.get('hidden_states') and teacher.get('hidden_states') and len(teacher['hidden_states']) > 0:
            num_layers = min(len(teacher['hidden_states']), len(student_outputs['hidden_states']))

            # Match features from specified layers or all layers
            layers_to_match = self.config.feature_layers or list(range(num_layers))
            layers_to_match = [l for l in layers_to_match if l < num_layers]

            if layers_to_match:
                for layer_idx in layers_to_match:
                    teacher_features = teacher['hidden_states'][layer_idx]
                    student_features = student_outputs['hidden_states'][layer_idx]

                    # L2 loss for feature matching
                    feature_loss = feature_loss + F.mse_loss(
                        student_features,
                        teacher_features,
                        reduction='mean'
                    )

                # Average over matched layers
                feature_loss = feature_loss / len(layers_to_match)

        # Combine losses
        total_loss = (
            self.config.alpha_output * kl_loss +
            self.config.alpha_feature * feature_loss
        )

        loss_components = {
            'total': total_loss.item(),
            'kl': kl_loss.item(),
            'feature': feature_loss.item() if isinstance(feature_loss, torch.Tensor) else feature_loss,
            'precision': current_bits
        }

        return total_loss, loss_components

    def _get_batch_key(self, input_ids):
        """Generate hash key for batch."""
        # Use first few tokens as key to avoid full tensor hashing
        key_tensor = input_ids[:, :min(16, input_ids.shape[1])].cpu().numpy()
        return hash(key_tensor.tobytes())

    def _add_to_cache(self, key, entry):
        """Add entry to cache with LRU eviction."""
        if len(self.cache_keys) >= self.config.cache_size:
            # Remove oldest entry
            oldest_key = self.cache_keys.pop(0)
            if oldest_key in self.teacher_cache:
                del self.teacher_cache[oldest_key]

            # Clear GPU memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.teacher_cache[key] = entry
        self.cache_keys.append(key)

    def _get_from_cache(self, input_ids):
        """Retrieve teacher outputs from cache."""
        key = self._get_batch_key(input_ids)
        return self.teacher_cache.get(key)

    def should_use_distillation(self, step):
        """Check if distillation should be used at current step."""
        self.step_count = step
        return step >= self.config.warmup_steps and self.config.use_distillation

    def clear_cache(self):
        """Clear teacher cache to free memory."""
        self.teacher_cache.clear()
        self.cache_keys.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_stats(self):
        """Return statistics for monitoring."""
        return self.stats.copy()
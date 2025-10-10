import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Any
import gc


class DistillationManagerQA:
    """
    Distillation Manager for Question Answering Task

    Manages knowledge distillation from 32-bit teacher to low-bit student models.
    Adapted for QA: distills start_logits and end_logits instead of vocabulary logits.
    """

    def __init__(self, model, full_precision_bits, config):
        """
        Initialize distillation manager

        Args:
            model: SPQuestionAnsweringModel
            full_precision_bits: Bit-width for teacher (usually 32)
            config: Training configuration
        """
        self.model = model
        self.full_precision_bits = full_precision_bits
        self.config = config

        # Get device from model (should always be CUDA)
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device('cuda')

        # Enhanced teacher output cache with per-sequence storage
        self.teacher_cache = {}
        self.cache_keys = []
        self.iteration_count = 0

        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0

        # Track when teacher was last updated
        self.last_teacher_update = -1
        self.pending_teacher_update = False

    def update_teacher_qa(self, input_ids, teacher_outputs):
        """
        Update teacher cache with QA outputs

        Caches:
        - start_logits: [batch_size, seq_length]
        - end_logits: [batch_size, seq_length]
        - hidden_states: List of hidden states from all layers (optional)

        Args:
            input_ids: Input token IDs
            teacher_outputs: Dict with 'start_logits', 'end_logits', 'hidden_states'
        """
        current_bits = self.model.get_current_precision()
        if current_bits != self.full_precision_bits:
            raise RuntimeError(
                f"Teacher update called at {current_bits}-bit precision, "
                f"expected {self.full_precision_bits}-bit"
            )

        with torch.no_grad():
            # Build cache entry with QA-specific outputs
            cache_entry = {
                'start_logits': teacher_outputs['start_logits'].detach().clone(),
                'end_logits': teacher_outputs['end_logits'].detach().clone(),
                'hidden_states': []
            }

            # Cache hidden states for feature distillation (optional)
            if teacher_outputs.get('hidden_states'):
                cache_entry['hidden_states'] = [
                    h.detach().clone() for h in teacher_outputs['hidden_states']
                ]

            batch_key = self._get_batch_key(input_ids)
            self._add_to_cache(batch_key, cache_entry)

            self.last_teacher_update = self.iteration_count
            self.pending_teacher_update = False

        return cache_entry

    def compute_distillation_loss_qa(self, student_outputs, input_ids, config, accumulative=False):
        """
        Compute distillation loss for QA task

        Components:
        1. KL divergence on start position distribution
        2. KL divergence on end position distribution
        3. Optional: MSE on hidden states (feature distillation)

        Args:
            student_outputs: Dict with 'start_logits', 'end_logits', 'hidden_states'
            input_ids: Input token IDs
            config: Training configuration
            accumulative: Whether to average feature loss over all layers (default: random layer)

        Returns:
            total_loss: Combined KL loss + feature loss
        """
        teacher = self._get_from_cache(input_ids)

        if teacher is None:
            raise ValueError("Teacher outputs not cached for student distillation!")

        T = config.distill_temperature

        # ===== KL DIVERGENCE ON START LOGITS =====
        teacher_start_logits = teacher['start_logits']  # [batch, seq_len]
        student_start_logits = student_outputs['start_logits']

        teacher_start_log_probs = F.log_softmax(teacher_start_logits / T, dim=-1)
        student_start_log_probs = F.log_softmax(student_start_logits / T, dim=-1)

        kl_start = F.kl_div(
            student_start_log_probs,
            teacher_start_log_probs,
            reduction='batchmean',
            log_target=True
        ) * (T * T)

        # ===== KL DIVERGENCE ON END LOGITS =====
        teacher_end_logits = teacher['end_logits']  # [batch, seq_len]
        student_end_logits = student_outputs['end_logits']

        teacher_end_log_probs = F.log_softmax(teacher_end_logits / T, dim=-1)
        student_end_log_probs = F.log_softmax(student_end_logits / T, dim=-1)

        kl_end = F.kl_div(
            student_end_log_probs,
            teacher_end_log_probs,
            reduction='batchmean',
            log_target=True
        ) * (T * T)

        # Average KL loss over start and end
        kl_loss = (kl_start + kl_end) / 2.0

        # ===== FEATURE DISTILLATION (hidden states) =====
        feature_loss = torch.tensor(0.0, device=self.device)

        if student_outputs.get('hidden_states') and teacher.get('hidden_states'):
            num_layers = min(len(teacher['hidden_states']), len(student_outputs['hidden_states']))

            layers_to_match = getattr(config, 'feature_layers', None) or list(range(num_layers))
            layers_to_match = [l for l in layers_to_match if l < num_layers]

            if layers_to_match:
                if accumulative:
                    # Average feature loss over all specified layers
                    for layer_idx in layers_to_match:
                        teacher_features = teacher['hidden_states'][layer_idx]
                        student_features = student_outputs['hidden_states'][layer_idx]

                        feature_loss = feature_loss + F.mse_loss(
                            student_features,
                            teacher_features,
                            reduction='mean'
                        )
                    feature_loss = feature_loss / len(layers_to_match)
                else:
                    # Randomly select one layer to compute feature loss
                    layer_idx = random.choice(layers_to_match)
                    teacher_features = teacher['hidden_states'][layer_idx]
                    student_features = student_outputs['hidden_states'][layer_idx]

                    feature_loss = F.mse_loss(
                        student_features,
                        teacher_features,
                        reduction='mean'
                    )

        # Combine losses
        total_loss = (
            config.distill_alpha_kl * kl_loss +
            config.distill_alpha_feature * feature_loss
        )

        return total_loss

    def _get_batch_key(self, input_ids):
        """
        Generate unique key for batch caching

        Uses shape and sample of tokens to create hash key
        """
        shape_key = tuple(input_ids.shape)
        sample_tokens = input_ids.flatten()[:min(32, input_ids.numel())].cpu().numpy()
        return hash((shape_key, sample_tokens.tobytes()))

    def _add_to_cache(self, key, entry):
        """
        Add entry to cache with LRU eviction

        Args:
            key: Cache key
            entry: Cache entry (dict with outputs)
        """
        cache_size_limit = getattr(self.config, 'cache_size', 32)

        if len(self.cache_keys) >= cache_size_limit:
            # Remove oldest entry (LRU)
            oldest_key = self.cache_keys.pop(0)
            if oldest_key in self.teacher_cache:
                del self.teacher_cache[oldest_key]

            # Force garbage collection to free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.teacher_cache[key] = entry
        self.cache_keys.append(key)

    def _get_from_cache(self, input_ids):
        """
        Retrieve cached teacher outputs

        Args:
            input_ids: Input token IDs

        Returns:
            Cached entry or None if not found
        """
        key = self._get_batch_key(input_ids)
        result = self.teacher_cache.get(key)

        if result is not None:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        return result

    def step(self):
        """Increment iteration counter"""
        self.iteration_count += 1

    def get_cache_stats(self):
        """
        Get cache statistics

        Returns:
            Dict with cache size, hits, misses, total requests
        """
        total_requests = self.cache_hits + self.cache_misses

        return {
            'cache_size': len(self.teacher_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': total_requests,
            'hit_rate': self.cache_hits / max(total_requests, 1)
        }

"""
Distillation Manager for Switchable Precision Training
Manages teacher-student distillation where full-precision teaches low-precision models.
Following the paper "Switchable Precision Neural Networks".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Any
import gc


class DistillationManager:
    """
    Manages self-distillation for switchable precision training.
    Teacher (full-precision) outputs guide student (low-precision) learning.
    """

    def __init__(self, model, full_precision_bits, config):
        """
        Initialize distillation manager.

        Args:
            model: The switchable precision model
            full_precision_bits: The bit-width considered as full precision (teacher)
            config: Training configuration with distillation parameters
        """
        self.model = model
        self.full_precision_bits = full_precision_bits
        self.config = config

        # Get device from model
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Teacher output cache
        self.teacher_cache = {}
        self.cache_keys = []
        self.iteration_count = 0

        # Track when teacher was last updated
        self.last_teacher_update = -1
        self.pending_teacher_update = False

    def should_update_teacher(self, current_bits, iteration):
        """
        Determine if teacher cache should be updated.

        Args:
            current_bits: Current model precision
            iteration: Current training iteration

        Returns:
            bool: Whether to update teacher
        """
        # Update teacher when:
        # 1. At full precision AND interval reached
        # 2. Pending update flag is set (switching from full precision)
        at_full_precision = (current_bits == self.full_precision_bits)
        interval_reached = (iteration - self.last_teacher_update) >= self.config.teacher_update_interval

        return at_full_precision and (interval_reached or self.pending_teacher_update)

    def update_teacher(self, input_ids, attention_mask=None):
        """
        Update teacher cache with current batch outputs.
        Should be called when model is at full precision.

        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Optional attention mask

        Returns:
            Dictionary with cached teacher outputs
        """
        # Ensure model is at full precision
        current_bits = self.model.get_current_precision()
        if current_bits != self.full_precision_bits:
            raise RuntimeError(f"Teacher update called at {current_bits}-bit precision, expected {self.full_precision_bits}-bit")

        # Compute teacher outputs without gradients
        with torch.no_grad():
            teacher_outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            # Immediately detach to prevent memory leaks
            cache_entry = {
                'logits': teacher_outputs['logits'].detach().clone(),
                'hidden_states': []
            }

            # Store hidden states if available
            if teacher_outputs.get('hidden_states'):
                cache_entry['hidden_states'] = [h.detach().clone() for h in teacher_outputs['hidden_states']]

            # Optionally move to CPU to save GPU memory
            if self.config.get('move_cache_to_cpu', False):
                cache_entry['logits'] = cache_entry['logits'].cpu()
                cache_entry['hidden_states'] = [h.cpu() for h in cache_entry['hidden_states']]

            # Store in cache with input hash as key
            batch_key = self._get_batch_key(input_ids)
            self._add_to_cache(batch_key, cache_entry)

            self.last_teacher_update = self.iteration_count
            self.pending_teacher_update = False

        return cache_entry

    def compute_distillation_loss(self, student_outputs, input_ids):
        """
        Compute distillation loss for student (low-precision) model.
        Following paper: L_q = α₁·L_out + α₂·L_f

        Args:
            student_outputs: Model outputs with 'logits' and 'hidden_states'
            input_ids: Input IDs for cache lookup

        Returns:
            Loss tensor (scalar)
        """
        # Get cached teacher outputs
        teacher = self._get_from_cache(input_ids)
        if teacher is None:
            # No teacher outputs available - fall back to standard loss
            # This should rarely happen if teacher updates are managed properly
            print(f"Warning: No teacher outputs in cache for batch")
            logits = student_outputs['logits']
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            return F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        # Move teacher tensors to GPU if they were on CPU
        if teacher['logits'].device != self.device:
            teacher['logits'] = teacher['logits'].to(self.device)
            if teacher['hidden_states']:
                teacher['hidden_states'] = [h.to(self.device) for h in teacher['hidden_states']]

        # 1. Output distillation (KL divergence)
        T = self.config.distill_temperature
        teacher_logits = teacher['logits'][..., :-1, :].contiguous()
        student_logits = student_outputs['logits'][..., :-1, :].contiguous()

        # Temperature-scaled softmax
        teacher_log_probs = F.log_softmax(teacher_logits / T, dim=-1)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)

        # KL divergence with temperature scaling
        kl_loss = F.kl_div(
            student_log_probs.view(-1, student_log_probs.size(-1)),
            teacher_log_probs.view(-1, teacher_log_probs.size(-1)),
            reduction='batchmean',
            log_target=True
        ) * (T * T)

        # 2. Feature matching (MSE on hidden states)
        feature_loss = torch.tensor(0.0, device=self.device)
        if student_outputs.get('hidden_states') and teacher.get('hidden_states'):
            num_layers = min(len(teacher['hidden_states']), len(student_outputs['hidden_states']))

            # Match all layers or specified layers
            layers_to_match = self.config.get('feature_layers') or list(range(num_layers))
            layers_to_match = [l for l in layers_to_match if l < num_layers]

            if layers_to_match:
                for layer_idx in layers_to_match:
                    teacher_features = teacher['hidden_states'][layer_idx]
                    student_features = student_outputs['hidden_states'][layer_idx]

                    # MSE loss for feature matching
                    feature_loss = feature_loss + F.mse_loss(
                        student_features,
                        teacher_features,
                        reduction='mean'
                    )

                # Average across layers
                feature_loss = feature_loss / len(layers_to_match)

        # Combine losses with weights from config
        total_loss = (
            self.config.distill_alpha_kl * kl_loss +
            self.config.distill_alpha_feature * feature_loss
        )

        return total_loss

    def _get_batch_key(self, input_ids):
        """Generate hash key for batch."""
        # Use shape and first few tokens to create unique key
        shape_key = tuple(input_ids.shape)
        sample_tokens = input_ids.flatten()[:min(32, input_ids.numel())].cpu().numpy()
        return hash((shape_key, sample_tokens.tobytes()))

    def _add_to_cache(self, key, entry):
        """Add entry to cache with LRU eviction."""
        cache_size_limit = self.config.get('cache_size', 32)

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
        """Retrieve teacher outputs from cache."""
        key = self._get_batch_key(input_ids)
        return self.teacher_cache.get(key)

    def mark_switch_from_teacher(self):
        """Mark that we're switching away from teacher precision."""
        self.pending_teacher_update = True

    def clear_cache(self):
        """Clear all cached teacher outputs to free memory."""
        self.teacher_cache.clear()
        self.cache_keys.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def step(self):
        """Increment iteration counter."""
        self.iteration_count += 1
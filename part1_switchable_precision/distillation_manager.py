import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Any
import gc


class DistillationManager:
    def __init__(self, model, full_precision_bits, config):
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

    def update_teacher(self, input_ids, attention_mask=None):
        current_bits = self.model.get_current_precision()
        if current_bits != self.full_precision_bits:
            raise RuntimeError(f"Teacher update called at {current_bits}-bit precision, expected {self.full_precision_bits}-bit")

        with torch.no_grad():
            teacher_outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

 
            cache_entry = {
                'logits': teacher_outputs['logits'].detach().clone(),
                'hidden_states': []
            }

            if teacher_outputs.get('hidden_states'):
                cache_entry['hidden_states'] = [h.detach().clone() for h in teacher_outputs['hidden_states']]

            batch_key = self._get_batch_key(input_ids)
            self._add_to_cache(batch_key, cache_entry)

            self.last_teacher_update = self.iteration_count
            self.pending_teacher_update = False

        return cache_entry

    def compute_distillation_loss(self, student_outputs, input_ids):
        teacher = self._get_from_cache(input_ids)

        T = self.config.distill_temperature
        teacher_logits = teacher['logits'][..., :-1, :].contiguous()
        student_logits = student_outputs['logits'][..., :-1, :].contiguous()

        teacher_log_probs = F.log_softmax(teacher_logits / T, dim=-1)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)

        kl_loss = F.kl_div(
            student_log_probs.view(-1, student_log_probs.size(-1)),
            teacher_log_probs.view(-1, teacher_log_probs.size(-1)),
            reduction='batchmean',
            log_target=True
        ) * (T * T)

        feature_loss = torch.tensor(0.0, device=self.device)
        if student_outputs.get('hidden_states') and teacher.get('hidden_states'):
            num_layers = min(len(teacher['hidden_states']), len(student_outputs['hidden_states']))

            layers_to_match = getattr(self.config, 'feature_layers', None) or list(range(num_layers))
            layers_to_match = [l for l in layers_to_match if l < num_layers]

            if layers_to_match:
                for layer_idx in layers_to_match:
                    teacher_features = teacher['hidden_states'][layer_idx]
                    student_features = student_outputs['hidden_states'][layer_idx]

                    feature_loss = feature_loss + F.mse_loss(
                        student_features,
                        teacher_features,
                        reduction='mean'
                    )

                feature_loss = feature_loss / len(layers_to_match)

        total_loss = (
            self.config.distill_alpha_kl * kl_loss +
            self.config.distill_alpha_feature * feature_loss
        )

        return total_loss

    def _get_batch_key(self, input_ids):
        shape_key = tuple(input_ids.shape)
        sample_tokens = input_ids.flatten()[:min(32, input_ids.numel())].cpu().numpy()
        return hash((shape_key, sample_tokens.tobytes()))

    def _add_to_cache(self, key, entry):
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


    ## cache the teacher model loss
    def _get_from_cache(self, input_ids):
        key = self._get_batch_key(input_ids)
        result = self.teacher_cache.get(key)

        if result is not None:
            # for debugging whether the teacher cache is working
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        return result


    def step(self):
        ## track the loss step
        self.iteration_count += 1

    def get_cache_stats(self):
        total_requests = self.cache_hits + self.cache_misses

        return {
            'cache_size': len(self.teacher_cache),
            'cache_misses': self.cache_misses,
            'total_requests': total_requests
        }
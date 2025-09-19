"""
Cyclic Precision Training GPT-2 Model Implementation
This module contains the model components specifically for Cyclic Precision Training (CPT).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Tuple
from transformers import GPT2Config
from torch.utils.checkpoint import checkpoint

# Import quantization and LoRA modules
try:
    from .quantization import LearnableFakeQuantize
    from .lora import LinearWithLoRA
except ImportError:
    from quantization import LearnableFakeQuantize
    from lora import LinearWithLoRA


class CPTAttention(nn.Module):
    """Attention module for Cyclic Precision Training with single LoRA that adapts to different precisions."""

    def __init__(self, config: GPT2Config, bits=8):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.current_bits = bits

        # Single LoRA adapter that will be trained with cyclic precision
        lora_rank = getattr(config, 'lora_rank', 16)
        lora_alpha = getattr(config, 'lora_alpha', 32)
        lora_dropout = getattr(config, 'lora_dropout', 0.1)

        self.c_attn = LinearWithLoRA(
            config.n_embd, 3 * config.n_embd,
            bits=bits,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        self.c_proj = LinearWithLoRA(
            config.n_embd, config.n_embd,
            bits=bits,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )

        # KV cache quantizer
        kv_bits = getattr(config, 'kv_cache_bits', bits)
        self.kv_quantizer = LearnableFakeQuantize(num_bits=kv_bits, symmetric=False)

        self.register_buffer("bias", torch.tril(torch.ones(config.n_positions, config.n_positions)))

    def set_precision(self, weight_bits, activation_bits, kv_bits=None):
        """Set precision for all components - used during cyclic training."""
        self.current_bits = weight_bits
        self.c_attn.set_precision(weight_bits, activation_bits)
        self.c_proj.set_precision(weight_bits, activation_bits)
        if kv_bits is not None:
            self.kv_quantizer.set_num_bits(kv_bits)

    def get_precision(self):
        """Get current precision setting."""
        return self.current_bits

    def forward(self, hidden_states, attention_mask=None):
        B, T, C = hidden_states.shape

        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Quantize KV cache
        k = self.kv_quantizer(k)
        v = self.kv_quantizer(v)

        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias_mask = self.bias[:T, :T].to(attn_weights.device)
        attn_weights = attn_weights.masked_fill(bias_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.c_proj(attn_output)

        return attn_output


class CPTMLP(nn.Module):
    """MLP module for Cyclic Precision Training with single LoRA that adapts to different precisions."""

    def __init__(self, config: GPT2Config, bits=8):
        super().__init__()
        self.current_bits = bits

        # Single LoRA adapter that will be trained with cyclic precision
        lora_rank = getattr(config, 'lora_rank', 16)
        lora_alpha = getattr(config, 'lora_alpha', 32)
        lora_dropout = getattr(config, 'lora_dropout', 0.1)

        self.c_fc = LinearWithLoRA(
            config.n_embd, 4 * config.n_embd,
            bits=bits,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        self.c_proj = LinearWithLoRA(
            4 * config.n_embd, config.n_embd,
            bits=bits,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        self.act = nn.GELU()

    def set_precision(self, weight_bits, activation_bits):
        """Set precision for all components - used during cyclic training."""
        self.current_bits = weight_bits
        self.c_fc.set_precision(weight_bits, activation_bits)
        self.c_proj.set_precision(weight_bits, activation_bits)

    def get_precision(self):
        """Get current precision setting."""
        return self.current_bits

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class CPTBlock(nn.Module):
    """Transformer block for Cyclic Precision Training."""

    def __init__(self, config: GPT2Config, bits=8):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CPTAttention(config, bits)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = CPTMLP(config, bits)

    def set_precision(self, attn_bits, mlp_bits, activation_bits, kv_bits):
        """Set precision for attention and MLP layers - allows layer-wise precision."""
        self.attn.set_precision(attn_bits, activation_bits, kv_bits)
        self.mlp.set_precision(mlp_bits, activation_bits)

    def forward(self, hidden_states, attention_mask=None, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self._forward, hidden_states, attention_mask)
        else:
            return self._forward(hidden_states, attention_mask)

    def _forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, attention_mask)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states


class CPTModel(nn.Module):
    """GPT-2 model for Cyclic Precision Training."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # Default precision for initialization
        self.current_bits = getattr(config, 'default_bits', 8)

        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer blocks
        self.h = nn.ModuleList([
            CPTBlock(config, self.current_bits)
            for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def set_layer_precision(self, layer_config):
        """
        Set precision for each layer - used for cyclic precision training.

        Args:
            layer_config: List of dicts with 'attn_bits', 'mlp_bits', 'activation_bits', 'kv_bits' for each layer
        """
        if not isinstance(layer_config, list):
            raise ValueError("layer_config must be a list of dictionaries")

        for i, config in enumerate(layer_config):
            if i < len(self.h):
                attn_bits = config.get('attn_bits', 8)
                mlp_bits = config.get('mlp_bits', 8)
                activation_bits = config.get('activation_bits', 8)
                kv_bits = config.get('kv_bits', 8)

                self.h[i].set_precision(attn_bits, mlp_bits, activation_bits, kv_bits)

    def set_global_precision(self, bits, activation_bits=None):
        """
        Set the same precision for all layers - simple cyclic training mode.

        Args:
            bits: Number of bits to use for weight quantization
            activation_bits: Number of bits for activation quantization (defaults to bits)
        """
        if activation_bits is None:
            activation_bits = bits

        self.current_bits = bits
        for block in self.h:
            block.set_precision(bits, bits, activation_bits, bits)

    def get_current_precision(self):
        """Get current global precision setting."""
        return self.current_bits

    def forward(self, input_ids, attention_mask=None, use_checkpoint=False):
        device = input_ids.device
        B, T = input_ids.shape

        # Token and position embeddings
        token_embeddings = self.wte(input_ids)
        position_ids = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        position_embeddings = self.wpe(position_ids)

        hidden_states = self.drop(token_embeddings + position_embeddings)

        # Pass through transformer blocks
        for block in self.h:
            hidden_states = block(hidden_states, attention_mask, use_checkpoint)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states

    def load_pretrained_weights(self, pretrained_model, device='cuda'):
        """Load weights from pretrained GPT-2 model."""
        pretrained_dict = pretrained_model.state_dict()
        model_dict = self.state_dict()

        # Map pretrained weights to our model
        mapped_dict = {}
        for name, param in pretrained_dict.items():
            # Handle embeddings and layer norms
            if 'wte' in name or 'wpe' in name or 'ln' in name:
                if name in model_dict and param.shape == model_dict[name].shape:
                    mapped_dict[name] = param
            # Map attention and MLP weights - handle Conv1D to Linear conversion
            elif 'attn.c_attn.weight' in name or 'attn.c_proj.weight' in name:
                new_name = name.replace('.weight', '.linear.weight')
                if len(param.shape) == 2:
                    mapped_dict[new_name] = param.t()  # Transpose for Conv1D to Linear
                else:
                    mapped_dict[new_name] = param
            elif 'mlp.c_fc.weight' in name or 'mlp.c_proj.weight' in name:
                new_name = name.replace('.weight', '.linear.weight')
                if len(param.shape) == 2:
                    mapped_dict[new_name] = param.t()  # Transpose for Conv1D to Linear
                else:
                    mapped_dict[new_name] = param

        # Update model weights
        model_dict.update(mapped_dict)
        self.load_state_dict(model_dict, strict=False)
        self.to(device)

        print(f"Loaded {len(mapped_dict)} weights from pretrained model")
        return self


class CPTLMHeadModel(nn.Module):
    """GPT-2 Language Model for Cyclic Precision Training."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = CPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights between token embeddings and lm_head
        self.lm_head.weight = self.transformer.wte.weight

    def set_layer_precision(self, layer_config):
        """Set layer-wise precision configuration."""
        self.transformer.set_layer_precision(layer_config)

    def set_global_precision(self, bits, activation_bits=None):
        """Set global precision for all layers."""
        self.transformer.set_global_precision(bits, activation_bits)

    def get_current_precision(self):
        """Get current precision setting."""
        return self.transformer.get_current_precision()

    def forward(self, input_ids, labels=None, attention_mask=None, use_checkpoint=False):
        # Get transformer outputs
        hidden_states = self.transformer(input_ids, attention_mask, use_checkpoint)

        # Get logits from language model head
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        return {'loss': loss, 'logits': logits} if loss is not None else logits

    def generate(self, input_ids, max_length=100, temperature=1.0,
                 do_sample=True, top_k=50, top_p=0.95, eos_token_id=None):
        """Generate text using the model."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                outputs = self.forward(input_ids)
                logits = outputs if not isinstance(outputs, dict) else outputs['logits']
                next_token_logits = logits[:, -1, :] / temperature

                if do_sample:
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')

                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')

                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                input_ids = torch.cat([input_ids, next_tokens], dim=1)

                if eos_token_id is not None and (next_tokens == eos_token_id).all():
                    break

                if input_ids.shape[1] >= self.config.n_positions:
                    break

        return input_ids

    def load_pretrained_weights(self, pretrained_model, device='cuda'):
        """Load weights from pretrained GPT-2 model."""
        self.transformer.load_pretrained_weights(pretrained_model.transformer, device)
        return self


class CyclicPrecisionScheduler:
    """
    Scheduler for cyclic precision training.
    Cycles through different bit-widths during training.
    """

    def __init__(self, bit_schedule=[8, 6, 4, 4, 6, 8], cycle_length=100,
                 annealing_type='cosine', warmup_steps=0):
        """
        Args:
            bit_schedule: List of bit-widths to cycle through
            cycle_length: Number of training steps per complete cycle
            annealing_type: Type of annealing within each bit period ('cosine', 'linear', 'step')
            warmup_steps: Number of warmup steps before cycling begins
        """
        self.bit_schedule = bit_schedule
        self.cycle_length = cycle_length
        self.annealing_type = annealing_type
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def get_bits(self, step=None):
        """Get bit-width for current or specified step."""
        if step is None:
            step = self.current_step

        # During warmup, use highest precision
        if step < self.warmup_steps:
            return max(self.bit_schedule)

        # Calculate position in cycle
        cycle_step = (step - self.warmup_steps) % self.cycle_length
        schedule_position = int(cycle_step / self.cycle_length * len(self.bit_schedule))

        return self.bit_schedule[min(schedule_position, len(self.bit_schedule) - 1)]

    def get_lr_scale(self, step=None):
        """Get learning rate scaling factor based on annealing type."""
        if step is None:
            step = self.current_step

        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)

        cycle_step = (step - self.warmup_steps) % self.cycle_length
        progress = cycle_step / self.cycle_length

        if self.annealing_type == 'cosine':
            return 0.5 * (1 + math.cos(math.pi * progress))
        elif self.annealing_type == 'linear':
            return 1.0 - progress
        else:  # step
            return 1.0

    def step(self):
        """Advance scheduler by one step."""
        self.current_step += 1

    def state_dict(self):
        """Return scheduler state."""
        return {
            'current_step': self.current_step,
            'bit_schedule': self.bit_schedule,
            'cycle_length': self.cycle_length,
            'annealing_type': self.annealing_type,
            'warmup_steps': self.warmup_steps
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.current_step = state_dict['current_step']
        self.bit_schedule = state_dict['bit_schedule']
        self.cycle_length = state_dict['cycle_length']
        self.annealing_type = state_dict['annealing_type']
        self.warmup_steps = state_dict['warmup_steps']
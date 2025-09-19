"""
Switchable Precision GPT-2 Model
Supports multiple bit-widths simultaneously with separate LoRA adapters per precision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List
from transformers import GPT2Config
from torch.utils.checkpoint import checkpoint

# Import quantization and LoRA modules
try:
    from .quantization import LearnableFakeQuantize
    from .lora import SwitchableLinearWithLoRA
except ImportError:
    from quantization import LearnableFakeQuantize
    from lora import SwitchableLinearWithLoRA


class SPAttention(nn.Module):
    """Attention module with switchable precision for multiple bit-widths."""

    def __init__(self, config: GPT2Config, bit_widths=[4, 8, 16]):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.bit_widths = bit_widths

        # Get LoRA configs from config - throw error if not provided
        try:
            lora_rank_per_bit = config.lora_rank_per_bit
            lora_alpha_per_bit = config.lora_alpha_per_bit
        except AttributeError as e:
            raise AttributeError(
                f"Config missing required switchable precision attributes: {e}\n"
                "Required: lora_rank_per_bit, lora_alpha_per_bit\n"
                "Example: config.lora_rank_per_bit = {4: 8, 8: 16, 16: 32}"
            )
        lora_dropout = getattr(config, 'lora_dropout', 0.1)

        # Switchable layers with per-bit-width LoRA modules
        self.c_attn = SwitchableLinearWithLoRA(
            config.n_embd, 3 * config.n_embd,
            bit_widths=bit_widths,
            lora_rank_per_bit=lora_rank_per_bit,
            lora_alpha_per_bit=lora_alpha_per_bit,
            lora_dropout=lora_dropout
        )
        self.c_proj = SwitchableLinearWithLoRA(
            config.n_embd, config.n_embd,
            bit_widths=bit_widths,
            lora_rank_per_bit=lora_rank_per_bit,
            lora_alpha_per_bit=lora_alpha_per_bit,
            lora_dropout=lora_dropout
        )

        # KV cache quantizer - use median bit width as default
        default_kv_bits = sorted(bit_widths)[len(bit_widths)//2]
        kv_bits = getattr(config, 'kv_cache_bits', default_kv_bits)
        self.kv_quantizer = LearnableFakeQuantize(num_bits=kv_bits, symmetric=False)

        self.register_buffer("bias", torch.tril(torch.ones(config.n_positions, config.n_positions)))

    def set_precision(self, bits):
        """Set precision for all layers to specified bit-width."""
        if bits not in self.bit_widths:
            raise ValueError(f"Bit width {bits} not in configured widths {self.bit_widths}")
        self.c_attn.set_precision(bits)
        self.c_proj.set_precision(bits)

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


class SPMLP(nn.Module):
    """MLP module with switchable precision for multiple bit-widths."""

    def __init__(self, config: GPT2Config, bit_widths=[4, 8, 16]):
        super().__init__()
        self.bit_widths = bit_widths

        # Get LoRA configs from config
        try:
            lora_rank_per_bit = config.lora_rank_per_bit
            lora_alpha_per_bit = config.lora_alpha_per_bit
        except AttributeError as e:
            raise AttributeError(
                f"Config missing required switchable precision attributes: {e}\n"
                "Required: lora_rank_per_bit, lora_alpha_per_bit"
            )
        lora_dropout = getattr(config, 'lora_dropout', 0.1)

        # Switchable layers with per-bit-width LoRA modules
        self.c_fc = SwitchableLinearWithLoRA(
            config.n_embd, 4 * config.n_embd,
            bit_widths=bit_widths,
            lora_rank_per_bit=lora_rank_per_bit,
            lora_alpha_per_bit=lora_alpha_per_bit,
            lora_dropout=lora_dropout
        )
        self.c_proj = SwitchableLinearWithLoRA(
            4 * config.n_embd, config.n_embd,
            bit_widths=bit_widths,
            lora_rank_per_bit=lora_rank_per_bit,
            lora_alpha_per_bit=lora_alpha_per_bit,
            lora_dropout=lora_dropout
        )
        self.act = nn.GELU()

    def set_precision(self, bits):
        """Set precision for all layers to specified bit-width."""
        if bits not in self.bit_widths:
            raise ValueError(f"Bit width {bits} not in configured widths {self.bit_widths}")
        self.c_fc.set_precision(bits)
        self.c_proj.set_precision(bits)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class SPBlock(nn.Module):
    """Transformer block with switchable precision."""

    def __init__(self, config: GPT2Config, bit_widths=[4, 8, 16]):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = SPAttention(config, bit_widths)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = SPMLP(config, bit_widths)

    def set_precision(self, bits):
        """Set precision for attention and MLP layers."""
        self.attn.set_precision(bits)
        self.mlp.set_precision(bits)

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


class SPModel(nn.Module):
    """GPT-2 model with switchable precision training support."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # Get bit widths from config
        self.bit_widths = getattr(config, 'bit_widths', [4, 8, 16])
        self.current_bit_width = max(self.bit_widths)  # Start with highest precision

        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer blocks with switchable precision
        self.h = nn.ModuleList([
            SPBlock(config, self.bit_widths)
            for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def set_precision(self, bits):
        """Set precision for all transformer blocks."""
        if bits not in self.bit_widths:
            raise ValueError(f"Bit width {bits} not in configured widths {self.bit_widths}")
        self.current_bit_width = bits
        for block in self.h:
            block.set_precision(bits)

    def get_current_precision(self):
        """Get current precision setting."""
        return self.current_bit_width

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
            # Handle the mapping of pretrained weights to our switchable model
            if 'wte' in name or 'wpe' in name or 'ln' in name:
                if name in model_dict and param.shape == model_dict[name].shape:
                    mapped_dict[name] = param
            # Map attention and MLP weights
            elif 'attn.c_attn.weight' in name or 'attn.c_proj.weight' in name:
                new_name = name.replace('.weight', '.linear.weight')
                if 'c_attn' in name or 'c_proj' in name:
                    # Handle Conv1D to Linear conversion
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


class SPLMHeadModel(nn.Module):
    """GPT-2 Language Model with switchable precision training support."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = SPModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights between token embeddings and lm_head
        self.lm_head.weight = self.transformer.wte.weight

    def set_precision(self, bits):
        """Set precision for the model."""
        self.transformer.set_precision(bits)

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
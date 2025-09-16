import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List
from transformers import GPT2Config
from torch.utils.checkpoint import checkpoint

# Try relative imports first, fall back to direct imports
try:
    from .quantization import LearnableFakeQuantize
    from .lora import QATLinearWithLoRA, SwitchableQATLinearWithLoRA
except ImportError:
    from quantization import LearnableFakeQuantize
    from lora import QATLinearWithLoRA, SwitchableQATLinearWithLoRA

class QATGPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config, bits=8):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        self.c_attn = QATLinearWithLoRA(config.n_embd, 3 * config.n_embd, bits=bits,
                                         lora_rank=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout)
        self.c_proj = QATLinearWithLoRA(config.n_embd, config.n_embd, bits=bits,
                                         lora_rank=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout)
        
        self.kv_quantizer = LearnableFakeQuantize(num_bits=8, symmetric=False)
        
        self.register_buffer("bias", torch.tril(torch.ones(config.n_positions, config.n_positions)))
        
    def forward(self, hidden_states, attention_mask=None):
        B, T, C = hidden_states.shape
        
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        k = self.kv_quantizer(k)
        v = self.kv_quantizer(v)
        
        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Ensure bias is on the same device as attn_weights
        bias_mask = self.bias[:T, :T].to(attn_weights.device)
        attn_weights = attn_weights.masked_fill(bias_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.c_proj(attn_output)
        return attn_output
    
    def set_precision(self, weight_bits, activation_bits, kv_bits=8):
        self.c_attn.set_precision(weight_bits, activation_bits)
        self.c_proj.set_precision(weight_bits, activation_bits)
        self.kv_quantizer.set_num_bits(kv_bits)

class QATGPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config, bits=8):
        super().__init__()
        self.c_fc = QATLinearWithLoRA(config.n_embd, 4 * config.n_embd, bits=bits,
                                       lora_rank=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout)
        self.c_proj = QATLinearWithLoRA(4 * config.n_embd, config.n_embd, bits=bits,
                                         lora_rank=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout)
        self.act = nn.GELU()
        
    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states
    
    def set_precision(self, weight_bits, activation_bits):
        self.c_fc.set_precision(weight_bits, activation_bits)
        self.c_proj.set_precision(weight_bits, activation_bits)

class QATGPT2Block(nn.Module):
    def __init__(self, config: GPT2Config, bits=8):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = QATGPT2Attention(config, bits)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = QATGPT2MLP(config, bits)
        
    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, attention_mask)
        hidden_states = residual + attn_output
        
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        return hidden_states
    
    def set_precision(self, attn_bits, mlp_bits, activation_bits=8, kv_bits=8):
        self.attn.set_precision(attn_bits, activation_bits, kv_bits)
        self.mlp.set_precision(mlp_bits, activation_bits)

class QATGPT2(nn.Module):
    """GPT-2 with QAT (single precision, fake quantization)."""
    def __init__(self, config: GPT2Config, quantization_bits, initialize_weights=True):
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = True

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        self.h = nn.ModuleList([
            QATGPT2Block(config, quantization_bits) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.lm_head.weight = self.wte.weight

        # Only apply random init if explicitly requested (for backward compatibility)
        if initialize_weights:
            self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids) #broad casted with a batch size of 1
        hidden_states = self.drop(inputs_embeds + position_embeds)

        for i, block in enumerate(self.h):
            if self.use_gradient_checkpointing and self.training:
                # Create a wrapper function to ensure proper cleanup
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states = checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    use_reentrant=False,
                    preserve_rng_state=False  # Reduce memory usage
                )
            else:
                hidden_states = block(hidden_states, attention_mask)

            # Force cleanup after every 4 layers to prevent memory accumulation
            if i % 4 == 3:
                torch.cuda.empty_cache()

        hidden_states = self.ln_f(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                  shift_labels.view(-1))
        return {'loss': loss, 'logits': logits}
    
    def forward_from_embeddings(self, inputs_embeds, attention_mask=None, labels=None):
        batch_size, seq_length = inputs_embeds.shape[:2]

        position_ids = torch.arange(seq_length, device=inputs_embeds.device).unsqueeze(0)
        position_embeds = self.wpe(position_ids)
        hidden_states = self.drop(inputs_embeds + position_embeds)

        for i, block in enumerate(self.h):
            if self.use_gradient_checkpointing and self.training:
                # Create a wrapper function to ensure proper cleanup
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states = checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    use_reentrant=False,
                    preserve_rng_state=False  # Reduce memory usage
                )
            else:
                hidden_states = block(hidden_states, attention_mask)

            # Force cleanup after every 4 layers to prevent memory accumulation
            if i % 4 == 3:
                torch.cuda.empty_cache()

        hidden_states = self.ln_f(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                  shift_labels.view(-1))

        return {'loss': loss, 'logits': logits}

    def generate(self, input_ids=None, max_new_tokens=50, temperature=1.0,
                 do_sample=False, top_k=50, top_p=0.95, pad_token_id=None,
                 eos_token_id=None, **kwargs):
        """Generate text using the model."""
        self.eval()
        device = input_ids.device if input_ids is not None else next(self.parameters()).device

        if input_ids is None:
            input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)

        batch_size = input_ids.shape[0]

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.forward(input_ids=input_ids)
                logits = outputs['logits']

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

    def set_layer_precision(self, layer_config):
        """
        Set precision for each layer - used for cyclic precision training.

        Args:
            layer_config: List of dicts with 'attn_bits' and 'mlp_bits' for each layer
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

    def set_global_precision(self, bits):
        """
        Set the same precision for all layers - used for QAT training.

        Args:
            bits: Number of bits to use for quantization
        """
        for block in self.h:
            block.set_precision(bits, bits, bits, bits)


class SwitchableQATGPT2Attention(nn.Module):
    """Attention module with switchable precision."""
    def __init__(self, config: GPT2Config, bit_widths=[4, 8, 16]):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.bit_widths = bit_widths

        # Get LoRA configs from config
        lora_rank_per_bit = {4: 32, 8: 16, 16: 8}
        lora_alpha_per_bit = {4: 64, 8: 32, 16: 16}
        lora_dropout = config.lora_dropout

        # Switchable layers
        self.c_attn = SwitchableQATLinearWithLoRA(
            config.n_embd, 3 * config.n_embd,
            bit_widths=bit_widths,
            lora_rank_per_bit=lora_rank_per_bit,
            lora_alpha_per_bit=lora_alpha_per_bit,
            lora_dropout=lora_dropout
        )
        self.c_proj = SwitchableQATLinearWithLoRA(
            config.n_embd, config.n_embd,
            bit_widths=bit_widths,
            lora_rank_per_bit=lora_rank_per_bit,
            lora_alpha_per_bit=lora_alpha_per_bit,
            lora_dropout=lora_dropout
        )

        self.kv_quantizer = LearnableFakeQuantize(num_bits=8, symmetric=False)
        self.register_buffer("bias", torch.tril(torch.ones(config.n_positions, config.n_positions)))

    def set_precision(self, bits):
        """Set precision for all layers."""
        self.c_attn.set_precision(bits)
        self.c_proj.set_precision(bits)

    def forward(self, hidden_states, attention_mask=None):
        B, T, C = hidden_states.shape

        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

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


class SwitchableQATGPT2MLP(nn.Module):
    """MLP module with switchable precision."""
    def __init__(self, config: GPT2Config, bit_widths=[4, 8, 16]):
        super().__init__()
        self.bit_widths = bit_widths

        # Get LoRA configs
        lora_rank_per_bit = {4: 32, 8: 16, 16: 8}
        lora_alpha_per_bit = {4: 64, 8: 32, 16: 16}
        lora_dropout = config.lora_dropout

        self.c_fc = SwitchableQATLinearWithLoRA(
            config.n_embd, 4 * config.n_embd,
            bit_widths=bit_widths,
            lora_rank_per_bit=lora_rank_per_bit,
            lora_alpha_per_bit=lora_alpha_per_bit,
            lora_dropout=lora_dropout
        )
        self.c_proj = SwitchableQATLinearWithLoRA(
            4 * config.n_embd, config.n_embd,
            bit_widths=bit_widths,
            lora_rank_per_bit=lora_rank_per_bit,
            lora_alpha_per_bit=lora_alpha_per_bit,
            lora_dropout=lora_dropout
        )
        self.act = nn.GELU()

    def set_precision(self, bits):
        """Set precision for all layers."""
        self.c_fc.set_precision(bits)
        self.c_proj.set_precision(bits)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class SwitchableQATGPT2Block(nn.Module):
    """Transformer block with switchable precision."""
    def __init__(self, config: GPT2Config, bit_widths=[4, 8, 16]):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = SwitchableQATGPT2Attention(config, bit_widths)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = SwitchableQATGPT2MLP(config, bit_widths)

    def set_precision(self, bits):
        """Set precision for attention and MLP."""
        self.attn.set_precision(bits)
        self.mlp.set_precision(bits)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, attention_mask)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states


class SwitchableQATGPT2(nn.Module):
    """GPT-2 with switchable precision QAT."""
    def __init__(self, config: GPT2Config, bit_widths=[4, 8, 16], initialize_weights=True):
        super().__init__()
        self.config = config
        self.bit_widths = bit_widths
        self.current_bits = 8  # Default
        self.n_layer = config.n_layer
        self.use_gradient_checkpointing = True

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        self.h = nn.ModuleList([
            SwitchableQATGPT2Block(config, bit_widths) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        # Only apply random init if explicitly requested (for backward compatibility)
        if initialize_weights:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def set_precision(self, bits):
        """Set precision for all blocks."""
        if bits not in self.bit_widths:
            raise ValueError(f"Bit-width {bits} not supported. Choose from {self.bit_widths}")
        self.current_bits = bits
        for block in self.h:
            block.set_precision(bits)

    def set_layer_precision(self, layer_configs):
        """Set per-layer bit-widths"""
        for i, bits in enumerate(layer_configs):
            if i < self.n_layer:
                self.h[i].set_precision(bits)

    def set_global_precision(self, bits):
        """Set same bit-width for all layers"""
        for block in self.h:
            block.set_precision(bits)

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = self.drop(inputs_embeds + position_embeds)

        for i, block in enumerate(self.h):
            if self.use_gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states = checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    use_reentrant=False,
                    preserve_rng_state=False
                )
            else:
                hidden_states = block(hidden_states, attention_mask)

            if i % 4 == 3:
                torch.cuda.empty_cache()

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                  shift_labels.view(-1))
        return {'loss': loss, 'logits': logits}

    def generate(self, input_ids=None, max_new_tokens=50, temperature=1.0,
                 do_sample=False, top_k=50, top_p=0.95, pad_token_id=None,
                 eos_token_id=None, **kwargs):
        """Generate text using the model."""
        self.eval()
        device = input_ids.device if input_ids is not None else next(self.parameters()).device

        if input_ids is None:
            input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)

        batch_size = input_ids.shape[0]

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.forward(input_ids=input_ids)
                logits = outputs['logits']
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

    def forward_from_embeddings(self, inputs_embeds, attention_mask=None, labels=None):
        batch_size, seq_length = inputs_embeds.shape[:2]
        position_ids = torch.arange(seq_length, device=inputs_embeds.device).unsqueeze(0)
        position_embeds = self.wpe(position_ids)
        hidden_states = self.drop(inputs_embeds + position_embeds)

        for i, block in enumerate(self.h):
            if self.use_gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states = checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    use_reentrant=False,
                    preserve_rng_state=False
                )
            else:
                hidden_states = block(hidden_states, attention_mask)

            if i % 4 == 3:
                torch.cuda.empty_cache()

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                  shift_labels.view(-1))
        return {'loss': loss, 'logits': logits}
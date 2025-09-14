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
    from .lora import QATLinearWithLoRA
except ImportError:
    from quantization import LearnableFakeQuantize
    from lora import QATLinearWithLoRA

class QATGPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config, bits=8):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        self.c_attn = QATLinearWithLoRA(config.n_embd, 3 * config.n_embd, bits=bits)
        self.c_proj = QATLinearWithLoRA(config.n_embd, config.n_embd, bits=bits)
        
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
        self.kv_quantizer.num_bits = kv_bits

class QATGPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config, bits=8):
        super().__init__()
        self.c_fc = QATLinearWithLoRA(config.n_embd, 4 * config.n_embd, bits=bits)
        self.c_proj = QATLinearWithLoRA(4 * config.n_embd, config.n_embd, bits=bits)
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
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = True
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        
        bits = getattr(config, 'quantization_bits', 8)
        self.h = nn.ModuleList([
            QATGPT2Block(config, bits) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.wte.weight
        
        self.apply(self._init_weights) ## handy trick learned from gemini
        
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
            if i % 4 == 3 and torch.cuda.is_available():
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
            if i % 4 == 3 and torch.cuda.is_available():
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
    
# Alias for compatibility
SwitchableQuantizedGPT2 = QATGPT2
QuantizedGPT2Attention = QATGPT2Attention  
QuantizedGPT2MLP = QATGPT2MLP
QuantizedGPT2Block = QATGPT2Block
                    
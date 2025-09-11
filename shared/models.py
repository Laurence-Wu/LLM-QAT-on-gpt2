import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List
from transformers import GPT2Config
from torch.utils.checkpoint import checkpoint

from part1_switchable_precision.train_switchable import log_memory_usage
from quantization import LearnableFakeQuantize
from lora import QuantizedLinearWithLoRA

class QuantizedGPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config, bit_widths=None):
        super().__init__()
        if bit_widths is None:
            bit_widths = [4, 8, 16]
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        self.c_attn = QuantizedLinearWithLoRA(config.n_embd, 3 * config.n_embd, bit_widths=bit_widths)
        self.c_proj = QuantizedLinearWithLoRA(config.n_embd, config.n_embd, bit_widths=bit_widths)
        
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

class QuantizedGPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config, bit_widths=None):
        super().__init__()
        if bit_widths is None:
            bit_widths = [4, 8, 16]
            
        self.c_fc = QuantizedLinearWithLoRA(config.n_embd, 4 * config.n_embd, bit_widths=bit_widths)
        self.c_proj = QuantizedLinearWithLoRA(4 * config.n_embd, config.n_embd, bit_widths=bit_widths)
        self.act = nn.GELU()
        
    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states
    
    def set_precision(self, weight_bits, activation_bits):
        self.c_fc.set_precision(weight_bits, activation_bits)
        self.c_proj.set_precision(weight_bits, activation_bits)

class QuantizedGPT2Block(nn.Module):
    def __init__(self, config: GPT2Config, bit_widths=None):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = QuantizedGPT2Attention(config, bit_widths)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = QuantizedGPT2MLP(config, bit_widths)
        
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

class SwitchableQuantizedGPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = True
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        
        bit_widths = config.bit_widths
        self.h = nn.ModuleList()
        for _ in range(config.n_layer):
            self.h.append(QuantizedGPT2Block(config, bit_widths=bit_widths))

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

        for block in self.h:
            if self.use_gradient_checkpointing and self.training:
                hidden_states = checkpoint(block, hidden_states, attention_mask, use_reentrant=False)
            else:
                hidden_states = block(hidden_states, attention_mask)
        
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
        
        for block in self.h:
            if self.use_gradient_checkpointing and self.training:
                hidden_states = checkpoint(block, hidden_states, attention_mask, use_reentrant=False)
            else:
                hidden_states = block(hidden_states, attention_mask)
        
        hidden_states = self.ln_f(hidden_states)
        
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                  shift_labels.view(-1))
        
        return {'loss': loss, 'logits': logits}
    
    def set_layer_precision(self, layer_configs: List[Dict]):
        """
        Set precision for each layer based on provided configurations.
        
        Args:
            layer_configs: List of dictionaries containing precision settings for each layer
                         Each dict should have 'attn_bits' and 'mlp_bits' keys
        """
        for i, config in enumerate(layer_configs):
            if i < len(self.h):
                
                # Set attention precision
                if hasattr(self.h[i].attn.c_attn, 'set_precision'):
                    self.h[i].attn.c_attn.set_precision(
                        config.get('attn_bits', 8), 
                        config.get('attn_bits', 8)
                    )
                if hasattr(self.h[i].attn.c_proj, 'set_precision'):
                    self.h[i].attn.c_proj.set_precision(
                        config.get('attn_bits', 8), 
                        config.get('attn_bits', 8)
                    )
                
                # Set MLP precision
                if hasattr(self.h[i].mlp.c_fc, 'set_precision'):
                    self.h[i].mlp.c_fc.set_precision(
                        config.get('mlp_bits', 8), 
                        config.get('mlp_bits', 8)
                    )
                if hasattr(self.h[i].mlp.c_proj, 'set_precision'):
                    self.h[i].mlp.c_proj.set_precision(
                        config.get('mlp_bits', 8), 
                        config.get('mlp_bits', 8)
                    )
                    
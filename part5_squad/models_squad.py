
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Tuple
from transformers import GPT2Config
from torch.utils.checkpoint import checkpoint

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from lora import SPLinearWithLoRA
from switchable_batchnorm import SwitchableLayerNorm

class SPAttention(nn.Module):

    def __init__(self, config: GPT2Config, bit_widths):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        self.bit_widths = bit_widths

        lora_rank_per_bit = config.lora_rank_per_bit
        lora_alpha_per_bit = config.lora_alpha_per_bit
        quantizer_per_bit = config.quantizer_per_bit
        per_channel = getattr(config, 'per_channel_quantization', True)

        self.c_attn = SPLinearWithLoRA(
            config.n_embd, 3 * config.n_embd,
            bit_widths=bit_widths,
            lora_rank_per_bit=lora_rank_per_bit,
            lora_alpha_per_bit=lora_alpha_per_bit,
            quantizer_per_bit=quantizer_per_bit,
            per_channel=per_channel
        )
        self.c_proj = SPLinearWithLoRA(
            config.n_embd, config.n_embd,
            bit_widths=bit_widths,
            lora_rank_per_bit=lora_rank_per_bit,
            lora_alpha_per_bit=lora_alpha_per_bit,
            quantizer_per_bit=quantizer_per_bit,
            per_channel=per_channel
        )

        self.register_buffer("bias", torch.tril(torch.ones(config.n_positions, config.n_positions)))

    def set_precision(self, bits) -> int:
        self.current_bit_width = bits
        self.c_attn.set_precision(bits)
        self.c_proj.set_precision(bits)
        return self.current_bit_width

    def forward(self, hidden_states, attention_mask=None):
        B, T, C = hidden_states.shape

        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias_mask = self.bias[:T, :T].to(attn_weights.device)
        attn_weights = attn_weights.masked_fill(bias_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.c_proj(attn_output)

        return attn_output

class SPMLP(nn.Module):

    def __init__(self, config: GPT2Config, bit_widths=None):
        super().__init__()

        if bit_widths is None:
            bit_widths = getattr(config, 'bit_widths', [6, 8, 16, 32])
        self.bit_widths = bit_widths

        try:
            lora_rank_per_bit = config.lora_rank_per_bit
            lora_alpha_per_bit = config.lora_alpha_per_bit
        except AttributeError as e:
            raise AttributeError(
                f"Config missing required switchable precision attributes: {e}\n"
                "Required: lora_rank_per_bit, lora_alpha_per_bit"
            )
        quantizer_per_bit = getattr(config, 'quantizer_per_bit', None)
        per_channel = getattr(config, 'per_channel_quantization', True)

        self.c_fc = SPLinearWithLoRA(
            config.n_embd, 4 * config.n_embd,
            bit_widths=bit_widths,
            lora_rank_per_bit=lora_rank_per_bit,
            lora_alpha_per_bit=lora_alpha_per_bit,
            quantizer_per_bit=quantizer_per_bit,
            per_channel=per_channel
        )
        self.c_proj = SPLinearWithLoRA(
            4 * config.n_embd, config.n_embd,
            bit_widths=bit_widths,
            lora_rank_per_bit=lora_rank_per_bit,
            lora_alpha_per_bit=lora_alpha_per_bit,
            quantizer_per_bit=quantizer_per_bit,
            per_channel=per_channel
        )
        self.act = nn.GELU()

    def set_precision(self, bits) -> int:

        if bits not in self.bit_widths:
            raise ValueError(f"Bit width {bits} not in configured widths {self.bit_widths}")
        self.c_fc.set_precision(bits)
        self.c_proj.set_precision(bits)
        return bits

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states

class SPBlock(nn.Module):

    def __init__(self, config: GPT2Config, bit_widths):
        super().__init__()
        self.ln_1 = SwitchableLayerNorm(
            config.n_embd,
            precision_levels=bit_widths,
            eps=config.layer_norm_epsilon
        )
        self.attn = SPAttention(config, bit_widths)
        self.ln_2 = SwitchableLayerNorm(
            config.n_embd,
            precision_levels=bit_widths,
            eps=config.layer_norm_epsilon
        )
        self.mlp = SPMLP(config, bit_widths)

    def set_precision(self, bits) -> int:
        self.ln_1.set_precision(bits)
        self.attn.set_precision(bits)
        self.ln_2.set_precision(bits)
        self.mlp.set_precision(bits)
        return bits

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

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.bit_widths = config.bit_widths
        self.current_bit_width = max(self.bit_widths)

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        self.h = nn.ModuleList([
            SPBlock(config, bit_widths=self.bit_widths)
            for _ in range(config.n_layer)
        ])

        self.ln_f = SwitchableLayerNorm(
            config.n_embd,
            precision_levels=self.bit_widths,
            eps=config.layer_norm_epsilon
        )

    def unfreeze_weights(self, bits):
        if bits == 32:
            for block in self.h:
                if isinstance(block.ln_1, SwitchableLayerNorm):
                    for precision in block.ln_1.precision_levels:
                        block.ln_1.weights[str(precision)].requires_grad = True
                        block.ln_1.biases[str(precision)].requires_grad = True
                if isinstance(block.ln_2, SwitchableLayerNorm):
                    for precision in block.ln_2.precision_levels:
                        block.ln_2.weights[str(precision)].requires_grad = True
                        block.ln_2.biases[str(precision)].requires_grad = True

                block.attn.c_attn.linear.weight.requires_grad = True
                block.attn.c_attn.linear.bias.requires_grad = True
                block.attn.c_proj.linear.weight.requires_grad = True
                block.attn.c_proj.linear.bias.requires_grad = True

                block.mlp.c_fc.linear.weight.requires_grad = True
                block.mlp.c_fc.linear.bias.requires_grad = True
                block.mlp.c_proj.linear.weight.requires_grad = True
                block.mlp.c_proj.linear.bias.requires_grad = True

            if isinstance(self.ln_f, SwitchableLayerNorm):
                for precision in self.ln_f.precision_levels:
                    self.ln_f.weights[str(precision)].requires_grad = True
                    self.ln_f.biases[str(precision)].requires_grad = True

    def set_precision(self, bits) -> int:
        if bits not in self.bit_widths:
            raise ValueError(f"Bit width {bits} not in configured widths {self.bit_widths}")
        self.current_bit_width = bits

        for block in self.h:
            block.set_precision(bits)

        self.ln_f.set_precision(bits)

        return self.current_bit_width

    def disable_lora_for_calibration(self):

        for module in self.modules():
            if module.__class__.__name__ == 'SPLinearWithLoRA':
                module.calibration_mode = True

    def enable_lora_after_calibration(self):

        for module in self.modules():
            if module.__class__.__name__ == 'SPLinearWithLoRA':
                module.calibration_mode = False

    def verify_precision_consistency(self) -> Tuple[bool, Dict]:

        details = {
            'expected': self.current_bit_width,
            'mismatches': [],
            'components': {}
        }

        for i, block in enumerate(self.h):
            if isinstance(block.ln_1, SwitchableLayerNorm):
                ln1_prec = block.ln_1.current_precision
                details['components'][f'block_{i}_ln_1'] = ln1_prec
                if ln1_prec != self.current_bit_width:
                    details['mismatches'].append(f'block_{i}_ln_1: {ln1_prec} (expected {self.current_bit_width})')

            if isinstance(block.ln_2, SwitchableLayerNorm):
                ln2_prec = block.ln_2.current_precision
                details['components'][f'block_{i}_ln_2'] = ln2_prec
                if ln2_prec != self.current_bit_width:
                    details['mismatches'].append(f'block_{i}_ln_2: {ln2_prec} (expected {self.current_bit_width})')

            if hasattr(block.attn, 'current_bit_width'):
                attn_prec = block.attn.current_bit_width
                details['components'][f'block_{i}_attn'] = attn_prec
                if attn_prec != self.current_bit_width:
                    details['mismatches'].append(f'block_{i}_attn: {attn_prec} (expected {self.current_bit_width})')

            if hasattr(block.mlp.c_fc, 'current_precision'):
                mlp_fc_prec = block.mlp.c_fc.current_precision
                details['components'][f'block_{i}_mlp_c_fc'] = mlp_fc_prec
                if mlp_fc_prec != self.current_bit_width:
                    details['mismatches'].append(f'block_{i}_mlp_c_fc: {mlp_fc_prec} (expected {self.current_bit_width})')

            if hasattr(block.mlp.c_proj, 'current_precision'):
                mlp_proj_prec = block.mlp.c_proj.current_precision
                details['components'][f'block_{i}_mlp_c_proj'] = mlp_proj_prec
                if mlp_proj_prec != self.current_bit_width:
                    details['mismatches'].append(f'block_{i}_mlp_c_proj: {mlp_proj_prec} (expected {self.current_bit_width})')

        if isinstance(self.ln_f, SwitchableLayerNorm):
            lnf_prec = self.ln_f.current_precision
            details['components']['ln_f'] = lnf_prec
            if lnf_prec != self.current_bit_width:
                details['mismatches'].append(f'ln_f: {lnf_prec} (expected {self.current_bit_width})')

        is_consistent = len(details['mismatches']) == 0
        return is_consistent, details

    def get_current_precision(self):

        return self.current_bit_width

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, use_checkpoint=False,
                output_hidden_states=False):

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
            B, T = hidden_states.shape[:2]
            device = hidden_states.device
        else:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            device = input_ids.device
            B, T = input_ids.shape

            token_embeddings = self.wte(input_ids)
            position_ids = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
            position_embeddings = self.wpe(position_ids)

            hidden_states = self.drop(token_embeddings + position_embeddings)

        all_hidden_states = [] if output_hidden_states else None

        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states.append(hidden_states.clone().detach())

            hidden_states = block(hidden_states, attention_mask, use_checkpoint)

            if i % 4 == 3 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states.clone().detach())
            return hidden_states, all_hidden_states
        else:
            return hidden_states


class SPQuestionAnsweringModel(nn.Module):
    """
    Switchable Precision Question Answering Model for SQuAD
    Uses SPModel transformer with separate QA heads for start/end positions
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = SPModel(config)

        # Separate heads for start and end positions (Option A - better performance)
        self.qa_dropout = nn.Dropout(0.1)
        self.qa_start = nn.Linear(config.n_embd, 1, bias=True)
        self.qa_end = nn.Linear(config.n_embd, 1, bias=True)

        # Initialize QA heads
        nn.init.normal_(self.qa_start.weight, std=0.02)
        nn.init.zeros_(self.qa_start.bias)
        nn.init.normal_(self.qa_end.weight, std=0.02)
        nn.init.zeros_(self.qa_end.bias)

    def set_precision(self, bits) -> int:
        """Set quantization precision for entire model"""
        return self.transformer.set_precision(bits)

    def disable_lora_for_calibration(self):
        """Disable LoRA during calibration phase"""
        self.transformer.disable_lora_for_calibration()

    def enable_lora_after_calibration(self):
        """Re-enable LoRA after calibration"""
        self.transformer.enable_lora_after_calibration()

    def verify_precision_consistency(self) -> Tuple[bool, Dict]:
        """Verify all components are at same precision"""
        return self.transformer.verify_precision_consistency()

    def get_current_precision(self):
        """Get current precision level"""
        return self.transformer.get_current_precision()

    def _compute_qa_loss(self, start_logits, end_logits, start_positions, end_positions):
        """
        Compute Question Answering loss

        Args:
            start_logits: [batch_size, seq_length] - logits for start position
            end_logits: [batch_size, seq_length] - logits for end position
            start_positions: [batch_size] - ground truth start positions
            end_positions: [batch_size] - ground truth end positions

        Returns:
            total_loss: Average of start and end losses
        """
        # Ignore positions marked as -1 (padding or invalid)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)

        total_loss = (start_loss + end_loss) / 2.0

        return total_loss

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                start_positions=None, end_positions=None,
                use_checkpoint=False, output_hidden_states=False,
                return_dict=False):
        """
        Forward pass for QA model

        Args:
            input_ids: [batch_size, seq_length] - Input token IDs
            attention_mask: [batch_size, seq_length] - Attention mask
            start_positions: [batch_size] - Ground truth start positions (optional)
            end_positions: [batch_size] - Ground truth end positions (optional)
            output_hidden_states: Whether to return hidden states from all layers
            return_dict: Whether to return dict output

        Returns:
            Dict with keys: 'loss', 'start_logits', 'end_logits', 'hidden_states'
        """
        # Get transformer outputs
        if output_hidden_states:
            hidden_states, all_hidden_states = self.transformer(
                input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                use_checkpoint=use_checkpoint,
                output_hidden_states=True
            )
        else:
            hidden_states = self.transformer(
                input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                use_checkpoint=use_checkpoint,
                output_hidden_states=False
            )
            all_hidden_states = None

        # Apply dropout
        hidden_states = self.qa_dropout(hidden_states)

        # Compute start and end logits separately (Option A - separate heads)
        start_logits = self.qa_start(hidden_states).squeeze(-1)  # [batch, seq_len]
        end_logits = self.qa_end(hidden_states).squeeze(-1)      # [batch, seq_len]

        # Compute loss if labels provided
        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self._compute_qa_loss(
                start_logits, end_logits,
                start_positions, end_positions
            )

        # Return output
        if return_dict or output_hidden_states:
            return {
                'loss': loss,
                'start_logits': start_logits,
                'end_logits': end_logits,
                'hidden_states': all_hidden_states
            }
        else:
            return {
                'loss': loss,
                'start_logits': start_logits,
                'end_logits': end_logits
            }

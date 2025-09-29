"""
CPT Model Architecture with cyclic precision switching.
Incorporates LoRA adapters, per-channel quantization, and Range BatchNorm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Dict, Tuple
import math

from quantization import LearnableFakeQuantize
from switchable_batchnorm import SwitchableLayerNorm


class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation for parameter-efficient fine-tuning."""

    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0

        if rank > 0:
            self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation."""
        if self.lora_A is not None and self.lora_B is not None:
            # x: [batch, seq_len, in_features]
            lora_output = x @ self.lora_A @ self.lora_B
            return lora_output * self.scaling
        return torch.zeros_like(x[..., :self.lora_B.size(1)] if self.lora_B is not None else x)


class CPTLinear(nn.Module):
    """
    Linear layer with cyclic precision support and LoRA adapters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bit_widths: list = [4, 6, 8],
        lora_rank_per_bit: dict = None,
        lora_alpha_per_bit: dict = None,
        quantizer_per_bit: dict = None,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bit_widths = bit_widths

        # Base linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA adapters for each precision
        self.lora_adapters = nn.ModuleDict()
        if lora_rank_per_bit is None:
            lora_rank_per_bit = {4: 32, 6: 24, 8: 16}
        if lora_alpha_per_bit is None:
            lora_alpha_per_bit = {4: 64, 6: 48, 8: 32}

        for bits in bit_widths:
            rank = lora_rank_per_bit.get(bits, 16)
            alpha = lora_alpha_per_bit.get(bits, 32)
            # Use same naming as part1 (without 'lora_' prefix)
            self.lora_adapters[f'{bits}bit'] = LoRAAdapter(
                in_features, out_features, rank, alpha
            )

        # Default quantizer types if not provided
        if quantizer_per_bit is None:
            quantizer_per_bit = {bits: 'log' for bits in bit_widths}

        student_bits = [b for b in bit_widths if b < 32]

        # Quantizers for weights - matching part1 naming
        self.quantizers_weight = nn.ModuleDict({
            f'{bits}bit': LearnableFakeQuantize(
                num_bits=bits,
                quantizer_type=quantizer_per_bit.get(bits, 'log'),
                channel_dim=0,  # Weight quantization along output channel
                per_channel=True
            )
            for bits in student_bits
        })

        # Quantizers for inputs - matching part1 naming
        self.quantizers_input = nn.ModuleDict({
            f'{bits}bit': LearnableFakeQuantize(
                num_bits=bits,
                quantizer_type=quantizer_per_bit.get(bits, 'log'),
                channel_dim=-1,  # Input quantization along hidden dim
                per_channel=True,
                is_input=True
            )
            for bits in student_bits
        })

        # Current precision
        self.current_bits = max(bit_widths)

        # Calibration mode flag (like part1)
        self.calibration_mode = False

    def set_precision(self, num_bits: int):
        """Set current precision for forward pass."""
        if num_bits not in self.bit_widths:
            raise ValueError(f"Precision {num_bits} not in configured widths {self.bit_widths}")
        self.current_bits = num_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with current precision."""
        # Handle 32-bit precision (no quantization)
        if self.current_bits >= 32:
            out = F.linear(x, self.linear.weight, self.linear.bias)
            return out

        # Get quantizers for current bit width
        bits_key = f'{self.current_bits}bit'

        # Quantize inputs and weights (matching part1 naming)
        x_quant = self.quantizers_input[bits_key](x)
        weight_quant = self.quantizers_weight[bits_key](self.linear.weight)

        # Base linear operation
        out = F.linear(x_quant, weight_quant, self.linear.bias)

        # Skip LoRA in calibration mode (like part1)
        if self.calibration_mode:
            return out

        # Add LoRA adaptation for current precision (using original input like part1)
        lora_key = f'{self.current_bits}bit'  # Match part1 naming
        if lora_key in self.lora_adapters:
            lora_out = self.lora_adapters[lora_key](x)
            out = out + lora_out

        return out


class CPTSelfAttention(nn.Module):
    """
    Self-attention module with cyclic precision support.
    """

    def __init__(self, config, bit_widths: list, lora_rank_per_bit: dict, lora_alpha_per_bit: dict, quantizer_per_bit: dict = None):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        # Combined QKV projection (like GPT-2 and part1)
        self.c_attn = CPTLinear(
            self.n_embd, 3 * self.n_embd,
            bit_widths, lora_rank_per_bit, lora_alpha_per_bit, quantizer_per_bit
        )
        # Output projection
        self.c_proj = CPTLinear(
            self.n_embd, self.n_embd,
            bit_widths, lora_rank_per_bit, lora_alpha_per_bit, quantizer_per_bit
        )

        # Attention dropout
        self.attn_dropout = nn.Dropout(config.embd_pdrop)
        self.resid_dropout = nn.Dropout(config.embd_pdrop)

    def set_precision(self, num_bits: int):
        """Set precision for all layers."""
        self.c_attn.set_precision(num_bits)
        self.c_proj.set_precision(num_bits)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # Compute combined QKV and split
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.n_embd, dim=-1)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # Handle past key values for generation
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        present_key_value = (key, value)

        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_embd
        )

        # Output projection
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present_key_value


class CPTBlock(nn.Module):
    """
    Transformer block with CPT support and Range LayerNorm.
    """

    def __init__(self, config, bit_widths: list, lora_rank_per_bit: dict, lora_alpha_per_bit: dict, quantizer_per_bit: dict = None):
        super().__init__()

        # Switchable LayerNorm with bit widths
        self.ln_1 = SwitchableLayerNorm(config.n_embd, precision_levels=bit_widths, eps=config.layer_norm_epsilon)
        self.ln_2 = SwitchableLayerNorm(config.n_embd, precision_levels=bit_widths, eps=config.layer_norm_epsilon)
        self.bit_widths = bit_widths

        # Self-attention with CPT
        self.attn = CPTSelfAttention(config, bit_widths, lora_rank_per_bit, lora_alpha_per_bit, quantizer_per_bit)

        # Feed-forward with CPT
        self.mlp = nn.ModuleDict({
            'fc_in': CPTLinear(
                config.n_embd, 4 * config.n_embd,
                bit_widths, lora_rank_per_bit, lora_alpha_per_bit, quantizer_per_bit
            ),
            'fc_out': CPTLinear(
                4 * config.n_embd, config.n_embd,
                bit_widths, lora_rank_per_bit, lora_alpha_per_bit, quantizer_per_bit
            )
        })
        self.mlp_dropout = nn.Dropout(config.embd_pdrop)

    def set_precision(self, num_bits: int):
        """Set precision for all layers."""
        self.ln_1.set_precision(num_bits)
        self.ln_2.set_precision(num_bits)
        self.attn.set_precision(num_bits)
        self.mlp['fc_in'].set_precision(num_bits)
        self.mlp['fc_out'].set_precision(num_bits)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        # Self-attention
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, present_key_value = self.attn(
            hidden_states, attention_mask, past_key_value
        )
        hidden_states = residual + attn_output

        # Feed-forward
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp['fc_in'](hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.mlp['fc_out'](hidden_states)
        hidden_states = self.mlp_dropout(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


class CPTModel(nn.Module):
    """
    GPT-2 model with Cyclic Precision Training support.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config['model']
        cpt_config = config['cpt']

        # Token and position embeddings
        self.wte = nn.Embedding(model_config.vocab_size, model_config.n_embd)
        self.wpe = nn.Embedding(model_config.n_positions, model_config.n_embd)
        self.drop = nn.Dropout(model_config.embd_pdrop)

        # Transformer blocks with CPT
        self.h = nn.ModuleList([
            CPTBlock(
                model_config,
                model_config.bit_widths,
                model_config.lora_rank_per_bit,
                model_config.lora_alpha_per_bit,
                model_config.quantizer_per_bit
            )
            for _ in range(model_config.n_layer)
        ])

        # Final layer norm
        self.ln_f = SwitchableLayerNorm(model_config.n_embd, precision_levels=model_config.bit_widths, eps=model_config.layer_norm_epsilon)

        # Language modeling head
        self.lm_head = CPTLinear(
            model_config.n_embd, model_config.vocab_size,
            model_config.bit_widths,
            model_config.lora_rank_per_bit,
            model_config.lora_alpha_per_bit,
            model_config.quantizer_per_bit,
            bias=False
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Current precision
        self.current_precision = model_config.default_bits

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def set_precision(self, num_bits: int):
        """Set precision for all CPT layers."""
        self.current_precision = num_bits
        for block in self.h:
            block.set_precision(num_bits)
        self.ln_f.set_precision(num_bits)
        self.lm_head.set_precision(num_bits)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> CausalLMOutputWithPast:
        batch_size, seq_len = input_ids.shape

        # Token and position embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Forward through transformer blocks
        presents = [] if use_cache else None
        for i, block in enumerate(self.h):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_key_value = block(
                hidden_states, attention_mask, past_key_value
            )
            if use_cache:
                presents.append(present_key_value)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Language modeling head
        logits = self.lm_head(hidden_states)

        # Calculate loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=hidden_states,
            attentions=None
        )

    def disable_lora_for_calibration(self):
        """
        Disable LoRA during calibration to get clean statistics.
        Sets calibration_mode=True on all CPTLinear layers.
        """
        for module in self.modules():
            if isinstance(module, CPTLinear):
                module.calibration_mode = True

    def enable_lora_after_calibration(self):
        """
        Re-enable LoRA after calibration.
        Sets calibration_mode=False on all CPTLinear layers.
        """
        for module in self.modules():
            if isinstance(module, CPTLinear):
                module.calibration_mode = False

    def replace_quantizers_for_evaluation(self):
        """
        Replace all quantizers with per-tensor versions for evaluation.
        This allows the model to handle variable-length sequences.
        """
        # Replace quantizers in all CPTLinear layers
        for module in self.modules():
            if isinstance(module, CPTLinear):
                # Replace weight quantizers
                for bits in module.weight_quantizers.bit_widths:
                    old_quantizer = module.weight_quantizers.quantizers[bits]
                    # Create new per-tensor quantizer
                    if module.weight_quantizers.quantizer_type == 'log':
                        new_quantizer = PerChannelLogQuantization(
                            num_bits=bits,
                            symmetric=old_quantizer.symmetric,
                            per_channel=False  # Use per-tensor for evaluation
                        )
                    else:  # sbm
                        new_quantizer = SBMQuantization(
                            num_bits=bits,
                            use_stochastic_rounding=old_quantizer.use_stochastic_rounding,
                            per_channel=False  # Use per-tensor for evaluation
                        )
                    # Copy device and calibration state if needed
                    new_quantizer = new_quantizer.to(old_quantizer.channel_scales.device if old_quantizer.channel_scales is not None else 'cpu')
                    module.weight_quantizers.quantizers[bits] = new_quantizer

                # Replace activation quantizers
                for bits in module.activation_quantizers.bit_widths:
                    old_quantizer = module.activation_quantizers.quantizers[bits]
                    # Create new per-tensor quantizer
                    if module.activation_quantizers.quantizer_type == 'log':
                        new_quantizer = PerChannelLogQuantization(
                            num_bits=bits,
                            symmetric=old_quantizer.symmetric,
                            per_channel=False  # Use per-tensor for evaluation
                        )
                    else:  # sbm
                        new_quantizer = SBMQuantization(
                            num_bits=bits,
                            use_stochastic_rounding=old_quantizer.use_stochastic_rounding,
                            per_channel=False  # Use per-tensor for evaluation
                        )
                    # Copy device
                    new_quantizer = new_quantizer.to(old_quantizer.channel_scales.device if old_quantizer.channel_scales is not None else 'cpu')
                    module.activation_quantizers.quantizers[bits] = new_quantizer


    def generate(self, input_ids, max_length=100, temperature=1.0, do_sample=True, **kwargs):
        """Simple generation method."""
        self.eval()
        batch_size = input_ids.size(0)

        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                outputs = self.forward(input_ids, use_cache=False)
                next_token_logits = outputs.logits[:, -1, :] / temperature

                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Check for EOS token
                if kwargs.get('eos_token_id') is not None:
                    if (next_token == kwargs['eos_token_id']).all():
                        break

        return input_ids
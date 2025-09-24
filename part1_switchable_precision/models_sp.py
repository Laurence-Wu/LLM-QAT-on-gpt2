"""
Switchable Precision GPT-2 Model
Supports multiple bit-widths simultaneously with separate LoRA adapters per precision.
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
    from .lora import SPLinearWithLoRA
    from .switchable_batchnorm import SwitchableLayerNorm
except ImportError:
    from quantization import LearnableFakeQuantize
    from lora import SPLinearWithLoRA
    from switchable_batchnorm import SwitchableLayerNorm


class SPAttention(nn.Module):
    """Attention module with switchable precision for multiple bit-widths."""

    def __init__(self, config: GPT2Config, bit_widths):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        self.bit_widths = bit_widths

        # Get LoRA configs from config - throw error if not provided
        lora_rank_per_bit = config.lora_rank_per_bit
        lora_alpha_per_bit = config.lora_alpha_per_bit
        quantizer_per_bit = config.quantizer_per_bit

        # Switchable layers with per-bit-width LoRA modules
        self.c_attn = SPLinearWithLoRA(
            config.n_embd, 3 * config.n_embd,
            bit_widths=bit_widths,
            lora_rank_per_bit=lora_rank_per_bit,
            lora_alpha_per_bit=lora_alpha_per_bit,
            quantizer_per_bit=quantizer_per_bit
        )
        self.c_proj = SPLinearWithLoRA(
            config.n_embd, config.n_embd,
            bit_widths=bit_widths,
            lora_rank_per_bit=lora_rank_per_bit,
            lora_alpha_per_bit=lora_alpha_per_bit,
            quantizer_per_bit=quantizer_per_bit
        )

        ## use this bias mask as the attention mask.
        self.register_buffer("bias", torch.tril(torch.ones(config.n_positions, config.n_positions)))

    def set_precision(self, bits) -> int:
        self.current_bit_width = bits
        self.c_attn.set_precision(bits)
        self.c_proj.set_precision(bits)
        return self.current_bit_width

    def forward(self, hidden_states, attention_mask=None):
        #standard attention implementation
        B, T, C = hidden_states.shape

        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Quantize KV cache (bypass for 16-bit to match GPT-2 exactly)

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

    def __init__(self, config: GPT2Config, bit_widths=None):
        super().__init__()

        # Get bit_widths from config if not provided
        if bit_widths is None:
            bit_widths = getattr(config, 'bit_widths', [6, 8, 16, 32])
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
        quantizer_per_bit = getattr(config, 'quantizer_per_bit', None)

        # Switchable layers with per-bit-width LoRA modules
        self.c_fc = SPLinearWithLoRA(
            config.n_embd, 4 * config.n_embd,
            bit_widths=bit_widths,
            lora_rank_per_bit=lora_rank_per_bit,
            lora_alpha_per_bit=lora_alpha_per_bit,
            quantizer_per_bit=quantizer_per_bit
        )
        self.c_proj = SPLinearWithLoRA(
            4 * config.n_embd, config.n_embd,
            bit_widths=bit_widths,
            lora_rank_per_bit=lora_rank_per_bit,
            lora_alpha_per_bit=lora_alpha_per_bit,
            quantizer_per_bit=quantizer_per_bit
        )
        self.act = nn.GELU()

    def set_precision(self, bits) -> int:
        """Set precision for all layers to specified bit-width.

        Returns:
            int: Current precision after setting
        """
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
    """Transformer block with switchable precision."""

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
    """GPT-2 model with switchable precision training support."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # Get bit widths from config
        self.bit_widths = config.bit_widths
        self.current_bit_width = max(self.bit_widths)  # Start with highest precision since we are distilling it

        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer blocks with switchable precision
        self.h = nn.ModuleList([
            SPBlock(config, bit_widths=self.bit_widths)
            for _ in range(config.n_layer)
        ])

        # Use Switchable LayerNorm for final layer
        self.ln_f = SwitchableLayerNorm(
            config.n_embd,
            precision_levels=self.bit_widths,
            eps=config.layer_norm_epsilon
        )
    
    def unfreeze_weights(self, bits):
        if bits == 32:
            # Unfreeze transformer block weights for teacher training
            for block in self.h:
                # Layer normalizations (SwitchableLayerNorm has weights and biases dicts)
                if isinstance(block.ln_1, SwitchableLayerNorm):
                    for precision in block.ln_1.precision_levels:
                        block.ln_1.weights[str(precision)].requires_grad = True
                        block.ln_1.biases[str(precision)].requires_grad = True
                if isinstance(block.ln_2, SwitchableLayerNorm):
                    for precision in block.ln_2.precision_levels:
                        block.ln_2.weights[str(precision)].requires_grad = True
                        block.ln_2.biases[str(precision)].requires_grad = True

                # Attention weights
                block.attn.c_attn.linear.weight.requires_grad = True
                block.attn.c_attn.linear.bias.requires_grad = True
                block.attn.c_proj.linear.weight.requires_grad = True
                block.attn.c_proj.linear.bias.requires_grad = True

                # MLP weights
                block.mlp.c_fc.linear.weight.requires_grad = True
                block.mlp.c_fc.linear.bias.requires_grad = True
                block.mlp.c_proj.linear.weight.requires_grad = True
                block.mlp.c_proj.linear.bias.requires_grad = True

            # Final layer norm (SwitchableLayerNorm)
            if isinstance(self.ln_f, SwitchableLayerNorm):
                for precision in self.ln_f.precision_levels:
                    self.ln_f.weights[str(precision)].requires_grad = True
                    self.ln_f.biases[str(precision)].requires_grad = True
    
    def set_precision(self, bits) -> int:
        if bits not in self.bit_widths:
            raise ValueError(f"Bit width {bits} not in configured widths {self.bit_widths}")
        self.current_bit_width = bits
        # Handle weight freezing/unfreezing based on precision
        # 32-bit teacher: Unfreeze all base weights (but not embeddings)
        # Other precisions: Keep base weights frozen, only train LoRA

        for block in self.h:
            block.set_precision(bits)

        # Set precision for final layer norm
        self.ln_f.set_precision(bits)

        return self.current_bit_width

    def disable_lora_for_calibration(self):
        """Disable LoRA during calibration by setting calibration_mode=True on all SPLinearWithLoRA layers."""
        for module in self.modules():
            if module.__class__.__name__ == 'SPLinearWithLoRA':
                module.calibration_mode = True

    def enable_lora_after_calibration(self):
        """Re-enable LoRA after calibration by setting calibration_mode=False on all SPLinearWithLoRA layers."""
        for module in self.modules():
            if module.__class__.__name__ == 'SPLinearWithLoRA':
                module.calibration_mode = False

    def verify_precision_consistency(self) -> Tuple[bool, Dict]:
        """Verify all components are at the same precision.

        Returns:
            Tuple[bool, Dict]: (is_consistent, details) where details contains:
                - expected: The expected precision
                - mismatches: List of components with wrong precision
                - components: Dict of all component precisions
        """
        details = {
            'expected': self.current_bit_width,
            'mismatches': [],
            'components': {}
        }

        # Check each block
        for i, block in enumerate(self.h):
            # Check LayerNorms
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

            # Check attention
            if hasattr(block.attn, 'current_bit_width'):
                attn_prec = block.attn.current_bit_width
                details['components'][f'block_{i}_attn'] = attn_prec
                if attn_prec != self.current_bit_width:
                    details['mismatches'].append(f'block_{i}_attn: {attn_prec} (expected {self.current_bit_width})')

            # Check MLP components
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

        # Check final layer norm
        if isinstance(self.ln_f, SwitchableLayerNorm):
            lnf_prec = self.ln_f.current_precision
            details['components']['ln_f'] = lnf_prec
            if lnf_prec != self.current_bit_width:
                details['mismatches'].append(f'ln_f: {lnf_prec} (expected {self.current_bit_width})')

        is_consistent = len(details['mismatches']) == 0
        return is_consistent, details

    def get_current_precision(self):
        """Get current precision setting."""
        return self.current_bit_width


    def forward(self, input_ids, attention_mask=None, use_checkpoint=False,
                output_hidden_states=False):
        """
        Modified forward pass to support feature extraction for distillation.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            use_checkpoint: Use gradient checkpointing
            output_hidden_states: Return intermediate hidden states

        Returns:
            If output_hidden_states=False: final hidden states
            If output_hidden_states=True: (final hidden states, all hidden states list)
        """
        device = input_ids.device
        B, T = input_ids.shape

        # Token and position embeddings
        token_embeddings = self.wte(input_ids)
        position_ids = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        position_embeddings = self.wpe(position_ids)

        hidden_states = self.drop(token_embeddings + position_embeddings)

        # Collect hidden states if requested
        all_hidden_states = [] if output_hidden_states else None

        # Pass through transformer blocks
        for i, block in enumerate(self.h):
            if output_hidden_states:
                # Store hidden state AFTER each block
                # Clone and detach to prevent gradient accumulation
                all_hidden_states.append(hidden_states.clone().detach())

            hidden_states = block(hidden_states, attention_mask, use_checkpoint)

            # Memory cleanup
            if i % 4 == 3 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            # Add final hidden state
            all_hidden_states.append(hidden_states.clone().detach())
            return hidden_states, all_hidden_states
        else:
            return hidden_states

    def load_pretrained_weights(self, pretrained_model, device='cuda'):
        """Load weights from pretrained GPT-2 model with S-BN support."""

        # Copy embeddings - frozen (never trained)
        self.wte.weight.data = pretrained_model.wte.weight.data.clone()
        self.wte.weight.requires_grad = False  # Keep frozen

        self.wpe.weight.data = pretrained_model.wpe.weight.data.clone()
        self.wpe.weight.requires_grad = False  # Keep frozen

        # Load transformer blocks with S-BN support
        for i in range(min(len(self.h), len(pretrained_model.h))):
            # Layer normalizations - load into ALL precision-specific layers
            # For SwitchableLayerNorm, copy weights to each precision's LayerNorm
            for ln_key in self.h[i].ln_1.ln_layers:
                self.h[i].ln_1.ln_layers[ln_key].weight.data = pretrained_model.h[i].ln_1.weight.data.clone()
                self.h[i].ln_1.ln_layers[ln_key].bias.data = pretrained_model.h[i].ln_1.bias.data.clone()
                self.h[i].ln_1.ln_layers[ln_key].weight.requires_grad = False
                self.h[i].ln_1.ln_layers[ln_key].bias.requires_grad = False

            for ln_key in self.h[i].ln_2.ln_layers:
                self.h[i].ln_2.ln_layers[ln_key].weight.data = pretrained_model.h[i].ln_2.weight.data.clone()
                self.h[i].ln_2.ln_layers[ln_key].bias.data = pretrained_model.h[i].ln_2.bias.data.clone()
                self.h[i].ln_2.ln_layers[ln_key].weight.requires_grad = False
                self.h[i].ln_2.ln_layers[ln_key].bias.requires_grad = False

            # Attention QKV weights - transpose and freeze by default
            self.h[i].attn.c_attn.linear.weight.data = pretrained_model.h[i].attn.c_attn.weight.data.t().contiguous()
            self.h[i].attn.c_attn.linear.bias.data = pretrained_model.h[i].attn.c_attn.bias.data.clone()
            self.h[i].attn.c_attn.linear.weight.requires_grad = False
            self.h[i].attn.c_attn.linear.bias.requires_grad = False

            # Attention projection weights - transpose and freeze by default
            self.h[i].attn.c_proj.linear.weight.data = pretrained_model.h[i].attn.c_proj.weight.data.t().contiguous()
            self.h[i].attn.c_proj.linear.bias.data = pretrained_model.h[i].attn.c_proj.bias.data.clone()
            self.h[i].attn.c_proj.linear.weight.requires_grad = False
            self.h[i].attn.c_proj.linear.bias.requires_grad = False

            # MLP weights - transpose and freeze by default
            self.h[i].mlp.c_fc.linear.weight.data = pretrained_model.h[i].mlp.c_fc.weight.data.t().contiguous()
            self.h[i].mlp.c_fc.linear.bias.data = pretrained_model.h[i].mlp.c_fc.bias.data.clone()
            self.h[i].mlp.c_fc.linear.weight.requires_grad = False
            self.h[i].mlp.c_fc.linear.bias.requires_grad = False

            self.h[i].mlp.c_proj.linear.weight.data = pretrained_model.h[i].mlp.c_proj.weight.data.t().contiguous()
            self.h[i].mlp.c_proj.linear.bias.data = pretrained_model.h[i].mlp.c_proj.bias.data.clone()
            self.h[i].mlp.c_proj.linear.weight.requires_grad = False
            self.h[i].mlp.c_proj.linear.bias.requires_grad = False

        # Final layer normalization - frozen by default
        # Copy to all precision-specific LayerNorms
        for ln_key in self.ln_f.ln_layers:
            self.ln_f.ln_layers[ln_key].weight.data = pretrained_model.ln_f.weight.data.clone()
            self.ln_f.ln_layers[ln_key].bias.data = pretrained_model.ln_f.bias.data.clone()
            self.ln_f.ln_layers[ln_key].weight.requires_grad = False
            self.ln_f.ln_layers[ln_key].bias.requires_grad = False

        print(f"✅ Loaded pretrained weights with S-BN support")
        print(f"   - All precision-specific LayerNorm layers initialized")

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

    def set_precision(self, bits) -> int:
        precision = self.transformer.set_precision(bits)

        # Verify consistency
        is_consistent, details = self.verify_precision_consistency()
        if not is_consistent:
            print(f"⚠️ Warning: Precision mismatch detected after setting to {bits}-bit:")
            for mismatch in details['mismatches']:
                print(f"   - {mismatch}")

        return precision


    def verify_precision_consistency(self) -> Tuple[bool, Dict]:
        """Verify all components are at the same precision.

        Returns:
            Tuple[bool, Dict]: (is_consistent, details)
        """
        return self.transformer.verify_precision_consistency()

        # Handle LM head weight freezing/unfreezing
        # IMPORTANT: lm_head.weight is tied to transformer.wte.weight
        # We should NOT unfreeze it since embeddings must remain frozen
        # The 32-bit teacher will train the transformer block weights but keep embeddings/lm_head frozen

    def get_current_precision(self):
        """Get current precision setting."""
        return self.transformer.get_current_precision()

    def forward(self, input_ids, labels=None, attention_mask=None,
                use_checkpoint=False, output_hidden_states=False,
                return_dict=False):
        """
        Enhanced forward pass supporting distillation.

        Returns:
            Dictionary with 'loss', 'logits', and optionally 'hidden_states'
        """
        # Get transformer outputs
        if output_hidden_states:
            hidden_states, all_hidden_states = self.transformer(
                input_ids, attention_mask, use_checkpoint,
                output_hidden_states=True
            )
        else:
            hidden_states = self.transformer(
                input_ids, attention_mask, use_checkpoint,
                output_hidden_states=False
            )
            all_hidden_states = None

        # Language model head
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        # Return format for distillation compatibility
        if return_dict or output_hidden_states:
            return {
                'loss': loss,
                'logits': logits,
                'hidden_states': all_hidden_states
            }
        else:
            # Backward compatibility
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
        # Load transformer weights
        self.transformer.load_pretrained_weights(pretrained_model.transformer, device)

        # Load LM head weight (already tied to embeddings via weight sharing)
        # Since weights are tied, this is already loaded through wte
        # But we need to ensure the tie is maintained
        self.lm_head.weight = self.transformer.wte.weight

        print(f"LM head weights tied to token embeddings")
        return self
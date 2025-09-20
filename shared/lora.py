import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Try relative imports first, fall back to direct imports
try:
    from .quantization import LearnableFakeQuantize
except ImportError:
    from quantization import LearnableFakeQuantize

class LoRALayer(nn.Module):
    """LoRA adapter with fake quantization."""
    def __init__(self, in_features, out_features, rank=8, alpha=16, bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.bits = bits

        # CRITICAL: For 16-bit, disable LoRA entirely to ensure exact GPT-2 behavior
        if bits >= 16 or rank <= 0:
            self.enabled = False
            self.scaling = 0
            # Create dummy parameters to avoid parameter counting issues
            self.register_buffer('lora_A', torch.zeros(1, 1))
            self.register_buffer('lora_B', torch.zeros(1, 1))
            # No quantizers for 16-bit (saves memory and avoids calibration issues)
            self.quantize_A = None
            self.quantize_B = None
        else:
            self.enabled = True
            self.scaling = alpha / rank

            # Initialize LoRA matrices - CRITICAL: Must start with zero contribution
            self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
            self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

            # CRITICAL: Initialize A with small values, B with zeros for zero initial contribution
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)  # This ensures zero contribution initially

            # Quantizers for LoRA weights (only for low-bit modes)
            self.quantize_A = LearnableFakeQuantize(num_bits=bits, symmetric=True)
            self.quantize_B = LearnableFakeQuantize(num_bits=bits, symmetric=True)

        # Only allocate buffers if LoRA is enabled to save memory
        if self.enabled:
            self.register_buffer('lora_A_quantized', torch.empty(in_features, rank))
            self.register_buffer('lora_B_quantized', torch.empty(rank, out_features))
        else:
            self.register_buffer('lora_A_quantized', torch.empty(1, 1))
            self.register_buffer('lora_B_quantized', torch.empty(1, 1))

    def forward(self, x):
        if not self.enabled or self.scaling == 0:
            # Return zeros for disabled LoRA (16-bit mode)
            batch_shape = x.shape[:-1]  # All dims except last
            return torch.zeros(*batch_shape, self.out_features, device=x.device, dtype=x.dtype)

        # Quantize LoRA weights (only for enabled low-bit modes)
        lora_A_q = self.quantize_A(self.lora_A)
        lora_B_q = self.quantize_B(self.lora_B)

        # Compute LoRA output: x @ A @ B * scaling
        output = torch.matmul(x, lora_A_q)
        output = torch.matmul(output, lora_B_q)
        output = output * self.scaling
        return output


class LinearWithLoRA(nn.Module):
    """Linear layer with LoRA adapter and quantization."""
    def __init__(self, in_features, out_features, bias=True, bits=8,
                 lora_rank=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.bits = bits
        self.in_features = in_features
        self.out_features = out_features

        # FP32 base layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Fake quantizers
        self.quantize_weight = LearnableFakeQuantize(num_bits=bits, symmetric=True)
        self.quantize_input = LearnableFakeQuantize(num_bits=bits, symmetric=True)

        # LoRA adapter with config parameters
        self.lora = LoRALayer(in_features, out_features,
                                 rank=lora_rank, alpha=lora_alpha, bits=bits)

        # Pre-allocate buffers for quantized tensors
        self.register_buffer('weight_quantized', torch.empty(out_features, in_features))
        self.register_buffer('input_quantized', None)  # Will be allocated on first use

    def forward(self, x):
        # Quantize input and weights - these need gradients
        x_q = self.quantize_input(x)
        w_q = self.quantize_weight(self.linear.weight)

        # Base output + LoRA
        base = F.linear(x_q, w_q, self.linear.bias)
        lora = self.lora(x)

        # Add outputs
        return base + lora

    def set_precision(self, weight_bits, activation_bits):
        """Set the precision (bit-width) for quantization."""
        # Update weight quantizer
        self.quantize_weight.set_num_bits(weight_bits)
        try:
            if self.quantize_weight.symmetric:
                self.quantize_weight.quant_min = -(2 ** (weight_bits - 1))
                self.quantize_weight.quant_max = 2 ** (weight_bits - 1) - 1
            else:
                self.quantize_weight.quant_min = 0
                self.quantize_weight.quant_max = 2 ** weight_bits - 1
        except AttributeError:
            pass  # quant_min/max attributes don't exist

        # Update activation quantizer
        self.quantize_input.set_num_bits(activation_bits)
        try:
            if self.quantize_input.symmetric:
                self.quantize_input.quant_min = -(2 ** (activation_bits - 1))
                self.quantize_input.quant_max = 2 ** (activation_bits - 1) - 1
            else:
                self.quantize_input.quant_min = 0
                self.quantize_input.quant_max = 2 ** activation_bits - 1
        except AttributeError:
            pass  # quant_min/max attributes don't exist

        # Update LoRA quantizers if they exist
        try:
            self.lora.quantize_A.set_num_bits(weight_bits)
            self.lora.quantize_B.set_num_bits(weight_bits)
        except AttributeError:
            pass  # LoRA quantizers not present


class SPLinearWithLoRA(nn.Module):
    """Linear layer with multiple LoRA adapters for switchable precision."""

    def __init__(self, in_features, out_features, bias=True,
                 bit_widths=[4, 8, 16],
                 lora_rank_per_bit={4: 32, 8: 16, 16: 8},
                 lora_alpha_per_bit={4: 64, 8: 32, 16: 16},
                 lora_dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bit_widths = bit_widths
        self.current_bits = bit_widths[1]  # Default to 8-bit

        # FP32 base layer (shared across all bit-widths)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Create separate quantizers for each bit-width
        self.quantizers_weight = nn.ModuleDict({
            f'{bits}bit': LearnableFakeQuantize(num_bits=bits, symmetric=True)
            for bits in bit_widths
        })
        self.quantizers_input = nn.ModuleDict({
            f'{bits}bit': LearnableFakeQuantize(num_bits=bits, symmetric=True)
            for bits in bit_widths
        })

        # Create separate LoRA adapters for each bit-width
        # CRITICAL: 16-bit LoRA must be disabled to ensure exact GPT-2 behavior
        self.lora_adapters = nn.ModuleDict({
            f'{bits}bit': LoRALayer(
                in_features, out_features,
                rank=lora_rank_per_bit.get(bits, 16) if bits < 16 else 0,  # Force rank=0 for 16-bit
                alpha=lora_alpha_per_bit.get(bits, 32),
                bits=bits
            )
            for bits in bit_widths
        })

        # Pre-allocate buffers for quantized tensors
        self.register_buffer('weight_quantized', torch.empty(out_features, in_features))
        self.register_buffer('input_quantized', None)

    def set_precision(self, bits):
        """Switch to specified bit-width."""
        if bits not in self.bit_widths:
            raise ValueError(f"Bit-width {bits} not supported. Choose from {self.bit_widths}")
        self.current_bits = bits

        # CRITICAL: Update the num_bits for the active quantizers
        # This ensures they actually quantize at the right precision
        bits_key = f'{bits}bit'

        # Ensure quantizers exist for this precision
        if bits_key not in self.quantizers_weight:
            raise ValueError(f"Weight quantizer for {bits}-bit not found. Available: {list(self.quantizers_weight.keys())}")
        if bits_key not in self.quantizers_input:
            raise ValueError(f"Input quantizer for {bits}-bit not found. Available: {list(self.quantizers_input.keys())}")
        if bits_key not in self.lora_adapters:
            raise ValueError(f"LoRA adapter for {bits}-bit not found. Available: {list(self.lora_adapters.keys())}")

        # Update quantizers
        self.quantizers_weight[bits_key].set_num_bits(bits)
        self.quantizers_input[bits_key].set_num_bits(bits)

        # Update LoRA adapter quantizers
        lora = self.lora_adapters[bits_key]
        if lora.quantize_A is not None:
            lora.quantize_A.set_num_bits(bits)
        if lora.quantize_B is not None:
            lora.quantize_B.set_num_bits(bits)

    def get_active_lora(self):
        """Get the currently active LoRA adapter."""
        return self.lora_adapters[f'{self.current_bits}bit']

    def start_stats_collection(self):
        """Start Pass 1: Begin collecting statistics for current bit-width."""
        if self.current_bits >= 16:
            return  # No quantization for 16-bit

        bits_key = f'{self.current_bits}bit'
        self.quantizers_weight[bits_key].start_stats_collection()
        self.quantizers_input[bits_key].start_stats_collection()

    def stop_stats_collection(self):
        """End Pass 1: Finalize statistics and freeze quantization parameters."""
        if self.current_bits >= 16:
            return  # No quantization for 16-bit

        bits_key = f'{self.current_bits}bit'
        self.quantizers_weight[bits_key].stop_stats_collection()
        self.quantizers_input[bits_key].stop_stats_collection()

    def unfreeze_stats(self):
        """End Pass 2: Allow statistics to be updated again."""
        if self.current_bits >= 16:
            return  # No quantization for 16-bit

        bits_key = f'{self.current_bits}bit'
        self.quantizers_weight[bits_key].unfreeze_stats()
        self.quantizers_input[bits_key].unfreeze_stats()

    def forward(self, x):
        """
        Forward pass with bit-width-specific behavior:
        - 16-bit: Direct computation (no quantization, no LoRA)
        - 8/4-bit: Quantization + LoRA compensation
        """
        # CRITICAL: 16-bit uses pure FP32 weights without modifications
        if self.current_bits >= 16:
            return F.linear(x, self.linear.weight, self.linear.bias)

        # Lower precision: Apply quantization and LoRA
        bits_key = f'{self.current_bits}bit'

        # DEBUG: Verify quantizers exist
        if bits_key not in self.quantizers_weight:
            print(f"ERROR: Missing weight quantizer for {bits_key}")
            print(f"Available: {list(self.quantizers_weight.keys())}")
            raise KeyError(f"No weight quantizer for {bits_key}")

        if bits_key not in self.quantizers_input:
            print(f"ERROR: Missing input quantizer for {bits_key}")
            print(f"Available: {list(self.quantizers_input.keys())}")
            raise KeyError(f"No input quantizer for {bits_key}")

        weight_quantizer = self.quantizers_weight[bits_key]
        input_quantizer = self.quantizers_input[bits_key]
        active_lora = self.lora_adapters[bits_key]

        # DEBUG: Check quantizer state before calling
        if not hasattr(self, '_debug_logged'):
            self._debug_logged = True
            print(f"\nDEBUG SPLinearWithLoRA.forward():")
            print(f"  current_bits: {self.current_bits}")
            print(f"  bits_key: {bits_key}")
            print(f"  weight_quantizer.num_bits: {weight_quantizer.num_bits}")
            print(f"  weight_quantizer.training: {weight_quantizer.training}")
            print(f"  weight_quantizer.calibrated (before): {weight_quantizer.calibrated}")
            print(f"  input_quantizer.num_bits: {input_quantizer.num_bits}")
            print(f"  input_quantizer.calibrated (before): {input_quantizer.calibrated}")

        # Quantize inputs and weights
        x_quantized = input_quantizer(x)
        weight_quantized = weight_quantizer(self.linear.weight)

        # DEBUG: Check if calibration happened and values are valid
        if not hasattr(self, '_calib_logged'):
            self._calib_logged = True
            print(f"  weight_quantizer.calibrated (after): {weight_quantizer.calibrated}")
            print(f"  input_quantizer.calibrated (after): {input_quantizer.calibrated}")
            if hasattr(weight_quantizer, 'scale'):
                print(f"  weight_quantizer.scale shape: {weight_quantizer.scale.shape}")
                print(f"  weight_quantizer.scale mean: {weight_quantizer.scale.mean().item():.6f}")
            if hasattr(input_quantizer, 'scale'):
                print(f"  input_quantizer.scale shape: {input_quantizer.scale.shape}")
                print(f"  input_quantizer.scale mean: {input_quantizer.scale.mean().item():.6f}")

            # Check for NaN/Inf
            if torch.isnan(x_quantized).any():
                print(f"  WARNING: NaN detected in x_quantized!")
            if torch.isinf(x_quantized).any():
                print(f"  WARNING: Inf detected in x_quantized!")
            if torch.isnan(weight_quantized).any():
                print(f"  WARNING: NaN detected in weight_quantized!")
            if torch.isinf(weight_quantized).any():
                print(f"  WARNING: Inf detected in weight_quantized!")

        # Base computation with quantized values
        base_output = F.linear(x_quantized, weight_quantized, self.linear.bias)

        # Add LoRA compensation for quantization errors
        lora_output = active_lora(x)

        return base_output + lora_output

    def get_all_parameters(self):
        """Get all parameters (base + all LoRAs) for optimizer."""
        params = list(self.linear.parameters())
        for lora in self.lora_adapters.values():
            params.extend(lora.parameters())
        for quantizer in self.quantizers_weight.values():
            params.extend(quantizer.parameters())
        for quantizer in self.quantizers_input.values():
            params.extend(quantizer.parameters())
        return params

    def get_active_parameters(self):
        """Get only parameters for current bit-width (for selective updates)."""
        bits_key = f'{self.current_bits}bit'
        params = list(self.linear.parameters())
        params.extend(self.lora_adapters[bits_key].parameters())
        params.extend(self.quantizers_weight[bits_key].parameters())
        params.extend(self.quantizers_input[bits_key].parameters())
        return params




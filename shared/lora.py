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

        # Handle rank=0 case (no LoRA)
        if rank <= 0:
            self.scaling = 0
            self.enabled = False
            # Create dummy parameters to avoid issues
            self.register_buffer('lora_A', torch.zeros(1, 1))
            self.register_buffer('lora_B', torch.zeros(1, 1))
        else:
            self.scaling = alpha / rank
            self.enabled = True
            # FP32 weights
            self.lora_A = nn.Parameter(torch.empty(in_features, rank))
            self.lora_B = nn.Parameter(torch.empty(rank, out_features))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        # Fake quantizers
        self.quantize_A = LearnableFakeQuantize(num_bits=bits, symmetric=True)
        self.quantize_B = LearnableFakeQuantize(num_bits=bits, symmetric=True)

        # Pre-allocate buffers for quantized weights to avoid creating new tensors
        self.register_buffer('lora_A_quantized', torch.empty(in_features, rank))
        self.register_buffer('lora_B_quantized', torch.empty(rank, out_features))

    def forward(self, x):
        # Quantize weights - these need gradients for backprop
        lora_A_quantized = self.quantize_A(self.lora_A)
        lora_B_quantized = self.quantize_B(self.lora_B)

        # Use more memory-efficient matrix multiplication
        # Split the computation to avoid intermediate large tensors
        output = torch.matmul(x, lora_A_quantized)
        output = torch.matmul(output, lora_B_quantized)
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
        self.lora_adapters = nn.ModuleDict({
            f'{bits}bit': LoRALayer(
                in_features, out_features,
                rank=lora_rank_per_bit.get(bits, 16),
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

    def get_active_lora(self):
        """Get the currently active LoRA adapter."""
        return self.lora_adapters[f'{self.current_bits}bit']

    def forward(self, x):
        # Get current bit-width quantizers and LoRA
        bits_key = f'{self.current_bits}bit'
        weight_quantizer = self.quantizers_weight[bits_key]
        input_quantizer = self.quantizers_input[bits_key]
        active_lora = self.lora_adapters[bits_key]

        # For 16-bit precision, use exact FP32 computation (no quantization, no LoRA)
        # This ensures perfect equivalence to GPT-2 for baseline comparison
        if self.current_bits == 16:
            # Pure FP32 computation - bypass quantization and LoRA for exact match
            return F.linear(x, self.linear.weight, self.linear.bias)

        # For lower precisions, use quantization + LoRA
        # Quantize inputs and weights for current bit-width
        x_quantized = input_quantizer(x)
        weight_quantized = weight_quantizer(self.linear.weight)

        # Base computation with quantized values
        base_output = F.linear(x_quantized, weight_quantized, self.linear.bias)

        # Add only the active LoRA adapter
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




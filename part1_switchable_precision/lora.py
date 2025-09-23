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
    def __init__(self, in_features, out_features, rank, alpha, bits, quantizer_type, eps=1e-5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.bits = bits

        if bits >= 32 or rank <= 0:
            self.enabled = False
            self.scaling = 0
            self.register_buffer('lora_A', torch.zeros(1, 1))
            self.register_buffer('lora_B', torch.zeros(1, 1))
            self.quantize_A = None
            self.quantize_B = None

        else:
            self.enabled = True
            self.scaling = alpha / rank

            self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
            self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            # A: [rank, in_features] -> quantize along in_features (dim=1)
            self.quantize_A = LearnableFakeQuantize(num_bits=bits, quantizer_type=quantizer_type, channel_dim=1, eps=eps)
            # B: [out_features, rank] -> quantize along out_features (dim=0)
            self.quantize_B = LearnableFakeQuantize(num_bits=bits, quantizer_type=quantizer_type, channel_dim=0, eps=eps)
            
            self.register_buffer('lora_A_quantized', torch.empty(in_features, rank))
            self.register_buffer('lora_B_quantized', torch.empty(rank, out_features))

    def forward(self, x):
        if not self.enabled or self.scaling == 0:
            # Return zeros for disabled LoRA (32-bit teacher mode)
            batch_shape = x.shape[:-1]  # All dims except last
            return torch.zeros(*batch_shape, self.out_features, device=x.device, dtype=x.dtype)
        lora_A_q = self.quantize_A(self.lora_A)
        lora_B_q = self.quantize_B(self.lora_B)
        output = torch.matmul(x, lora_A_q)
        output = torch.matmul(output, lora_B_q)
        output = output * self.scaling
        return output


class SPLinearWithLoRA(nn.Module):
    """Linear layer with multiple LoRA adapters for switchable precision."""

    def __init__(self, in_features, out_features,
                 bit_widths,
                 lora_rank_per_bit,
                 lora_alpha_per_bit,
                 quantizer_per_bit,
                 eps=1e-5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bit_widths = bit_widths
        self.lora_rank_per_bit = lora_rank_per_bit
        # Default to middle precision (skip 32-bit teacher)
        student_bits = [b for b in bit_widths if b < 32]
        self.current_bits = sorted(bit_widths, reverse=True)[1] # biggest student bit-width to start with

        # FP32 teacher layer
        self.linear = nn.Linear(in_features, out_features, bias=True)

        # Weight quantizers: [out_features, in_features] -> channel_dim=0 (per output channel)
        self.quantizers_weight = nn.ModuleDict({
            f'{bits}bit': LearnableFakeQuantize(num_bits=bits,
                                                quantizer_type=quantizer_per_bit[bits], channel_dim=0, eps=eps)
            for bits in student_bits
        })
        # Input quantizers: [batch, in_features] -> channel_dim=1 (per feature)
        self.quantizers_input = nn.ModuleDict({
            f'{bits}bit': LearnableFakeQuantize(num_bits=bits,
                                                quantizer_type=quantizer_per_bit[bits], channel_dim=1, eps=eps)
            for bits in student_bits
        })

        # Create separate LoRA adapters for each bit-width
        # CRITICAL: 32-bit LoRA must be disabled for teacher (no quantization)
        self.lora_adapters = nn.ModuleDict({
            f'{bits}bit': LoRALayer(
                in_features, out_features,
                rank=lora_rank_per_bit[bits],
                alpha=lora_alpha_per_bit[bits],
                bits=bits,
                quantizer_type=quantizer_per_bit[bits],
                eps=eps
            )
            for bits in student_bits
        })

        # Pre-allocate buffers for quantized tensors
        self.register_buffer('weight_quantized', torch.empty(out_features, in_features))
        self.register_buffer('input_quantized', None)

    def set_precision(self, bits) -> int:
        self.current_bits = bits
        bits_key = f'{bits}bit'
        # Update quantizers
        self.quantizers_weight[bits_key].set_num_bits(bits)
        self.quantizers_input[bits_key].set_num_bits(bits)

        # Update LoRA adapter quantizers
        lora = self.lora_adapters[bits_key]
        if lora.quantize_A is not None:
            lora.quantize_A.set_num_bits(bits)
        if lora.quantize_B is not None:
            lora.quantize_B.set_num_bits(bits)

        return self.current_bits

    def get_active_lora(self):
        """Get the currently active LoRA adapter."""
        return self.lora_adapters[f'{self.current_bits}bit']


    def forward(self, x):
        """
        Forward pass with bit-width-specific behavior:
        - 32-bit: Pure FP32 teacher (no quantization, no LoRA)
        - 16-bit: Student with quantization and LoRA
        - 8/-bit: Student with stronger quantization and LoRA
        """
        # Forward for the 32 bits is handled here. Use student bit width for students.
        if self.current_bits >= 32:
            output = F.linear(x, self.linear.weight, self.linear.bias)
            return output

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
        x_quantized = input_quantizer(x)
        weight_quantized = weight_quantizer(self.linear.weight)

        # Base computation with quantized values
        base_output = F.linear(x_quantized, weight_quantized, self.linear.bias)
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




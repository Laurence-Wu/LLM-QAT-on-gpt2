import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Try relative imports first, fall back to direct imports
try:
    from .quantization import LearnableFakeQuantize
except ImportError:
    from quantization import LearnableFakeQuantize

class QATLoRALayer(nn.Module):
    """Simple QAT LoRA: FP32 weights with fake quantization during forward."""
    def __init__(self, in_features, out_features, rank=8, alpha=16, bits=8):
        super().__init__()
        self.scaling = alpha / rank
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

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

class QATLinearWithLoRA(nn.Module):
    """QAT Linear layer: FP32 weights, fake quantization, LoRA adaptation."""
    def __init__(self, in_features, out_features, bias=True, bits=8):
        super().__init__()
        self.bits = bits
        self.in_features = in_features
        self.out_features = out_features

        # FP32 base layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Fake quantizers
        self.quantize_weight = LearnableFakeQuantize(num_bits=bits, symmetric=True)
        self.quantize_input = LearnableFakeQuantize(num_bits=bits, symmetric=False)

        # LoRA adapter
        rank = max(4, min(16, in_features // 64))
        self.lora = QATLoRALayer(in_features, out_features, rank=rank, alpha=rank*2, bits=bits)

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
        """For compatibility - Part 1 uses fixed precision."""
        pass

# Alias for compatibility
QuantizedLinearWithLoRA = QATLinearWithLoRA




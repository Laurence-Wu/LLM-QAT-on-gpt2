import torch
import torch.nn as nn
import math
from typing import List, Optional

try:
    from .quantization import QuantizedLinear
except ImportError:
    from quantization import QuantizedLinear

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank if rank > 0 else 1.0
        
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.empty(rank, out_features))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling

class MultiPrecisionLoRA(nn.Module):
    def __init__(self, in_features, out_features, bit_widths=None):
        super().__init__()
        if bit_widths is None:
            bit_widths = [4, 8, 16]
        self.bit_widths = bit_widths
        
        self.lora_modules = nn.ModuleDict()
        for bits in bit_widths:
            if bits <= 4:
                rank = max(1, min(4, in_features // 128))
            elif bits <= 8:
                rank = max(2, min(8, in_features // 64))
            else:  # 16-bit
                rank = max(4, min(16, in_features // 32))
            
            alpha = rank * bits // 2
            
            self.lora_modules[f'lora_{bits}bit'] = LoRALayer(
                in_features, out_features, rank=rank, alpha=alpha
            )
        
        self.current_bits = 8
        
    def forward(self, x, bits=None):
        if bits is None:
            bits = self.current_bits
        
        key = f'lora_{bits}bit'
        if key in self.lora_modules:
            return self.lora_modules[key](x)
        else:
            nearest_bits = min(self.bit_widths, key=lambda b: abs(b - bits))
            return self.lora_modules[f'lora_{nearest_bits}bit'](x)
    
    def set_bits(self, bits):
        self.current_bits = bits

class QuantizedLinearWithLoRA(nn.Module):
    def __init__(self, in_features, out_features, bias=True, bit_widths=None):
        super().__init__()
        if bit_widths is None:
            bit_widths = [4, 8, 16]
            
        self.quantized_linear = QuantizedLinear(
            in_features, out_features, bias=bias,
            weight_bits=8, activation_bits=8
        )
        
        self.lora = MultiPrecisionLoRA(in_features, out_features, bit_widths)
        self.current_weight_bits = 8
        self.current_activation_bits = 8
        
    def forward(self, x):
        base_output = self.quantized_linear(x)
        lora_output = self.lora(x, bits=self.current_weight_bits)
        return base_output + lora_output
    
    def set_precision(self, weight_bits, activation_bits):
        self.current_weight_bits = weight_bits
        self.current_activation_bits = activation_bits
        self.lora.set_bits(weight_bits)
        
        # Properly handle FP32 by disabling quantization
        if weight_bits >= 32:
            # For FP32, set quantizer to passthrough mode
            self.quantized_linear.weight_quantizer.num_bits = 32
            self.quantized_linear.weight_quantizer.calibrated = False
        else:
            self.quantized_linear.weight_quantizer.num_bits = max(1, min(weight_bits, 32))
            
        if activation_bits >= 32:
            self.quantized_linear.activation_quantizer.num_bits = 32
            self.quantized_linear.activation_quantizer.calibrated = False
        else:
            self.quantized_linear.activation_quantizer.num_bits = max(1, min(activation_bits, 32))
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from .quantization import LearnableFakeQuantize
except ImportError:
    from quantization import LearnableFakeQuantize

class QATLoRALayer(nn.Module):
    """Simple QAT LoRA: FP32 weights with fake quantization during forward."""
    def __init__(self, in_features, out_features, rank=8, alpha=16, bits=8):
        super().__init__()
        self.scaling = alpha / rank
        
        # FP32 weights
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.empty(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Fake quantizers
        self.quantize_A = LearnableFakeQuantize(num_bits=bits, symmetric=True)
        self.quantize_B = LearnableFakeQuantize(num_bits=bits, symmetric=True)
        
    def forward(self, x):
        lora_A = self.quantize_A(self.lora_A.to(x.device))
        lora_B = self.quantize_B(self.lora_B.to(x.device))
        return (x @ lora_A @ lora_B) * self.scaling

class QATLinearWithLoRA(nn.Module):
    """QAT Linear layer: FP32 weights, fake quantization, LoRA adaptation."""
    def __init__(self, in_features, out_features, bias=True, bits=8):
        super().__init__()
        self.bits = bits
        
        # FP32 base layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Fake quantizers
        self.quantize_weight = LearnableFakeQuantize(num_bits=bits, symmetric=True)
        self.quantize_input = LearnableFakeQuantize(num_bits=bits, symmetric=False)
        
        # LoRA adapter
        rank = max(4, min(16, in_features // 64))
        self.lora = QATLoRALayer(in_features, out_features, rank=rank, alpha=rank*2, bits=bits)
        
    def forward(self, x):
        # Quantize input and weights
        x_q = self.quantize_input(x)
        w_q = self.quantize_weight(self.linear.weight)
        
        # Base output + LoRA
        base = F.linear(x_q, w_q, self.linear.bias)
        lora = self.lora(x)
        return base + lora
    
    def set_precision(self, weight_bits, activation_bits):
        """For compatibility - Part 1 uses fixed precision."""
        pass

# Alias for compatibility
QuantizedLinearWithLoRA = QATLinearWithLoRA




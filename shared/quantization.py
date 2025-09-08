import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class QuantizationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, num_bits, symmetric):
        ctx.save_for_backward(input, scale, zero_point)
        ctx.num_bits = num_bits
        ctx.symmetric = symmetric
        
        # Symmetric MinMax quantization without clipping (following LLM-QAT paper)
        if symmetric:
            # Paper formula: X_Q = α⌊X_R/α⌉, α = max(|X_R|)/(2^(N-1) - 1)
            quantized = torch.round(input / scale)
            output = quantized * scale
            # NO CLIPPING - preserve outliers as recommended in paper
        else:
            output = torch.round(input / scale + zero_point)
            output = torch.clamp(output, 0, 2**num_bits - 1)
            output = (output - zero_point) * scale
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, scale, zero_point = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None

class LearnableFakeQuantize(nn.Module):
    def __init__(self, num_bits=8, symmetric=True, per_channel=False, channel_dim=0):
        super().__init__()
        self.num_bits = max(1, min(num_bits, 32))  # Clamp to valid range
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_dim = channel_dim
        self.eps = 1e-7
        
        self.quant_min = 0
        self.quant_max = 2 ** self.num_bits - 1
        
        if symmetric:
            self.quant_min = -(2 ** (self.num_bits - 1))
            self.quant_max = 2 ** (self.num_bits - 1) - 1
        
        # Store quantization statistics on CPU to save GPU memory
        # These will be moved to GPU only when needed for computation
        if per_channel:
            self.scale = torch.ones(1)  # Will be resized during first forward pass
            self.zero_point = torch.zeros(1)
            self.running_min = torch.zeros(1)
            self.running_max = torch.zeros(1)
        else:
            self.scale = torch.ones(1)
            self.zero_point = torch.zeros(1) 
            self.running_min = torch.zeros(1)
            self.running_max = torch.zeros(1)
        
        self.calibrated = False
    
    def forward(self, x):
        # Pass through for FP32 mode
        if self.num_bits >= 32:
            return x
            
        if self.training:
            if self.per_channel:
                # Per-channel statistics - calculate min/max along non-channel dimensions
                dims_to_reduce = list(range(x.dim()))
                dims_to_reduce.remove(self.channel_dim)
                
                # Calculate min/max by iteratively reducing dimensions
                min_val = x
                max_val = x
                for dim in sorted(dims_to_reduce, reverse=True):  # Start from highest dim
                    min_val = min_val.min(dim=dim, keepdim=True)[0]
                    max_val = max_val.max(dim=dim, keepdim=True)[0]
                
                # Store statistics on CPU to save GPU memory
                if not self.calibrated:
                    self.running_min = min_val.cpu().clone()  # Move to CPU
                    self.running_max = max_val.cpu().clone()  # Move to CPU
                    self.scale = torch.ones_like(self.running_min)  # Keep on CPU
                    self.zero_point = torch.zeros_like(self.running_min)  # Keep on CPU
                    self.calibrated = True
                else:
                    # Update on CPU
                    min_cpu = min_val.cpu()
                    max_cpu = max_val.cpu()
                    self.running_min = self.running_min * 0.9 + min_cpu * 0.1
                    self.running_max = self.running_max * 0.9 + max_cpu * 0.1
            else:
                # Per-tensor statistics - store on CPU
                min_val = x.min().cpu()  # Move to CPU
                max_val = x.max().cpu()  # Move to CPU
                
                if not self.calibrated:
                    self.running_min = min_val.clone()
                    self.running_max = max_val.clone()
                    self.calibrated = True
                else:
                    self.running_min = self.running_min * 0.9 + min_val * 0.1
                    self.running_max = self.running_max * 0.9 + max_val * 0.1
        
        if self.symmetric:
            # Paper formula: α = max(|X_R|)/(2^(N-1) - 1)
            max_val = torch.max(torch.abs(self.running_min), torch.abs(self.running_max))
            max_val = torch.clamp(max_val, min=self.eps)
            self.scale = max_val / (2**(self.num_bits-1) - 1)
            self.zero_point = torch.zeros_like(self.scale)
        else:
            range_val = torch.clamp(self.running_max - self.running_min, min=self.eps)
            denom = self.quant_max - self.quant_min
            if denom > 0:
                self.scale = range_val / denom
                if torch.any(self.scale > 0):
                    self.zero_point = torch.round(self.quant_min - self.running_min / self.scale)
                else:
                    self.zero_point = torch.zeros_like(self.zero_point)
            else:
                self.scale = torch.ones_like(self.scale)
                self.zero_point = torch.zeros_like(self.zero_point)
        
        # Move scale and zero_point to GPU only when needed for computation
        # Use detach() to break any potential gradient links
        scale_gpu = self.scale.detach().to(x.device)
        zero_point_gpu = self.zero_point.detach().to(x.device)
        
        return QuantizationFunction.apply(x, scale_gpu, zero_point_gpu, 
                                         self.num_bits, self.symmetric)

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, 
                 weight_bits=8, activation_bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Per-channel weight quantization, per-token activation quantization (paper approach)
        self.weight_quantizer = LearnableFakeQuantize(weight_bits, symmetric=True, per_channel=True, channel_dim=0)
        self.activation_quantizer = LearnableFakeQuantize(activation_bits, symmetric=True)  # Paper uses symmetric for both
        
    def forward(self, input):
        input_q = self.activation_quantizer(input)
        weight_q = self.weight_quantizer(self.weight)
        return F.linear(input_q, weight_q, self.bias)

class CyclicPrecisionScheduler:
    def __init__(self, min_bits=4, max_bits=8, cycle_length=200, warmup_steps=50):
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.cycle_length = cycle_length
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def get_bits(self):
        if self.current_step < self.warmup_steps:
            return self.max_bits
        
        adjusted_step = self.current_step - self.warmup_steps
        cycle_position = (adjusted_step % self.cycle_length) / max(self.cycle_length, 1)
        cos_value = (1 + math.cos(math.pi * cycle_position)) / 2
        current_bits = self.min_bits + (self.max_bits - self.min_bits) * cos_value
        
        return int(round(current_bits))
    
    def step(self):
        self.current_step += 1
        return self.get_bits()
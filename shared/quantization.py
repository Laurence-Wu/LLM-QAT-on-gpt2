import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class QuantizationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, num_bits, symmetric):
        ctx.save_for_backward(input, scale.clone(), zero_point.clone())
        ctx.num_bits = num_bits
        ctx.symmetric = symmetric
        
        # Symmetric MinMax quantization without clipping
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

        # Initialize quantization ranges
        self._update_quant_range()

        # Register buffers for proper memory management and persistence
        # Buffers are automatically moved with model.to(device) and saved with state_dict
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))

        self.calibrated = False  # Should be False initially, will be set True after first calibration

        # Manual calibration mode support
        self.collecting_stats = False  # True during calibration
        self.num_batches_collected = 0

        # Temporary statistics for calibration (NOT registered as buffers to save memory)
        self.temp_min = None  # Will be created only when needed
        self.temp_max = None  # Will be created only when needed

    def set_num_bits(self, value):
        """Update num_bits and recalculate quantization ranges."""
        old_bits = self.num_bits
        self.num_bits = max(1, min(value, 32))
        self._update_quant_range()

        # Reset calibration when bit-width changes
        # Different bit-widths need different scale/zero_point calibration
        if old_bits != self.num_bits:
            print(f"    Reset calibration for LearnableFakeQuantize: {old_bits}")
            self.calibrated = False
            # Optionally reset statistics for clean recalibration
            # self.running_min.zero_()
            # self.running_max.zero_()

    def start_calibration(self):
        """Start calibration mode."""
        self.collecting_stats = True
        self.calibrated = False
        self.num_batches_collected = 0
        self.temp_min = None
        self.temp_max = None

    def finish_calibration(self):
        """Finish calibration and compute final scale/zero_point."""
        if self.num_batches_collected > 0 and self.temp_min is not None:
            # Move temp stats to running stats
            self.running_min.resize_as_(self.temp_min).copy_(self.temp_min)
            self.running_max.resize_as_(self.temp_max).copy_(self.temp_max)

            # Compute scale and zero_point
            with torch.no_grad():
                if self.symmetric:
                    abs_max = torch.max(torch.abs(self.running_min), torch.abs(self.running_max))
                    abs_max = torch.clamp(abs_max, min=self.eps)
                    self.scale.resize_as_(abs_max).copy_(abs_max / (2**(self.num_bits-1) - 1))
                    self.zero_point.resize_as_(abs_max).zero_()
                else:
                    range_val = torch.clamp(self.running_max - self.running_min, min=self.eps)
                    self.scale.resize_as_(range_val).copy_(range_val / (2**self.num_bits - 1))
                    self.zero_point.resize_as_(range_val)
                    self.zero_point.copy_(torch.round(self.quant_min - self.running_min / self.scale))

            self.calibrated = True
            self.collecting_stats = False
            # Clear temp buffers
            self.temp_min = None
            self.temp_max = None
        else:
            # If no statistics collected, just turn off collection mode
            self.collecting_stats = False

    def _collect_statistics_batch(self, x):
        """Collect min/max statistics for current batch (Pass 1)."""
        with torch.no_grad():
            if self.per_channel:
                # Per-channel statistics
                dims_to_reduce = list(range(x.dim()))
                dims_to_reduce.remove(self.channel_dim)

                min_val = x
                max_val = x
                for dim in sorted(dims_to_reduce, reverse=True):
                    min_val = min_val.min(dim=dim, keepdim=True)[0]
                    max_val = max_val.max(dim=dim, keepdim=True)[0]

                if self.num_batches_collected == 0:
                    # First batch: create and initialize temporary tensors
                    self.temp_min = min_val.clone().detach()
                    self.temp_max = max_val.clone().detach()
                else:
                    # Subsequent batches: track global min/max
                    self.temp_min = torch.minimum(self.temp_min, min_val)
                    self.temp_max = torch.maximum(self.temp_max, max_val)
            else:
                # Per-tensor statistics
                min_val = x.min()
                max_val = x.max()

                if self.num_batches_collected == 0:
                    # First batch: create and initialize temporary tensors
                    self.temp_min = min_val.clone().detach()
                    self.temp_max = max_val.clone().detach()
                else:
                    self.temp_min = torch.minimum(self.temp_min, min_val)
                    self.temp_max = torch.maximum(self.temp_max, max_val)

            self.num_batches_collected += 1

    def _compute_scale_zero_point(self):
        """Compute scale and zero_point from statistics."""
        with torch.no_grad():
            if self.symmetric:
                # Symmetric quantization
                max_abs = torch.max(torch.abs(self.running_min), torch.abs(self.running_max))
                max_abs = torch.clamp(max_abs, min=self.eps)
                self.scale.resize_as_(max_abs)
                self.scale.copy_(max_abs / (2**(self.num_bits-1) - 1))
                self.zero_point.resize_as_(max_abs)
                self.zero_point.zero_()
            else:
                # Asymmetric quantization
                range_val = torch.clamp(self.running_max - self.running_min, min=self.eps)
                self.scale.resize_as_(range_val)
                self.scale.copy_(range_val / (self.quant_max - self.quant_min))
                self.zero_point.resize_as_(range_val)
                if torch.any(self.scale > 0):
                    self.zero_point.copy_(torch.round(self.quant_min - self.running_min / self.scale))
                else:
                    self.zero_point.zero_()

    def _update_quant_range(self):
        """Update quantization range based on current num_bits."""
        self.quant_min = 0
        self.quant_max = 2 ** self.num_bits - 1

        if self.symmetric:
            self.quant_min = -(2 ** (self.num_bits - 1))
            self.quant_max = 2 ** (self.num_bits - 1) - 1
    
    def forward(self, x):
        # Skip quantization for 16-bit and above
        if self.num_bits >= 16:
            return x

        # MANUAL CALIBRATION MODE (controlled externally)
        if self.collecting_stats:
            # Collect statistics during calibration
            self._collect_statistics_batch(x)
            return x  # No quantization during calibration

        # If not calibrated, do one-shot calibration
        if not self.calibrated:
            with torch.no_grad():
                if self.per_channel:
                    dims = list(range(x.dim()))
                    dims.remove(self.channel_dim)
                    min_val = x.amin(dim=dims, keepdim=True)
                    max_val = x.amax(dim=dims, keepdim=True)
                else:
                    min_val = x.min()
                    max_val = x.max()

                self.running_min.resize_as_(min_val).copy_(min_val)
                self.running_max.resize_as_(max_val).copy_(max_val)

                # Compute scale and zero_point immediately
                if self.symmetric:
                    abs_max = torch.max(torch.abs(min_val), torch.abs(max_val))
                    abs_max = torch.clamp(abs_max, min=self.eps)
                    self.scale.resize_as_(abs_max).copy_(abs_max / (2**(self.num_bits-1) - 1))
                    self.zero_point.resize_as_(abs_max).zero_()
                else:
                    range_val = torch.clamp(max_val - min_val, min=self.eps)
                    self.scale.resize_as_(range_val).copy_(range_val / (2**self.num_bits - 1))
                    self.zero_point.resize_as_(range_val)
                    self.zero_point.copy_(torch.round(self.quant_min - min_val / self.scale))

                self.calibrated = True

        # Apply quantization with current scale/zero_point
        return QuantizationFunction.apply(
            x, self.scale, self.zero_point,
            self.num_bits, self.symmetric
        )

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
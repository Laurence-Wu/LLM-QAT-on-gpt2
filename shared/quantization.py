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
    def __init__(self, num_bits=8, symmetric=True, per_channel=False, channel_dim=0, gradient_accumulation_steps=8):
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

        # Two-pass mode support
        self.collecting_stats = False  # True during Pass 1 (statistics collection)
        self.stats_frozen = False  # True during Pass 2 (fixed parameters)
        self.num_batches_collected = 0

        # Temporary statistics for two-pass mode (NOT registered as buffers to save memory)
        self.temp_min = None  # Will be created only when needed
        self.temp_max = None  # Will be created only when needed

        # Auto two-pass configuration
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.forward_counter = 0  # Counts forward passes

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
        # Pass through for high precision modes (16-bit and above)
        # 16-bit provides sufficient precision without quantization artifacts
        if self.num_bits >= 16:
            return x

        # Automatic two-pass mode in training
        if self.training:
            self.forward_counter += 1

            # Pass 1: Collect statistics for gradient_accumulation_steps batches
            if self.forward_counter <= self.gradient_accumulation_steps:
                if not self.collecting_stats:
                    # Start Pass 1
                    self.collecting_stats = True
                    self.stats_frozen = False
                    self.num_batches_collected = 0
                    self.temp_min = None
                    self.temp_max = None

                # Collect statistics
                self._collect_statistics_batch(x)

                # Check if Pass 1 is complete
                if self.forward_counter == self.gradient_accumulation_steps:
                    # End Pass 1: Compute scale/zero_point and freeze them
                    if self.num_batches_collected > 0 and self.temp_min is not None:
                        # Copy statistics to running buffers
                        with torch.no_grad():
                            self.running_min.resize_as_(self.temp_min).copy_(self.temp_min)
                            self.running_max.resize_as_(self.temp_max).copy_(self.temp_max)
                            self._compute_scale_zero_point()
                        self.calibrated = True
                        self.stats_frozen = True
                        # Clear temp buffers
                        self.temp_min = None
                        self.temp_max = None
                    self.collecting_stats = False

                return x  # Return original tensor during Pass 1

            # Pass 2: Use frozen quantization parameters for next gradient_accumulation_steps
            elif self.forward_counter <= 2 * self.gradient_accumulation_steps:
                # Keep stats frozen during Pass 2
                if self.forward_counter == 2 * self.gradient_accumulation_steps:
                    # End Pass 2: Reset for next cycle
                    self.stats_frozen = False
                    self.forward_counter = 0
                # Continue to quantization below
            else:
                # Should not reach here, reset counter
                self.forward_counter = 0

        # Two-pass mode: Pass 1 - Statistics Collection (manual mode for compatibility)
        if self.collecting_stats:
            self._collect_statistics_batch(x)
            return x  # Return original tensor during collection

        # If uncalibrated in eval mode, do one-shot calibration on first input
        if not self.calibrated and not self.training:
            with torch.no_grad():
                if self.per_channel:
                    # Per-channel one-shot calibration
                    dims_to_reduce = list(range(x.dim()))
                    dims_to_reduce.remove(self.channel_dim)

                    min_val = x
                    max_val = x
                    for dim in sorted(dims_to_reduce, reverse=True):
                        min_val = min_val.min(dim=dim, keepdim=True)[0]
                        max_val = max_val.max(dim=dim, keepdim=True)[0]

                    self.running_min.resize_as_(min_val).copy_(min_val)
                    self.running_max.resize_as_(max_val).copy_(max_val)
                    self.scale.resize_as_(min_val)
                    self.zero_point.resize_as_(min_val)
                else:
                    # Per-tensor one-shot calibration
                    self.running_min.copy_(x.min())
                    self.running_max.copy_(x.max())

                self.calibrated = True  # Mark as calibrated

        if self.training and not self.stats_frozen and self.forward_counter == 0:
            with torch.no_grad():  # Ensure no gradient tracking for statistics
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

                    # Use in-place operations to avoid creating new tensors
                    if not self.calibrated:
                        # Resize buffers and copy values
                        self.running_min.resize_as_(min_val).copy_(min_val)
                        self.running_max.resize_as_(max_val).copy_(max_val)
                        self.scale.resize_as_(min_val).fill_(1.0)
                        self.zero_point.resize_as_(min_val).zero_()
                        self.calibrated = True
                    else:
                        # In-place exponential moving average update
                        self.running_min.mul_(0.9).add_(min_val, alpha=0.1)
                        self.running_max.mul_(0.9).add_(max_val, alpha=0.1)
                else:
                    # Per-tensor statistics
                    min_val = x.min()
                    max_val = x.max()

                    if not self.calibrated:
                        # Direct assignment for scalars
                        self.running_min.copy_(min_val)
                        self.running_max.copy_(max_val)
                        self.calibrated = True
                    else:
                        # In-place exponential moving average update
                        self.running_min.mul_(0.9).add_(min_val, alpha=0.1)
                        self.running_max.mul_(0.9).add_(max_val, alpha=0.1)
        
        # Only update scale and zero_point if not in frozen state (Pass 2)
        if not self.stats_frozen and (self.calibrated or self.training):
            with torch.no_grad():
                if self.symmetric:
                    # Paper formula: α = max(|X_R|)/(2^(N-1) - 1)
                    max_val = torch.max(torch.abs(self.running_min), torch.abs(self.running_max))
                    max_val = torch.clamp(max_val, min=self.eps)
                    # In-place update of scale
                    self.scale.copy_(max_val / (2**(self.num_bits-1) - 1))
                    self.zero_point.zero_()  # In-place zero
                else:
                    range_val = torch.clamp(self.running_max - self.running_min, min=self.eps)
                    denom = self.quant_max - self.quant_min
                    if denom > 0:
                        # In-place update of scale
                        self.scale.copy_(range_val / denom)
                        if torch.any(self.scale > 0):
                            # In-place update of zero_point
                            self.zero_point.copy_(torch.round(self.quant_min - self.running_min / self.scale))
                        else:
                            self.zero_point.zero_()
                    else:
                        self.scale.fill_(1.0)
                        self.zero_point.zero_()

        # Scale and zero_point are already registered buffers, so they're on the right device
        # Just use them directly without creating new tensors
        return QuantizationFunction.apply(x, self.scale, self.zero_point,
                                         self.num_bits, self.symmetric)

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 weight_bits=8, activation_bits=8, gradient_accumulation_steps=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Per-channel weight quantization, per-token activation quantization (paper approach)
        self.weight_quantizer = LearnableFakeQuantize(weight_bits, symmetric=True, per_channel=True, channel_dim=0, gradient_accumulation_steps=gradient_accumulation_steps)
        self.activation_quantizer = LearnableFakeQuantize(activation_bits, symmetric=True, gradient_accumulation_steps=gradient_accumulation_steps)  # Paper uses symmetric for both
        
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
"""
LearnableFakeQuantize module with support for different quantization methods.
Supports multiple quantization strategies: minmax, relu_clip, tanh, and log.
"""

import torch
import torch.nn as nn

# Import quantization functions
try:
    from .quantization_methods import (
        apply_minmax_quantization,
        apply_log_quantization
    )
except ImportError:
    from quantization_methods import (
        apply_minmax_quantization,
        apply_log_quantization
    )


class LearnableFakeQuantize(nn.Module):
    def __init__(self, num_bits=8, symmetric=False, per_channel=True,
                 channel_dim=0, quantizer_type='minmax', eps=1e-5):
        """
        Initialize the quantization layer with configurable quantizer type.

        Args:
            num_bits: Number of quantization bits
            symmetric: Whether to use symmetric quantization
            per_channel: Whether to use per-channel quantization
            channel_dim: Channel dimension for per-channel quantization
            quantizer_type: 'minmax', 'relu_clip', 'tanh', or 'log'
        """
        super().__init__()
        self.num_bits = max(1, min(num_bits, 32))
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_dim = channel_dim
        self.quantizer_type = quantizer_type
        self.eps = eps

        # Update quantization range
        self._update_quant_range()

        # Initialize buffers
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))

        # Calibration state
        self.calibrated = False
        self.collecting_stats = False
        self.num_batches_collected = 0
        self.temp_min = None
        self.temp_max = None

    def set_num_bits(self, value):
        """Update num_bits and recalculate quantization ranges."""
        old_bits = self.num_bits
        self.num_bits = max(1, min(value, 32))
        self._update_quant_range()

        if old_bits != self.num_bits:
            print(f"    Reset calibration for {self.quantizer_type} quantizer: {old_bits} -> {self.num_bits} bits")
            self.calibrated = False

    def _update_quant_range(self):
        """Update quantization range based on current num_bits."""
        self.quant_min = 0
        self.quant_max = 2 ** self.num_bits - 1

        if self.symmetric:
            self.quant_min = -(2 ** (self.num_bits - 1))
            self.quant_max = 2 ** (self.num_bits - 1) - 1

    def start_calibration(self):
        """Start calibration mode."""
        self.collecting_stats = True
        self.calibrated = False
        self.num_batches_collected = 0
        self.temp_min = None
        self.temp_max = None

    def finish_calibration(self, debug=False):
        """Finish calibration and compute final scale/zero_point."""
        if self.num_batches_collected > 0 and self.temp_min is not None:
            # Move temp stats to running stats
            self.running_min.resize_as_(self.temp_min).copy_(self.temp_min)
            self.running_max.resize_as_(self.temp_max).copy_(self.temp_max)

            # Debug output
            if debug:
                print(f"\n      📊 Calibration Debug Stats for {self.num_bits}-bit {self.quantizer_type}:")
                if self.per_channel:
                    print(f"         Running min (per-channel): mean={self.running_min.mean().item():.6f}")
                    print(f"         Running max (per-channel): mean={self.running_max.mean().item():.6f}")
                else:
                    print(f"         Running min: {self.running_min.item():.6f}")
                    print(f"         Running max: {self.running_max.item():.6f}")
                print(f"         Range: [{self.running_min.min().item() if self.per_channel else self.running_min.item():.6f}, "
                      f"{self.running_max.max().item() if self.per_channel else self.running_max.item():.6f}]")
                print(f"         Batches collected: {self.num_batches_collected}")
                print(f"         Quantizer type: {self.quantizer_type}")

            # Compute scale and zero_point based on quantizer type
            with torch.no_grad():
                if self.quantizer_type == 'relu_clip':
                    # For ReLU style, force min to 0
                    self.running_min.zero_()
                    range_val = torch.clamp(self.running_max, min=self.eps)
                    self.scale.resize_as_(range_val).copy_(range_val / (2**self.num_bits - 1))
                    self.zero_point.resize_as_(range_val).zero_()
                elif self.quantizer_type == 'log':
                    # For log quantization, running_min and running_max already contain log₂ values
                    # Just compute the range directly
                    log_min = self.running_min
                    log_max = self.running_max
                    log_range = log_max - log_min

                    # Store log_min in zero_point and log_range in scale for efficiency
                    self.zero_point.resize_as_(self.running_max).copy_(log_min)
                    self.scale.resize_as_(self.running_max).copy_(log_range)
                elif self.quantizer_type == 'tanh':
                    # For tanh quantization, scale represents the input range
                    abs_max = torch.max(torch.abs(self.running_min), torch.abs(self.running_max))
                    abs_max = torch.clamp(abs_max, min=self.eps)
                    self.scale.resize_as_(abs_max).copy_(abs_max)
                    self.zero_point.resize_as_(abs_max).zero_()
                elif self.symmetric:
                    abs_max = torch.max(torch.abs(self.running_min), torch.abs(self.running_max))
                    abs_max = torch.clamp(abs_max, min=self.eps)
                    self.scale.resize_as_(abs_max).copy_(abs_max / (2**(self.num_bits-1) - 1))
                    self.zero_point.resize_as_(abs_max).zero_()
                else:
                    range_val = torch.clamp(self.running_max - self.running_min, min=self.eps)
                    self.scale.resize_as_(range_val).copy_(range_val / (2**self.num_bits - 1))
                    self.zero_point.resize_as_(range_val)
                    self.zero_point.copy_(torch.round(self.quant_min - self.running_min / self.scale))

                if debug:
                    print(f"         Computed scale: mean={self.scale.mean().item():.6f}")

            self.calibrated = True
            self.collecting_stats = False
            # Clear temp buffers
            self.temp_min = None
            self.temp_max = None
        else:
            self.collecting_stats = False
            if debug:
                print(f"      ⚠️ No statistics collected for {self.num_bits}-bit {self.quantizer_type} quantizer")

    def _collect_statistics_batch(self, x):
        """Collect min/max statistics for current batch."""
        with torch.no_grad():
            if self.quantizer_type == 'log':
                # For log quantization, collect log₂ statistics directly
                # Handle zeros and get absolute values
                abs_x = torch.abs(x)
                non_zero_mask = abs_x > self.eps

                if non_zero_mask.any():
                    # Get log₂ of non-zero absolute values
                    abs_x_clamped = torch.clamp(abs_x, min=self.eps)
                    log_x = torch.log2(abs_x_clamped)

                    if self.per_channel:
                        dims_to_reduce = list(range(log_x.dim()))
                        dims_to_reduce.remove(self.channel_dim)

                        min_val = log_x
                        max_val = log_x
                        for dim in sorted(dims_to_reduce, reverse=True):
                            min_val = min_val.min(dim=dim, keepdim=True)[0]
                            max_val = max_val.max(dim=dim, keepdim=True)[0]
                    else:
                        min_val = log_x.min()
                        max_val = log_x.max()

                    # Update temp min/max with log₂ values
                    if self.num_batches_collected == 0:
                        self.temp_min = min_val.clone().detach()
                        self.temp_max = max_val.clone().detach()
                    else:
                        self.temp_min = torch.minimum(self.temp_min, min_val)
                        self.temp_max = torch.maximum(self.temp_max, max_val)
                # If all values are zero, keep existing temp_min/temp_max or use defaults
            else:
                # For other quantization methods, collect linear statistics
                if self.per_channel:
                    dims_to_reduce = list(range(x.dim()))
                    dims_to_reduce.remove(self.channel_dim)

                    min_val = x
                    max_val = x
                    for dim in sorted(dims_to_reduce, reverse=True):
                        min_val = min_val.min(dim=dim, keepdim=True)[0]
                        max_val = max_val.max(dim=dim, keepdim=True)[0]

                    if self.num_batches_collected == 0:
                        self.temp_min = min_val.clone().detach()
                        self.temp_max = max_val.clone().detach()
                    else:
                        self.temp_min = torch.minimum(self.temp_min, min_val)
                        self.temp_max = torch.maximum(self.temp_max, max_val)
                else:
                    min_val = x.min()
                    max_val = x.max()

                    if self.num_batches_collected == 0:
                        self.temp_min = min_val.clone().detach()
                        self.temp_max = max_val.clone().detach()
                    else:
                        self.temp_min = torch.minimum(self.temp_min, min_val)
                        self.temp_max = torch.maximum(self.temp_max, max_val)

            self.num_batches_collected += 1

    def forward(self, x):
        """
        Forward pass with multiple quantization strategies.
        The quantizer type determines how we transform and quantize the input.
        """
        # Skip quantization for 32-bit FP32 teacher
        if self.num_bits >= 32:
            return x

        # Calibration mode - just collect stats, no quantization
        if self.collecting_stats:
            self._collect_statistics_batch(x)
            return x

        # Require manual calibration - no fallback
        if not self.calibrated:
            raise RuntimeError(f"Quantizer not calibrated. Please run calibration first for {self.quantizer_type} quantizer.")

        # Apply the appropriate quantization strategy
        if self.quantizer_type == 'minmax':
            return self._quantize_minmax(x)
        elif self.quantizer_type == 'log':
            return self._quantize_log(x)
        else:
            raise ValueError(f"Unknown quantizer type: {self.quantizer_type}. Supported types: 'minmax', 'log'")

    def _quantize_minmax(self, x):
        """Apply standard min-max quantization."""
        return apply_minmax_quantization(
            x, self.scale, self.zero_point,
            self.num_bits, self.symmetric
        )

    def _quantize_log(self, x):
        """Apply logarithmic quantization using precomputed log stats."""
        # log_min and log_range are stored in zero_point and scale respectively
        return apply_log_quantization(
            x, self.zero_point, self.scale, self.num_bits
        )


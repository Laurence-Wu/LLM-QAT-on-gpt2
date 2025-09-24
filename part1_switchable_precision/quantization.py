"""
LearnableFakeQuantize module with support for different quantization methods.
Supports quantization strategies: minmax and log.
"""

import torch
import torch.nn as nn

from quantization_methods import (
    apply_minmax_quantization,
    apply_log_quantization
)



class LearnableFakeQuantize(nn.Module):
    def __init__(self, num_bits,
                 channel_dim=0, quantizer_type='minmax', eps=1e-5, symmetric=True):
        super().__init__()
        self.num_bits = max(1, min(num_bits, 32))
        self.symmetric = symmetric
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

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Override to resize buffers to match checkpoint shapes."""
        # Resize our buffers to match the shapes in the checkpoint
        for buffer_name in ['scale', 'zero_point', 'running_min', 'running_max']:
            key = prefix + buffer_name
            if key in state_dict:
                # Get the buffer and resize it to match loaded shape
                buffer = getattr(self, buffer_name, None)
                if buffer is not None:
                    buffer.resize_as_(state_dict[key])

        # Call parent implementation to actually load the values
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

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
        if self.symmetric:
            # Symmetric quantization: range centered at 0
            self.quant_min = -(2 ** (self.num_bits - 1))
            self.quant_max = 2 ** (self.num_bits - 1) - 1
        else:
            # Asymmetric quantization: full positive range
            self.quant_min = 0
            self.quant_max = 2 ** self.num_bits - 1

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
            # Compute scale and zero_point based on quantizer type
            with torch.no_grad():
                if self.quantizer_type == 'log':
                    # For log quantization, running_min and running_max already contain log₂ values
                    # Just compute the range directly
                    log_min = self.running_min
                    log_max = self.running_max
                    log_range = log_max - log_min

                    # Store log_min in zero_point and log_range in scale for efficiency
                    self.zero_point.resize_as_(self.running_max).copy_(log_min)
                    self.scale.resize_as_(self.running_max).copy_(log_range)
                else:
                    # Minmax quantization
                    if self.symmetric:
                        # Symmetric: scale based on max absolute value
                        abs_max = torch.max(torch.abs(self.running_min), torch.abs(self.running_max))
                        abs_max = torch.clamp(abs_max, min=self.eps)
                        self.scale.resize_as_(abs_max).copy_(abs_max / (2**(self.num_bits-1) - 1))
                        self.zero_point.resize_as_(abs_max).zero_()
                    else:
                        # Asymmetric: scale based on full range with zero_point
                        range_val = torch.clamp(self.running_max - self.running_min, min=self.eps)
                        self.scale.resize_as_(range_val).copy_(range_val / (2**self.num_bits - 1))
                        self.zero_point.resize_as_(range_val)
                        self.zero_point.copy_(torch.round(-self.running_min / self.scale))

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
                    dims_to_reduce = list(range(log_x.dim()))
                    dims_to_reduce.remove(self.channel_dim)

                    min_val = log_x
                    max_val = log_x
                    for dim in sorted(dims_to_reduce, reverse=True):
                        min_val = min_val.min(dim=dim, keepdim=True)[0]
                        max_val = max_val.max(dim=dim, keepdim=True)[0]

                    # Update temp min/max with log₂ values
                    if self.num_batches_collected == 0:
                        self.temp_min = min_val.clone().detach()
                        self.temp_max = max_val.clone().detach()
                    else:
                        self.temp_min = torch.minimum(self.temp_min, min_val)
                        self.temp_max = torch.maximum(self.temp_max, max_val)
                # If all values are zero, keep existing temp_min/temp_max or use defaults
                elif self.num_batches_collected == 0:
                    # Initialize with default values for all-zero input
                    # Use log2(eps) as both min and max
                    log_eps = torch.log2(torch.tensor(self.eps))
                    shape = list(x.shape)
                    shape[self.channel_dim] = 1
                    self.temp_min = torch.full(shape, log_eps, device=x.device)
                    self.temp_max = torch.full(shape, log_eps, device=x.device)
            else:
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

            self.num_batches_collected += 1

    def forward(self, x):
        if self.num_bits >= 32:
            return x
        if self.collecting_stats:
            self._collect_statistics_batch(x)
            return x

        # You always need the manual calibration
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
            x, self.zero_point, self.scale, self.num_bits, self.symmetric
        )

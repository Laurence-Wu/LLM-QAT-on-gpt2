"""
LearnableFakeQuantize module with support for different quantization methods.
Supports quantization strategies: minmax and log.
"""

import torch
import torch.nn as nn

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
    def __init__(self, num_bits,
                 channel_dim=0, quantizer_type='minmax', eps=1e-5, symmetric=True, per_channel=True, is_input=False):
        super().__init__()
        self.num_bits = max(1, min(num_bits, 32))
        self.symmetric = symmetric
        self.per_channel = per_channel
        # If per_channel is False, set channel_dim to None for global statistics
        self.channel_dim = channel_dim if per_channel else None
        self.quantizer_type = quantizer_type
        self.eps = eps
        self.is_input = is_input  # Flag to differentiate input vs weight quantizers

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
        """Override to resize buffers to match checkpoint shapes and set calibrated flag."""
        # Special handling for input quantizers to handle sequence length mismatches
        if self.is_input:
            for buffer_name in ['scale', 'zero_point', 'running_min', 'running_max']:
                key = prefix + buffer_name
                if key in state_dict:
                    loaded_tensor = state_dict[key]
                    buffer = getattr(self, buffer_name, None)
                    if buffer is not None:
                        # For input quantizers, handle potential sequence length mismatch
                        # If loaded shape has sequence dimension (dim 1), reduce it
                        if loaded_tensor.dim() == 3 and loaded_tensor.shape[1] > 1:
                            # Old format: [1, seq_len, 1] -> reduce to [1, 1, features]
                            if self.quantizer_type == 'log':
                                # For log quantization, use max for conservative quantization
                                reduced = loaded_tensor.max(dim=1, keepdim=True)[0]
                            else:
                                # For minmax, use appropriate reduction
                                if 'min' in buffer_name:
                                    reduced = loaded_tensor.min(dim=1, keepdim=True)[0]
                                elif 'max' in buffer_name:
                                    reduced = loaded_tensor.max(dim=1, keepdim=True)[0]
                                else:
                                    # For scale/zero_point, use max for conservative quantization
                                    reduced = loaded_tensor.max(dim=1, keepdim=True)[0]
                            state_dict[key] = reduced
                        buffer.resize_as_(state_dict[key])
        else:
            # Standard handling for weight quantizers
            for buffer_name in ['scale', 'zero_point', 'running_min', 'running_max']:
                key = prefix + buffer_name
                if key in state_dict:
                    buffer = getattr(self, buffer_name, None)
                    if buffer is not None:
                        buffer.resize_as_(state_dict[key])

        # Call parent implementation to actually load the values
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

        # Set calibrated flag if we loaded calibration parameters
        # This indicates the quantizer has valid calibration data from checkpoint
        if (prefix + 'scale' in state_dict and
            prefix + 'zero_point' in state_dict):
            self.calibrated = True

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

    def _get_reduction_dims(self, tensor):
        """Get dimensions to reduce based on per_channel setting."""
        if self.per_channel and self.channel_dim is not None:
            # Per-channel: reduce all dims except channel_dim
            dims_to_reduce = list(range(tensor.dim()))
            # Handle negative indexing for channel_dim
            actual_dim = self.channel_dim if self.channel_dim >= 0 else tensor.dim() + self.channel_dim
            if actual_dim in dims_to_reduce:
                dims_to_reduce.remove(actual_dim)
        else:
            # Per-tensor: reduce all dimensions
            dims_to_reduce = list(range(tensor.dim()))
        return dims_to_reduce

    def _reduce_min_max(self, tensor, dims_to_reduce):
        """Reduce tensor along specified dimensions to get min/max."""
        if not dims_to_reduce:
            return tensor, tensor

        min_val = tensor
        max_val = tensor
        for dim in sorted(dims_to_reduce, reverse=True):
            min_val = min_val.min(dim=dim, keepdim=True)[0]
            max_val = max_val.max(dim=dim, keepdim=True)[0]
        return min_val, max_val

    def _get_default_shape(self, x, default_value):
        """Get appropriate tensor shape for default values based on per_channel setting."""
        if self.per_channel and self.channel_dim is not None:
            shape = list(x.shape)
            # Handle negative indexing for channel_dim
            actual_dim = self.channel_dim if self.channel_dim >= 0 else len(shape) + self.channel_dim
            shape[actual_dim] = 1
            return torch.full(shape, default_value, device=x.device)
        else:
            # Per-tensor: return scalar tensor
            return torch.tensor(default_value, device=x.device)

    def _collect_statistics_batch(self, x):
        """Collect min/max statistics for current batch."""
        with torch.no_grad():
            if self.quantizer_type == 'log':
                # For log quantization, collect log₂ statistics directly
                abs_x = torch.abs(x)
                non_zero_mask = abs_x > self.eps

                if non_zero_mask.any():
                    # Get log₂ of non-zero absolute values
                    abs_x_clamped = torch.clamp(abs_x, min=self.eps)
                    log_x = torch.log2(abs_x_clamped)

                    # Get dimensions to reduce and compute min/max
                    dims_to_reduce = self._get_reduction_dims(log_x)
                    min_val, max_val = self._reduce_min_max(log_x, dims_to_reduce)

                    # Update temp min/max with log₂ values
                    if self.num_batches_collected == 0:
                        self.temp_min = min_val.clone().detach()
                        self.temp_max = max_val.clone().detach()
                    else:
                        self.temp_min = torch.minimum(self.temp_min, min_val)
                        self.temp_max = torch.maximum(self.temp_max, max_val)
                # If all values are zero and this is the first batch, use defaults
                elif self.num_batches_collected == 0:
                    log_eps = torch.log2(torch.tensor(self.eps))
                    self.temp_min = self._get_default_shape(x, log_eps)
                    self.temp_max = self._get_default_shape(x, log_eps)
            else:
                # Standard min-max statistics collection
                dims_to_reduce = self._get_reduction_dims(x)
                min_val, max_val = self._reduce_min_max(x, dims_to_reduce)

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

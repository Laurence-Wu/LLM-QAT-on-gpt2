"""
Different quantization methods for LearnableFakeQuantize.
Supports multiple quantization strategies: minmax, relu_clip, tanh, and log.
"""

import torch
import torch.nn as nn


class QuantizationFunction(torch.autograd.Function):
    """Custom quantization function with straight-through estimator."""
    @staticmethod
    def forward(ctx, input, scale, zero_point, num_bits, symmetric):
        ctx.save_for_backward(input, scale.clone(), zero_point.clone())
        ctx.num_bits = num_bits
        ctx.symmetric = symmetric

        if symmetric:
            # Symmetric quantization
            quantized = torch.round(input / scale)
            output = quantized * scale
        else:
            # Asymmetric quantization
            output = torch.round(input / scale + zero_point)
            output = torch.clamp(output, 0, 2**num_bits - 1)
            output = (output - zero_point) * scale

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        input, scale, zero_point = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


class LearnableFakeQuantize(nn.Module):
    def __init__(self, num_bits=8, symmetric=True, per_channel=False,
                 channel_dim=0, quantizer_type='minmax'):
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
        self.eps = 1e-7

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

        # If not calibrated, perform one-shot calibration
        if not self.calibrated:
            self._perform_one_shot_calibration(x)

        # Apply the appropriate quantization strategy
        if self.quantizer_type == 'minmax':
            return self._quantize_minmax(x)
        elif self.quantizer_type == 'relu_clip':
            return self._quantize_relu_clip(x)
        elif self.quantizer_type == 'tanh':
            return self._quantize_tanh(x)
        elif self.quantizer_type == 'log':
            return self._quantize_log(x)
        else:
            raise ValueError(f"Unknown quantizer type: {self.quantizer_type}")

    def _quantize_minmax(self, x):
        """
        Standard min-max quantization (Paper's default).
        Uses the formula: X_Q = α⌊X_R/α⌉ where α = max(|X_R|)/(2^(N-1) - 1)
        """
        # Apply standard quantization
        return QuantizationFunction.apply(
            x, self.scale, self.zero_point,
            self.num_bits, self.symmetric
        )

    def _quantize_relu_clip(self, x):
        """
        ReLU-style quantization - clips to [0, max] range.
        Useful for activations after ReLU layers.
        """
        # Clip to non-negative range
        x_clipped = torch.clamp(x, min=0.0)

        # Clip to calibrated maximum
        if self.calibrated:
            max_val = self.running_max
            x_clipped = torch.clamp(x_clipped, max=max_val)

        # Quantize in [0, max] range
        x_int = torch.round(x_clipped / self.scale)
        x_int = torch.clamp(x_int, 0, 2**self.num_bits - 1)
        x_quant = x_int * self.scale

        # Straight-through estimator
        return x + (x_quant - x).detach()

    def _quantize_tanh(self, x):
        """
        Tanh-based quantization - maps inputs through tanh to [-1, 1] range.
        Useful for bounded inputs and helps with outliers.
        """
        # Scale input based on calibration stats
        if self.calibrated:
            input_scale = torch.max(torch.abs(self.running_min), torch.abs(self.running_max))
            x_scaled = x / (input_scale + self.eps)
        else:
            x_scaled = x

        # Apply tanh to bound the input to [-1, 1]
        x_tanh = torch.tanh(x_scaled)

        # Quantize in [-1, 1] range
        if self.symmetric:
            x_int = torch.round(x_tanh * (2**(self.num_bits-1) - 1))
            x_int = torch.clamp(x_int, -(2**(self.num_bits-1) - 1), 2**(self.num_bits-1) - 1)
            x_quant = x_int / (2**(self.num_bits-1) - 1)
        else:
            # Map to [0, 1] for asymmetric
            x_normalized = (x_tanh + 1) / 2
            x_int = torch.round(x_normalized * (2**self.num_bits - 1))
            x_int = torch.clamp(x_int, 0, 2**self.num_bits - 1)
            x_quant = x_int / (2**self.num_bits - 1)
            x_quant = x_quant * 2 - 1  # Back to [-1, 1]

        # Scale back to original range
        if self.calibrated:
            x_quant = x_quant * input_scale

        # Straight-through estimator
        return x + (x_quant - x_tanh * (input_scale if self.calibrated else 1.0)).detach()

    def _quantize_log(self, x):
        """
        Logarithmic quantization - non-uniform quantization.
        Allocates more levels to smaller values.
        """
        # Separate sign and magnitude
        sign_x = torch.sign(x)
        abs_x = torch.abs(x) + self.eps

        # Apply logarithm to magnitude
        log_x = torch.log2(abs_x)

        # Find range for normalization
        if self.calibrated:
            abs_min = torch.abs(self.running_min) + self.eps
            abs_max = torch.abs(self.running_max) + self.eps
            log_min = torch.log2(abs_min)
            log_max = torch.log2(abs_max)
        else:
            log_min = log_x.min()
            log_max = log_x.max()

        # Normalize to [0, 1]
        log_range = log_max - log_min + self.eps
        log_normalized = (log_x - log_min) / log_range

        # Quantize uniformly in log space
        log_int = torch.round(log_normalized * (2**self.num_bits - 1))
        log_int = torch.clamp(log_int, 0, 2**self.num_bits - 1)

        # Dequantize
        log_quant_normalized = log_int / (2**self.num_bits - 1)
        log_quant = log_quant_normalized * log_range + log_min

        # Convert back to linear space
        abs_quant = torch.pow(2, log_quant)
        x_quant = sign_x * abs_quant

        # Handle zeros
        x_quant = torch.where(torch.abs(x) < self.eps, torch.zeros_like(x), x_quant)

        # Straight-through estimator
        return x + (x_quant - x).detach()

    def _perform_one_shot_calibration(self, x):
        """
        One-shot calibration when not manually calibrated.
        """
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

            # Compute scale based on quantizer type
            if self.quantizer_type == 'relu_clip':
                self.running_min.zero_()
                range_val = torch.clamp(self.running_max, min=self.eps)
                self.scale.resize_as_(range_val).copy_(range_val / (2**self.num_bits - 1))
                self.zero_point.resize_as_(range_val).zero_()
            elif self.symmetric:
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
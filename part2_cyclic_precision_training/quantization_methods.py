"""
Quantization methods for CPT with per-channel calibration.
Includes log quantization and SBM's GEMMLOWP scheme.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PerChannelLogQuantization(nn.Module):
    """
    Log-scale quantization with per-channel calibration.
    Non-uniform quantization that better preserves information.
    """

    def __init__(self, num_bits: int = 8, symmetric: bool = True):
        super().__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.register_buffer('calibrated', torch.tensor(False))

        # Per-channel calibration parameters
        self.register_buffer('channel_scales', None)
        self.register_buffer('channel_zero_points', None)

    def calibrate_per_channel(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calibrate scale and zero-point per channel.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim] or [out_features, in_features]

        Returns:
            Tuple of (scale, zero_point) per channel
        """
        # Handle different tensor shapes
        if x.dim() == 3:
            # Activation tensor [batch, seq_len, hidden_dim]
            # Calibrate per hidden dimension (channel)
            channel_dim = -1
            reduce_dims = (0, 1)  # Reduce over batch and sequence
        elif x.dim() == 2:
            # Weight tensor [out_features, in_features]
            # Calibrate per output channel
            channel_dim = 0
            reduce_dims = (1,)  # Reduce over input features
        else:
            raise ValueError(f"Unexpected tensor dimension: {x.dim()}")

        # Get min/max per channel
        channel_max = x.max(dim=reduce_dims[0], keepdim=True)[0]
        for dim in reduce_dims[1:]:
            channel_max = channel_max.max(dim=dim, keepdim=True)[0]

        channel_min = x.min(dim=reduce_dims[0], keepdim=True)[0]
        for dim in reduce_dims[1:]:
            channel_min = channel_min.min(dim=dim, keepdim=True)[0]

        if self.symmetric:
            # Symmetric quantization
            channel_abs_max = torch.max(channel_max.abs(), channel_min.abs())
            scale = channel_abs_max / (2 ** (self.num_bits - 1) - 1)
            zero_point = torch.zeros_like(scale)
        else:
            # Asymmetric quantization
            scale = (channel_max - channel_min) / (2 ** self.num_bits - 1)
            zero_point = -channel_min / scale

        # Avoid division by zero
        scale = torch.clamp(scale, min=1e-8)

        return scale, zero_point

    def quantize(self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        """
        Apply log-scale quantization with per-channel parameters.
        """
        # Apply log transformation for non-uniform quantization
        x_sign = torch.sign(x)
        x_abs = torch.abs(x)

        # Log-scale transformation
        x_log = torch.log(1 + x_abs) * x_sign

        # Quantize in log domain
        if self.symmetric:
            x_int = torch.round(x_log / scale)
            x_int = torch.clamp(x_int, -(2 ** (self.num_bits - 1)), 2 ** (self.num_bits - 1) - 1)
            x_quant_log = x_int * scale
        else:
            x_int = torch.round(x_log / scale + zero_point)
            x_int = torch.clamp(x_int, 0, 2 ** self.num_bits - 1)
            x_quant_log = (x_int - zero_point) * scale

        # Inverse log transformation
        x_quant = torch.sign(x_quant_log) * (torch.exp(torch.abs(x_quant_log)) - 1)

        return x_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantization.
        """
        if not self.calibrated or self.channel_scales is None:
            # Calibrate on first pass
            self.channel_scales, self.channel_zero_points = self.calibrate_per_channel(x)
            self.calibrated = True

        # Apply quantization
        x_quant = self.quantize(x, self.channel_scales, self.channel_zero_points)

        # Straight-through estimator for gradients
        if self.training:
            return x + (x_quant - x).detach()
        else:
            return x_quant


class SBMQuantization(nn.Module):
    """
    SBM's GEMMLOWP quantization scheme with stochastic rounding.
    Based on the SBM paper Section 4 and Appendix B.
    """

    def __init__(self, num_bits: int = 8, use_stochastic_rounding: bool = True):
        super().__init__()
        self.num_bits = num_bits
        self.use_stochastic_rounding = use_stochastic_rounding

        # Per-channel calibration parameters
        self.register_buffer('channel_vmin', None)
        self.register_buffer('channel_vmax', None)
        self.register_buffer('calibrated', torch.tensor(False))

    def calibrate_per_channel(self, x: torch.Tensor, percentile: float = 99.9):
        """
        Calibrate using GEMMLOWP scheme with per-channel statistics.
        """
        # Get channel dimension
        if x.dim() == 3:
            # [batch, seq_len, hidden_dim]
            x_flat = x.view(-1, x.size(-1))  # Flatten batch and seq
        elif x.dim() == 2:
            # [out_features, in_features]
            x_flat = x
        else:
            raise ValueError(f"Unexpected tensor dimension: {x.dim()}")

        # Calculate per-channel min/max with percentile clipping
        if percentile < 100:
            # Use percentile for better robustness
            channel_vmax = torch.quantile(x_flat, percentile / 100, dim=0)
            channel_vmin = torch.quantile(x_flat, (100 - percentile) / 100, dim=0)
        else:
            channel_vmax = x_flat.max(dim=0)[0]
            channel_vmin = x_flat.min(dim=0)[0]

        self.channel_vmin = channel_vmin.view(1, 1, -1) if x.dim() == 3 else channel_vmin.view(-1, 1)
        self.channel_vmax = channel_vmax.view(1, 1, -1) if x.dim() == 3 else channel_vmax.view(-1, 1)
        self.calibrated = True

    def quantize_gemmlowp(self, x: torch.Tensor) -> torch.Tensor:
        """
        GEMMLOWP quantization with per-channel calibration.
        """
        if not self.calibrated:
            self.calibrate_per_channel(x)

        # GEMMLOWP scheme from SBM paper
        scale = (self.channel_vmax - self.channel_vmin) / (2 ** self.num_bits)
        zero_point = torch.round(
            torch.clamp(-self.channel_vmin / scale, 0, 2 ** self.num_bits)
        )

        # Quantize with optional stochastic rounding
        if self.use_stochastic_rounding and self.training:
            # Stochastic rounding for gradients
            noise = torch.rand_like(x) - 0.5
            x_int = torch.round(x / scale + zero_point + noise)
        else:
            x_int = torch.round(x / scale + zero_point)

        # Clamp to bit range
        x_int = torch.clamp(x_int, 0, 2 ** self.num_bits - 1)

        # Dequantize
        x_quant = (x_int - zero_point) * scale

        return x_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with GEMMLOWP quantization.
        """
        x_quant = self.quantize_gemmlowp(x)

        # Straight-through estimator
        if self.training:
            return x + (x_quant - x).detach()
        else:
            return x_quant


class GradientBifurcation:
    """
    Gradient bifurcation from SBM paper.
    Uses different precision for weight gradients vs activation gradients.
    """

    def __init__(self, weight_grad_bits: int = 16, activation_grad_bits: int = 8):
        self.weight_grad_bits = weight_grad_bits
        self.activation_grad_bits = activation_grad_bits

        # Quantizers for each gradient type
        self.weight_grad_quantizer = SBMQuantization(
            num_bits=weight_grad_bits,
            use_stochastic_rounding=True
        )
        self.activation_grad_quantizer = PerChannelLogQuantization(
            num_bits=activation_grad_bits,
            symmetric=True
        )

    def quantize_weight_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """Quantize weight gradients with higher precision."""
        if self.weight_grad_bits == 32:
            return grad
        return self.weight_grad_quantizer(grad)

    def quantize_activation_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """Quantize activation gradients with lower precision."""
        if self.activation_grad_bits == 32:
            return grad
        return self.activation_grad_quantizer(grad)


class MultiPrecisionQuantizer:
    """
    Manager for multiple quantizers at different bit-widths.
    Used by CPT model to switch between precisions.
    """

    def __init__(self, bit_widths: list = [4, 6, 8], quantizer_type: str = 'log'):
        self.bit_widths = bit_widths
        self.quantizer_type = quantizer_type

        # Create quantizer for each bit-width
        self.quantizers = {}
        for bits in bit_widths:
            if quantizer_type == 'log':
                self.quantizers[bits] = PerChannelLogQuantization(num_bits=bits)
            elif quantizer_type == 'sbm':
                self.quantizers[bits] = SBMQuantization(num_bits=bits)
            else:
                raise ValueError(f"Unknown quantizer type: {quantizer_type}")

    def get_quantizer(self, num_bits: int):
        """Get quantizer for specific bit-width."""
        if num_bits not in self.quantizers:
            raise ValueError(f"No quantizer for {num_bits} bits")
        return self.quantizers[num_bits]

    def quantize(self, x: torch.Tensor, num_bits: int) -> torch.Tensor:
        """Quantize tensor with specified precision."""
        return self.quantizers[num_bits](x)
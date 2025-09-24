"""
Range Batch Normalization from SBM paper.
Replaces variance computation with range for better numerical stability in INT8 training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class RangeBatchNorm(nn.Module):
    """
    Range Batch Normalization from SBM paper Section 3.
    Uses range (max - min) instead of variance for normalization.
    More stable for low-precision training.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # Affine parameters
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # Running statistics
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_range', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_range', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with range normalization.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim] or [batch, hidden_dim]

        Returns:
            Normalized tensor
        """
        # Handle different input shapes
        if x.dim() == 3:
            # [batch, seq_len, hidden_dim]
            batch_size = x.size(0) * x.size(1)
            x_reshaped = x.view(-1, self.num_features)
        elif x.dim() == 2:
            # [batch, hidden_dim]
            batch_size = x.size(0)
            x_reshaped = x
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        # Calculate statistics
        if self.training:
            # Calculate mean
            mean = x_reshaped.mean(dim=0)

            # Calculate range per feature
            x_centered = x_reshaped - mean.unsqueeze(0)
            max_vals = x_centered.max(dim=0)[0]
            min_vals = x_centered.min(dim=0)[0]
            range_vals = max_vals - min_vals

            # Scale adjustment C(n) from SBM paper Equation 2
            # C(n) = 1 / sqrt(2 * ln(n))
            C_n = 1.0 / math.sqrt(2 * math.log(batch_size + 1))

            # Adjusted range
            adjusted_range = C_n * range_vals + self.eps

            # Update running statistics
            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked += 1
                    if self.momentum is None:
                        # Cumulative moving average
                        exponential_average_factor = 1.0 / self.num_batches_tracked.item()
                    else:
                        exponential_average_factor = self.momentum

                    self.running_mean = (
                        (1 - exponential_average_factor) * self.running_mean +
                        exponential_average_factor * mean
                    )
                    self.running_range = (
                        (1 - exponential_average_factor) * self.running_range +
                        exponential_average_factor * adjusted_range
                    )
        else:
            # Use running statistics
            if self.track_running_stats:
                mean = self.running_mean
                adjusted_range = self.running_range
            else:
                # Calculate statistics for evaluation
                mean = x_reshaped.mean(dim=0)
                x_centered = x_reshaped - mean.unsqueeze(0)
                max_vals = x_centered.max(dim=0)[0]
                min_vals = x_centered.min(dim=0)[0]
                range_vals = max_vals - min_vals
                C_n = 1.0 / math.sqrt(2 * math.log(batch_size + 1))
                adjusted_range = C_n * range_vals + self.eps

        # Normalize using range
        x_normalized = (x_reshaped - mean.unsqueeze(0)) / adjusted_range.unsqueeze(0)

        # Apply affine transformation
        if self.affine:
            x_normalized = x_normalized * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

        # Reshape back to original shape
        if x.dim() == 3:
            x_normalized = x_normalized.view(x.size(0), x.size(1), self.num_features)

        return x_normalized

    def extra_repr(self) -> str:
        return (
            f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, '
            f'affine={self.affine}, track_running_stats={self.track_running_stats}'
        )


class RangeBatchNorm1d(RangeBatchNorm):
    """1D Range Batch Normalization for linear layers."""
    pass


class RangeLayerNorm(nn.Module):
    """
    Range-based Layer Normalization.
    Similar to RangeBN but normalizes across hidden dimension for each token.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with range-based layer normalization.

        Args:
            x: Input tensor [..., normalized_shape]

        Returns:
            Normalized tensor
        """
        # Calculate mean
        mean = x.mean(dim=-1, keepdim=True)

        # Center the input
        x_centered = x - mean

        # Calculate range
        max_val = x_centered.max(dim=-1, keepdim=True)[0]
        min_val = x_centered.min(dim=-1, keepdim=True)[0]
        range_val = max_val - min_val

        # Scale adjustment based on dimension size
        # Using sequence length or hidden dimension as 'n'
        n = x.size(-1)
        C_n = 1.0 / math.sqrt(2 * math.log(n + 1))

        # Normalize by adjusted range
        x_normalized = x_centered / (C_n * range_val + self.eps)

        # Apply affine transformation
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


def replace_batchnorm_with_rangebn(model: nn.Module):
    """
    Utility function to replace all BatchNorm layers with RangeBatchNorm.
    """
    for name, child in model.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            # Create RangeBatchNorm with same parameters
            range_bn = RangeBatchNorm(
                num_features=child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats
            )

            # Copy weights if they exist
            if child.affine:
                range_bn.weight.data = child.weight.data.clone()
                range_bn.bias.data = child.bias.data.clone()

            # Copy running stats if tracked
            if child.track_running_stats:
                range_bn.running_mean.data = child.running_mean.data.clone()
                # Convert variance to range approximation
                range_bn.running_range.data = 2 * torch.sqrt(child.running_var.data + child.eps)
                range_bn.num_batches_tracked.data = child.num_batches_tracked.data.clone()

            # Replace the module
            setattr(model, name, range_bn)
        else:
            # Recursively replace in child modules
            replace_batchnorm_with_rangebn(child)


def replace_layernorm_with_rangeln(model: nn.Module):
    """
    Utility function to replace all LayerNorm layers with RangeLayerNorm.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.LayerNorm):
            # Create RangeLayerNorm with same parameters
            range_ln = RangeLayerNorm(
                normalized_shape=child.normalized_shape[0] if isinstance(child.normalized_shape, tuple) else child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.elementwise_affine
            )

            # Copy weights if they exist
            if child.elementwise_affine:
                range_ln.weight.data = child.weight.data.clone()
                range_ln.bias.data = child.bias.data.clone()

            # Replace the module
            setattr(model, name, range_ln)
        else:
            # Recursively replace in child modules
            replace_layernorm_with_rangeln(child)
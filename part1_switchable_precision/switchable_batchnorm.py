
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union


class SwitchableLayerNorm(nn.Module):
    """
    Ensure privacy for each bit width by maintaining separate
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        precision_levels: List[int] = [6, 8, 16, 32],
        eps: float = 1e-5
    ):

        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.precision_levels = sorted(precision_levels)
        self.eps = eps
        self.weights = nn.ParameterDict()
        self.biases = nn.ParameterDict()

        for precision in self.precision_levels:
            # Initialize with dict keys to ensure privacy.
            self.weights[str(precision)] = nn.Parameter(
                torch.ones(normalized_shape)
            )
            self.biases[str(precision)] = nn.Parameter(
                torch.zeros(normalized_shape)
            )
        # Default to highest precision
        self.current_precision = max(self.precision_levels)

    def set_precision(self, precision: int) -> int:
        if precision not in self.precision_levels:
            raise ValueError(f"Precision {precision} not supported. Available: {self.precision_levels}")
        self.current_precision = precision
        return self.current_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weights[str(self.current_precision)]
        bias = self.biases[str(self.current_precision)]
        return weight * x_normalized + bias

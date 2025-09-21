
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union


class SwitchableLayerNorm(nn.Module):
    """
    Switchable Layer Normalization for Transformer models.
    Maintains separate LN parameters for each precision level.
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        precision_levels: List[int] = [4, 8, 16, 32],
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        """
        Args:
            normalized_shape: Shape for normalization
            precision_levels: List of supported bit widths
            eps: Small constant for numerical stability
            elementwise_affine: Whether to learn affine parameters
        """
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.precision_levels = sorted(precision_levels)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        # Create LN layers for each precision
        self.ln_layers = nn.ModuleDict()
        for precision in self.precision_levels:
            self.ln_layers[f'ln_{precision}bit'] = nn.LayerNorm(
                normalized_shape=normalized_shape,
                eps=eps,
                elementwise_affine=elementwise_affine
            )

        # Default to highest precision
        self.current_precision = max(self.precision_levels)

    def set_precision(self, precision: int) -> int:
        """Set the current precision level.

        Returns:
            int: The current precision after setting
        """
        if precision not in self.precision_levels:
            raise ValueError(f"Precision {precision} not supported. Available: {self.precision_levels}")
        self.current_precision = precision
        return self.current_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using LN parameters for current precision."""
        ln_key = f'ln_{self.current_precision}bit'
        return self.ln_layers[ln_key](x)

    def extra_repr(self) -> str:
        return (f'{self.normalized_shape}, precisions={self.precision_levels}, '
                f'eps={self.eps}, elementwise_affine={self.elementwise_affine}')


def replace_bn_with_switchable(
    module: nn.Module,
    precision_levels: List[int] = [4, 8, 16, 32],
    inplace: bool = False
) -> nn.Module:
    """
    Replace all BatchNorm layers with Switchable versions.

    Args:
        module: Module to process
        precision_levels: Supported precision levels
        inplace: Whether to modify in-place or create copy

    Returns:
        Module with switchable BN layers
    """
    if not inplace:
        import copy
        module = copy.deepcopy(module)

    def replace_bn(m):
        for name, child in m.named_children():
            if isinstance(child, nn.BatchNorm1d):
                setattr(m, name, SwitchableBatchNorm1d(
                    num_features=child.num_features,
                    precision_levels=precision_levels,
                    eps=child.eps,
                    momentum=child.momentum,
                    affine=child.affine,
                    track_running_stats=child.track_running_stats
                ))
            # Skip BatchNorm2d - not needed for transformer models
            elif isinstance(child, nn.LayerNorm):
                setattr(m, name, SwitchableLayerNorm(
                    normalized_shape=child.normalized_shape,
                    precision_levels=precision_levels,
                    eps=child.eps,
                    elementwise_affine=child.elementwise_affine
                ))
            else:
                replace_bn(child)

    replace_bn(module)
    return module
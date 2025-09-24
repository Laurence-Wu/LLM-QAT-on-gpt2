#!/usr/bin/env python3
"""
Utility functions for testing the Switchable Precision model.
"""

import sys
import os
import torch

# Add parent directory (part1_switchable_precision) to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


def get_configured_bit_widths(model=None, config=None):
    """
    Get configured bit widths from model or config.

    Args:
        model: SPLMHeadModel instance (optional)
        config: ModelConfig instance (optional)

    Returns:
        List of configured bit widths
    """
    # Try to get from model first
    if model is not None:
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'bit_widths'):
            return model.transformer.bit_widths
        # Try to get from model's layers
        for name, module in model.named_modules():
            if hasattr(module, 'bit_widths'):
                return module.bit_widths

    # Try to get from config
    if config is not None:
        if hasattr(config, 'bit_widths'):
            return config.bit_widths

    # Fallback: import from config_sp
    try:
        from config_sp import ModelConfig
        return ModelConfig().bit_widths
    except ImportError:
        # Ultimate fallback - return the current default
        print("Warning: Could not determine configured bit widths, using default [6, 8, 16, 32]")
        return [6, 8, 16, 32]


def get_student_precisions(bit_widths):
    """
    Get student precisions (all precisions except 32-bit teacher).

    Args:
        bit_widths: List of all bit widths

    Returns:
        List of student precisions
    """
    return [b for b in bit_widths if b < 32]


def get_quantizer_type(model, precision):
    """
    Get the quantizer type for a given precision from the model.

    Args:
        model: SPLMHeadModel instance
        precision: Bit width to check

    Returns:
        str: Quantizer type ('log', 'minmax', 'relu_clip', 'tanh')
    """
    bits_key = f'{precision}bit'

    # Find a quantizer to check its type
    for name, module in model.named_modules():
        if hasattr(module, 'quantizers_weight') and bits_key in module.quantizers_weight:
            quantizer = module.quantizers_weight[bits_key]
            if hasattr(quantizer, 'quantizer_type'):
                return quantizer.quantizer_type

    # Fallback to config
    try:
        from config_sp import ModelConfig
        config = ModelConfig()
        if hasattr(config, 'quantizer_per_bit') and precision in config.quantizer_per_bit:
            return config.quantizer_per_bit[precision]
        elif hasattr(config, 'quantizer_type'):
            return config.quantizer_type
    except ImportError:
        pass

    return 'minmax'  # Default fallback
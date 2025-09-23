"""
Different quantization function implementations for various quantization methods.
Each method has its own custom autograd function with straight-through estimator.
"""

import torch
import torch.nn as nn


class MinMaxQuantizationFunction(torch.autograd.Function):
    """Standard min-max quantization with straight-through estimator."""
    @staticmethod
    def forward(ctx, input, scale, zero_point, num_bits, symmetric):
        ctx.save_for_backward(input, scale.clone(), zero_point.clone())
        ctx.num_bits = num_bits
        ctx.symmetric = symmetric

        if symmetric:
            # Symmetric quantization
            quantized = torch.round(input / scale)
            quantized = torch.clamp(quantized, -(2**(num_bits-1)), 2**(num_bits-1) - 1)
            output = quantized * scale
        else:
            # Asymmetric quantization
            quantized = torch.round(input / scale + zero_point)
            quantized = torch.clamp(quantized, 0, 2**num_bits - 1)
            output = (quantized - zero_point) * scale

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        input, scale, zero_point = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


class ReLUClipQuantizationFunction(torch.autograd.Function):
    """ReLU-style quantization that clips to [0, max] range."""
    @staticmethod
    def forward(ctx, input, scale, max_val, num_bits):
        ctx.save_for_backward(input, scale.clone(), max_val.clone())
        ctx.num_bits = num_bits

        # Clip to non-negative range
        input_clipped = torch.clamp(input, min=0.0, max=max_val)

        # Quantize in [0, max] range
        quantized = torch.round(input_clipped / scale)
        quantized = torch.clamp(quantized, 0, 2**num_bits - 1)
        output = quantized * scale

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator with ReLU gradient
        input, scale, max_val = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Zero gradient for negative inputs (ReLU behavior)
        grad_input = grad_input * (input >= 0).float()
        # Zero gradient for inputs above max
        grad_input = grad_input * (input <= max_val).float()
        return grad_input, None, None, None


class TanhQuantizationFunction(torch.autograd.Function):
    """Tanh-based quantization that maps inputs through tanh to bounded range."""
    @staticmethod
    def forward(ctx, input, input_scale, num_bits, symmetric):
        ctx.save_for_backward(input, input_scale.clone())
        ctx.num_bits = num_bits
        ctx.symmetric = symmetric

        # Scale input based on calibration
        eps = ctx.eps if hasattr(ctx, 'eps') else 1e-5
        input_scaled = input / (input_scale + eps)

        # Apply tanh to bound to [-1, 1]
        input_tanh = torch.tanh(input_scaled)

        # Quantize in [-1, 1] or [0, 1] range
        if symmetric:
            quantized = torch.round(input_tanh * (2**(num_bits-1) - 1))
            quantized = torch.clamp(quantized, -(2**(num_bits-1) - 1), 2**(num_bits-1) - 1)
            output_normalized = quantized / (2**(num_bits-1) - 1)
        else:
            # Map to [0, 1] for asymmetric
            input_normalized = (input_tanh + 1) / 2
            quantized = torch.round(input_normalized * (2**num_bits - 1))
            quantized = torch.clamp(quantized, 0, 2**num_bits - 1)
            output_normalized = quantized / (2**num_bits - 1)
            output_normalized = output_normalized * 2 - 1  # Back to [-1, 1]

        # Scale back to original range
        output = output_normalized * input_scale

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator with tanh derivative consideration
        input, input_scale = ctx.saved_tensors
        eps = ctx.eps if hasattr(ctx, 'eps') else 1e-5
        input_scaled = input / (input_scale + eps)
        input_tanh = torch.tanh(input_scaled)

        # Tanh derivative: d/dx tanh(x) = 1 - tanh^2(x)
        # But we use straight-through for simplicity
        grad_input = grad_output.clone()

        # Optional: Scale gradient by tanh derivative for better flow
        # tanh_derivative = 1 - input_tanh ** 2
        # grad_input = grad_output * tanh_derivative

        return grad_input, None, None, None


class LogQuantizationFunction(torch.autograd.Function):
    """
    Logarithmic quantization following the SP-Net paper formula:
    LogQuant(x) = 0 if x = 0, else 2^x_hat · sign(x)
    where x_hat = rescale(Q(normalize(|log₂(x)|)))
    """
    @staticmethod
    def forward(ctx, input, log_min, log_range, num_bits):
        """
        Args:
            input: Tensor to quantize
            log_min: Pre-calibrated min for log₂ space
            log_range: Pre-calibrated range for log₂ space
            num_bits: Number of bits for quantization
        """
        eps = ctx.eps if hasattr(ctx, 'eps') else 1e-5

        # Save for backward pass
        ctx.save_for_backward(input)
        ctx.num_bits = num_bits

        # Step 1: Handle exact zeros (LogQuant(x) = 0 if x = 0)
        zero_mask = (torch.abs(input) < eps)

        # Step 2: Extract sign s(x) = sign(x)
        sign_input = torch.sign(input)

        # Step 3: Get absolute value and prevent log(0)
        abs_input = torch.abs(input).clamp(min=eps)

        # Step 4: Apply base-2 logarithm as specified in the paper
        # Paper states: "logarithm base-2 was used"
        log_abs_input = torch.log2(abs_input)  # Critical: use log₂

        # Step 5: Normalize using calibrated stats
        # normalize(x) = (x - min(x))/(max(x) - min(x))
        log_normalized = (log_abs_input - log_min) / (log_range.clamp(min=eps))
        log_normalized = torch.clamp(log_normalized, 0, 1)

        # Step 6: Quantize Q(normalized) with num_bits levels
        n_levels = 2**num_bits - 1
        quantized = torch.round(log_normalized * n_levels)
        quantized = torch.clamp(quantized, 0, n_levels)

        # Step 7: Dequantize back to normalized space [0, 1]
        q_normalized = quantized / n_levels

        # Step 8: Rescale back to original log space
        # rescale(x) = x · (max(x) - min(x)) + min(x)
        x_hat = q_normalized * log_range + log_min

        # Step 9: Apply 2^x_hat (NOT e^x_hat!) to return to linear space
        # This is critical - must use same base as the logarithm
        magnitude = torch.pow(2, x_hat)  # 2^x_hat, not e^x_hat

        # Step 10: Apply sign: 2^x_hat · sign(x)
        output = magnitude * sign_input

        # Step 11: Restore exact zeros
        output = torch.where(zero_mask, torch.zeros_like(input), output)

        # Store statistics for potential reuse
        ctx.log_min = log_min
        ctx.log_range = log_range

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Straight-through estimator as mentioned in the paper
        """
        input, = ctx.saved_tensors

        # Basic STE: pass gradients through
        grad_input = grad_output.clone()

        # Optional: clip gradients for stability with logarithmic quantization
        # This helps prevent gradient explosion near zero values
        grad_input = torch.clamp(grad_input, -10, 10)

        return grad_input, None, None, None


def apply_minmax_quantization(x, scale, zero_point, num_bits, symmetric):
    """
    Apply standard min-max quantization.
    Uses the formula: X_Q = α⌊X_R/α⌉ where α = max(|X_R|)/(2^(N-1) - 1)

    Args:
        x: Input tensor
        scale: Quantization scale
        zero_point: Quantization zero point
        num_bits: Number of quantization bits
        symmetric: Whether to use symmetric quantization

    Returns:
        Quantized tensor
    """
    return MinMaxQuantizationFunction.apply(x, scale, zero_point, num_bits, symmetric)


def apply_log_quantization(x, log_min, log_range, num_bits):
    """
    Apply logarithmic quantization following LogQuant formula.
    LogQuant(x) = 0 if x = 0, else 2^x_hat · sign(x)

    Args:
        x: Input tensor
        log_min: Minimum log value from calibration
        log_range: Log range from calibration
        num_bits: Number of quantization bits

    Returns:
        Quantized tensor
    """
    return LogQuantizationFunction.apply(x, log_min, log_range, num_bits)
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
        eps = 1e-7
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
        eps = 1e-7
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
    """Logarithmic quantization for non-uniform quantization levels."""
    @staticmethod
    def forward(ctx, input, log_min, log_range, num_bits):
        ctx.save_for_backward(input, log_min.clone(), log_range.clone())
        ctx.num_bits = num_bits

        eps = 1e-7

        # Separate sign and magnitude
        sign_input = torch.sign(input)
        abs_input = torch.abs(input) + eps

        # Apply logarithm to magnitude
        log_input = torch.log2(abs_input)

        # Normalize to [0, 1] using calibrated range
        log_normalized = (log_input - log_min) / (log_range + eps)
        log_normalized = torch.clamp(log_normalized, 0, 1)

        # Quantize uniformly in log space
        quantized = torch.round(log_normalized * (2**num_bits - 1))
        quantized = torch.clamp(quantized, 0, 2**num_bits - 1)

        # Dequantize
        log_dequant_normalized = quantized / (2**num_bits - 1)
        log_dequant = log_dequant_normalized * log_range + log_min

        # Convert back to linear space
        abs_output = torch.pow(2, log_dequant)
        output = sign_input * abs_output

        # Handle zeros (input values that were very close to zero)
        output = torch.where(torch.abs(input) < eps, torch.zeros_like(input), output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        input, log_min, log_range = ctx.saved_tensors
        grad_input = grad_output.clone()

        # Optional: Scale gradient by logarithmic derivative
        # eps = 1e-7
        # abs_input = torch.abs(input) + eps
        # log_derivative = 1 / (abs_input * torch.log(2))
        # grad_input = grad_output * log_derivative

        return grad_input, None, None, None
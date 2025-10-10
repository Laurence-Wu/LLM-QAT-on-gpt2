
import torch
import torch.nn as nn

class MinMaxQuantizationFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, scale, zero_point, num_bits, symmetric):
        ctx.save_for_backward(input, scale.clone(), zero_point.clone())
        ctx.num_bits = num_bits
        ctx.symmetric = symmetric

        if symmetric:
            quantized = torch.round(input / scale)
            quantized = torch.clamp(quantized, -(2**(num_bits-1) - 1), 2**(num_bits-1) - 1)
            output = quantized * scale
        else:
            quantized = torch.round(input / scale + zero_point)
            quantized = torch.clamp(quantized, 0, 2**num_bits - 1)
            output = (quantized - zero_point) * scale

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, scale, zero_point = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None, None

class LogQuantizationFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, log_min, log_range, num_bits, symmetric):
        
        eps = ctx.eps if hasattr(ctx, 'eps') else 1e-5

        ctx.save_for_backward(input)
        ctx.num_bits = num_bits
        ctx.symmetric = symmetric

        zero_mask = (torch.abs(input) < eps)

        sign_input = torch.sign(input)

        abs_input = torch.abs(input).clamp(min=eps)

        log_abs_input = torch.log2(abs_input)

        log_normalized = (log_abs_input - log_min) / (log_range.clamp(min=eps))
        log_normalized = torch.clamp(log_normalized, 0, 1)

        if symmetric:
            n_levels = 2**(num_bits - 1) - 1
            centered = log_normalized - 0.5
            quantized = torch.round(centered * 2 * n_levels)
            quantized = torch.clamp(quantized, -n_levels, n_levels)
            quantized = (quantized / (2 * n_levels) + 0.5) * (2**num_bits - 1)
        else:
            n_levels = 2**num_bits - 1
            quantized = torch.round(log_normalized * n_levels)
            quantized = torch.clamp(quantized, 0, n_levels)

        if symmetric:
            q_normalized = quantized / (2**num_bits - 1)
        else:
            q_normalized = quantized / n_levels

        x_hat = q_normalized * log_range + log_min

        magnitude = torch.pow(2, x_hat)

        output = magnitude * sign_input

        output = torch.where(zero_mask, torch.zeros_like(input), output)

        ctx.log_min = log_min
        ctx.log_range = log_range

        return output

    @staticmethod
    def backward(ctx, grad_output):
        
        input, = ctx.saved_tensors

        grad_input = grad_output.clone()

        grad_input = torch.clamp(grad_input, -10, 10)

        return grad_input, None, None, None, None

def apply_minmax_quantization(x, scale, zero_point, num_bits, symmetric=True):
    
    return MinMaxQuantizationFunction.apply(x, scale, zero_point, num_bits, symmetric)

def apply_log_quantization(x, log_min, log_range, num_bits, symmetric=True):
    
    return LogQuantizationFunction.apply(x, log_min, log_range, num_bits, symmetric)
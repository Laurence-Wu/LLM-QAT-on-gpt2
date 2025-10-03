import torch

class MinMaxQuantizationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, num_bits, symmetric):
        ctx.save_for_backward(input, scale.clone(), zero_point.clone())
        ctx.num_bits = num_bits
        ctx.symmetric = symmetric

        if symmetric:
            quantized = torch.round(input / scale)
            quantized = torch.clamp(quantized, -(2**(num_bits-1) - 1), 2**(num_bits-1) - 1)
            return quantized * scale
        else:
            quantized = torch.round(input / scale + zero_point)
            quantized = torch.clamp(quantized, 0, 2**num_bits - 1)
            return (quantized - zero_point) * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None, None, None, None

class LogQuantizationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, log_min, log_range, num_bits, symmetric):
        eps = 1e-5
        ctx.save_for_backward(input)
        ctx.num_bits = num_bits
        ctx.symmetric = symmetric

        zero_mask = (torch.abs(input) < eps)
        sign_input = torch.sign(input)
        log_abs_input = torch.log2(torch.abs(input).clamp(min=eps))
        log_normalized = torch.clamp((log_abs_input - log_min) / log_range.clamp(min=eps), 0, 1)

        if symmetric:
            n_levels = 2**(num_bits - 1) - 1
            quantized = torch.round((log_normalized - 0.5) * 2 * n_levels)
            quantized = torch.clamp(quantized, -n_levels, n_levels)
            q_normalized = (quantized / (2 * n_levels) + 0.5)
        else:
            n_levels = 2**num_bits - 1
            quantized = torch.clamp(torch.round(log_normalized * n_levels), 0, n_levels)
            q_normalized = quantized / n_levels

        output = torch.pow(2, q_normalized * log_range + log_min) * sign_input
        return torch.where(zero_mask, torch.zeros_like(input), output)

    @staticmethod
    def backward(ctx, grad_output):
        return torch.clamp(grad_output.clone(), -10, 10), None, None, None, None

def apply_minmax_quantization(x, scale, zero_point, num_bits, symmetric=True):
    return MinMaxQuantizationFunction.apply(x, scale, zero_point, num_bits, symmetric)

def apply_log_quantization(x, log_min, log_range, num_bits, symmetric=True):
    return LogQuantizationFunction.apply(x, log_min, log_range, num_bits, symmetric)

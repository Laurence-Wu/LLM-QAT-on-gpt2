
import os
import sys
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from quantization_methods import (
    apply_minmax_quantization,
    apply_log_quantization
)


class GradientQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, quantizer):
        ctx.quantizer = quantizer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.quantizer is not None and ctx.quantizer.collecting_stats:
            return ctx.quantizer(grad_output), None
        if ctx.quantizer is not None and ctx.quantizer.calibrated:
            return ctx.quantizer(grad_output), None
        return grad_output, None

class LearnableFakeQuantize(nn.Module):
    def __init__(self, num_bits,
                 channel_dim=0, quantizer_type='minmax', eps=1e-5, symmetric=True, per_channel=True, is_input=False):
        super().__init__()
        self.num_bits = max(1, min(num_bits, 32))
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_dim = channel_dim if per_channel else None
        self.quantizer_type = quantizer_type
        self.eps = eps
        self.is_input = is_input

        self._update_quant_range()

        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))

        self.calibrated = False
        self.collecting_stats = False
        self.num_batches_collected = 0
        self.temp_min = None
        self.temp_max = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        
        if self.is_input:
            for buffer_name in ['scale', 'zero_point', 'running_min', 'running_max']:
                key = prefix + buffer_name
                if key in state_dict:
                    loaded_tensor = state_dict[key]
                    buffer = getattr(self, buffer_name, None)
                    if buffer is not None:
                        if loaded_tensor.dim() == 3 and loaded_tensor.shape[1] > 1:
                            if self.quantizer_type == 'log':
                                reduced = loaded_tensor.max(dim=1, keepdim=True)[0]
                            else:
                                if 'min' in buffer_name:
                                    reduced = loaded_tensor.min(dim=1, keepdim=True)[0]
                                elif 'max' in buffer_name:
                                    reduced = loaded_tensor.max(dim=1, keepdim=True)[0]
                                else:
                                    reduced = loaded_tensor.max(dim=1, keepdim=True)[0]
                            state_dict[key] = reduced
                        buffer.resize_as_(state_dict[key])
        else:
            for buffer_name in ['scale', 'zero_point', 'running_min', 'running_max']:
                key = prefix + buffer_name
                if key in state_dict:
                    buffer = getattr(self, buffer_name, None)
                    if buffer is not None:
                        buffer.resize_as_(state_dict[key])

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

        if (prefix + 'scale' in state_dict and
            prefix + 'zero_point' in state_dict):
            self.calibrated = True

    def set_num_bits(self, value):
        old_bits = self.num_bits
        self.num_bits = max(1, min(value, 32))
        self._update_quant_range()
        if old_bits != self.num_bits:
            self.calibrated = False

    def _update_quant_range(self):
        if self.symmetric:
            self.quant_min = -(2 ** (self.num_bits - 1))
            self.quant_max = 2 ** (self.num_bits - 1) - 1
        else:
            self.quant_min = 0
            self.quant_max = 2 ** self.num_bits - 1

    def start_calibration(self):
        self.collecting_stats = True
        self.calibrated = False
        self.num_batches_collected = 0
        self.temp_min = None
        self.temp_max = None

    def finish_calibration(self, debug=False):
        if self.num_batches_collected > 0 and self.temp_min is not None:
            self.running_min.resize_as_(self.temp_min).copy_(self.temp_min)
            self.running_max.resize_as_(self.temp_max).copy_(self.temp_max)
            with torch.no_grad():
                if self.quantizer_type == 'log':
                    log_range = self.running_max - self.running_min
                    self.zero_point.resize_as_(self.running_max).copy_(self.running_min)
                    self.scale.resize_as_(self.running_max).copy_(log_range)
                else:
                    if self.symmetric:
                        abs_max = torch.clamp(torch.max(torch.abs(self.running_min), torch.abs(self.running_max)), min=self.eps)
                        self.scale.resize_as_(abs_max).copy_(abs_max / (2**(self.num_bits-1) - 1))
                        self.zero_point.resize_as_(abs_max).zero_()
                    else:
                        range_val = torch.clamp(self.running_max - self.running_min, min=self.eps)
                        self.scale.resize_as_(range_val).copy_(range_val / (2**self.num_bits - 1))
                        self.zero_point.resize_as_(range_val).copy_(torch.round(-self.running_min / self.scale))

                # Validate scales for numerical stability
                scale_min = self.scale.min().item()
                scale_max = self.scale.max().item()
                scale_mean = self.scale.mean().item()

                if scale_min < 1e-6:
                    print(f"⚠️ WARNING: Very small scale detected!")
                    print(f"  Scale min: {scale_min:.2e}")
                    print(f"  Scale max: {scale_max:.2e}")
                    print(f"  Scale mean: {scale_mean:.2e}")
                    print(f"  Quantizer: {self.num_bits}-bit {self.quantizer_type}")
                    print(f"  Running min: {self.running_min.min().item():.2e}")
                    print(f"  Running max: {self.running_max.max().item():.2e}")

                if debug:
                    print(f"         Computed scale: mean={self.scale.mean().item():.6f}")

            self.calibrated = True
            self.collecting_stats = False
            self.temp_min = None
            self.temp_max = None
        else:
            self.collecting_stats = False
            if debug:
                print(f"      ⚠️ No statistics collected for {self.num_bits}-bit {self.quantizer_type} quantizer")

    def _get_reduction_dims(self, tensor):
        if self.per_channel and self.channel_dim is not None:
            dims_to_reduce = list(range(tensor.dim()))
            actual_dim = self.channel_dim if self.channel_dim >= 0 else tensor.dim() + self.channel_dim
            if actual_dim in dims_to_reduce:
                dims_to_reduce.remove(actual_dim)
        else:
            dims_to_reduce = list(range(tensor.dim()))
        return dims_to_reduce

    def _reduce_min_max(self, tensor, dims_to_reduce):
        if not dims_to_reduce:
            return tensor, tensor
        min_val = tensor
        max_val = tensor
        for dim in sorted(dims_to_reduce, reverse=True):
            min_val = min_val.min(dim=dim, keepdim=True)[0]
            max_val = max_val.max(dim=dim, keepdim=True)[0]
        return min_val, max_val

    def _get_default_shape(self, x, default_value):
        if self.per_channel and self.channel_dim is not None:
            shape = list(x.shape)
            actual_dim = self.channel_dim if self.channel_dim >= 0 else len(shape) + self.channel_dim
            shape[actual_dim] = 1
            return torch.full(shape, default_value, device=x.device)
        else:
            return torch.tensor(default_value, device=x.device)

    def _collect_statistics_batch(self, x):
        with torch.no_grad():
            if self.quantizer_type == 'log':
                abs_x = torch.abs(x)
                non_zero_mask = abs_x > self.eps

                if non_zero_mask.any():
                    log_x = torch.log2(torch.clamp(abs_x, min=self.eps))
                    dims_to_reduce = self._get_reduction_dims(log_x)
                    min_val, max_val = self._reduce_min_max(log_x, dims_to_reduce)

                    if self.num_batches_collected == 0:
                        self.temp_min = min_val.clone().detach()
                        self.temp_max = max_val.clone().detach()
                    else:
                        self.temp_min = torch.minimum(self.temp_min, min_val)
                        self.temp_max = torch.maximum(self.temp_max, max_val)
                elif self.num_batches_collected == 0:
                    log_eps = torch.log2(torch.tensor(self.eps))
                    self.temp_min = self._get_default_shape(x, log_eps)
                    self.temp_max = self._get_default_shape(x, log_eps)
            else:
                dims_to_reduce = self._get_reduction_dims(x)
                min_val, max_val = self._reduce_min_max(x, dims_to_reduce)

                if self.num_batches_collected == 0:
                    self.temp_min = min_val.clone().detach()
                    self.temp_max = max_val.clone().detach()
                else:
                    self.temp_min = torch.minimum(self.temp_min, min_val)
                    self.temp_max = torch.maximum(self.temp_max, max_val)

            self.num_batches_collected += 1

    def forward(self, x):
        if self.num_bits >= 32:
            return x
        if self.collecting_stats:
            self._collect_statistics_batch(x)
            return x

        if not self.calibrated:
            return x

        if self.quantizer_type == 'minmax':
            return self._quantize_minmax(x)
        elif self.quantizer_type == 'log':
            return self._quantize_log(x)
        else:
            raise ValueError(f"Unknown quantizer type: {self.quantizer_type}. Supported types: 'minmax', 'log'")

    def _quantize_minmax(self, x):
        return apply_minmax_quantization(x, self.scale, self.zero_point, self.num_bits, self.symmetric)

    def _quantize_log(self, x):
        return apply_log_quantization(x, self.zero_point, self.scale, self.num_bits, self.symmetric)

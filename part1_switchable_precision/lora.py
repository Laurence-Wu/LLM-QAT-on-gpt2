import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from part1_switchable_precision.quantization import LearnableFakeQuantize

class LoRALayer(nn.Module):
    
    def __init__(self, in_features, out_features, rank, alpha, bits, quantizer_type, eps=1e-5, per_channel=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.bits = bits

        if bits >= 32 or rank <= 0:
            self.enabled = False
            self.scaling = 0
            self.register_buffer('lora_A', torch.zeros(1, 1))
            self.register_buffer('lora_B', torch.zeros(1, 1))
            self.quantize_A = None
            self.quantize_B = None

        else:
            self.enabled = True
            self.scaling = alpha / rank

            self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
            self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.quantize_A = LearnableFakeQuantize(num_bits=bits, quantizer_type=quantizer_type, channel_dim=1, eps=eps, per_channel=per_channel)
            self.quantize_B = LearnableFakeQuantize(num_bits=bits, quantizer_type=quantizer_type, channel_dim=1, eps=eps, per_channel=per_channel)
            
            self.register_buffer('lora_A_quantized', torch.empty(in_features, rank))
            self.register_buffer('lora_B_quantized', torch.empty(rank, out_features))

    def forward(self, x):
        if not self.enabled or self.scaling == 0:
            batch_shape = x.shape[:-1]
            return torch.zeros(*batch_shape, self.out_features, device=x.device, dtype=x.dtype)
        lora_A_q = self.quantize_A(self.lora_A)
        lora_B_q = self.quantize_B(self.lora_B)
        output = torch.matmul(x, lora_A_q)
        output = torch.matmul(output, lora_B_q)
        output = output * self.scaling
        return output

class SPLinearWithLoRA(nn.Module):
    
    def __init__(self, in_features, out_features,
                 bit_widths,
                 lora_rank_per_bit,
                 lora_alpha_per_bit,
                 quantizer_per_bit,
                 eps=1e-5,
                 per_channel=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bit_widths = bit_widths
        self.lora_rank_per_bit = lora_rank_per_bit
        student_bits = [b for b in bit_widths if b < 32]
        self.current_bits = sorted(bit_widths, reverse=True)[1]

        self.linear = nn.Linear(in_features, out_features, bias=True)

        self.quantizers_weight = nn.ModuleDict({
            f'{bits}bit': LearnableFakeQuantize(num_bits=bits,
                                                quantizer_type=quantizer_per_bit[bits], channel_dim=0, eps=eps, per_channel=per_channel)
            for bits in student_bits
        })
        self.quantizers_input = nn.ModuleDict({
            f'{bits}bit': LearnableFakeQuantize(num_bits=bits,
                                                quantizer_type=quantizer_per_bit[bits], channel_dim=-1, eps=eps,
                                                per_channel=per_channel, is_input=True)
            for bits in student_bits
        })

        self.lora_adapters = nn.ModuleDict({
            f'{bits}bit': LoRALayer(
                in_features, out_features,
                rank=lora_rank_per_bit[bits],
                alpha=lora_alpha_per_bit[bits],
                bits=bits,
                quantizer_type=quantizer_per_bit[bits],
                eps=eps,
                per_channel=per_channel
            )
            for bits in student_bits
        })

        self.register_buffer('weight_quantized', torch.empty(out_features, in_features))
        self.register_buffer('input_quantized', None)

        self.calibration_mode = False

    def set_precision(self, bits) -> int:
        if bits >= 32:
            self.current_bits = 32
            bits_key = f"{bits}bits"
            return bits
        self.current_bits = bits
        bits_key = f'{bits}bit'
        self.quantizers_weight[bits_key].set_num_bits(bits)
        self.quantizers_input[bits_key].set_num_bits(bits)

        lora = self.lora_adapters[bits_key]
        if lora.quantize_A is not None:
            lora.quantize_A.set_num_bits(bits)
        if lora.quantize_B is not None:
            lora.quantize_B.set_num_bits(bits)

        return self.current_bits

    def get_active_lora(self):
        
        return self.lora_adapters[f'{self.current_bits}bit']

    def forward(self, x):
        
        if self.current_bits >= 32:
            output = F.linear(x, self.linear.weight, self.linear.bias)
            return output

        bits_key = f'{self.current_bits}bit'

        if bits_key not in self.quantizers_weight or bits_key not in self.quantizers_input:
            raise KeyError(f"No weight quantizer for {bits_key}")

        weight_quantizer = self.quantizers_weight[bits_key]
        input_quantizer = self.quantizers_input[bits_key]
        active_lora = self.lora_adapters[bits_key]
        x_quantized = input_quantizer(x)
        weight_quantized = weight_quantizer(self.linear.weight)

        base_output = F.linear(x_quantized, weight_quantized, self.linear.bias)

        if self.calibration_mode:
            return base_output

        lora_output = active_lora(x)
        return base_output + lora_output
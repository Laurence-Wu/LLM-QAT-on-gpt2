"""
Debug 8-bit quantization to identify why activations have 100% error.
"""

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config as HFConfig
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import QATGPT2
from shared.quantization import LearnableFakeQuantize
from shared.lora import QATLinearWithLoRA
from fix_weight_loading import load_pretrained_weights_fixed


def debug_quantizer_state():
    """Debug the state of quantizers in the model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create config
    config = HFConfig()
    config.n_positions = 256
    config.n_layer = 12
    config.lora_rank = 8
    config.lora_alpha = 16
    config.lora_dropout = 0.0
    config.kv_cache_bits = 8

    # Create model with 8-bit precision
    qat_model = QATGPT2(config, quantization_bits=8, initialize_weights=False)
    load_pretrained_weights_fixed(qat_model, debug=False)
    qat_model = qat_model.to(device)
    qat_model.eval()

    print("="*60)
    print("CHECKING QUANTIZER INITIALIZATION")
    print("="*60)

    # Check first layer's quantizers
    first_attn = qat_model.h[0].attn.c_attn

    print("\nWeight Quantizer (symmetric):")
    wq = first_attn.quantize_weight
    print(f"  Num bits: {wq.num_bits}")
    print(f"  Symmetric: {wq.symmetric}")
    print(f"  Calibrated: {wq.calibrated}")
    print(f"  Scale: {wq.scale.item():.6f}")
    print(f"  Zero point: {wq.zero_point.item():.6f}")
    print(f"  Running min: {wq.running_min.item():.6f}")
    print(f"  Running max: {wq.running_max.item():.6f}")
    print(f"  Quant min: {wq.quant_min}")
    print(f"  Quant max: {wq.quant_max}")

    print("\nInput/Activation Quantizer (asymmetric):")
    iq = first_attn.quantize_input
    print(f"  Num bits: {iq.num_bits}")
    print(f"  Symmetric: {iq.symmetric}")
    print(f"  Calibrated: {iq.calibrated}")
    print(f"  Scale: {iq.scale.item():.6f}")
    print(f"  Zero point: {iq.zero_point.item():.6f}")
    print(f"  Running min: {iq.running_min.item():.6f}")
    print(f"  Running max: {iq.running_max.item():.6f}")
    print(f"  Quant min: {iq.quant_min}")
    print(f"  Quant max: {iq.quant_max}")

    print("\n" + "="*60)
    print("TESTING QUANTIZATION ON SAMPLE DATA")
    print("="*60)

    # Create test input
    test_input = torch.randn(1, 10, 768).to(device)
    print(f"\nTest input stats:")
    print(f"  Mean: {test_input.mean().item():.6f}")
    print(f"  Std: {test_input.std().item():.6f}")
    print(f"  Min: {test_input.min().item():.6f}")
    print(f"  Max: {test_input.max().item():.6f}")

    # Test weight quantization
    test_weight = first_attn.linear.weight
    print(f"\nOriginal weight stats:")
    print(f"  Mean: {test_weight.mean().item():.6f}")
    print(f"  Std: {test_weight.std().item():.6f}")
    print(f"  Min: {test_weight.min().item():.6f}")
    print(f"  Max: {test_weight.max().item():.6f}")

    # Quantize weight
    with torch.no_grad():
        quantized_weight = first_attn.quantize_weight(test_weight)
        weight_diff = (test_weight - quantized_weight).abs()
        print(f"\nWeight quantization:")
        print(f"  Mean abs diff: {weight_diff.mean().item():.6f}")
        print(f"  Max abs diff: {weight_diff.max().item():.6f}")
        print(f"  Relative error: {(weight_diff.mean() / test_weight.abs().mean()).item()*100:.2f}%")

    # Quantize input
    with torch.no_grad():
        quantized_input = first_attn.quantize_input(test_input)
        input_diff = (test_input - quantized_input).abs()
        print(f"\nInput quantization:")
        print(f"  Mean abs diff: {input_diff.mean().item():.6f}")
        print(f"  Max abs diff: {input_diff.max().item():.6f}")
        print(f"  Relative error: {(input_diff.mean() / test_input.abs().mean()).item()*100:.2f}%")

    # Check quantized values
    print(f"\nQuantized input stats:")
    print(f"  Mean: {quantized_input.mean().item():.6f}")
    print(f"  Std: {quantized_input.std().item():.6f}")
    print(f"  Min: {quantized_input.min().item():.6f}")
    print(f"  Max: {quantized_input.max().item():.6f}")
    print(f"  Unique values: {len(torch.unique(quantized_input))}")

    print("\n" + "="*60)
    print("TESTING CALIBRATION")
    print("="*60)

    # Create a fresh quantizer and calibrate it
    test_quantizer = LearnableFakeQuantize(num_bits=8, symmetric=False)
    test_quantizer.to(device)

    print("\nBefore calibration:")
    print(f"  Scale: {test_quantizer.scale.item():.6f}")
    print(f"  Zero point: {test_quantizer.zero_point.item():.6f}")
    print(f"  Running min: {test_quantizer.running_min.item():.6f}")
    print(f"  Running max: {test_quantizer.running_max.item():.6f}")
    print(f"  Calibrated: {test_quantizer.calibrated}")

    # Put in training mode to calibrate
    test_quantizer.train()
    calibration_data = torch.randn(100, 10, 768).to(device)

    print("\nCalibrating with 100 samples...")
    with torch.no_grad():
        for i in range(100):
            _ = test_quantizer(calibration_data[i:i+1])

    print(f"\nAfter calibration:")
    print(f"  Scale: {test_quantizer.scale.item():.6f}")
    print(f"  Zero point: {test_quantizer.zero_point.item():.6f}")
    print(f"  Running min: {test_quantizer.running_min.item():.6f}")
    print(f"  Running max: {test_quantizer.running_max.item():.6f}")
    print(f"  Calibrated: {test_quantizer.calibrated}")

    # Test quantization after calibration
    test_quantizer.eval()
    test_sample = torch.randn(1, 10, 768).to(device)
    quantized_sample = test_quantizer(test_sample)
    calib_diff = (test_sample - quantized_sample).abs()
    print(f"\nQuantization after calibration:")
    print(f"  Mean abs diff: {calib_diff.mean().item():.6f}")
    print(f"  Relative error: {(calib_diff.mean() / test_sample.abs().mean()).item()*100:.2f}%")

    print("\n" + "="*60)
    print("TESTING REAL FORWARD PASS")
    print("="*60)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    test_text = "The capital"
    inputs = tokenizer(test_text, return_tensors='pt').to(device)

    with torch.no_grad():
        # Get embeddings
        embeddings = qat_model.wte(inputs['input_ids'])
        pos_embeds = qat_model.wpe(torch.arange(inputs['input_ids'].shape[1], device=device))
        hidden = embeddings + pos_embeds

        print(f"Hidden state stats:")
        print(f"  Mean: {hidden.mean().item():.6f}")
        print(f"  Std: {hidden.std().item():.6f}")
        print(f"  Min: {hidden.min().item():.6f}")
        print(f"  Max: {hidden.max().item():.6f}")

        # Pass through layer norm
        ln_output = qat_model.h[0].ln_1(hidden)
        print(f"\nAfter LayerNorm:")
        print(f"  Mean: {ln_output.mean().item():.6f}")
        print(f"  Std: {ln_output.std().item():.6f}")
        print(f"  Min: {ln_output.min().item():.6f}")
        print(f"  Max: {ln_output.max().item():.6f}")

        # Check what happens in c_attn
        print("\nInside c_attn forward:")

        # Get quantized input
        x_q = first_attn.quantize_input(ln_output)
        print(f"  After input quantization:")
        print(f"    Mean: {x_q.mean().item():.6f}")
        print(f"    Std: {x_q.std().item():.6f}")
        print(f"    Diff from original: {(ln_output - x_q).abs().mean().item():.6f}")

        # Get quantized weight
        w_q = first_attn.quantize_weight(first_attn.linear.weight)
        print(f"  After weight quantization:")
        print(f"    Diff from original: {(first_attn.linear.weight - w_q).abs().mean().item():.6f}")

        # Linear operation
        output = nn.functional.linear(x_q, w_q, first_attn.linear.bias)
        print(f"  After linear:")
        print(f"    Mean: {output.mean().item():.6f}")
        print(f"    Std: {output.std().item():.6f}")


if __name__ == "__main__":
    debug_quantizer_state()
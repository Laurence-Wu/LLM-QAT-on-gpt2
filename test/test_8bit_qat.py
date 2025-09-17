"""
Test 8-bit QAT model performance and compare with reference GPT-2.
"""

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config as HFConfig
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import QATGPT2
from part1_switchable_precision.main_qat import load_pretrained_weights


def test_8bit_qat():
    """Test 8-bit QAT model against reference."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create config with all necessary attributes
    config = HFConfig()
    config.n_positions = 256
    config.n_layer = 12
    config.lora_rank = 8
    config.lora_alpha = 16
    config.lora_dropout = 0.0
    config.kv_cache_bits = 8  # Explicitly set to 8-bit

    print("="*60)
    print("TESTING 8-BIT QAT MODEL")
    print("="*60)

    # Create model with 8-bit precision
    qat_model = QATGPT2(config, quantization_bits=8, initialize_weights=False)

    # Load weights
    load_pretrained_weights(qat_model)

    # Zero all LoRA weights to isolate quantization effects
    with torch.no_grad():
        lora_count = 0
        for name, param in qat_model.named_parameters():
            if 'lora' in name.lower():
                param.data.zero_()
                lora_count += 1
        print(f"Zeroed {lora_count} LoRA parameters\n")

    qat_model = qat_model.to(device)
    qat_model.eval()

    # Reference model
    ref_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    ref_model.eval()

    # Test tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Test texts
    test_texts = [
        "The capital of France is",
        "Machine learning is",
        "Once upon a time",
        "In the beginning",
        "The quick brown fox"
    ]

    print("Comparing 8-bit QAT vs Reference GPT-2:")
    print("-"*60)

    total_qat_loss = 0
    total_ref_loss = 0
    count = 0

    for text in test_texts:
        inputs = tokenizer(text, return_tensors='pt', max_length=50, truncation=True).to(device)

        with torch.no_grad():
            # QAT model
            qat_outputs = qat_model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
            qat_loss = qat_outputs['loss'] if isinstance(qat_outputs, dict) else qat_outputs.loss

            # Reference model
            ref_outputs = ref_model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
            ref_loss = ref_outputs.loss

            if qat_loss is not None and ref_loss is not None:
                qat_perplexity = math.exp(qat_loss.item()) if qat_loss.item() < 20 else float('inf')
                ref_perplexity = math.exp(ref_loss.item()) if ref_loss.item() < 20 else float('inf')

                total_qat_loss += qat_loss.item()
                total_ref_loss += ref_loss.item()
                count += 1

                print(f"\nText: '{text[:30]}...'")
                print(f"  8-bit QAT  - Loss: {qat_loss.item():.4f}, Perplexity: {qat_perplexity:.1f}")
                print(f"  Reference  - Loss: {ref_loss.item():.4f}, Perplexity: {ref_perplexity:.1f}")
                print(f"  Difference - Loss: {abs(qat_loss.item() - ref_loss.item()):.4f} ({(abs(qat_loss.item() - ref_loss.item())/ref_loss.item()*100):.1f}%)")

    if count > 0:
        avg_qat_loss = total_qat_loss / count
        avg_ref_loss = total_ref_loss / count
        avg_qat_perplexity = math.exp(avg_qat_loss) if avg_qat_loss < 20 else float('inf')
        avg_ref_perplexity = math.exp(avg_ref_loss) if avg_ref_loss < 20 else float('inf')

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Average 8-bit QAT Loss: {avg_qat_loss:.4f}")
        print(f"Average Reference Loss: {avg_ref_loss:.4f}")
        print(f"Average 8-bit QAT Perplexity: {avg_qat_perplexity:.1f}")
        print(f"Average Reference Perplexity: {avg_ref_perplexity:.1f}")
        print(f"Loss Degradation: {(avg_qat_loss - avg_ref_loss):.4f} ({(avg_qat_loss/avg_ref_loss - 1)*100:.1f}% worse)")
        print(f"Perplexity Degradation: {(avg_qat_perplexity/avg_ref_perplexity):.2f}x")

    # Detailed analysis of quantization effects
    print("\n" + "="*60)
    print("DETAILED QUANTIZATION ANALYSIS")
    print("="*60)

    test_input = tokenizer("The", return_tensors='pt').to(device)

    with torch.no_grad():
        # Check weight quantization effects
        first_layer = qat_model.h[0].attn.c_attn

        # Get original and quantized weights
        original_weight = first_layer.linear.weight
        quantized_weight = first_layer.quantize_weight(original_weight)

        weight_diff = (original_weight - quantized_weight).abs().mean().item()
        weight_relative_error = weight_diff / original_weight.abs().mean().item()

        print(f"First attention layer weight quantization:")
        print(f"  Original weight norm: {original_weight.norm().item():.4f}")
        print(f"  Quantized weight norm: {quantized_weight.norm().item():.4f}")
        print(f"  Mean absolute difference: {weight_diff:.6f}")
        print(f"  Relative error: {weight_relative_error*100:.2f}%")

        # Check activation quantization
        test_activation = torch.randn(1, 10, 768).to(device)
        quantized_activation = first_layer.quantize_input(test_activation)

        act_diff = (test_activation - quantized_activation).abs().mean().item()
        act_relative_error = act_diff / test_activation.abs().mean().item()

        print(f"\nActivation quantization (8-bit):")
        print(f"  Mean absolute difference: {act_diff:.6f}")
        print(f"  Relative error: {act_relative_error*100:.2f}%")

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)

    if avg_qat_loss < avg_ref_loss * 1.5:
        print("✅ 8-bit QAT model shows reasonable performance!")
        print("   The model can benefit from quantization-aware training to improve further.")
    else:
        print("⚠️  8-bit QAT model shows significant degradation.")
        print("   Quantization-aware training is essential for good 8-bit performance.")


if __name__ == "__main__":
    test_8bit_qat()
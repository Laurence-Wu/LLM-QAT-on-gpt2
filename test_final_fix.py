"""
Final test to verify the fixes and identify remaining issues.
"""

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config as HFConfig
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import QATGPT2
from fix_weight_loading import load_pretrained_weights_fixed


def test_with_32bit():
    """Test with 32-bit (no quantization) to isolate the issue."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing with 32-bit precision (no quantization)\n")

    # Create config with all necessary attributes
    config = HFConfig()
    config.n_positions = 256
    config.n_layer = 12
    config.lora_rank = 8
    config.lora_alpha = 16
    config.lora_dropout = 0.0
    config.kv_cache_bits = 32  # Explicitly set to 32

    # Create model with 32-bit precision
    qat_model = QATGPT2(config, quantization_bits=32, initialize_weights=False)

    # Load weights
    load_pretrained_weights_fixed(qat_model, debug=False)

    # Explicitly zero all LoRA weights
    with torch.no_grad():
        for name, param in qat_model.named_parameters():
            if 'lora' in name.lower():
                param.data.zero_()
                print(f"Zeroed: {name}")

    qat_model = qat_model.to(device)
    qat_model.eval()

    # Reference model
    ref_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    ref_model.eval()

    # Test
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    test_texts = [
        "The capital of France is",
        "Machine learning is",
        "Once upon a time"
    ]

    print("\n" + "="*60)
    print("32-BIT QAT MODEL vs REFERENCE")
    print("="*60)

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

                print(f"\nText: '{text[:30]}...'")
                print(f"  QAT  - Loss: {qat_loss.item():.4f}, Perplexity: {qat_perplexity:.1f}")
                print(f"  Ref  - Loss: {ref_loss.item():.4f}, Perplexity: {ref_perplexity:.1f}")
                print(f"  Diff - Loss: {abs(qat_loss.item() - ref_loss.item()):.4f}")

    # Debug attention mechanism at 32-bit
    print("\n" + "="*60)
    print("DEBUGGING ATTENTION AT 32-BIT")
    print("="*60)

    test_input = tokenizer("Test", return_tensors='pt').to(device)

    with torch.no_grad():
        # Get embeddings
        qat_embeds = qat_model.wte(test_input['input_ids'])
        qat_pos = qat_model.wpe(torch.arange(test_input['input_ids'].shape[1], device=device))
        qat_hidden = qat_embeds + qat_pos

        ref_embeds = ref_model.transformer.wte(test_input['input_ids'])
        ref_pos = ref_model.transformer.wpe(torch.arange(test_input['input_ids'].shape[1], device=device))
        ref_hidden = ref_embeds + ref_pos

        print(f"Embedding difference: {(qat_hidden - ref_hidden).abs().mean().item():.6f}")

        # First block
        qat_ln1 = qat_model.h[0].ln_1(qat_hidden)
        ref_ln1 = ref_model.transformer.h[0].ln_1(ref_hidden)
        print(f"After LN1 difference: {(qat_ln1 - ref_ln1).abs().mean().item():.6f}")

        # Check c_attn output
        qat_qkv = qat_model.h[0].attn.c_attn(qat_ln1)
        ref_qkv = ref_model.transformer.h[0].attn.c_attn(ref_ln1)
        print(f"After c_attn projection: {(qat_qkv - ref_qkv).abs().mean().item():.6f}")

        # Check if it's the linear layer or the LoRA
        qat_linear_only = qat_model.h[0].attn.c_attn.linear(qat_ln1)
        print(f"Linear only output: {(qat_linear_only - ref_qkv).abs().mean().item():.6f}")


if __name__ == "__main__":
    test_with_32bit()
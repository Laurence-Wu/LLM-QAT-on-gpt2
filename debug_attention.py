"""
Debug the attention mechanism to find the discrepancy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config as HFConfig
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import QATGPT2, QATGPT2Attention
from shared.lora import QATLinearWithLoRA
from transformers import GPT2Model
import math


def debug_attention_forward():
    """Debug the attention mechanism step by step."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create models
    config = HFConfig()
    config.n_positions = 256
    config.n_layer = 12
    config.lora_rank = 0
    config.lora_alpha = 0
    config.lora_dropout = 0.0

    # Create QAT model
    qat_model = QATGPT2(config, quantization_bits=16, initialize_weights=False)

    # Load weights using the existing function
    from fix_weight_loading import load_pretrained_weights_fixed
    load_pretrained_weights_fixed(qat_model, debug=False)
    qat_model = qat_model.to(device)
    qat_model.eval()

    # Reference model
    ref_model = GPT2Model.from_pretrained('gpt2').to(device)
    ref_model.eval()

    # Test input
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    test_text = "The capital"
    inputs = tokenizer(test_text, return_tensors='pt').to(device)

    print("="*60)
    print("DEBUGGING ATTENTION MECHANISM")
    print("="*60)

    with torch.no_grad():
        # Get embeddings
        qat_embeds = qat_model.wte(inputs['input_ids'])
        qat_pos = qat_model.wpe(torch.arange(inputs['input_ids'].shape[1], device=device))
        qat_hidden = qat_embeds + qat_pos

        ref_embeds = ref_model.wte(inputs['input_ids'])
        ref_pos = ref_model.wpe(torch.arange(inputs['input_ids'].shape[1], device=device))
        ref_hidden = ref_embeds + ref_pos

        embed_diff = (qat_hidden - ref_hidden).abs().mean().item()
        print(f"Embedding difference: {embed_diff:.6f}")

        # First block layer norm
        qat_ln1 = qat_model.h[0].ln_1(qat_hidden)
        ref_ln1 = ref_model.h[0].ln_1(ref_hidden)
        ln1_diff = (qat_ln1 - ref_ln1).abs().mean().item()
        print(f"After LN1 difference: {ln1_diff:.6f}")

        # Debug attention step by step
        print("\n--- Attention Debug ---")

        # QAT attention
        B, T, C = qat_ln1.shape

        # QKV projection in QAT
        qat_qkv = qat_model.h[0].attn.c_attn(qat_ln1)  # Should be [B, T, 3*C]
        print(f"QAT QKV shape: {qat_qkv.shape}")

        # Split Q, K, V
        qat_q, qat_k, qat_v = qat_qkv.split(C, dim=-1)
        print(f"QAT Q shape: {qat_q.shape}, K shape: {qat_k.shape}, V shape: {qat_v.shape}")

        # Reshape for multi-head attention
        n_head = 12
        head_dim = C // n_head
        qat_q = qat_q.view(B, T, n_head, head_dim).transpose(1, 2)  # [B, n_head, T, head_dim]
        qat_k = qat_k.view(B, T, n_head, head_dim).transpose(1, 2)
        qat_v = qat_v.view(B, T, n_head, head_dim).transpose(1, 2)

        print(f"QAT reshaped Q shape: {qat_q.shape}")

        # Reference attention QKV
        ref_qkv = ref_model.h[0].attn.c_attn(ref_ln1)
        ref_q, ref_k, ref_v = ref_qkv.split(C, dim=-1)
        ref_q = ref_q.view(B, T, n_head, head_dim).transpose(1, 2)
        ref_k = ref_k.view(B, T, n_head, head_dim).transpose(1, 2)
        ref_v = ref_v.view(B, T, n_head, head_dim).transpose(1, 2)

        # Compare Q, K, V
        q_diff = (qat_q - ref_q).abs().mean().item()
        k_diff = (qat_k - ref_k).abs().mean().item()
        v_diff = (qat_v - ref_v).abs().mean().item()
        print(f"Q difference: {q_diff:.6f}")
        print(f"K difference: {k_diff:.6f}")
        print(f"V difference: {v_diff:.6f}")

        # Attention scores
        scale = 1.0 / math.sqrt(head_dim)
        qat_scores = torch.matmul(qat_q, qat_k.transpose(-2, -1)) * scale
        ref_scores = torch.matmul(ref_q, ref_k.transpose(-2, -1)) * scale

        scores_diff = (qat_scores - ref_scores).abs().mean().item()
        print(f"Attention scores difference: {scores_diff:.6f}")

        # Apply causal mask
        causal_mask = torch.ones(T, T, device=device).tril().view(1, 1, T, T)
        qat_scores = qat_scores.masked_fill(causal_mask == 0, -1e9)
        ref_scores = ref_scores.masked_fill(causal_mask == 0, -1e9)

        # Softmax
        qat_probs = F.softmax(qat_scores, dim=-1)
        ref_probs = F.softmax(ref_scores, dim=-1)

        probs_diff = (qat_probs - ref_probs).abs().mean().item()
        print(f"Attention probs difference: {probs_diff:.6f}")

        # Attention output
        qat_attn_out = torch.matmul(qat_probs, qat_v)
        ref_attn_out = torch.matmul(ref_probs, ref_v)

        attn_out_diff = (qat_attn_out - ref_attn_out).abs().mean().item()
        print(f"Attention output (before proj) difference: {attn_out_diff:.6f}")

        # Reshape back
        qat_attn_out = qat_attn_out.transpose(1, 2).contiguous().view(B, T, C)
        ref_attn_out = ref_attn_out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        qat_attn_out = qat_model.h[0].attn.c_proj(qat_attn_out)
        ref_attn_out = ref_model.h[0].attn.c_proj(ref_attn_out)[0]  # GPT2 returns tuple

        final_attn_diff = (qat_attn_out - ref_attn_out).abs().mean().item()
        print(f"Attention output (after proj) difference: {final_attn_diff:.6f}")

        # Full block output
        qat_block_out = qat_model.h[0](qat_hidden)
        ref_block_out = ref_model.h[0](ref_hidden)[0]

        block_diff = (qat_block_out - ref_block_out).abs().mean().item()
        print(f"\nFull block output difference: {block_diff:.6f}")

        # Check if quantization is happening
        print("\n--- Checking Quantization ---")

        # Check if quantizers are active
        for name, module in qat_model.h[0].named_modules():
            if hasattr(module, 'quantize_weight'):
                bits = module.quantize_weight.num_bits
                print(f"{name}: weight bits = {bits}")

                # Check if quantization is actually happening
                if hasattr(module, 'linear'):
                    orig_weight = module.linear.weight
                    quantized_weight = module.quantize_weight(orig_weight)
                    weight_diff = (orig_weight - quantized_weight).abs().mean().item()
                    print(f"  Weight quantization difference: {weight_diff:.6f}")

        # Check KV quantization
        if hasattr(qat_model.h[0].attn, 'kv_quantizer'):
            kv_bits = qat_model.h[0].attn.kv_quantizer.num_bits
            print(f"KV quantizer bits: {kv_bits}")

            # Test KV quantization
            test_tensor = torch.randn(1, 10, device=device)
            quantized_test = qat_model.h[0].attn.kv_quantizer(test_tensor)
            kv_quant_diff = (test_tensor - quantized_test).abs().mean().item()
            print(f"KV quantization difference on test tensor: {kv_quant_diff:.6f}")


if __name__ == "__main__":
    debug_attention_forward()
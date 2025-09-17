#!/usr/bin/env python3
"""
Debug script to identify why perplexity is still extremely high.
"""

import os
import sys
import torch
import torch.nn as nn
import math
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import QATGPT2, SwitchableQATGPT2
from part1_switchable_precision.main_qat import load_pretrained_weights
from part1_switchable_precision.config_qat import ModelConfig


def diagnose_model_issues():
    """Systematically diagnose what's causing high perplexity."""

    print("\n" + "="*70)
    print("DIAGNOSING HIGH PERPLEXITY ISSUE")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create reference GPT-2 model
    print("\n1. Creating reference GPT-2 model...")
    reference = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    reference.eval()

    # Create QAT model
    print("\n2. Creating QAT model...")
    config = ModelConfig()
    gpt2_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        layer_norm_epsilon=config.layer_norm_epsilon,
        embd_pdrop=config.embd_pdrop,
        quantization_bits=8,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout
    )

    qat_model = SwitchableQATGPT2(gpt2_config, bit_widths=[4, 8, 16], initialize_weights=False)
    qat_model = qat_model.to(device)
    qat_model.eval()

    # Test with tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    test_text = "The capital of France is"
    inputs = tokenizer(test_text, return_tensors='pt').to(device)

    # ============ TEST 1: Before any weight loading ============
    print("\n" + "-"*50)
    print("TEST 1: Model with random weights")
    print("-"*50)

    with torch.no_grad():
        outputs = qat_model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
        if loss is not None:
            perplexity = math.exp(loss.item()) if loss.item() < 20 else float('inf')
            print(f"Loss: {loss.item():.4f}")
            print(f"Perplexity: {perplexity:.1f}")

    # ============ TEST 2: Load ONLY embeddings ============
    print("\n" + "-"*50)
    print("TEST 2: After loading ONLY embeddings")
    print("-"*50)

    # Load embeddings from reference
    qat_model.wte.weight.data = reference.transformer.wte.weight.data.clone()
    min_pos = min(qat_model.wpe.weight.shape[0], reference.transformer.wpe.weight.shape[0])
    qat_model.wpe.weight.data[:min_pos] = reference.transformer.wpe.weight.data[:min_pos].clone()

    # Tie LM head to embeddings
    qat_model.lm_head.weight = qat_model.wte.weight

    with torch.no_grad():
        outputs = qat_model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
        if loss is not None:
            perplexity = math.exp(loss.item()) if loss.item() < 20 else float('inf')
            print(f"Loss: {loss.item():.4f}")
            print(f"Perplexity: {perplexity:.1f}")

    # ============ TEST 3: Load first transformer block ============
    print("\n" + "-"*50)
    print("TEST 3: After loading first transformer block")
    print("-"*50)

    # Load first block
    qat_model.h[0].ln_1.weight.data = reference.transformer.h[0].ln_1.weight.data.clone()
    qat_model.h[0].ln_1.bias.data = reference.transformer.h[0].ln_1.bias.data.clone()
    qat_model.h[0].ln_2.weight.data = reference.transformer.h[0].ln_2.weight.data.clone()
    qat_model.h[0].ln_2.bias.data = reference.transformer.h[0].ln_2.bias.data.clone()

    # Check if weights need transposing
    ref_attn_weight = reference.transformer.h[0].attn.c_attn.weight
    print(f"Reference c_attn weight shape: {ref_attn_weight.shape}")

    if hasattr(qat_model.h[0].attn.c_attn, 'linear'):
        print(f"QAT c_attn.linear weight shape: {qat_model.h[0].attn.c_attn.linear.weight.shape}")

        # Try WITHOUT transpose first
        print("\nTrying WITHOUT transpose...")
        qat_model.h[0].attn.c_attn.linear.weight.data = ref_attn_weight.data.clone()

    with torch.no_grad():
        try:
            outputs = qat_model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
            if loss is not None:
                perplexity = math.exp(loss.item()) if loss.item() < 20 else float('inf')
                print(f"Loss (no transpose): {loss.item():.4f}")
                print(f"Perplexity (no transpose): {perplexity:.1f}")
        except Exception as e:
            print(f"Error without transpose: {e}")

            # Try WITH transpose
            print("\nTrying WITH transpose...")
            qat_model.h[0].attn.c_attn.linear.weight.data = ref_attn_weight.data.t().contiguous()

            outputs = qat_model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
            if loss is not None:
                perplexity = math.exp(loss.item()) if loss.item() < 20 else float('inf')
                print(f"Loss (with transpose): {loss.item():.4f}")
                print(f"Perplexity (with transpose): {perplexity:.1f}")

    # ============ TEST 4: Check LoRA interference ============
    print("\n" + "-"*50)
    print("TEST 4: Check LoRA interference")
    print("-"*50)

    # Set precision to 16-bit
    qat_model.set_precision(16)

    # Check if LoRA is adding to outputs
    for name, module in qat_model.named_modules():
        if hasattr(module, 'lora_adapters') and module.lora_adapters:
            if 16 in module.lora_adapters:
                lora = module.lora_adapters[16]
                if hasattr(lora, 'lora_B'):
                    b_sum = lora.lora_B.abs().sum().item()
                    a_sum = lora.lora_A.abs().sum().item() if hasattr(lora, 'lora_A') else 0
                    print(f"{name}: B sum={b_sum:.6f}, A sum={a_sum:.6f}")

    # ============ TEST 5: Compare intermediate outputs ============
    print("\n" + "-"*50)
    print("TEST 5: Compare intermediate outputs")
    print("-"*50)

    # Get embeddings from both models
    with torch.no_grad():
        # Reference embeddings
        ref_embeds = reference.transformer.wte(inputs['input_ids'])
        ref_pos_ids = torch.arange(inputs['input_ids'].shape[1], device=device).unsqueeze(0)
        ref_pos_embeds = reference.transformer.wpe(ref_pos_ids)
        ref_hidden = reference.transformer.drop(ref_embeds + ref_pos_embeds)

        # QAT embeddings
        qat_embeds = qat_model.wte(inputs['input_ids'])
        qat_pos_ids = torch.arange(inputs['input_ids'].shape[1], device=device).unsqueeze(0)
        qat_pos_embeds = qat_model.wpe(qat_pos_ids)
        qat_hidden = qat_model.drop(qat_embeds + qat_pos_embeds)

        embed_diff = torch.mean(torch.abs(ref_hidden - qat_hidden)).item()
        print(f"Embedding difference: {embed_diff:.6f}")

        # Check first layer output
        ref_block_out = reference.transformer.h[0](ref_hidden)[0]

        # For QAT model, need to check attention mask
        qat_block_out = qat_model.h[0](qat_hidden, attention_mask=None)

        block_diff = torch.mean(torch.abs(ref_block_out - qat_block_out)).item()
        print(f"First block output difference: {block_diff:.6f}")

    # ============ TEST 6: Test reference model perplexity ============
    print("\n" + "-"*50)
    print("TEST 6: Reference GPT-2 perplexity (expected ~30-50)")
    print("-"*50)

    with torch.no_grad():
        ref_outputs = reference(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
        ref_loss = ref_outputs.loss
        ref_perplexity = math.exp(ref_loss.item()) if ref_loss.item() < 20 else float('inf')
        print(f"Reference Loss: {ref_loss.item():.4f}")
        print(f"Reference Perplexity: {ref_perplexity:.1f}")

    # ============ TEST 7: Check quantization interference ============
    print("\n" + "-"*50)
    print("TEST 7: Check quantization at 16-bit")
    print("-"*50)

    # Check if quantization is active even at 16-bit
    for name, module in qat_model.named_modules():
        if hasattr(module, 'weight_quantizer'):
            if hasattr(module.weight_quantizer, 'enabled'):
                print(f"{name}: Quantization enabled = {module.weight_quantizer.enabled}")
            if hasattr(module.weight_quantizer, 'num_bits'):
                print(f"{name}: Quantization bits = {module.weight_quantizer.num_bits}")

    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)
    print("\nPossible issues to investigate:")
    print("1. Weight transpose direction (with or without .t())")
    print("2. LoRA interference even when zeroed")
    print("3. Quantization active at 16-bit")
    print("4. Attention mask handling difference")
    print("5. Layer norm or bias initialization")


def test_minimal_model():
    """Test a minimal version without any QAT features."""
    print("\n" + "="*70)
    print("TESTING MINIMAL MODEL (No QAT)")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use standard QATGPT2 without switchable precision
    config = ModelConfig()
    gpt2_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        layer_norm_epsilon=config.layer_norm_epsilon,
        embd_pdrop=config.embd_pdrop,
        quantization_bits=8,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout
    )

    print("\n1. Testing standard QATGPT2 (not switchable)...")
    model = QATGPT2(gpt2_config, quantization_bits=8, initialize_weights=False)
    load_pretrained_weights(model)

    # Move to device AFTER loading weights
    model = model.to(device)
    model.eval()

    # Test
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    test_text = "The capital of France is"
    inputs = tokenizer(test_text, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
        if loss is not None:
            perplexity = math.exp(loss.item()) if loss.item() < 20 else float('inf')
            print(f"QATGPT2 Loss: {loss.item():.4f}")
            print(f"QATGPT2 Perplexity: {perplexity:.1f}")

            if perplexity < 100:
                print("✅ Standard QATGPT2 works fine!")
                print("→ Issue is with SwitchableQATGPT2")
            else:
                print("❌ Issue exists in base QATGPT2 too")
                print("→ Problem is in weight loading or base model structure")


if __name__ == "__main__":
    diagnose_model_issues()
    test_minimal_model()
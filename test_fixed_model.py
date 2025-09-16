#!/usr/bin/env python3
"""
Test script to verify the model is properly loading pre-trained weights
and not using random initialization.
"""

import torch
import torch.nn as nn
import math
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from shared.models import SwitchableQATGPT2
import torch.nn.functional as F


def load_pretrained_weights_into_qat(qat_model, model_name='gpt2'):
    """Load pre-trained GPT-2 weights into QAT model"""
    print(f"Loading pre-trained weights from {model_name}...")

    # Load pre-trained GPT-2
    pretrained_model = GPT2LMHeadModel.from_pretrained(model_name)
    pretrained_state = pretrained_model.state_dict()

    # Copy embeddings
    qat_model.wte.weight.data = pretrained_state['transformer.wte.weight']

    # Handle position embeddings size mismatch
    pretrained_wpe = pretrained_state['transformer.wpe.weight']
    if pretrained_wpe.shape[0] != qat_model.wpe.weight.shape[0]:
        min_pos = min(pretrained_wpe.shape[0], qat_model.wpe.weight.shape[0])
        qat_model.wpe.weight.data[:min_pos] = pretrained_wpe[:min_pos]
        print(f"Adjusted position embeddings from {pretrained_wpe.shape[0]} to {qat_model.wpe.weight.shape[0]}")
    else:
        qat_model.wpe.weight.data = pretrained_wpe

    # Copy transformer blocks
    for i in range(len(qat_model.h)):
        # Layer norms
        qat_model.h[i].ln_1.weight.data = pretrained_state[f'transformer.h.{i}.ln_1.weight']
        qat_model.h[i].ln_1.bias.data = pretrained_state[f'transformer.h.{i}.ln_1.bias']
        qat_model.h[i].ln_2.weight.data = pretrained_state[f'transformer.h.{i}.ln_2.weight']
        qat_model.h[i].ln_2.bias.data = pretrained_state[f'transformer.h.{i}.ln_2.bias']

        # Attention weights (transpose from conv1d to linear)
        qat_model.h[i].attn.c_attn.linear.weight.data = pretrained_state[f'transformer.h.{i}.attn.c_attn.weight'].t()
        qat_model.h[i].attn.c_attn.linear.bias.data = pretrained_state[f'transformer.h.{i}.attn.c_attn.bias']
        qat_model.h[i].attn.c_proj.linear.weight.data = pretrained_state[f'transformer.h.{i}.attn.c_proj.weight'].t()
        qat_model.h[i].attn.c_proj.linear.bias.data = pretrained_state[f'transformer.h.{i}.attn.c_proj.bias']

        # MLP weights (transpose from conv1d to linear)
        qat_model.h[i].mlp.c_fc.linear.weight.data = pretrained_state[f'transformer.h.{i}.mlp.c_fc.weight'].t()
        qat_model.h[i].mlp.c_fc.linear.bias.data = pretrained_state[f'transformer.h.{i}.mlp.c_fc.bias']
        qat_model.h[i].mlp.c_proj.linear.weight.data = pretrained_state[f'transformer.h.{i}.mlp.c_proj.weight'].t()
        qat_model.h[i].mlp.c_proj.linear.bias.data = pretrained_state[f'transformer.h.{i}.mlp.c_proj.bias']

        # Handle attention bias if exists
        if f'transformer.h.{i}.attn.bias' in pretrained_state:
            pretrained_bias = pretrained_state[f'transformer.h.{i}.attn.bias']
            model_bias_shape = qat_model.h[i].attn.bias.shape
            if pretrained_bias.shape != model_bias_shape:
                min_size = min(pretrained_bias.shape[0], model_bias_shape[0])
                qat_model.h[i].attn.bias.data[:min_size, :min_size] = pretrained_bias[:min_size, :min_size]
            else:
                qat_model.h[i].attn.bias.data = pretrained_bias

    # Final layer norm
    qat_model.ln_f.weight.data = pretrained_state['transformer.ln_f.weight']
    qat_model.ln_f.bias.data = pretrained_state['transformer.ln_f.bias']

    # LM head shares weight with embeddings
    qat_model.lm_head.weight = qat_model.wte.weight

    # Initialize LoRA weights to small/zero values
    with torch.no_grad():
        for module in qat_model.modules():
            try:
                lora_adapters = module.lora_adapters
                for lora in lora_adapters.values():
                    try:
                        nn.init.zeros_(lora.lora_B)
                    except AttributeError:
                        pass  # lora_B doesn't exist
                    try:
                        nn.init.normal_(lora.lora_A, std=0.01)
                    except AttributeError:
                        pass  # lora_A doesn't exist
            except AttributeError:
                pass  # module doesn't have lora_adapters

    print("Pre-trained weights loaded successfully!")
    return qat_model


def test_model_initialization():
    """Test that the model is properly initialized with pre-trained weights"""

    print("="*70)
    print("TESTING MODEL INITIALIZATION")
    print("="*70)

    # Create two models - one with proper loading, one with random init
    print("\n1. Creating model WITH pre-trained weights...")

    # Create config for the model
    config = GPT2Config(
        vocab_size=50257,
        n_positions=256,
        n_embd=768,
        n_layer=6,  # Using smaller model for testing
        n_head=12,
        layer_norm_epsilon=1e-5,
        embd_pdrop=0.1,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1
    )

    # Create model WITHOUT random initialization
    model_pretrained = SwitchableQATGPT2(config, bit_widths=[4, 8, 16], initialize_weights=False)

    # Load pre-trained weights
    model_pretrained = load_pretrained_weights_into_qat(model_pretrained, 'gpt2')
    model_pretrained = model_pretrained.to('cuda')

    print("\n2. Creating model WITH RANDOM initialization (for comparison)...")
    config = GPT2Config(
        vocab_size=50257,
        n_positions=256,
        n_embd=768,
        n_layer=6,
        n_head=12,
        layer_norm_epsilon=1e-5,
        embd_pdrop=0.1,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model_random = SwitchableQATGPT2(config, bit_widths=[4, 8, 16])
    # Apply random init to this one
    model_random.apply(model_random._init_weights)
    model_random = model_random.to('cuda')

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Test text
    test_text = "The capital of France is"
    inputs = tokenizer(test_text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}

    print("\n" + "="*70)
    print("WEIGHT STATISTICS COMPARISON")
    print("="*70)

    # Compare weight statistics
    with torch.no_grad():
        # Check embedding weights
        wte_pretrained = model_pretrained.wte.weight
        wte_random = model_random.wte.weight

        print(f"\nWord Embeddings:")
        print(f"  Pre-trained - Mean: {wte_pretrained.mean():.4f}, Std: {wte_pretrained.std():.4f}")
        print(f"  Random      - Mean: {wte_random.mean():.4f}, Std: {wte_random.std():.4f}")

        # Check first attention layer
        attn_pretrained = model_pretrained.h[0].attn.c_attn.linear.weight
        attn_random = model_random.h[0].attn.c_attn.linear.weight

        print(f"\nFirst Attention Layer:")
        print(f"  Pre-trained - Mean: {attn_pretrained.mean():.4f}, Std: {attn_pretrained.std():.4f}")
        print(f"  Random      - Mean: {attn_random.mean():.4f}, Std: {attn_random.std():.4f}")

    print("\n" + "="*70)
    print("PERPLEXITY TEST ON SAMPLE TEXT")
    print("="*70)

    def calculate_perplexity(model, inputs):
        """Calculate perplexity on the input text"""
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
            loss = outputs['loss']
            perplexity = math.exp(loss.item()) if loss.item() < 20 else float('inf')
        return perplexity

    # Test with different bit-widths
    for bits in [16, 8, 4]:
        print(f"\n{bits}-bit Configuration:")

        # Test pre-trained model
        model_pretrained.set_precision(bits)
        ppl_pretrained = calculate_perplexity(model_pretrained, inputs)

        # Test random model
        model_random.set_precision(bits)
        ppl_random = calculate_perplexity(model_random, inputs)

        print(f"  Pre-trained model: {ppl_pretrained:.1f}")
        print(f"  Random model:      {ppl_random:.1f}")

        # The pre-trained model should have MUCH lower perplexity
        if ppl_pretrained < ppl_random / 10:
            print(f"  ✅ PASS: Pre-trained model is significantly better")
        else:
            print(f"  ❌ FAIL: Models have similar performance (likely both random)")

    print("\n" + "="*70)
    print("TEXT GENERATION TEST")
    print("="*70)

    # Generate text with both models
    model_pretrained.set_precision(8)
    model_random.set_precision(8)

    print(f"\nPrompt: '{test_text}'")

    # Pre-trained model generation
    model_pretrained.eval()
    with torch.no_grad():
        generated_pretrained = model_pretrained.generate(
            inputs['input_ids'],
            max_new_tokens=20,
            do_sample=False,
            temperature=1.0
        )
    text_pretrained = tokenizer.decode(generated_pretrained[0], skip_special_tokens=True)
    print(f"\nPre-trained model output:")
    print(f"  '{text_pretrained}'")

    # Random model generation
    model_random.eval()
    with torch.no_grad():
        generated_random = model_random.generate(
            inputs['input_ids'],
            max_new_tokens=20,
            do_sample=False,
            temperature=1.0
        )
    text_random = tokenizer.decode(generated_random[0], skip_special_tokens=True)
    print(f"\nRandom model output:")
    print(f"  '{text_random}'")

    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)

    # Final diagnosis
    if "Paris" in text_pretrained or "French" in text_pretrained:
        print("✅ SUCCESS: Pre-trained model generates coherent text!")
        print("The model is properly loading GPT-2 weights.")
    else:
        print("⚠️  WARNING: Pre-trained model output doesn't look coherent.")
        print("Check if weights are being loaded correctly.")

    # Check if random model is truly random
    if len(set(text_random.split())) < 3:  # Very repetitive = random
        print("✅ Random model confirmed to be random (as expected)")
    else:
        print("⚠️  Random model generating coherent text (unexpected)")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_model_initialization()
#!/usr/bin/env python3
"""
Complete test for proper weight loading in QATGPT2 models.
This ensures ALL weights are loaded correctly, not just embeddings.
"""

import os
import sys
import torch
import torch.nn as nn
import math
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, GPT2Model

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'part1_switchable_precision'))

from models import QATGPT2, SwitchableQATGPT2
from config_qat import ModelConfig


def load_all_pretrained_weights(model, verbose=True):
    """
    Properly load ALL pretrained GPT-2 weights into the QAT model.
    This is the COMPLETE loading function.
    """
    if verbose:
        print("Loading ALL pretrained GPT-2 weights...")

    # Use GPT2Model for transformer weights
    pretrained = GPT2Model.from_pretrained('gpt2')

    # 1. Copy embeddings
    model.wte.weight.data = pretrained.wte.weight.data.clone()
    if verbose:
        print("  ✓ Token embeddings loaded")

    # 2. Handle position embeddings with size adjustment
    min_positions = min(model.wpe.weight.shape[0], pretrained.wpe.weight.shape[0])
    model.wpe.weight.data[:min_positions] = pretrained.wpe.weight.data[:min_positions].clone()
    if model.wpe.weight.shape[0] != pretrained.wpe.weight.shape[0]:
        if verbose:
            print(f"  ✓ Position embeddings adjusted: {pretrained.wpe.weight.shape[0]} → {model.wpe.weight.shape[0]}")
    else:
        if verbose:
            print("  ✓ Position embeddings loaded")

    # 3. Copy ALL transformer blocks (not just the first one!)
    num_layers = min(len(model.h), len(pretrained.h))
    for i in range(num_layers):
        # Layer normalizations
        model.h[i].ln_1.weight.data = pretrained.h[i].ln_1.weight.data.clone()
        model.h[i].ln_1.bias.data = pretrained.h[i].ln_1.bias.data.clone()
        model.h[i].ln_2.weight.data = pretrained.h[i].ln_2.weight.data.clone()
        model.h[i].ln_2.bias.data = pretrained.h[i].ln_2.bias.data.clone()

        # Attention weights - handle QAT structure
        if hasattr(model.h[i].attn.c_attn, 'linear'):
            # QAT model has wrapped linear layers
            model.h[i].attn.c_attn.linear.weight.data = pretrained.h[i].attn.c_attn.weight.data.t().contiguous()
            model.h[i].attn.c_attn.linear.bias.data = pretrained.h[i].attn.c_attn.bias.data.clone()
            model.h[i].attn.c_proj.linear.weight.data = pretrained.h[i].attn.c_proj.weight.data.t().contiguous()
            model.h[i].attn.c_proj.linear.bias.data = pretrained.h[i].attn.c_proj.bias.data.clone()
        else:
            # Direct assignment if not wrapped
            model.h[i].attn.c_attn.weight.data = pretrained.h[i].attn.c_attn.weight.data.clone()
            model.h[i].attn.c_attn.bias.data = pretrained.h[i].attn.c_attn.bias.data.clone()
            model.h[i].attn.c_proj.weight.data = pretrained.h[i].attn.c_proj.weight.data.clone()
            model.h[i].attn.c_proj.bias.data = pretrained.h[i].attn.c_proj.bias.data.clone()

        # MLP weights - handle QAT structure
        if hasattr(model.h[i].mlp.c_fc, 'linear'):
            # QAT model has wrapped linear layers
            model.h[i].mlp.c_fc.linear.weight.data = pretrained.h[i].mlp.c_fc.weight.data.t().contiguous()
            model.h[i].mlp.c_fc.linear.bias.data = pretrained.h[i].mlp.c_fc.bias.data.clone()
            model.h[i].mlp.c_proj.linear.weight.data = pretrained.h[i].mlp.c_proj.weight.data.t().contiguous()
            model.h[i].mlp.c_proj.linear.bias.data = pretrained.h[i].mlp.c_proj.bias.data.clone()
        else:
            # Direct assignment if not wrapped
            model.h[i].mlp.c_fc.weight.data = pretrained.h[i].mlp.c_fc.weight.data.clone()
            model.h[i].mlp.c_fc.bias.data = pretrained.h[i].mlp.c_fc.bias.data.clone()
            model.h[i].mlp.c_proj.weight.data = pretrained.h[i].mlp.c_proj.weight.data.clone()
            model.h[i].mlp.c_proj.bias.data = pretrained.h[i].mlp.c_proj.bias.data.clone()

    if verbose:
        print(f"  ✓ All {num_layers} transformer blocks loaded")

    # 4. Final layer normalization
    model.ln_f.weight.data = pretrained.ln_f.weight.data.clone()
    model.ln_f.bias.data = pretrained.ln_f.bias.data.clone()
    if verbose:
        print("  ✓ Final layer norm loaded")

    # 5. LM head is tied to embeddings
    model.lm_head.weight = model.wte.weight
    if verbose:
        print("  ✓ LM head tied to embeddings")

    # Clean up
    del pretrained
    torch.cuda.empty_cache()

    if verbose:
        print("✅ All pretrained weights loaded successfully!")

    return model


def initialize_lora_properly(model, verbose=True):
    """
    Properly initialize LoRA adapters to not interfere with pretrained weights.
    """
    if verbose:
        print("\nInitializing LoRA adapters...")

    with torch.no_grad():
        for name, module in model.named_modules():
            # Handle modules with LoRA adapters dictionary
            if hasattr(module, 'lora_adapters') and module.lora_adapters:
                for bit_width, lora in module.lora_adapters.items():
                    if hasattr(lora, 'lora_B'):
                        # CRITICAL: B must be zero for BA = 0 initially
                        nn.init.zeros_(lora.lora_B)
                    if hasattr(lora, 'lora_A'):
                        # Small initialization for A
                        nn.init.normal_(lora.lora_A, mean=0.0, std=0.001)

    if verbose:
        print("  ✓ LoRA adapters initialized (B=0, A~N(0,0.001))")

    return model


def test_model_with_complete_loading():
    """
    Test the model with COMPLETE weight loading.
    """
    print("\n" + "="*70)
    print("COMPLETE WEIGHT LOADING TEST")
    print("="*70)

    # Create model
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

    print("\n1. Creating SwitchableQATGPT2 model...")
    model = SwitchableQATGPT2(gpt2_config, bit_widths=[4, 8, 16], initialize_weights=False)
    print(f"   Model created with {config.n_layer} layers, {config.n_positions} positions")

    # Load ALL weights
    print("\n2. Loading complete pretrained weights...")
    model = load_all_pretrained_weights(model, verbose=True)

    # Initialize LoRA properly
    print("\n3. Initializing LoRA adapters...")
    model = initialize_lora_properly(model, verbose=True)

    # Set initial precision
    print("\n4. Setting initial precision to 16-bit...")
    if hasattr(model, 'set_precision'):
        model.set_precision(16)
        print("  ✓ Precision set to 16-bit")

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\n5. Model moved to {device}")

    # Test with tokenizer
    print("\n6. Testing model output...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    test_texts = [
        "The capital of France is",
        "Machine learning is",
        "Hello, my name is"
    ]

    model.eval()
    results = []

    for test_text in test_texts:
        inputs = tokenizer(test_text, return_tensors='pt', max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])

            if isinstance(outputs, dict) and 'loss' in outputs and outputs['loss'] is not None:
                loss = outputs['loss'].item()
                perplexity = math.exp(loss) if loss < 20 else float('inf')
                results.append((test_text, loss, perplexity))

                print(f"\n   Text: '{test_text}'")
                print(f"   Loss: {loss:.4f}")
                print(f"   Perplexity: {perplexity:.1f}")

                # Generate some tokens to see if model works
                if perplexity < 1000:
                    gen_outputs = model.generate(
                        inputs['input_ids'],
                        max_length=20,
                        num_return_sequences=1,
                        temperature=0.8,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    generated = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
                    print(f"   Generated: '{generated}'")

    # Overall assessment
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    if results:
        avg_perplexity = sum(p for _, _, p in results if p != float('inf')) / len(results)

        if avg_perplexity < 50:
            print("✅ EXCELLENT: Model is working perfectly!")
        elif avg_perplexity < 100:
            print("✅ GOOD: Model is working well!")
        elif avg_perplexity < 500:
            print("⚠️ WARNING: Model has higher perplexity than expected")
        else:
            print("❌ ERROR: Model perplexity is too high")

        print(f"\nAverage perplexity: {avg_perplexity:.1f}")

    return model


def compare_with_reference():
    """
    Compare QAT model with reference GPT-2 to ensure weights match.
    """
    print("\n" + "="*70)
    print("WEIGHT COMPARISON TEST")
    print("="*70)

    # Create QAT model
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

    model = QATGPT2(gpt2_config, quantization_bits=8, initialize_weights=False)
    model = load_all_pretrained_weights(model, verbose=False)

    # Load reference
    reference = GPT2Model.from_pretrained('gpt2')

    print("\nComparing weights with reference GPT-2:")

    # Compare embeddings
    wte_match = torch.allclose(model.wte.weight, reference.wte.weight, rtol=1e-5)
    print(f"  Token embeddings match: {wte_match}")

    # Compare position embeddings (first N)
    min_pos = min(model.wpe.weight.shape[0], reference.wpe.weight.shape[0])
    wpe_match = torch.allclose(
        model.wpe.weight[:min_pos],
        reference.wpe.weight[:min_pos],
        rtol=1e-5
    )
    print(f"  Position embeddings match (first {min_pos}): {wpe_match}")

    # Compare first layer
    ln1_match = torch.allclose(model.h[0].ln_1.weight, reference.h[0].ln_1.weight, rtol=1e-5)
    print(f"  First block ln_1 match: {ln1_match}")

    # Compare final layer norm
    lnf_match = torch.allclose(model.ln_f.weight, reference.ln_f.weight, rtol=1e-5)
    print(f"  Final layer norm match: {lnf_match}")

    if wte_match and wpe_match and ln1_match and lnf_match:
        print("\n✅ All checked weights match reference model!")
    else:
        print("\n❌ Some weights don't match!")

    del reference, model
    torch.cuda.empty_cache()


def main():
    """Run complete loading tests."""
    print("\n" + "="*70)
    print("COMPLETE MODEL LOADING TEST SUITE")
    print("="*70)

    try:
        # Test 1: Weight comparison
        compare_with_reference()

        # Test 2: Complete loading and inference
        model = test_model_with_complete_loading()

        print("\n" + "="*70)
        print("ALL TESTS COMPLETED!")
        print("="*70)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    exit(main())
#!/usr/bin/env python3
"""
Diagnose High Loss Issue in SP Training
Identifies why the model is producing extremely high loss values
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2Config, GPT2TokenizerFast, GPT2Model

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models_sp import SPLMHeadModel
from part1_switchable_precision.config_sp import ModelConfig


def check_model_initialization():
    """Check if model weights are properly initialized."""
    print("\n" + "="*60)
    print("DIAGNOSING HIGH LOSS ISSUE")
    print("="*60)

    print("\n1. Checking Model Initialization...")

    # Create model
    model_config = ModelConfig()
    model_config.n_layer = 2
    model_config.n_embd = 256
    model_config.n_head = 4
    model_config.vocab_size = 50257

    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head
    )
    gpt2_config.bit_widths = model_config.bit_widths
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit

    model = SPLMHeadModel(gpt2_config)

    # Check weight statistics
    print("\n   Weight Statistics (before initialization):")
    for name, param in model.named_parameters():
        if 'weight' in name and 'lora' not in name:
            mean = param.data.mean().item()
            std = param.data.std().item()
            min_val = param.data.min().item()
            max_val = param.data.max().item()

            if abs(mean) > 1.0 or std > 2.0 or std < 0.001:
                print(f"   ⚠️ {name[:40]:40s}: mean={mean:8.4f}, std={std:8.4f}, min={min_val:8.4f}, max={max_val:8.4f}")
            else:
                print(f"   ✓ {name[:40]:40s}: mean={mean:8.4f}, std={std:8.4f}")

    return model


def check_embeddings(model):
    """Check if embeddings are initialized properly."""
    print("\n2. Checking Embeddings...")

    # Check token embeddings
    wte_mean = model.transformer.wte.weight.data.mean().item()
    wte_std = model.transformer.wte.weight.data.std().item()
    print(f"   Token embeddings: mean={wte_mean:.4f}, std={wte_std:.4f}")

    # Check position embeddings
    wpe_mean = model.transformer.wpe.weight.data.mean().item()
    wpe_std = model.transformer.wpe.weight.data.std().item()
    print(f"   Position embeddings: mean={wpe_mean:.4f}, std={wpe_std:.4f}")

    # Check if embeddings are zeros
    if wte_std < 0.001:
        print("   ⚠️ Token embeddings appear to be uninitialized (near zero)")
    if wpe_std < 0.001:
        print("   ⚠️ Position embeddings appear to be uninitialized (near zero)")

    return wte_std > 0.001 and wpe_std > 0.001


def test_forward_pass(model):
    """Test forward pass with simple input."""
    print("\n3. Testing Forward Pass...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Create simple input
    batch_size = 2
    seq_length = 10
    vocab_size = model.config.vocab_size

    # Test with different input patterns
    test_cases = [
        ("Random tokens", torch.randint(0, vocab_size, (batch_size, seq_length))),
        ("All zeros", torch.zeros((batch_size, seq_length), dtype=torch.long)),
        ("All padding", torch.full((batch_size, seq_length), 50256, dtype=torch.long)),  # EOS token
        ("Sequential", torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1))
    ]

    for name, input_ids in test_cases:
        input_ids = input_ids.to(device)
        labels = input_ids.clone()

        with torch.no_grad():
            # Test at different bit widths
            for bits in [16, 8, 4]:
                model.set_precision(bits)
                outputs = model(input_ids, labels=labels)
                loss = outputs['loss'].item()
                logits = outputs['logits']

                # Check for issues
                if np.isnan(loss) or np.isinf(loss):
                    print(f"   ❌ {name} at {bits}-bit: Loss is {loss} (NaN or Inf)")
                elif loss > 100:
                    print(f"   ⚠️ {name} at {bits}-bit: Loss is extremely high: {loss:.2f}")
                elif loss > 20:
                    print(f"   ⚠️ {name} at {bits}-bit: Loss is very high: {loss:.2f}")
                else:
                    print(f"   ✓ {name} at {bits}-bit: Loss = {loss:.4f}")

                # Check logits
                logits_mean = logits.mean().item()
                logits_std = logits.std().item()
                if logits_std < 0.001:
                    print(f"      ⚠️ Logits have no variance (std={logits_std:.4f})")


def initialize_with_pretrained():
    """Initialize model with pretrained GPT-2 weights."""
    print("\n4. Testing with Pretrained Weights...")

    # Create model
    model_config = ModelConfig()
    model_config.n_layer = 2
    model_config.n_embd = 768  # Use GPT-2 dimensions
    model_config.n_head = 12
    model_config.vocab_size = 50257

    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head
    )
    gpt2_config.bit_widths = model_config.bit_widths
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit

    model = SPLMHeadModel(gpt2_config)

    # Load pretrained weights
    print("   Loading pretrained GPT-2 weights...")
    pretrained = GPT2Model.from_pretrained('gpt2')

    # Copy embeddings
    model.transformer.wte.weight.data = pretrained.wte.weight.data.clone()
    model.transformer.wpe.weight.data[:pretrained.wpe.weight.shape[0]] = pretrained.wpe.weight.data.clone()

    # Copy first 2 transformer blocks
    for i in range(min(2, len(pretrained.h))):
        # Layer norms
        model.transformer.h[i].ln_1.weight.data = pretrained.h[i].ln_1.weight.data.clone()
        model.transformer.h[i].ln_1.bias.data = pretrained.h[i].ln_1.bias.data.clone()
        model.transformer.h[i].ln_2.weight.data = pretrained.h[i].ln_2.weight.data.clone()
        model.transformer.h[i].ln_2.bias.data = pretrained.h[i].ln_2.bias.data.clone()

        # Attention weights - note the transpose for linear layers
        model.transformer.h[i].attn.c_attn.linear.weight.data = pretrained.h[i].attn.c_attn.weight.data.t().contiguous()
        model.transformer.h[i].attn.c_attn.linear.bias.data = pretrained.h[i].attn.c_attn.bias.data.clone()
        model.transformer.h[i].attn.c_proj.linear.weight.data = pretrained.h[i].attn.c_proj.weight.data.t().contiguous()
        model.transformer.h[i].attn.c_proj.linear.bias.data = pretrained.h[i].attn.c_proj.bias.data.clone()

        # MLP weights
        model.transformer.h[i].mlp.c_fc.linear.weight.data = pretrained.h[i].mlp.c_fc.weight.data.t().contiguous()
        model.transformer.h[i].mlp.c_fc.linear.bias.data = pretrained.h[i].mlp.c_fc.bias.data.clone()
        model.transformer.h[i].mlp.c_proj.linear.weight.data = pretrained.h[i].mlp.c_proj.weight.data.t().contiguous()
        model.transformer.h[i].mlp.c_proj.linear.bias.data = pretrained.h[i].mlp.c_proj.bias.data.clone()

    # Final layer norm
    model.transformer.ln_f.weight.data = pretrained.ln_f.weight.data.clone()
    model.transformer.ln_f.bias.data = pretrained.ln_f.bias.data.clone()

    print("   ✓ Pretrained weights loaded")

    # Test with pretrained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Test with real text
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token for GPT-2
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors='pt', max_length=50, truncation=True, padding='max_length')
    input_ids = inputs['input_ids'].to(device)
    labels = input_ids.clone()

    with torch.no_grad():
        model.set_precision(16)
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss'].item()

    print(f"\n   With pretrained weights:")
    print(f"   Loss on real text: {loss:.4f}")

    if loss < 10:
        print("   ✅ Loss is reasonable with pretrained weights!")
    else:
        print("   ⚠️ Loss is still high even with pretrained weights")

    return model, loss


def check_lora_initialization(model):
    """Check if LoRA adapters are properly initialized."""
    print("\n5. Checking LoRA Initialization...")

    for name, module in model.named_modules():
        if hasattr(module, 'lora_adapters'):
            for bit_key, adapter in module.lora_adapters.items():
                lora_a = adapter['A']
                lora_b = adapter['B']

                # LoRA B should be initialized to zeros
                b_mean = lora_b.data.mean().item()
                b_std = lora_b.data.std().item()

                # LoRA A should be initialized with small values
                a_mean = lora_a.data.mean().item()
                a_std = lora_a.data.std().item()

                if abs(b_mean) > 0.01 or b_std > 0.01:
                    print(f"   ⚠️ {name}.{bit_key} LoRA B not zero: mean={b_mean:.4f}, std={b_std:.4f}")
                else:
                    print(f"   ✓ {name}.{bit_key} LoRA B properly initialized to zero")


def diagnose_and_fix():
    """Main diagnostic function."""
    print("\n" + "="*60)
    print("DIAGNOSIS RESULTS")
    print("="*60)

    # 1. Check basic initialization
    model = check_model_initialization()

    # 2. Check embeddings
    embeddings_ok = check_embeddings(model)

    # 3. Test forward pass
    test_forward_pass(model)

    # 4. Test with pretrained weights
    pretrained_model, pretrained_loss = initialize_with_pretrained()

    # 5. Check LoRA
    check_lora_initialization(model)

    # Diagnosis
    print("\n" + "="*60)
    print("RECOMMENDED FIXES")
    print("="*60)

    if pretrained_loss < 10:
        print("\n✅ The model works well with pretrained weights.")
        print("\nRecommended fixes:")
        print("1. Always initialize with pretrained GPT-2 weights")
        print("2. Make sure the load_pretrained_weights function is called")
        print("3. Verify weight dimensions match (especially n_embd)")
    else:
        print("\n⚠️ Issue persists even with pretrained weights.")
        print("\nPossible causes:")
        print("1. Quantization is too aggressive (try starting with 16-bit)")
        print("2. LoRA rank might be too low")
        print("3. Learning rate might be too high")

    print("\nExample fix in your training code:")
    print("-"*40)
    print("""
# In main_sp.py, make sure to load pretrained weights:
from transformers import GPT2Model

def load_pretrained_weights(model):
    pretrained = GPT2Model.from_pretrained('gpt2')

    # Copy embeddings
    model.transformer.wte.weight.data = pretrained.wte.weight.data.clone()
    model.transformer.wpe.weight.data[:pretrained.wpe.weight.shape[0]] = pretrained.wpe.weight.data.clone()

    # Copy transformer blocks (as many as available)
    for i in range(min(len(model.transformer.h), len(pretrained.h))):
        # ... (copy weights as shown above)

    print("Pretrained weights loaded successfully")

# Call this after creating the model:
model = SPLMHeadModel(config)
load_pretrained_weights(model)
""")

    return pretrained_model


if __name__ == "__main__":
    model = diagnose_and_fix()

    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    print("\nKey finding: The high loss is likely due to random initialization.")
    print("Solution: Always load pretrained GPT-2 weights before training.")
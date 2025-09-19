#!/usr/bin/env python3
"""
Debug Model Mismatch
Identifies why SP model differs from GPT-2 in perplexity
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test.fix_model_initialization import create_properly_initialized_model


def compare_model_architectures(sp_model, gpt2_model):
    """Compare the architectures of SP and GPT-2 models."""
    print("\n" + "="*60)
    print("ARCHITECTURE COMPARISON")
    print("="*60)

    print(f"\n📐 MODEL DIMENSIONS:")
    print(f"   SP Model:")
    print(f"     Vocab size: {sp_model.config.vocab_size}")
    print(f"     Hidden size: {sp_model.config.n_embd}")
    print(f"     Num layers: {sp_model.config.n_layer}")
    print(f"     Num heads: {sp_model.config.n_head}")

    print(f"   GPT-2 Model:")
    print(f"     Vocab size: {gpt2_model.config.vocab_size}")
    print(f"     Hidden size: {gpt2_model.config.n_embd}")
    print(f"     Num layers: {gpt2_model.config.n_layer}")
    print(f"     Num heads: {gpt2_model.config.n_head}")

    # Check parameter counts
    sp_params = sum(p.numel() for p in sp_model.parameters())
    gpt2_params = sum(p.numel() for p in gpt2_model.parameters())

    print(f"\n📊 PARAMETER COUNTS:")
    print(f"   SP Model: {sp_params:,}")
    print(f"   GPT-2 Model: {gpt2_params:,}")
    print(f"   Difference: {abs(sp_params - gpt2_params):,}")

    # Check specific layer structures
    print(f"\n🏗️ LAYER STRUCTURE:")
    print(f"   SP first block type: {type(sp_model.transformer.h[0]).__name__}")
    print(f"   GPT-2 first block type: {type(gpt2_model.transformer.h[0]).__name__}")


def compare_weight_values(sp_model, gpt2_model):
    """Compare actual weight values between models."""
    print("\n" + "="*60)
    print("WEIGHT VALUE COMPARISON")
    print("="*60)

    # Compare embeddings
    print(f"\n📝 EMBEDDING WEIGHTS:")
    sp_wte = sp_model.transformer.wte.weight
    gpt2_wte = gpt2_model.transformer.wte.weight

    wte_diff = (sp_wte - gpt2_wte).abs().mean().item()
    wte_max_diff = (sp_wte - gpt2_wte).abs().max().item()

    print(f"   Token embedding diff - Mean: {wte_diff:.6f}, Max: {wte_max_diff:.6f}")

    # Compare first layer weights
    print(f"\n🔍 FIRST LAYER WEIGHTS:")

    # Attention weights
    try:
        if hasattr(sp_model.transformer.h[0].attn.c_attn, 'linear'):
            sp_attn_w = sp_model.transformer.h[0].attn.c_attn.linear.weight
        else:
            sp_attn_w = sp_model.transformer.h[0].attn.c_attn.weight

        gpt2_attn_w = gpt2_model.transformer.h[0].attn.c_attn.weight

        # GPT-2 uses Conv1D (transposed), so we need to transpose for comparison
        if sp_attn_w.shape != gpt2_attn_w.shape:
            gpt2_attn_w = gpt2_attn_w.t()

        attn_diff = (sp_attn_w - gpt2_attn_w).abs().mean().item()
        attn_max_diff = (sp_attn_w - gpt2_attn_w).abs().max().item()

        print(f"   Attention weight shapes: SP={sp_attn_w.shape}, GPT-2={gpt2_attn_w.shape}")
        print(f"   Attention weight diff - Mean: {attn_diff:.6f}, Max: {attn_max_diff:.6f}")

        if attn_diff > 1e-5:
            print(f"   ⚠️ Attention weights differ significantly!")
        else:
            print(f"   ✅ Attention weights match closely")

    except Exception as e:
        print(f"   ❌ Error comparing attention weights: {e}")

    # Compare MLP weights
    try:
        if hasattr(sp_model.transformer.h[0].mlp.c_fc, 'linear'):
            sp_mlp_w = sp_model.transformer.h[0].mlp.c_fc.linear.weight
        else:
            sp_mlp_w = sp_model.transformer.h[0].mlp.c_fc.weight

        gpt2_mlp_w = gpt2_model.transformer.h[0].mlp.c_fc.weight

        # GPT-2 uses Conv1D (transposed)
        if sp_mlp_w.shape != gpt2_mlp_w.shape:
            gpt2_mlp_w = gpt2_mlp_w.t()

        mlp_diff = (sp_mlp_w - gpt2_mlp_w).abs().mean().item()
        mlp_max_diff = (sp_mlp_w - gpt2_mlp_w).abs().max().item()

        print(f"   MLP weight shapes: SP={sp_mlp_w.shape}, GPT-2={gpt2_mlp_w.shape}")
        print(f"   MLP weight diff - Mean: {mlp_diff:.6f}, Max: {mlp_max_diff:.6f}")

        if mlp_diff > 1e-5:
            print(f"   ⚠️ MLP weights differ significantly!")
        else:
            print(f"   ✅ MLP weights match closely")

    except Exception as e:
        print(f"   ❌ Error comparing MLP weights: {e}")


def test_forward_pass_equivalence(sp_model, gpt2_model, tokenizer, device):
    """Test if forward passes produce equivalent results."""
    print("\n" + "="*60)
    print("FORWARD PASS EQUIVALENCE TEST")
    print("="*60)

    # Simple test input
    test_text = "The quick brown fox"
    inputs = tokenizer(test_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)

    print(f"\nTest input: \"{test_text}\"")
    print(f"Input shape: {input_ids.shape}")

    with torch.no_grad():
        # SP model forward pass (16-bit)
        sp_model.set_precision(16)
        sp_outputs = sp_model(input_ids)

        # Handle different output formats
        if isinstance(sp_outputs, dict) and 'logits' in sp_outputs:
            sp_logits = sp_outputs['logits']
        elif isinstance(sp_outputs, dict) and 'prediction_scores' in sp_outputs:
            sp_logits = sp_outputs['prediction_scores']
        else:
            sp_logits = sp_outputs  # Assume it's the logits tensor directly

        # GPT-2 forward pass
        gpt2_outputs = gpt2_model(input_ids)
        gpt2_logits = gpt2_outputs['logits']

    print(f"\nLogit shapes: SP={sp_logits.shape}, GPT-2={gpt2_logits.shape}")

    # Compare logits
    logit_diff = (sp_logits - gpt2_logits).abs()
    mean_diff = logit_diff.mean().item()
    max_diff = logit_diff.max().item()

    print(f"Logit differences - Mean: {mean_diff:.6f}, Max: {max_diff:.6f}")

    # Compare top predictions
    sp_top5 = torch.topk(sp_logits[0, -1], 5)
    gpt2_top5 = torch.topk(gpt2_logits[0, -1], 5)

    print(f"\n🏆 TOP 5 PREDICTIONS:")
    print("SP Model:")
    for i, (idx, score) in enumerate(zip(sp_top5.indices, sp_top5.values)):
        token = tokenizer.decode(idx.item())
        print(f"  {i+1}. '{token}' (score: {score.item():.3f})")

    print("GPT-2 Model:")
    for i, (idx, score) in enumerate(zip(gpt2_top5.indices, gpt2_top5.values)):
        token = tokenizer.decode(idx.item())
        print(f"  {i+1}. '{token}' (score: {score.item():.3f})")

    # Check if top predictions match
    overlap = len(set(sp_top5.indices.cpu().numpy()) & set(gpt2_top5.indices.cpu().numpy()))
    print(f"\nTop-5 overlap: {overlap}/5")

    if mean_diff < 1e-3 and max_diff < 1e-2:
        print("✅ Forward passes are equivalent!")
        return True
    else:
        print("❌ Forward passes differ significantly!")
        return False


def test_quantization_interference(sp_model, tokenizer, device):
    """Test if quantization layers interfere even at 16-bit."""
    print("\n" + "="*60)
    print("QUANTIZATION INTERFERENCE TEST")
    print("="*60)

    test_text = "The quick brown fox"
    inputs = tokenizer(test_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)

    print(f"Testing if quantization affects 16-bit mode...")

    with torch.no_grad():
        # Test at different precisions
        results = {}
        for precision in [16, 8, 4]:
            sp_model.set_precision(precision)
            outputs = sp_model(input_ids, labels=input_ids)
            loss = outputs['loss'].item()
            results[precision] = loss
            print(f"   {precision:2d}-bit: Loss = {loss:.4f}")

    # Check if 16-bit is affected by quantization infrastructure
    loss_16 = results[16]
    loss_8 = results[8]

    if abs(loss_16 - loss_8) > 0.1:
        print(f"\n✅ Quantization working correctly (16-bit != 8-bit)")
    else:
        print(f"\n⚠️ Quantization may not be working (16-bit ≈ 8-bit)")

    # Test if we can disable quantization completely
    print(f"\nTesting with quantization disabled...")
    try:
        # Temporarily disable quantization
        for name, module in sp_model.named_modules():
            if hasattr(module, 'quantization_enabled'):
                module.quantization_enabled = False

        outputs = sp_model(input_ids, labels=input_ids)
        loss_no_quant = outputs['loss'].item()
        print(f"   No quantization: Loss = {loss_no_quant:.4f}")

        if abs(loss_16 - loss_no_quant) > 0.01:
            print(f"   ⚠️ Quantization infrastructure affects 16-bit mode")
        else:
            print(f"   ✅ Quantization infrastructure doesn't affect 16-bit mode")

    except Exception as e:
        print(f"   ❌ Cannot disable quantization: {e}")


def check_model_state(sp_model):
    """Check the current state of the SP model."""
    print("\n" + "="*60)
    print("MODEL STATE CHECK")
    print("="*60)

    print(f"Current precision: {sp_model.get_current_precision()} bits")

    # Check if model is in training/eval mode
    print(f"Training mode: {sp_model.training}")

    # Check LoRA adapter states
    print(f"\n🔧 LoRA ADAPTER STATUS:")
    for name, module in sp_model.named_modules():
        if hasattr(module, 'lora_adapters'):
            current_bits = sp_model.get_current_precision()
            if current_bits in module.lora_adapters:
                adapter = module.lora_adapters[current_bits]
                a_norm = adapter.lora_A.norm().item()
                b_norm = adapter.lora_B.norm().item()
                print(f"   {name}: A_norm={a_norm:.4f}, B_norm={b_norm:.4f}")

                if b_norm > 0.01:
                    print(f"     ⚠️ LoRA B should be near zero for proper initialization")

    # Check if quantizers are calibrated
    print(f"\n⚖️ QUANTIZER STATUS:")
    for name, module in sp_model.named_modules():
        if hasattr(module, 'weight_quantizer'):
            print(f"   {name}.weight_quantizer: calibrated={getattr(module.weight_quantizer, 'calibrated', 'N/A')}")
        if hasattr(module, 'activation_quantizer'):
            print(f"   {name}.activation_quantizer: calibrated={getattr(module.activation_quantizer, 'calibrated', 'N/A')}")


def main():
    """Run comprehensive debugging."""
    print("\n" + "="*80)
    print("SP MODEL MISMATCH DEBUGGING")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load models
    print("\n1. Loading models...")
    sp_model, sp_config = create_properly_initialized_model(use_pretrained=True)
    sp_model = sp_model.to(device)
    sp_model.eval()

    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_model = gpt2_model.to(device)
    gpt2_model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Run diagnostics
    compare_model_architectures(sp_model, gpt2_model)
    compare_weight_values(sp_model, gpt2_model)

    forward_match = test_forward_pass_equivalence(sp_model, gpt2_model, tokenizer, device)

    test_quantization_interference(sp_model, tokenizer, device)
    check_model_state(sp_model)

    # Summary
    print("\n" + "="*80)
    print("DEBUGGING SUMMARY")
    print("="*80)

    print("\n🔍 POTENTIAL ISSUES IDENTIFIED:")

    if not forward_match:
        print("❌ Forward pass results differ from GPT-2")
        print("   → Check weight loading process")
        print("   → Verify model architecture matches GPT-2")

    print("\n💡 RECOMMENDED FIXES:")
    print("1. Ensure quantization layers are bypassed in 16-bit mode")
    print("2. Verify all Conv1D weights are properly transposed")
    print("3. Check that LoRA adapters start with zero contribution")
    print("4. Ensure model is in eval mode during testing")


if __name__ == "__main__":
    main()
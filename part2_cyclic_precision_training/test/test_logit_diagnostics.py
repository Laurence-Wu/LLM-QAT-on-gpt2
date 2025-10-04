#!/usr/bin/env python3
"""
Diagnostic tests for negative logit bias issue.

This script systematically checks:
1. FP32 vs quantized logit comparison
2. LayerNorm output analysis
3. lm_head weight verification
4. LoRA contribution measurement
5. Hidden state progression through layers
"""

import torch
import torch.nn.functional as F
import sys
import os
from pathlib import Path

# Add parent directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
part2_dir = os.path.join(parent_dir, 'part2_cyclic_precision_training')
sys.path.insert(0, part2_dir)

from cpt_model import CPTModel
from config_cpt import get_config
from transformers import GPT2Tokenizer


def test_simple_prompt_confidence():
    """Test 1: Simple prompt should have high confidence next token prediction."""
    print("\n" + "="*80)
    print("TEST 1: Simple Prompt Confidence")
    print("="*80)

    config = get_config()
    model = CPTModel(config)
    model.eval()
    model.cuda()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Test a simple, predictable prompt
    test_prompts = [
        "The cat",
        "Hello world",
        "I am a",
        "The quick brown"
    ]

    for prompt in test_prompts:
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]  # Last token logits

            probs = F.softmax(logits, dim=-1)
            top5 = probs.topk(5)

            print(f"\nPrompt: '{prompt}'")
            print(f"  Logits: mean={logits.mean():.2f}, max={logits.max():.2f}, min={logits.min():.2f}")
            print(f"  Top-5 predictions:")
            for i, (prob, token_id) in enumerate(zip(top5.values, top5.indices)):
                token = tokenizer.decode([token_id])
                print(f"    {i+1}. '{token}' (prob={prob:.4f}, logit={logits[token_id]:.2f})")


def test_fp32_vs_quantized():
    """Test 2: Compare FP32 vs quantized logits."""
    print("\n" + "="*80)
    print("TEST 2: FP32 vs Quantized Logits")
    print("="*80)

    config = get_config()
    model = CPTModel(config)
    model.eval()
    model.cuda()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer("The cat sat on the mat", return_tensors='pt').input_ids.cuda()

    # Test at FP32 (no quantization)
    print("\nTesting at 32-bit (FP32, no quantization):")
    model.set_precision(32)
    with torch.no_grad():
        outputs_fp32 = model(input_ids)
        logits_fp32 = outputs_fp32.logits

    print(f"  Logits mean: {logits_fp32.mean():.4f}")
    print(f"  Logits std: {logits_fp32.std():.4f}")
    print(f"  Logits max: {logits_fp32.max():.4f}")
    print(f"  Logits min: {logits_fp32.min():.4f}")
    print(f"  Positive logits: {(logits_fp32 > 0).sum().item()}/{logits_fp32.numel()}")

    # Test at different quantization levels
    for bits in [8, 6, 4]:
        print(f"\nTesting at {bits}-bit quantization:")
        model.set_precision(bits)
        with torch.no_grad():
            outputs_quant = model(input_ids)
            logits_quant = outputs_quant.logits

        print(f"  Logits mean: {logits_quant.mean():.4f}")
        print(f"  Logits std: {logits_quant.std():.4f}")
        print(f"  Logits max: {logits_quant.max():.4f}")
        print(f"  Logits min: {logits_quant.min():.4f}")
        print(f"  Positive logits: {(logits_quant > 0).sum().item()}/{logits_quant.numel()}")

        # Compute difference from FP32
        diff = (logits_quant - logits_fp32).abs()
        print(f"  Diff from FP32: mean={diff.mean():.4f}, max={diff.max():.4f}")


def test_layernorm_output():
    """Test 3: Check if LayerNorm is shifting values negative."""
    print("\n" + "="*80)
    print("TEST 3: LayerNorm Output Analysis")
    print("="*80)

    config = get_config()
    model = CPTModel(config)
    model.eval()
    model.cuda()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer("The cat sat on the mat", return_tensors='pt').input_ids.cuda()

    # Hook to capture hidden states before ln_f
    hidden_before_ln_f = []

    def hook_fn(module, input, output):
        hidden_before_ln_f.append(input[0].clone())

    hook = model.ln_f.register_forward_hook(hook_fn)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    hook.remove()

    # Analyze hidden states before and after ln_f
    hidden_before = hidden_before_ln_f[0]
    hidden_after = outputs.hidden_states

    print(f"\nBefore ln_f:")
    print(f"  Mean: {hidden_before.mean():.4f}")
    print(f"  Std: {hidden_before.std():.4f}")
    print(f"  Max: {hidden_before.max():.4f}")
    print(f"  Min: {hidden_before.min():.4f}")

    print(f"\nAfter ln_f:")
    print(f"  Mean: {hidden_after.mean():.4f}")
    print(f"  Std: {hidden_after.std():.4f}")
    print(f"  Max: {hidden_after.max():.4f}")
    print(f"  Min: {hidden_after.min():.4f}")

    # Check LayerNorm parameters
    print(f"\nLayerNorm parameters:")
    print(f"  Gamma (weight) mean: {model.ln_f.weight.mean():.4f}")
    print(f"  Gamma (weight) std: {model.ln_f.weight.std():.4f}")
    print(f"  Beta (bias) mean: {model.ln_f.bias.mean():.4f}")
    print(f"  Beta (bias) std: {model.ln_f.bias.std():.4f}")

    print(f"\nFinal logits:")
    print(f"  Mean: {logits.mean():.4f}")
    print(f"  Max: {logits.max():.4f}")
    print(f"  Min: {logits.min():.4f}")


def test_lm_head_weights():
    """Test 4: Verify lm_head weights are from pretrained model."""
    print("\n" + "="*80)
    print("TEST 4: lm_head Weight Verification")
    print("="*80)

    config = get_config()
    model = CPTModel(config)
    model.eval()
    model.cuda()

    # Check lm_head weight statistics
    lm_head_weight = model.lm_head.linear.weight

    print(f"\nlm_head.linear.weight statistics:")
    print(f"  Shape: {lm_head_weight.shape}")
    print(f"  Mean: {lm_head_weight.mean():.6f}")
    print(f"  Std: {lm_head_weight.std():.6f}")
    print(f"  Max: {lm_head_weight.max():.6f}")
    print(f"  Min: {lm_head_weight.min():.6f}")

    # Check if weights are initialized (should NOT be all zeros or random)
    if lm_head_weight.abs().max() < 0.001:
        print("  ⚠️ WARNING: Weights appear to be near zero!")
    elif lm_head_weight.std() < 0.01:
        print("  ⚠️ WARNING: Very low weight variance!")
    else:
        print("  ✓ Weights appear reasonable")

    # Check bias (should be None for GPT-2 lm_head)
    if model.lm_head.linear.bias is None:
        print(f"  Bias: None (correct for GPT-2)")
    else:
        print(f"  ⚠️ WARNING: Bias exists (mean={model.lm_head.linear.bias.mean():.6f})")


def test_lora_contribution():
    """Test 5: Measure LoRA's actual effect on logits."""
    print("\n" + "="*80)
    print("TEST 5: LoRA Contribution Analysis")
    print("="*80)

    config = get_config()
    model = CPTModel(config)
    model.eval()
    model.cuda()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer("The cat sat on the mat", return_tensors='pt').input_ids.cuda()

    # Test with LoRA enabled (calibration_mode=False)
    print("\nWith LoRA enabled (calibration_mode=False):")
    model.enable_lora_after_calibration()
    with torch.no_grad():
        outputs_with_lora = model(input_ids)
        logits_with_lora = outputs_with_lora.logits

    print(f"  Logits mean: {logits_with_lora.mean():.4f}")
    print(f"  Logits max: {logits_with_lora.max():.4f}")
    print(f"  Logits min: {logits_with_lora.min():.4f}")

    # Test with LoRA disabled (calibration_mode=True)
    print("\nWith LoRA disabled (calibration_mode=True):")
    model.disable_lora_for_calibration()
    with torch.no_grad():
        outputs_no_lora = model(input_ids)
        logits_no_lora = outputs_no_lora.logits

    print(f"  Logits mean: {logits_no_lora.mean():.4f}")
    print(f"  Logits max: {logits_no_lora.max():.4f}")
    print(f"  Logits min: {logits_no_lora.min():.4f}")

    # Compare difference
    diff = logits_with_lora - logits_no_lora
    print(f"\nLoRA contribution (difference):")
    print(f"  Mean shift: {diff.mean():.4f}")
    print(f"  Std: {diff.std():.4f}")
    print(f"  Max shift: {diff.max():.4f}")
    print(f"  Min shift: {diff.min():.4f}")

    # Check LoRA weight norms
    lm_head_lora = model.lm_head.shared_lora
    if lm_head_lora.lora_A is not None:
        print(f"\nlm_head LoRA weights:")
        print(f"  lora_A norm: {lm_head_lora.lora_A.norm():.4f}")
        print(f"  lora_B norm: {lm_head_lora.lora_B.norm():.4f}")
        print(f"  lora_A mean: {lm_head_lora.lora_A.mean():.6f}")
        print(f"  lora_B mean: {lm_head_lora.lora_B.mean():.6f}")


def test_hidden_state_progression():
    """Test 6: Track hidden state statistics through all layers."""
    print("\n" + "="*80)
    print("TEST 6: Hidden State Progression Through Layers")
    print("="*80)

    config = get_config()
    model = CPTModel(config)
    model.eval()
    model.cuda()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer("The cat sat on the mat", return_tensors='pt').input_ids.cuda()

    # Register hooks to capture hidden states at each layer
    layer_outputs = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            layer_outputs.append((layer_idx, hidden_states.clone()))
        return hook_fn

    hooks = []
    for i, block in enumerate(model.h):
        hook = block.register_forward_hook(make_hook(i))
        hooks.append(hook)

    with torch.no_grad():
        outputs = model(input_ids)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Analyze progression
    print(f"\nHidden state statistics by layer:")
    print(f"{'Layer':<6} {'Mean':<10} {'Std':<10} {'Max':<10} {'Min':<10}")
    print("-" * 50)

    for layer_idx, hidden in layer_outputs:
        mean = hidden.mean().item()
        std = hidden.std().item()
        max_val = hidden.max().item()
        min_val = hidden.min().item()
        print(f"{layer_idx:<6} {mean:<10.4f} {std:<10.4f} {max_val:<10.4f} {min_val:<10.4f}")

    # Check final output
    logits = outputs.logits
    print(f"\nFinal logits:")
    print(f"  Mean: {logits.mean():.4f}")
    print(f"  Std: {logits.std():.4f}")
    print(f"  Max: {logits.max():.4f}")
    print(f"  Min: {logits.min():.4f}")


def test_quantizer_scales():
    """Test 7: Check quantizer scale values for lm_head."""
    print("\n" + "="*80)
    print("TEST 7: Quantizer Scale Analysis")
    print("="*80)

    config = get_config()
    model = CPTModel(config)
    model.eval()
    model.cuda()

    lm_head = model.lm_head

    print(f"\nlm_head Weight Quantizer:")
    print(f"  Calibrated: {lm_head.quantizer_weight.calibrated}")
    print(f"  Num bits: {lm_head.quantizer_weight.num_bits}")
    print(f"  Scale shape: {lm_head.quantizer_weight.scale.shape}")
    print(f"  Scale mean: {lm_head.quantizer_weight.scale.mean():.6f}")
    print(f"  Scale std: {lm_head.quantizer_weight.scale.std():.6f}")
    print(f"  Scale min: {lm_head.quantizer_weight.scale.min():.6f}")
    print(f"  Scale max: {lm_head.quantizer_weight.scale.max():.6f}")

    if lm_head.quantizer_weight.scale.min() < 1e-6:
        print(f"  ⚠️ WARNING: Very small scale detected (< 1e-6)!")

    print(f"\nlm_head Input Quantizer:")
    print(f"  Calibrated: {lm_head.quantizer_input.calibrated}")
    print(f"  Num bits: {lm_head.quantizer_input.num_bits}")
    print(f"  Scale shape: {lm_head.quantizer_input.scale.shape}")
    print(f"  Scale mean: {lm_head.quantizer_input.scale.mean():.6f}")
    print(f"  Scale std: {lm_head.quantizer_input.scale.std():.6f}")
    print(f"  Scale min: {lm_head.quantizer_input.scale.min():.6f}")
    print(f"  Scale max: {lm_head.quantizer_input.scale.max():.6f}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("LOGIT DIAGNOSTIC TEST SUITE")
    print("Testing for negative logit bias issue")
    print("="*80)

    try:
        test_simple_prompt_confidence()
        test_fp32_vs_quantized()
        test_layernorm_output()
        test_lm_head_weights()
        test_lora_contribution()
        test_hidden_state_progression()
        test_quantizer_scales()

        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)

    except Exception as e:
        print(f"\n⚠️ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()

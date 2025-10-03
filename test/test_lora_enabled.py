#!/usr/bin/env python3
"""
Test script to verify LoRA is enabled during evaluation.

This script checks:
1. LoRA calibration_mode status (should be False)
2. LoRA weights are non-zero (trained)
3. LoRA contribution to output is non-zero
4. Model produces reasonable logits with LoRA enabled vs disabled
"""

import torch
import sys
import os
from pathlib import Path

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
part3_dir = os.path.join(parent_dir, 'part3_eval_cpt')
sys.path.insert(0, part3_dir)

from load_cpt_model import load_cpt_model
from transformers import GPT2Tokenizer


def test_lora_calibration_mode(model):
    """Test 1: Check if LoRA is in calibration mode (should be False)."""
    print("\n" + "="*70)
    print("TEST 1: LoRA Calibration Mode Status")
    print("="*70)

    lora_count = 0
    calibration_mode_count = 0

    for name, module in model.named_modules():
        if module.__class__.__name__ == 'LoRAAdapter':
            lora_count += 1
            if hasattr(module, 'calibration_mode') and module.calibration_mode:
                calibration_mode_count += 1
                print(f"  ❌ {name}: calibration_mode = True (LoRA DISABLED)")

    print(f"\nSummary:")
    print(f"  Total LoRA adapters: {lora_count}")
    print(f"  In calibration mode: {calibration_mode_count}")

    if calibration_mode_count == 0:
        print(f"  ✅ PASS: All LoRA adapters enabled (calibration_mode=False)")
        return True
    else:
        print(f"  ❌ FAIL: {calibration_mode_count} LoRA adapters disabled!")
        return False


def test_lora_weights_nonzero(model):
    """Test 2: Check if LoRA weights are non-zero (trained)."""
    print("\n" + "="*70)
    print("TEST 2: LoRA Weights Non-Zero Check")
    print("="*70)

    lora_weights = []
    zero_lora_count = 0

    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            norm = param.norm().item()
            mean = param.mean().item()
            std = param.std().item()

            lora_weights.append((name, norm, mean, std))

            if norm < 1e-6:
                zero_lora_count += 1
                print(f"  ❌ {name}: norm={norm:.2e} (ZERO!)")

    print(f"\nLoRA Weight Statistics (first 5):")
    for name, norm, mean, std in lora_weights[:5]:
        print(f"  {name}:")
        print(f"    Norm: {norm:.4f}, Mean: {mean:.6f}, Std: {std:.6f}")

    print(f"\nSummary:")
    print(f"  Total LoRA weights: {len(lora_weights)}")
    print(f"  Zero weights: {zero_lora_count}")

    if zero_lora_count == 0:
        print(f"  ✅ PASS: All LoRA weights are non-zero (trained)")
        return True
    else:
        print(f"  ❌ FAIL: {zero_lora_count} LoRA weights are zero!")
        return False


def test_lora_contribution(model, tokenizer):
    """Test 3: Measure LoRA's contribution to model output."""
    print("\n" + "="*70)
    print("TEST 3: LoRA Contribution to Output")
    print("="*70)

    device = next(model.parameters()).device
    test_input = "The quick brown fox jumps over the lazy dog"
    input_ids = tokenizer(test_input, return_tensors='pt').input_ids.to(device)

    model.eval()

    # Get output WITH LoRA enabled
    model.enable_lora_after_calibration()
    with torch.no_grad():
        outputs_with_lora = model(input_ids)
        logits_with_lora = outputs_with_lora.logits

    # Get output WITHOUT LoRA (calibration mode)
    model.disable_lora_for_calibration()
    with torch.no_grad():
        outputs_no_lora = model(input_ids)
        logits_no_lora = outputs_no_lora.logits

    # Re-enable LoRA
    model.enable_lora_after_calibration()

    # Calculate difference
    diff = (logits_with_lora - logits_no_lora).abs()

    print(f"\nLogits WITH LoRA:")
    print(f"  Mean: {logits_with_lora.mean():.4f}")
    print(f"  Std: {logits_with_lora.std():.4f}")
    print(f"  Max: {logits_with_lora.max():.4f}")
    print(f"  Min: {logits_with_lora.min():.4f}")

    print(f"\nLogits WITHOUT LoRA (base model only):")
    print(f"  Mean: {logits_no_lora.mean():.4f}")
    print(f"  Std: {logits_no_lora.std():.4f}")
    print(f"  Max: {logits_no_lora.max():.4f}")
    print(f"  Min: {logits_no_lora.min():.4f}")

    print(f"\nLoRA Contribution (absolute difference):")
    print(f"  Mean diff: {diff.mean():.4f}")
    print(f"  Max diff: {diff.max():.4f}")
    print(f"  Std diff: {diff.std():.4f}")

    # Check if LoRA is making a difference
    threshold = 0.01  # LoRA should change logits by at least this much

    if diff.mean() > threshold:
        print(f"\n  ✅ PASS: LoRA is contributing (mean diff {diff.mean():.4f} > {threshold})")
        return True
    else:
        print(f"\n  ❌ FAIL: LoRA contribution too small (mean diff {diff.mean():.4f} <= {threshold})")
        return False


def test_model_sanity(model, tokenizer):
    """Test 4: Basic sanity check - model produces reasonable logits."""
    print("\n" + "="*70)
    print("TEST 4: Model Sanity Check")
    print("="*70)

    device = next(model.parameters()).device
    test_input = "The capital of France is"
    input_ids = tokenizer(test_input, return_tensors='pt').input_ids.to(device)

    model.eval()
    model.enable_lora_after_calibration()

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    last_token_logits = logits[0, -1, :]
    probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
    top5 = probs.topk(5)

    print(f"\nInput: '{test_input}'")
    print(f"\nLogits statistics:")
    print(f"  Mean: {logits.mean():.4f}")
    print(f"  Std: {logits.std():.4f}")
    print(f"  Max: {logits.max():.4f}")
    print(f"  Min: {logits.min():.4f}")

    positive_count = (logits > 0).sum().item()
    total_count = logits.numel()
    print(f"  Positive logits: {positive_count}/{total_count} ({100*positive_count/total_count:.1f}%)")

    print(f"\nTop-5 next token predictions:")
    for i, (prob, token_id) in enumerate(zip(top5.values, top5.indices)):
        token = tokenizer.decode([token_id])
        print(f"  {i+1}. '{token}' (prob={prob:.4f})")

    # Sanity checks
    checks_passed = 0
    total_checks = 4

    # Check 1: Logits mean should be reasonable (-10 to +10)
    if -10 < logits.mean() < 10:
        print(f"\n  ✅ Check 1/4: Logits mean is reasonable ({logits.mean():.2f})")
        checks_passed += 1
    else:
        print(f"\n  ❌ Check 1/4: Logits mean is abnormal ({logits.mean():.2f})")

    # Check 2: Should have some positive logits (at least 10%)
    positive_pct = 100 * positive_count / total_count
    if positive_pct > 10:
        print(f"  ✅ Check 2/4: Has positive logits ({positive_pct:.1f}%)")
        checks_passed += 1
    else:
        print(f"  ❌ Check 2/4: Too few positive logits ({positive_pct:.1f}%)")

    # Check 3: Top prediction should have decent confidence (>0.05)
    if top5.values[0] > 0.05:
        print(f"  ✅ Check 3/4: Top prediction has confidence ({top5.values[0]:.4f})")
        checks_passed += 1
    else:
        print(f"  ❌ Check 3/4: Top prediction has low confidence ({top5.values[0]:.4f})")

    # Check 4: Probabilities should sum to ~1.0
    prob_sum = probs.sum().item()
    if 0.99 < prob_sum < 1.01:
        print(f"  ✅ Check 4/4: Probabilities sum to 1.0 ({prob_sum:.6f})")
        checks_passed += 1
    else:
        print(f"  ❌ Check 4/4: Probabilities sum is wrong ({prob_sum:.6f})")

    print(f"\n  Summary: {checks_passed}/{total_checks} sanity checks passed")

    return checks_passed == total_checks


def test_lora_parameters_trainable(model):
    """Test 5: Check if LoRA parameters are marked as trainable."""
    print("\n" + "="*70)
    print("TEST 5: LoRA Parameters Trainable Status")
    print("="*70)

    lora_params = []
    trainable_count = 0
    frozen_count = 0

    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append((name, param.requires_grad))
            if param.requires_grad:
                trainable_count += 1
            else:
                frozen_count += 1

    print(f"\nLoRA Parameter Status (first 5):")
    for name, requires_grad in lora_params[:5]:
        status = "✅ Trainable" if requires_grad else "❌ Frozen"
        print(f"  {name}: {status}")

    print(f"\nSummary:")
    print(f"  Total LoRA params: {len(lora_params)}")
    print(f"  Trainable: {trainable_count}")
    print(f"  Frozen: {frozen_count}")

    # During evaluation, requires_grad can be True or False (doesn't matter)
    # What matters is that weights are loaded and non-zero
    print(f"\n  ℹ️  NOTE: requires_grad status doesn't affect evaluation")
    print(f"  ℹ️  What matters: weights loaded & calibration_mode=False")

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test if LoRA is enabled during evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to CPT checkpoint (.pth file)')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("LoRA Enabled Verification Test Suite")
    print("="*70)
    print(f"Model: {args.model_path}\n")

    # Load model
    print("Loading model...")
    model, checkpoint_bit_width, model_config, training_config = load_cpt_model(args.model_path)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print(f"✅ Model loaded at {checkpoint_bit_width}-bit precision\n")

    # Run tests
    results = {}

    results['calibration_mode'] = test_lora_calibration_mode(model)
    results['weights_nonzero'] = test_lora_weights_nonzero(model)
    results['contribution'] = test_lora_contribution(model, tokenizer)
    results['sanity'] = test_model_sanity(model, tokenizer)
    results['trainable'] = test_lora_parameters_trainable(model)

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\nTest Results:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:20s}: {status}")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print(f"\n🎉 SUCCESS: LoRA is properly enabled for evaluation!")
        return 0
    else:
        print(f"\n⚠️  WARNING: {total_tests - passed_tests} test(s) failed!")
        print(f"   LoRA may not be working correctly during evaluation.")
        return 1


if __name__ == "__main__":
    exit(main())

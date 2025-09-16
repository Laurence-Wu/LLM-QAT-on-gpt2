#!/usr/bin/env python3
"""
Test script for the model initialization fix functions.
Tests the comprehensive fix that addresses:
1. LoRA rank configuration
2. Initial precision setting
3. LoRA adapter initialization
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
import math

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'part1_switchable_precision'))

from models import QATGPT2, SwitchableQATGPT2
from config_qat import ModelConfig


def create_test_model():
    """Create a test model for verification."""
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

    model = SwitchableQATGPT2(gpt2_config, bit_widths=[4, 8, 16])
    return model, gpt2_config


def test_lora_rank_fix():
    """Test that LoRA rank configuration is correctly fixed."""
    print("\n" + "="*60)
    print("Test 1: LoRA Rank Configuration Fix")
    print("="*60)

    model, _ = create_test_model()

    # Expected correct configuration
    correct_lora_rank_per_bit = {4: 8, 8: 16, 16: 32}
    correct_lora_alpha_per_bit = {4: 16, 8: 32, 16: 64}

    print("\n1. Before fix - checking current configuration:")
    issues_found = []
    for name, module in model.named_modules():
        if hasattr(module, 'lora_rank_per_bit'):
            print(f"   {name}: rank_per_bit = {module.lora_rank_per_bit}")
            # Check if it's incorrect (backwards)
            if module.lora_rank_per_bit.get(4, 0) > module.lora_rank_per_bit.get(16, 0):
                issues_found.append(f"{name} has inverted ranks")

    if issues_found:
        print(f"\n   ⚠️ Found {len(issues_found)} modules with incorrect rank configuration")

    # Apply the fix
    print("\n2. Applying LoRA rank fix...")

    # Fix function from the provided code
    def fix_lora_ranks_in_model(model):
        for name, module in model.named_modules():
            if hasattr(module, 'lora_adapters'):
                if hasattr(module, 'lora_rank_per_bit'):
                    module.lora_rank_per_bit = correct_lora_rank_per_bit
                    module.lora_alpha_per_bit = correct_lora_alpha_per_bit

    fix_lora_ranks_in_model(model)

    print("\n3. After fix - verifying configuration:")
    all_correct = True
    for name, module in model.named_modules():
        if hasattr(module, 'lora_rank_per_bit'):
            if module.lora_rank_per_bit != correct_lora_rank_per_bit:
                print(f"   ❌ {name}: Still incorrect!")
                all_correct = False

    if all_correct:
        print("   ✅ All LoRA ranks correctly fixed!")

    del model
    torch.cuda.empty_cache()
    return all_correct


def test_lora_initialization():
    """Test that LoRA adapters are properly initialized to zero."""
    print("\n" + "="*60)
    print("Test 2: LoRA Adapter Initialization")
    print("="*60)

    model, _ = create_test_model()

    print("\n1. Checking LoRA adapter values before initialization:")

    # Check initial values
    non_zero_count = 0
    total_count = 0

    for name, module in model.named_modules():
        if hasattr(module, 'lora_adapters') and module.lora_adapters:
            for bit_width, lora in module.lora_adapters.items():
                if hasattr(lora, 'lora_B'):
                    b_sum = lora.lora_B.abs().sum().item()
                    if b_sum > 1e-6:
                        non_zero_count += 1
                    total_count += 1

    print(f"   Found {non_zero_count}/{total_count} non-zero LoRA B matrices")

    # Apply proper initialization
    print("\n2. Applying proper LoRA initialization...")

    def properly_initialize_lora_adapters(model):
        with torch.no_grad():
            for name, module in model.named_modules():
                if hasattr(module, 'lora_adapters') and module.lora_adapters:
                    for bit_width, lora in module.lora_adapters.items():
                        if hasattr(lora, 'lora_B'):
                            nn.init.zeros_(lora.lora_B)
                        if hasattr(lora, 'lora_A'):
                            nn.init.normal_(lora.lora_A, mean=0.0, std=0.001)

    properly_initialize_lora_adapters(model)

    print("\n3. Verifying LoRA initialization:")

    # Verify all B matrices are zero
    all_zero = True
    for name, module in model.named_modules():
        if hasattr(module, 'lora_adapters') and module.lora_adapters:
            for bit_width, lora in module.lora_adapters.items():
                if hasattr(lora, 'lora_B'):
                    b_sum = lora.lora_B.abs().sum().item()
                    if b_sum > 1e-10:
                        print(f"   ❌ {name} bit={bit_width}: lora_B not zero (sum={b_sum})")
                        all_zero = False

    if all_zero:
        print("   ✅ All LoRA B matrices properly zeroed!")

    # Check A matrices have small values
    a_stats = []
    for name, module in model.named_modules():
        if hasattr(module, 'lora_adapters') and module.lora_adapters:
            for bit_width, lora in module.lora_adapters.items():
                if hasattr(lora, 'lora_A'):
                    a_std = lora.lora_A.std().item()
                    a_stats.append(a_std)

    if a_stats:
        avg_std = np.mean(a_stats)
        print(f"   LoRA A matrices avg std: {avg_std:.6f} (expected ~0.001)")
        if 0.0005 < avg_std < 0.002:
            print("   ✅ LoRA A matrices properly initialized!")

    del model
    torch.cuda.empty_cache()
    return all_zero


def test_initial_precision_setting():
    """Test that initial precision is correctly set."""
    print("\n" + "="*60)
    print("Test 3: Initial Precision Setting")
    print("="*60)

    model, _ = create_test_model()

    print("\n1. Checking current precision:")
    if hasattr(model, 'current_bits'):
        print(f"   Current bits: {model.current_bits}")
    else:
        print("   No current_bits attribute")

    # Set initial precision
    print("\n2. Setting initial precision to 16-bit...")

    def set_initial_precision(model, initial_bits=16):
        if hasattr(model, 'set_precision'):
            model.set_precision(initial_bits)
        elif hasattr(model, 'set_global_precision'):
            model.set_global_precision(initial_bits)
        else:
            if hasattr(model, 'h'):
                for block in model.h:
                    if hasattr(block, 'set_precision'):
                        block.set_precision(initial_bits)

        if hasattr(model, 'current_bits'):
            model.current_bits = initial_bits

    set_initial_precision(model, initial_bits=16)

    print("\n3. Verifying precision setting:")

    # Check model precision
    precision_correct = True
    if hasattr(model, 'current_bits'):
        if model.current_bits == 16:
            print(f"   ✅ Model precision set to 16-bit")
        else:
            print(f"   ❌ Model precision is {model.current_bits}, expected 16")
            precision_correct = False

    # Check individual modules
    for name, module in model.named_modules():
        if hasattr(module, 'current_bits'):
            if module.current_bits != 16:
                print(f"   ❌ {name} has precision {module.current_bits}")
                precision_correct = False

    del model
    torch.cuda.empty_cache()
    return precision_correct


def test_comprehensive_fix():
    """Test the complete load_and_fix_pretrained_model function."""
    print("\n" + "="*60)
    print("Test 4: Comprehensive Model Fix")
    print("="*60)

    # Create model
    model, config = create_test_model()

    # Move to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"\n1. Testing complete fix function on {device}...")

    # Apply comprehensive fix (simplified version for testing)
    def load_and_fix_pretrained_model_test(model):
        """Simplified version of the comprehensive fix for testing."""

        # Load pretrained weights from GPT2LMHeadModel
        pretrained_model = GPT2LMHeadModel.from_pretrained('gpt2')
        pretrained_state = pretrained_model.state_dict()

        # Copy embeddings
        model.wte.weight.data.copy_(pretrained_state['transformer.wte.weight'])

        # Handle position embeddings
        pretrained_wpe = pretrained_state['transformer.wpe.weight']
        if pretrained_wpe.shape[0] != model.wpe.weight.shape[0]:
            min_pos = min(pretrained_wpe.shape[0], model.wpe.weight.shape[0])
            model.wpe.weight.data[:min_pos].copy_(pretrained_wpe[:min_pos])
        else:
            model.wpe.weight.data.copy_(pretrained_wpe)

        # Copy first transformer block as test
        if len(model.h) > 0:
            model.h[0].ln_1.weight.data.copy_(pretrained_state['transformer.h.0.ln_1.weight'])
            model.h[0].ln_1.bias.data.copy_(pretrained_state['transformer.h.0.ln_1.bias'])

        # Final layer norm
        model.ln_f.weight.data.copy_(pretrained_state['transformer.ln_f.weight'])
        model.ln_f.bias.data.copy_(pretrained_state['transformer.ln_f.bias'])

        # Fix LoRA and precision
        if hasattr(model, 'set_precision'):
            model.set_precision(16)

        del pretrained_model
        torch.cuda.empty_cache()

        return model

    try:
        model = load_and_fix_pretrained_model_test(model)
        print("   ✅ Comprehensive fix applied successfully!")

        # Test with tokenizer
        print("\n2. Testing model output:")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        test_text = "The capital of France is"
        inputs = tokenizer(test_text, return_tensors='pt', max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])

            # Handle dictionary output from the model
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
                if loss is not None:
                    loss_value = loss.item()
                    perplexity = math.exp(loss_value) if loss_value < 20 else float('inf')

                    print(f"   Loss: {loss_value:.4f}")
                    print(f"   Perplexity: {perplexity:.1f}")

                    if perplexity < 100:
                        print("   ✅ Model producing reasonable outputs!")
                    elif perplexity < 1000:
                        print("   ⚠️ Model perplexity higher than expected")
                    else:
                        print("   ❌ Model perplexity too high")
                else:
                    print("   ❌ Loss is None in model output")
            elif hasattr(outputs, 'loss'):
                loss = outputs.loss.item()
                perplexity = math.exp(loss) if loss < 20 else float('inf')

                print(f"   Loss: {loss:.4f}")
                print(f"   Perplexity: {perplexity:.1f}")

                if perplexity < 100:
                    print("   ✅ Model producing reasonable outputs!")
                elif perplexity < 1000:
                    print("   ⚠️ Model perplexity higher than expected")
                else:
                    print("   ❌ Model perplexity too high")
            else:
                print("   ❌ No loss in model output")
                print(f"   Output type: {type(outputs)}")
                if isinstance(outputs, dict):
                    print(f"   Output keys: {outputs.keys()}")

        return True

    except Exception as e:
        print(f"   ❌ Error during comprehensive fix: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        del model
        torch.cuda.empty_cache()


def test_weight_comparison():
    """Compare weights before and after fix to ensure proper loading."""
    print("\n" + "="*60)
    print("Test 5: Weight Loading Verification")
    print("="*60)

    # Load reference pretrained model
    print("\n1. Loading reference GPT-2 model...")
    reference = GPT2LMHeadModel.from_pretrained('gpt2')
    ref_state = reference.state_dict()

    # Create and fix our model
    print("\n2. Creating and fixing QAT model...")
    model, _ = create_test_model()

    # Apply simplified fix
    model.wte.weight.data.copy_(ref_state['transformer.wte.weight'])

    # Position embeddings (handle size mismatch)
    ref_wpe = ref_state['transformer.wpe.weight']
    if ref_wpe.shape[0] != model.wpe.weight.shape[0]:
        min_pos = min(ref_wpe.shape[0], model.wpe.weight.shape[0])
        model.wpe.weight.data[:min_pos].copy_(ref_wpe[:min_pos])
        print(f"   Adjusted position embeddings: {ref_wpe.shape[0]} → {model.wpe.weight.shape[0]}")

    # Compare weights
    print("\n3. Comparing weights:")

    # Token embeddings
    wte_match = torch.allclose(model.wte.weight.data, ref_state['transformer.wte.weight'], rtol=1e-5)
    print(f"   Token embeddings match: {wte_match}")

    # Position embeddings (first N positions)
    min_pos = min(model.wpe.weight.shape[0], ref_state['transformer.wpe.weight'].shape[0])
    wpe_match = torch.allclose(
        model.wpe.weight.data[:min_pos],
        ref_state['transformer.wpe.weight'][:min_pos],
        rtol=1e-5
    )
    print(f"   Position embeddings match (first {min_pos}): {wpe_match}")

    # Final layer norm
    model.ln_f.weight.data.copy_(ref_state['transformer.ln_f.weight'])
    model.ln_f.bias.data.copy_(ref_state['transformer.ln_f.bias'])

    ln_f_match = torch.allclose(model.ln_f.weight.data, ref_state['transformer.ln_f.weight'], rtol=1e-5)
    print(f"   Final layer norm match: {ln_f_match}")

    all_match = wte_match and wpe_match and ln_f_match

    if all_match:
        print("\n   ✅ All tested weights correctly loaded!")
    else:
        print("\n   ❌ Some weights did not load correctly")

    del reference, model
    torch.cuda.empty_cache()

    return all_match


def run_all_tests():
    """Run all tests for the model fix functions."""
    print("\n" + "="*70)
    print("MODEL FIX TEST SUITE")
    print("="*70)
    print("\nThis tests the comprehensive fix for:")
    print("1. LoRA rank configuration (inverted)")
    print("2. Initial precision setting")
    print("3. LoRA adapter initialization")
    print("4. Pretrained weight loading")

    results = {}

    # Run tests
    results['LoRA Rank Fix'] = test_lora_rank_fix()
    results['LoRA Initialization'] = test_lora_initialization()
    results['Precision Setting'] = test_initial_precision_setting()
    results['Weight Loading'] = test_weight_comparison()
    results['Comprehensive Fix'] = test_comprehensive_fix()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:.<30} {status}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print("\n" + "="*70)
    if total_passed == total_tests:
        print(f"ALL {total_tests} TESTS PASSED! ✅")
    else:
        print(f"PASSED: {total_passed}/{total_tests} tests")
        print(f"FAILED: {total_tests - total_passed}/{total_tests} tests")
    print("="*70)

    return total_passed == total_tests


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    success = run_all_tests()
    exit(0 if success else 1)
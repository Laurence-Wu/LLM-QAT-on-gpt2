"""
Unit tests for BF16/FP16 precision conversion via set_precision() method

Tests the precision conversion functionality added to support:
- set_precision(16.0) → FP16 (torch.float16)
- set_precision(16.5) → BF16 (torch.bfloat16)
- set_precision(32) → FP32 (torch.float32)
- set_precision(4, 8, 16, etc.) → INT quantization

Usage:
    python test_precision_conversion.py
"""

import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
part5_dir = os.path.dirname(current_dir)
sys.path.insert(0, part5_dir)

import torch
from transformers import GPT2Config
from part5_squad.models_squad import SPQuestionAnsweringModel, SPModel


def create_mini_config():
    """Create minimal GPT-2 config for fast testing"""
    config = GPT2Config(
        vocab_size=1000,
        n_positions=128,
        n_embd=128,
        n_layer=2,
        n_head=2,
        layer_norm_epsilon=1e-5,
        embd_pdrop=0.0
    )

    # Switchable precision config
    config.bit_widths = [4, 8, 16, 32]
    config.lora_rank_per_bit = {4: 8, 8: 8, 16: 8, 32: 0}
    config.lora_alpha_per_bit = {4: 8, 8: 8, 16: 8, 32: 0}
    config.quantizer_per_bit = {4: 'minmax', 8: 'log', 16: 'log', 32: None}
    config.activation_bits_per_bit = {4: 4, 8: 8, 16: 16, 32: 32}

    return config


def check_model_dtype(model, expected_dtype, test_name):
    """Check all model parameters and buffers have expected dtype"""
    mismatches = []

    for name, param in model.named_parameters():
        if param.dtype != expected_dtype:
            mismatches.append(f"Parameter {name}: {param.dtype} != {expected_dtype}")

    for name, buffer in model.named_buffers():
        if buffer.dtype != expected_dtype:
            mismatches.append(f"Buffer {name}: {buffer.dtype} != {expected_dtype}")

    if mismatches:
        print(f"  ❌ {test_name} - Dtype mismatches found:")
        for mismatch in mismatches[:5]:  # Show first 5
            print(f"     {mismatch}")
        if len(mismatches) > 5:
            print(f"     ... and {len(mismatches) - 5} more")
        return False

    return True


def test_fp16_conversion():
    """Test set_precision(16.0) converts model to FP16"""
    print("Testing FP16 conversion (16.0 flag)...")

    config = create_mini_config()
    model = SPQuestionAnsweringModel(config)

    # Convert to FP16
    model.set_precision(16.0)

    # Check dtype
    if not check_model_dtype(model, torch.float16, "FP16 conversion"):
        return False

    # Check current_bit_width is set correctly
    if model.transformer.current_bit_width != 16.0:
        print(f"  ❌ current_bit_width = {model.transformer.current_bit_width}, expected 16.0")
        return False

    print("  ✓ FP16 conversion works correctly")
    return True


def test_bf16_conversion():
    """Test set_precision(16.5) converts model to BF16"""
    print("Testing BF16 conversion (16.5 flag)...")

    # Skip if BF16 not supported
    if not torch.cuda.is_bf16_supported() and torch.cuda.is_available():
        print("  ⚠️  BF16 not supported on this device, skipping")
        return True

    config = create_mini_config()
    model = SPQuestionAnsweringModel(config)

    # Convert to BF16
    model.set_precision(16.5)

    # Check dtype
    if not check_model_dtype(model, torch.bfloat16, "BF16 conversion"):
        return False

    # Check current_bit_width is set correctly
    if model.transformer.current_bit_width != 16.5:
        print(f"  ❌ current_bit_width = {model.transformer.current_bit_width}, expected 16.5")
        return False

    print("  ✓ BF16 conversion works correctly")
    return True


def test_fp32_conversion():
    """Test set_precision(32) keeps model at FP32"""
    print("Testing FP32 conversion (32 flag)...")

    config = create_mini_config()
    model = SPQuestionAnsweringModel(config)

    # Set to FP32 (should already be FP32)
    model.set_precision(32)

    # Check dtype
    if not check_model_dtype(model, torch.float32, "FP32 conversion"):
        return False

    print("  ✓ FP32 conversion works correctly")
    return True


def test_int_quantization():
    """Test integer precision values activate INT quantization"""
    print("Testing INT quantization (integer flags 4, 8, 16)...")

    config = create_mini_config()
    model = SPQuestionAnsweringModel(config)

    for bits in [4, 8, 16]:
        model.set_precision(bits)

        # Should remain FP32 but activate quantizers
        if not check_model_dtype(model, torch.float32, f"INT{bits} quantization"):
            return False

        # Check current_bit_width
        if model.transformer.current_bit_width != bits:
            print(f"  ❌ current_bit_width = {model.transformer.current_bit_width}, expected {bits}")
            return False

    print("  ✓ INT quantization activation works correctly")
    return True


def test_dtype_consistency():
    """Test all model components have matching dtype after conversion"""
    print("Testing dtype consistency across model components...")

    config = create_mini_config()
    model = SPQuestionAnsweringModel(config)

    test_cases = [
        (16.0, torch.float16, "FP16"),
        (32, torch.float32, "FP32"),
    ]

    # Add BF16 if supported
    if torch.cuda.is_bf16_supported() or not torch.cuda.is_available():
        test_cases.append((16.5, torch.bfloat16, "BF16"))

    for precision, expected_dtype, name in test_cases:
        model.set_precision(precision)

        # Check transformer components
        for i, block in enumerate(model.transformer.h):
            for param in block.parameters():
                if param.dtype != expected_dtype:
                    print(f"  ❌ {name}: Block {i} parameter has dtype {param.dtype}")
                    return False

        # Check QA heads
        for param in model.qa_start.parameters():
            if param.dtype != expected_dtype:
                print(f"  ❌ {name}: qa_start parameter has dtype {param.dtype}")
                return False

        for param in model.qa_end.parameters():
            if param.dtype != expected_dtype:
                print(f"  ❌ {name}: qa_end parameter has dtype {param.dtype}")
                return False

    print("  ✓ Dtype consistency check passed")
    return True


def test_forward_fp16():
    """Test forward pass works in FP16"""
    print("Testing forward pass in FP16...")

    config = create_mini_config()
    model = SPQuestionAnsweringModel(config)
    model.set_precision(16.0)
    model.eval()

    # Create dummy input
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    try:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        # Check output dtypes
        if outputs['start_logits'].dtype != torch.float16:
            print(f"  ❌ start_logits dtype = {outputs['start_logits'].dtype}, expected float16")
            return False

        if outputs['end_logits'].dtype != torch.float16:
            print(f"  ❌ end_logits dtype = {outputs['end_logits'].dtype}, expected float16")
            return False

        # Check for NaN/Inf
        if torch.isnan(outputs['start_logits']).any() or torch.isinf(outputs['start_logits']).any():
            print("  ❌ start_logits contains NaN/Inf")
            return False

        print("  ✓ Forward pass in FP16 works correctly")
        return True

    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        return False


def test_forward_bf16():
    """Test forward pass works in BF16"""
    print("Testing forward pass in BF16...")

    # Skip if BF16 not supported
    if not torch.cuda.is_bf16_supported() and torch.cuda.is_available():
        print("  ⚠️  BF16 not supported on this device, skipping")
        return True

    config = create_mini_config()
    model = SPQuestionAnsweringModel(config)
    model.set_precision(16.5)
    model.eval()

    # Create dummy input
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    try:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        # Check output dtypes
        if outputs['start_logits'].dtype != torch.bfloat16:
            print(f"  ❌ start_logits dtype = {outputs['start_logits'].dtype}, expected bfloat16")
            return False

        # Check for NaN/Inf
        if torch.isnan(outputs['start_logits']).any() or torch.isinf(outputs['start_logits']).any():
            print("  ❌ start_logits contains NaN/Inf")
            return False

        print("  ✓ Forward pass in BF16 works correctly")
        return True

    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        return False


def test_backward_fp16():
    """Test backward pass works in FP16"""
    print("Testing backward pass in FP16...")

    config = create_mini_config()
    model = SPQuestionAnsweringModel(config)
    model.set_precision(16.0)
    model.train()

    # Create dummy input
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    start_positions = torch.randint(0, seq_len, (batch_size,))
    end_positions = torch.randint(0, seq_len, (batch_size,))

    try:
        outputs = model(input_ids, attention_mask=attention_mask,
                       start_positions=start_positions, end_positions=end_positions)

        loss = outputs['loss']
        loss.backward()

        # Check gradients exist and are FP16
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                if param.grad.dtype != torch.float16:
                    print(f"  ❌ Gradient for {name} has dtype {param.grad.dtype}, expected float16")
                    return False

        if not has_gradients:
            print("  ❌ No gradients computed")
            return False

        print("  ✓ Backward pass in FP16 works correctly")
        return True

    except Exception as e:
        print(f"  ❌ Backward pass failed: {e}")
        return False


def test_backward_bf16():
    """Test backward pass works in BF16"""
    print("Testing backward pass in BF16...")

    # Skip if BF16 not supported
    if not torch.cuda.is_bf16_supported() and torch.cuda.is_available():
        print("  ⚠️  BF16 not supported on this device, skipping")
        return True

    config = create_mini_config()
    model = SPQuestionAnsweringModel(config)
    model.set_precision(16.5)
    model.train()

    # Create dummy input
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    start_positions = torch.randint(0, seq_len, (batch_size,))
    end_positions = torch.randint(0, seq_len, (batch_size,))

    try:
        outputs = model(input_ids, attention_mask=attention_mask,
                       start_positions=start_positions, end_positions=end_positions)

        loss = outputs['loss']
        loss.backward()

        # Check gradients exist and are BF16
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                if param.grad.dtype != torch.bfloat16:
                    print(f"  ❌ Gradient for {name} has dtype {param.grad.dtype}, expected bfloat16")
                    return False

        if not has_gradients:
            print("  ❌ No gradients computed")
            return False

        print("  ✓ Backward pass in BF16 works correctly")
        return True

    except Exception as e:
        print(f"  ❌ Backward pass failed: {e}")
        return False


def test_precision_switching():
    """Test switching between precisions multiple times"""
    print("Testing precision switching...")

    config = create_mini_config()
    model = SPQuestionAnsweringModel(config)

    # Test sequence: FP32 -> FP16 -> FP32 -> BF16 -> FP16 -> FP32
    precision_sequence = [
        (32, torch.float32, "FP32"),
        (16.0, torch.float16, "FP16"),
        (32, torch.float32, "FP32"),
        (16.0, torch.float16, "FP16"),
        (32, torch.float32, "FP32"),
    ]

    # Add BF16 if supported
    if torch.cuda.is_bf16_supported() or not torch.cuda.is_available():
        precision_sequence.insert(3, (16.5, torch.bfloat16, "BF16"))

    for precision, expected_dtype, name in precision_sequence:
        model.set_precision(precision)

        if not check_model_dtype(model, expected_dtype, f"Switch to {name}"):
            return False

    print("  ✓ Precision switching works correctly")
    return True


def test_lora_bypass_fp16_bf16():
    """Test LoRA is bypassed in FP16/BF16 mode"""
    print("Testing LoRA bypass in FP16/BF16...")

    config = create_mini_config()
    model = SPQuestionAnsweringModel(config)

    # Test FP16
    model.set_precision(16.0)

    # Check that linear layers use bypass path
    for module in model.modules():
        if module.__class__.__name__ == 'SPLinearWithLoRA':
            if module.current_bits != 16.0:
                print(f"  ❌ SPLinearWithLoRA.current_bits = {module.current_bits}, expected 16.0")
                return False

    # Test BF16 (if supported)
    if torch.cuda.is_bf16_supported() or not torch.cuda.is_available():
        model.set_precision(16.5)

        for module in model.modules():
            if module.__class__.__name__ == 'SPLinearWithLoRA':
                if module.current_bits != 16.5:
                    print(f"  ❌ SPLinearWithLoRA.current_bits = {module.current_bits}, expected 16.5")
                    return False

    print("  ✓ LoRA bypass works correctly in FP16/BF16")
    return True


def test_quantizer_access_protection():
    """Test that quantizers are not accessed for 16.0/16.5 flags"""
    print("Testing quantizer access protection...")

    config = create_mini_config()
    model = SPQuestionAnsweringModel(config)
    model.eval()

    # Create dummy input
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    test_cases = [16.0, 32]
    if torch.cuda.is_bf16_supported() or not torch.cuda.is_available():
        test_cases.append(16.5)

    for precision in test_cases:
        model.set_precision(precision)

        try:
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)

            # If we get here, quantizers were not accessed (good!)

        except KeyError as e:
            # This would happen if quantizers were incorrectly accessed
            print(f"  ❌ Quantizer access error at precision {precision}: {e}")
            return False
        except Exception as e:
            print(f"  ❌ Unexpected error at precision {precision}: {e}")
            return False

    print("  ✓ Quantizer access protection works correctly")
    return True


def main():
    """Run all tests"""
    print("="*70)
    print("Precision Conversion Unit Tests")
    print("="*70)
    print()

    tests = [
        test_fp16_conversion,
        test_bf16_conversion,
        test_fp32_conversion,
        test_int_quantization,
        test_dtype_consistency,
        test_forward_fp16,
        test_forward_bf16,
        test_backward_fp16,
        test_backward_bf16,
        test_precision_switching,
        test_lora_bypass_fp16_bf16,
        test_quantizer_access_protection,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Test crashed: {e}")
            failed += 1
        print()

    print("="*70)
    print("Test Summary")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"Failed: {failed}")
    print()

    if failed == 0:
        print("✅ All tests passed!")
        return 0
    else:
        print(f"❌ {failed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())

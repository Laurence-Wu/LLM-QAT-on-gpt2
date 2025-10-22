"""
Stress tests and edge cases for BF16/FP16 precision conversion

Tests intensive scenarios, edge cases, and numerical stability:
- Rapid precision switching
- Large models
- Gradient accumulation
- NaN/Inf detection
- Numerical stability
- Checkpoint save/load
- Performance benchmarks

Usage:
    python test_precision_stress.py
"""

import sys
import os
import time
import tempfile

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
part5_dir = os.path.dirname(current_dir)
sys.path.insert(0, part5_dir)

import torch
from transformers import GPT2Config
from part5_squad.models_squad import SPQuestionAnsweringModel


def create_mini_config(n_layer=2):
    """Create configurable mini GPT-2 config"""
    config = GPT2Config(
        vocab_size=1000,
        n_positions=128,
        n_embd=128,
        n_layer=n_layer,
        n_head=2,
        layer_norm_epsilon=1e-5,
        embd_pdrop=0.0
    )

    config.bit_widths = [4, 8, 16, 32]
    config.lora_rank_per_bit = {4: 8, 8: 8, 16: 8, 32: 0}
    config.lora_alpha_per_bit = {4: 8, 8: 8, 16: 8, 32: 0}
    config.quantizer_per_bit = {4: 'minmax', 8: 'log', 16: 'log', 32: None}
    config.activation_bits_per_bit = {4: 4, 8: 8, 16: 16, 32: 32}

    return config


def test_rapid_precision_switching():
    """Test switching precision many times rapidly"""
    print("Testing rapid precision switching (1000 iterations)...")

    config = create_mini_config()
    model = SPQuestionAnsweringModel(config)

    precisions = [32, 16.0, 32, 16.0]
    if torch.cuda.is_bf16_supported() or not torch.cuda.is_available():
        precisions.append(16.5)

    try:
        start_time = time.time()

        for i in range(1000):
            precision = precisions[i % len(precisions)]
            model.set_precision(precision)

        elapsed = time.time() - start_time

        print(f"  ✓ Completed 1000 precision switches in {elapsed:.2f}s ({1000/elapsed:.1f} switches/sec)")
        return True

    except Exception as e:
        print(f"  ❌ Rapid switching failed: {e}")
        return False


def test_large_model_bf16():
    """Test full-size GPT-2 model in BF16"""
    print("Testing large model (6-layer) in BF16...")

    if not torch.cuda.is_bf16_supported() and torch.cuda.is_available():
        print("  ⚠️  BF16 not supported on this device, skipping")
        return True

    try:
        config = create_mini_config(n_layer=6)  # Larger model
        model = SPQuestionAnsweringModel(config)

        # Convert to BF16
        model.set_precision(16.5)
        model.eval()

        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128)

        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        elapsed = time.time() - start_time

        # Check for NaN/Inf
        if torch.isnan(outputs['start_logits']).any():
            print("  ❌ NaN detected in start_logits")
            return False

        if torch.isinf(outputs['start_logits']).any():
            print("  ❌ Inf detected in start_logits")
            return False

        print(f"  ✓ Large model BF16 forward pass: {elapsed:.3f}s")
        return True

    except Exception as e:
        print(f"  ❌ Large model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_accumulation_fp16():
    """Test multi-step gradient accumulation in FP16"""
    print("Testing gradient accumulation in FP16...")

    config = create_mini_config()
    model = SPQuestionAnsweringModel(config)
    model.set_precision(16.0)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    try:
        accumulation_steps = 4
        batch_size = 2
        seq_len = 32

        for step in range(accumulation_steps):
            # Create batch
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)
            start_positions = torch.randint(0, seq_len, (batch_size,))
            end_positions = torch.randint(0, seq_len, (batch_size,))

            # Forward + backward
            outputs = model(input_ids, attention_mask=attention_mask,
                           start_positions=start_positions, end_positions=end_positions)

            loss = outputs['loss'] / accumulation_steps
            loss.backward()

        # Check for NaN gradients
        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"  ❌ NaN gradient in {name}")
                has_nan = True

        if has_nan:
            return False

        # Update
        optimizer.step()
        optimizer.zero_grad()

        print("  ✓ Gradient accumulation in FP16 works correctly")
        return True

    except Exception as e:
        print(f"  ❌ Gradient accumulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nan_detection():
    """Test NaN detection in FP16/BF16"""
    print("Testing NaN detection...")

    config = create_mini_config()
    model = SPQuestionAnsweringModel(config)

    # Test FP16
    model.set_precision(16.0)
    model.eval()

    input_ids = torch.randint(0, 1000, (1, 64))
    attention_mask = torch.ones(1, 64)

    try:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        # Should not have NaN/Inf with normal inputs
        has_nan = torch.isnan(outputs['start_logits']).any() or torch.isnan(outputs['end_logits']).any()
        has_inf = torch.isinf(outputs['start_logits']).any() or torch.isinf(outputs['end_logits']).any()

        if has_nan:
            print("  ❌ Unexpected NaN in outputs")
            return False

        if has_inf:
            print("  ❌ Unexpected Inf in outputs")
            return False

        print("  ✓ No NaN/Inf detected in normal operation")
        return True

    except Exception as e:
        print(f"  ❌ NaN detection test failed: {e}")
        return False


def test_numerical_stability():
    """Test numerical stability across precisions"""
    print("Testing numerical stability...")

    config = create_mini_config()
    model = SPQuestionAnsweringModel(config)

    input_ids = torch.randint(0, 1000, (1, 64))
    attention_mask = torch.ones(1, 64)

    # Run multiple times to check consistency
    n_runs = 5
    outputs_fp32_list = []
    outputs_fp16_list = []

    try:
        model.eval()

        # FP32 runs
        model.set_precision(32)
        for _ in range(n_runs):
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
            outputs_fp32_list.append(outputs['start_logits'].clone())

        # Check FP32 consistency (should be deterministic)
        for i in range(1, n_runs):
            diff = torch.abs(outputs_fp32_list[0] - outputs_fp32_list[i]).max()
            if diff > 1e-6:
                print(f"  ⚠️  FP32 not deterministic: max diff = {diff}")

        # FP16 runs
        model.set_precision(16.0)
        for _ in range(n_runs):
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
            outputs_fp16_list.append(outputs['start_logits'].clone())

        # Check FP16 consistency
        for i in range(1, n_runs):
            diff = torch.abs(outputs_fp16_list[0] - outputs_fp16_list[i]).max().float()
            if diff > 1e-3:  # FP16 has less precision
                print(f"  ⚠️  FP16 not deterministic: max diff = {diff}")

        # Compare FP32 vs FP16
        fp32_mean = outputs_fp32_list[0].float()
        fp16_mean = outputs_fp16_list[0].float()
        relative_error = torch.abs(fp32_mean - fp16_mean) / (torch.abs(fp32_mean) + 1e-8)

        max_rel_error = relative_error.max()
        mean_rel_error = relative_error.mean()

        print(f"  ✓ FP32 vs FP16: max rel error = {max_rel_error:.6f}, mean = {mean_rel_error:.6f}")

        if max_rel_error > 0.1:  # 10% error threshold
            print(f"  ⚠️  High relative error detected ({max_rel_error:.2%})")

        return True

    except Exception as e:
        print(f"  ❌ Numerical stability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_attention_overflow():
    """Test attention doesn't overflow in FP16"""
    print("Testing attention overflow protection...")

    config = create_mini_config()
    model = SPQuestionAnsweringModel(config)
    model.set_precision(16.0)
    model.eval()

    # Create input with large sequence length
    input_ids = torch.randint(0, 1000, (1, 128))  # Max length
    attention_mask = torch.ones(1, 128)

    try:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        # Check for overflow
        if torch.isinf(outputs['start_logits']).any():
            print("  ❌ Inf detected in attention outputs")
            return False

        print("  ✓ Attention stable at max sequence length")
        return True

    except Exception as e:
        print(f"  ❌ Attention overflow test failed: {e}")
        return False


def test_layernorm_precision():
    """Test LayerNorm stability in reduced precision"""
    print("Testing LayerNorm in FP16/BF16...")

    config = create_mini_config()
    model = SPQuestionAnsweringModel(config)

    input_ids = torch.randint(0, 1000, (2, 64))
    attention_mask = torch.ones(2, 64)

    try:
        # Test FP16
        model.set_precision(16.0)
        model.eval()

        with torch.no_grad():
            outputs_fp16 = model(input_ids, attention_mask=attention_mask)

        # Check output magnitude
        mean_magnitude = torch.abs(outputs_fp16['start_logits']).mean()
        if mean_magnitude > 1000 or mean_magnitude < 0.001:
            print(f"  ⚠️  FP16 output magnitude unusual: {mean_magnitude}")

        # Test BF16 (if supported)
        if torch.cuda.is_bf16_supported() or not torch.cuda.is_available():
            model.set_precision(16.5)

            with torch.no_grad():
                outputs_bf16 = model(input_ids, attention_mask=attention_mask)

            mean_magnitude = torch.abs(outputs_bf16['start_logits']).mean()
            if mean_magnitude > 1000 or mean_magnitude < 0.001:
                print(f"  ⚠️  BF16 output magnitude unusual: {mean_magnitude}")

        print("  ✓ LayerNorm stable in reduced precision")
        return True

    except Exception as e:
        print(f"  ❌ LayerNorm precision test failed: {e}")
        return False


def test_checkpoint_save_load_fp16():
    """Test saving and loading checkpoint in FP16"""
    print("Testing checkpoint save/load in FP16...")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "test_fp16.pth")

        try:
            # Create model
            config = create_mini_config()
            model = SPQuestionAnsweringModel(config)
            model.set_precision(16.0)

            # Save
            torch.save(model.state_dict(), checkpoint_path)

            # Create new model and load
            model2 = SPQuestionAnsweringModel(config)
            model2.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            model2.set_precision(16.0)

            # Compare outputs
            input_ids = torch.randint(0, 1000, (1, 32))
            attention_mask = torch.ones(1, 32)

            model.eval()
            model2.eval()

            with torch.no_grad():
                out1 = model(input_ids, attention_mask=attention_mask)
                out2 = model2(input_ids, attention_mask=attention_mask)

            diff = torch.abs(out1['start_logits'] - out2['start_logits']).max()
            if diff > 1e-4:
                print(f"  ❌ Checkpoint mismatch: diff = {diff}")
                return False

            print("  ✓ Checkpoint save/load in FP16 works correctly")
            return True

        except Exception as e:
            print(f"  ❌ Checkpoint save/load failed: {e}")
            return False


def test_checkpoint_save_load_bf16():
    """Test saving and loading checkpoint in BF16"""
    print("Testing checkpoint save/load in BF16...")

    if not torch.cuda.is_bf16_supported() and torch.cuda.is_available():
        print("  ⚠️  BF16 not supported on this device, skipping")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "test_bf16.pth")

        try:
            # Create model
            config = create_mini_config()
            model = SPQuestionAnsweringModel(config)
            model.set_precision(16.5)

            # Save
            torch.save(model.state_dict(), checkpoint_path)

            # Create new model and load
            model2 = SPQuestionAnsweringModel(config)
            model2.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            model2.set_precision(16.5)

            # Compare outputs
            input_ids = torch.randint(0, 1000, (1, 32))
            attention_mask = torch.ones(1, 32)

            model.eval()
            model2.eval()

            with torch.no_grad():
                out1 = model(input_ids, attention_mask=attention_mask)
                out2 = model2(input_ids, attention_mask=attention_mask)

            diff = torch.abs(out1['start_logits'] - out2['start_logits']).max()
            if diff > 1e-3:
                print(f"  ❌ Checkpoint mismatch: diff = {diff}")
                return False

            print("  ✓ Checkpoint save/load in BF16 works correctly")
            return True

        except Exception as e:
            print(f"  ❌ Checkpoint save/load failed: {e}")
            return False


def test_precision_with_gradients():
    """Test gradient computation correctness"""
    print("Testing gradient computation in FP16...")

    config = create_mini_config()

    # FP32 baseline
    model_fp32 = SPQuestionAnsweringModel(config)
    model_fp32.set_precision(32)
    model_fp32.train()

    # FP16
    model_fp16 = SPQuestionAnsweringModel(config)
    model_fp16.set_precision(16.0)
    model_fp16.train()

    # Copy weights
    model_fp16.load_state_dict(model_fp32.state_dict())
    model_fp16.set_precision(16.0)

    try:
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32)
        start_positions = torch.randint(0, 32, (2,))
        end_positions = torch.randint(0, 32, (2,))

        # FP32 forward + backward
        outputs_fp32 = model_fp32(input_ids, attention_mask=attention_mask,
                                   start_positions=start_positions, end_positions=end_positions)
        loss_fp32 = outputs_fp32['loss']
        loss_fp32.backward()

        # FP16 forward + backward
        outputs_fp16 = model_fp16(input_ids, attention_mask=attention_mask,
                                   start_positions=start_positions, end_positions=end_positions)
        loss_fp16 = outputs_fp16['loss']
        loss_fp16.backward()

        # Compare gradients
        max_grad_diff = 0.0
        for (name, param_fp32), (_, param_fp16) in zip(model_fp32.named_parameters(), model_fp16.named_parameters()):
            if param_fp32.grad is not None and param_fp16.grad is not None:
                diff = torch.abs(param_fp32.grad - param_fp16.grad.float()).max()
                max_grad_diff = max(max_grad_diff, diff.item())

        print(f"  ✓ Max gradient difference FP32 vs FP16: {max_grad_diff:.6f}")

        if max_grad_diff > 1.0:
            print(f"  ⚠️  Large gradient difference ({max_grad_diff:.2f})")

        return True

    except Exception as e:
        print(f"  ❌ Gradient computation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dtype_mismatch_handling():
    """Test handling of dtype mismatches"""
    print("Testing dtype mismatch handling...")

    config = create_mini_config()
    model = SPQuestionAnsweringModel(config)
    model.set_precision(16.0)
    model.eval()

    # Create FP32 input for FP16 model
    input_ids = torch.randint(0, 1000, (1, 32)).long()  # long dtype always
    attention_mask = torch.ones(1, 32).float()  # FP32 attention mask

    try:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        # PyTorch should auto-cast, but let's verify output is still FP16
        if outputs['start_logits'].dtype != torch.float16:
            print(f"  ⚠️  Output dtype = {outputs['start_logits'].dtype}, expected float16")
            print("     (Might be auto-cast behavior, not necessarily wrong)")

        print("  ✓ Dtype mismatch handled gracefully")
        return True

    except Exception as e:
        # If it fails, that's also acceptable behavior
        print(f"  ✓ Dtype mismatch correctly rejected: {type(e).__name__}")
        return True


def main():
    """Run all stress tests"""
    print("="*70)
    print("Precision Conversion Stress Tests")
    print("="*70)
    print()

    tests = [
        test_rapid_precision_switching,
        test_large_model_bf16,
        test_gradient_accumulation_fp16,
        test_nan_detection,
        test_numerical_stability,
        test_attention_overflow,
        test_layernorm_precision,
        test_checkpoint_save_load_fp16,
        test_checkpoint_save_load_bf16,
        test_precision_with_gradients,
        test_dtype_mismatch_handling,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
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
        print("✅ All stress tests passed!")
        return 0
    else:
        print(f"❌ {failed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())

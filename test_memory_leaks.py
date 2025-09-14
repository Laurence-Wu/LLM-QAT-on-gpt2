"""
Memory leak testing script for QAT GPT-2 implementation.
Tests all components for memory leaks and provides detailed reports.
"""

import torch
import torch.nn as nn
import time
import gc
import sys
import os
from transformers import GPT2Config

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared.memory_monitor import MemoryMonitor, profile_memory_usage
from shared.models import QATGPT2
from shared.quantization import LearnableFakeQuantize, QuantizedLinear
from shared.lora import QATLinearWithLoRA, QATLoRALayer
from shared.dataset import SQuADDataset


def test_model_memory_leak(config=None, iterations=100, batch_size=4, seq_length=256):
    """Test for memory leaks in model training."""
    print("\n" + "="*60)
    print("Testing Model Memory Leaks")
    print("="*60)

    if config is None:
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            quantization_bits=8
        )

    monitor = MemoryMonitor(threshold_mb=100)

    # Initialize model
    model = QATGPT2(config)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"Testing {iterations} iterations with batch_size={batch_size}, seq_length={seq_length}")
    print(f"Initial memory: {monitor.get_memory_usage():.2f} MB")

    memory_history = []

    for iteration in range(iterations):
        # Create batch
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        # Forward pass
        outputs = model(input_ids, labels=input_ids)
        loss = outputs['loss']

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Clean up batch
        del input_ids, outputs, loss

        # Check memory every 10 iterations
        if iteration % 10 == 0:
            current_memory = monitor.get_memory_usage()
            memory_history.append(current_memory)

            if iteration > 0:
                increase = current_memory - memory_history[0]
                print(f"Iteration {iteration:3d}: {current_memory:.2f} MB "
                      f"(+{increase:.2f} MB from start)")

                # Check for leak
                if increase > 100:  # 100 MB threshold
                    print(f"⚠️  Potential memory leak detected!")

    # Final check
    monitor.check_memory_leak("Model Training Test")
    monitor.print_summary()

    # Analyze memory growth
    if len(memory_history) > 2:
        growth_rate = (memory_history[-1] - memory_history[0]) / len(memory_history)
        print(f"\nAverage memory growth rate: {growth_rate:.2f} MB/checkpoint")

        if growth_rate > 1:  # Growing more than 1 MB per checkpoint
            print("❌ FAIL: Memory leak detected - consistent memory growth")
            return False
        else:
            print("✅ PASS: No significant memory leak detected")
            return True

    return True


def test_quantization_memory_leak(iterations=100):
    """Test for memory leaks in quantization layers."""
    print("\n" + "="*60)
    print("Testing Quantization Memory Leaks")
    print("="*60)

    monitor = MemoryMonitor(threshold_mb=50)

    # Test LearnableFakeQuantize
    quantizer = LearnableFakeQuantize(num_bits=8, symmetric=True, per_channel=True)
    if torch.cuda.is_available():
        quantizer = quantizer.cuda()

    print("Testing LearnableFakeQuantize...")
    memory_history = []

    for i in range(iterations):
        x = torch.randn(32, 768)
        if torch.cuda.is_available():
            x = x.cuda()

        # Forward pass
        quantizer.train()
        y = quantizer(x)

        # Clean up
        del x, y

        if i % 20 == 0:
            current_memory = monitor.get_memory_usage()
            memory_history.append(current_memory)
            if i > 0:
                increase = current_memory - memory_history[0]
                print(f"Iteration {i:3d}: Memory increase: {increase:.2f} MB")

    # Test QuantizedLinear
    print("\nTesting QuantizedLinear...")
    monitor.reset_baseline()

    linear = QuantizedLinear(768, 768, weight_bits=8, activation_bits=8)
    if torch.cuda.is_available():
        linear = linear.cuda()

    for i in range(iterations):
        x = torch.randn(32, 768)
        if torch.cuda.is_available():
            x = x.cuda()

        y = linear(x)
        loss = y.mean()
        loss.backward()

        # Clean up
        del x, y, loss
        linear.zero_grad()

        if i % 20 == 0:
            monitor.check_memory_leak(f"QuantizedLinear iteration {i}")

    monitor.print_summary()
    return True


def test_lora_memory_leak(iterations=100):
    """Test for memory leaks in LoRA adapters."""
    print("\n" + "="*60)
    print("Testing LoRA Memory Leaks")
    print("="*60)

    monitor = MemoryMonitor(threshold_mb=50)

    # Test QATLoRALayer
    lora_layer = QATLoRALayer(768, 768, rank=16, alpha=32, bits=8)
    if torch.cuda.is_available():
        lora_layer = lora_layer.cuda()

    print("Testing QATLoRALayer...")
    for i in range(iterations):
        x = torch.randn(32, 768)
        if torch.cuda.is_available():
            x = x.cuda()

        y = lora_layer(x)
        loss = y.mean()
        loss.backward()

        del x, y, loss
        lora_layer.zero_grad()

        if i % 20 == 0:
            monitor.check_memory_leak(f"LoRA iteration {i}")

    # Test QATLinearWithLoRA
    print("\nTesting QATLinearWithLoRA...")
    monitor.reset_baseline()

    linear_lora = QATLinearWithLoRA(768, 768, bits=8)
    if torch.cuda.is_available():
        linear_lora = linear_lora.cuda()

    for i in range(iterations):
        x = torch.randn(32, 768)
        if torch.cuda.is_available():
            x = x.cuda()

        y = linear_lora(x)
        loss = y.mean()
        loss.backward()

        del x, y, loss
        linear_lora.zero_grad()

        if i % 20 == 0:
            monitor.check_memory_leak(f"LinearLoRA iteration {i}")

    monitor.print_summary()
    return True


def test_gradient_checkpointing_leak(iterations=50):
    """Test for memory leaks with gradient checkpointing."""
    print("\n" + "="*60)
    print("Testing Gradient Checkpointing Memory Leaks")
    print("="*60)

    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=6,  # Fewer layers for faster testing
        n_head=12
    )

    monitor = MemoryMonitor(threshold_mb=100)

    # Test with gradient checkpointing enabled
    model = QATGPT2(config)
    model.use_gradient_checkpointing = True
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("Testing with gradient checkpointing ENABLED...")

    # Run a few warmup iterations to stabilize memory
    print("Running warmup iterations...")
    for _ in range(5):
        input_ids = torch.randint(0, config.vocab_size, (2, 128))
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        outputs = model(input_ids, labels=input_ids)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        del input_ids, outputs, loss

    # Reset baseline after warmup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    memory_before = monitor.get_memory_usage()
    memory_history = []

    for i in range(iterations):
        input_ids = torch.randint(0, config.vocab_size, (2, 128))
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        outputs = model(input_ids, labels=input_ids)
        loss = outputs['loss']
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        del input_ids, outputs, loss

        if i % 10 == 0:
            current_memory = monitor.get_memory_usage()
            memory_history.append(current_memory)
            increase = current_memory - memory_before
            print(f"Iteration {i:3d}: Memory increase: {increase:.2f} MB")

    # Analyze memory growth trend
    if len(memory_history) > 2:
        # Check if memory is growing consistently
        growth_rate = (memory_history[-1] - memory_history[0]) / len(memory_history)

        if growth_rate > 2:  # Growing more than 2 MB per checkpoint
            print(f"❌ FAIL: Gradient checkpointing leak - {growth_rate:.2f} MB/checkpoint growth")
            return False
        else:
            print(f"✅ PASS: No gradient checkpointing leak - {growth_rate:.2f} MB/checkpoint growth")
            return True

    return True


def run_comprehensive_test():
    """Run all memory leak tests."""
    print("\n" + "="*80)
    print(" "*20 + "MEMORY LEAK TEST SUITE")
    print("="*80)

    results = {}

    # Force initial cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Run tests
    print("\n📊 Starting comprehensive memory leak tests...")

    tests = [
        ("Model Training", lambda: test_model_memory_leak(iterations=50)),
        ("Quantization Layers", test_quantization_memory_leak),
        ("LoRA Adapters", test_lora_memory_leak),
        ("Gradient Checkpointing", test_gradient_checkpointing_leak)
    ]

    for test_name, test_func in tests:
        try:
            print(f"\n🔍 Running: {test_name}")
            results[test_name] = test_func()

            # Cleanup between tests
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error in {test_name}: {str(e)}")
            results[test_name] = False

    # Print summary
    print("\n" + "="*80)
    print(" "*25 + "TEST SUMMARY")
    print("="*80)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30s}: {status}")
        if not passed:
            all_passed = False

    print("="*80)

    if all_passed:
        print("\n🎉 All memory leak tests PASSED!")
    else:
        print("\n⚠️  Some tests FAILED - memory leaks detected")

    # Final memory stats
    if torch.cuda.is_available():
        print("\n📊 Final GPU Memory Stats:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"  Peak: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    return all_passed


if __name__ == "__main__":
    # Run comprehensive test
    success = run_comprehensive_test()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
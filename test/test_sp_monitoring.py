#!/usr/bin/env python3
"""
Test script for SP training monitoring
Verifies that all monitoring components work correctly
"""

import sys
import os
import torch
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2Config, GPT2TokenizerFast
from shared.models_sp import SPLMHeadModel
from part1_switchable_precision.config_sp import ModelConfig, TrainingConfig
from monitor_sp_complete_training import ComprehensiveTrainingMonitor


def test_monitor_initialization():
    """Test monitor initialization."""
    print("\n1. Testing monitor initialization...")

    monitor = ComprehensiveTrainingMonitor("test_monitor_output")

    assert "system_info" in monitor.log
    assert "model_architecture" in monitor.log
    assert "training_progress" in monitor.log

    print("   ✓ Monitor initialized correctly")
    return True


def test_model_architecture_monitoring():
    """Test model architecture monitoring."""
    print("\n2. Testing model architecture monitoring...")

    monitor = ComprehensiveTrainingMonitor("test_monitor_output")

    # Create small test model
    model_config = ModelConfig()
    model_config.n_layer = 1
    model_config.n_embd = 128
    model_config.n_head = 2
    model_config.vocab_size = 1000

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

    # Monitor architecture
    monitor.monitor_model_architecture(model, model_config)

    assert monitor.log["model_architecture"]["total_parameters"] > 0
    assert "lora_parameters_by_bit" in monitor.log["model_architecture"]

    print("   ✓ Model architecture monitored correctly")
    print(f"     Total params: {monitor.log['model_architecture']['total_parameters']:,}")

    return True


def test_training_step_monitoring():
    """Test training step monitoring."""
    print("\n3. Testing training step monitoring...")

    monitor = ComprehensiveTrainingMonitor("test_monitor_output")

    # Create small test model
    model_config = ModelConfig()
    model_config.n_layer = 1
    model_config.n_embd = 128
    model_config.n_head = 2
    model_config.vocab_size = 1000

    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=128,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head
    )
    gpt2_config.bit_widths = model_config.bit_widths
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit

    model = SPLMHeadModel(gpt2_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create dummy input
    input_ids = torch.randint(0, 1000, (2, 32)).to(device)
    labels = input_ids.clone()

    # Forward pass
    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']

    # Backward pass
    loss.backward()

    # Monitor the step
    step_info = monitor.monitor_training_step(
        step=0,
        model=model,
        loss=loss.item(),
        current_bits=8,
        optimizer=optimizer,
        input_ids=input_ids
    )

    assert step_info["step"] == 0
    assert "loss" in step_info
    assert "learning_rate" in step_info

    print("   ✓ Training step monitored correctly")
    print(f"     Loss: {step_info['loss']:.4f}")

    # Test gradient monitoring (step 10 triggers gradient analysis)
    for i in range(11):
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        loss.backward()

        step_info = monitor.monitor_training_step(
            step=i,
            model=model,
            loss=loss.item(),
            current_bits=8,
            optimizer=optimizer,
            input_ids=input_ids
        )

        if i == 10:
            assert "gradient_norm" in step_info
            assert "gradient_stats" in step_info
            print(f"     Gradient norm at step 10: {step_info['gradient_norm']:.4f}")

    return True


def test_precision_switching_monitoring():
    """Test precision switching monitoring."""
    print("\n4. Testing precision switching monitoring...")

    monitor = ComprehensiveTrainingMonitor("test_monitor_output")

    # Monitor precision switches
    monitor.monitor_precision_switch(step=10, old_bits=8, new_bits=4)
    monitor.monitor_precision_switch(step=20, old_bits=4, new_bits=16)
    monitor.monitor_precision_switch(step=30, old_bits=16, new_bits=8)

    assert len(monitor.log["precision_switches"]) == 3
    assert monitor.log["precision_switches"][0]["old_bits"] == 8
    assert monitor.log["precision_switches"][0]["new_bits"] == 4

    print("   ✓ Precision switching monitored correctly")
    print(f"     Total switches: {len(monitor.log['precision_switches'])}")

    return True


def test_issue_detection():
    """Test issue detection."""
    print("\n5. Testing issue detection...")

    monitor = ComprehensiveTrainingMonitor("test_monitor_output")

    # Test NaN/Inf detection
    monitor.monitor_training_step(
        step=0,
        model=None,  # Won't be used for this test
        loss=float('nan'),
        current_bits=8,
        optimizer=type('obj', (object,), {'param_groups': [{'lr': 1e-4}]})(),
        input_ids=torch.randn(2, 32)
    )

    assert len(monitor.log["issues_detected"]) > 0
    assert "Invalid loss" in monitor.log["issues_detected"][0]["issue"]

    print("   ✓ Issue detection works correctly")
    print(f"     Issues detected: {len(monitor.log['issues_detected'])}")

    return True


def test_log_saving():
    """Test log saving functionality."""
    print("\n6. Testing log saving...")

    monitor = ComprehensiveTrainingMonitor("test_monitor_output")

    # Add some dummy data
    monitor.log["training_progress"] = [
        {"step": 0, "loss": 10.0},
        {"step": 1, "loss": 9.5},
        {"step": 2, "loss": 9.0}
    ]

    # Finalize and save
    monitor.finalize()

    # Check if files were created
    log_file = os.path.join(monitor.output_dir, "training_monitor.json")
    summary_file = os.path.join(monitor.output_dir, "training_summary.txt")

    assert os.path.exists(log_file), "Log file not created"
    assert os.path.exists(summary_file), "Summary file not created"

    # Verify JSON is valid
    with open(log_file, 'r') as f:
        loaded_log = json.load(f)
        assert "performance_metrics" in loaded_log

    print("   ✓ Log saving works correctly")
    print(f"     Files saved to: {monitor.output_dir}")

    # Clean up
    import shutil
    if os.path.exists(monitor.output_dir):
        shutil.rmtree(monitor.output_dir)

    return True


def run_all_tests():
    """Run all monitoring tests."""
    print("\n" + "="*60)
    print("SP TRAINING MONITORING TEST SUITE")
    print("="*60)

    tests = [
        ("Monitor Initialization", test_monitor_initialization),
        ("Model Architecture Monitoring", test_model_architecture_monitoring),
        ("Training Step Monitoring", test_training_step_monitoring),
        ("Precision Switching Monitoring", test_precision_switching_monitoring),
        ("Issue Detection", test_issue_detection),
        ("Log Saving", test_log_saving)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n   ✗ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("\n✅ All monitoring tests passed successfully!")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please review the errors above.")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
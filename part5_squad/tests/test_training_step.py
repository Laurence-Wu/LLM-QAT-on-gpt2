"""
Test training step and integration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GPT2Config, GPT2TokenizerFast
from models_squad import SPQuestionAnsweringModel
from distillation_manager_qa import DistillationManagerQA
from train_squad import compute_loss_single_precision_qa


class MockConfig:
    """Mock training config"""
    distill_temperature = 3.0
    distill_alpha_kl = 1.0
    distill_alpha_feature = 1e-7
    cache_size = 32
    gradient_accumulation_steps = 4


def calibrate_model_simple(model, bits):
    """Simple calibration for testing - calibrates weight and input quantizers"""
    if bits >= 32:
        return  # No calibration needed for full precision

    bits_key = f'{bits}bit'
    model.set_precision(bits)

    # Calibrate weight quantizers
    for module in model.modules():
        if hasattr(module, 'quantizers_weight') and bits_key in module.quantizers_weight:
            quantizer = module.quantizers_weight[bits_key]
            quantizer.start_calibration()
            with torch.no_grad():
                # Get weight from module
                if hasattr(module, 'linear') and hasattr(module.linear, 'weight'):
                    _ = quantizer(module.linear.weight)
            quantizer.finish_calibration()

    # Calibrate LoRA quantizers
    for module in model.modules():
        if hasattr(module, 'lora_adapters') and bits_key in module.lora_adapters:
            lora_layer = module.lora_adapters[bits_key]
            if hasattr(lora_layer, 'quantize_A') and lora_layer.quantize_A is not None:
                lora_layer.quantize_A.start_calibration()
                with torch.no_grad():
                    _ = lora_layer.quantize_A(lora_layer.lora_A)
                lora_layer.quantize_A.finish_calibration()
            if hasattr(lora_layer, 'quantize_B') and lora_layer.quantize_B is not None:
                lora_layer.quantize_B.start_calibration()
                with torch.no_grad():
                    _ = lora_layer.quantize_B(lora_layer.lora_B)
                lora_layer.quantize_B.finish_calibration()

    # Calibrate input quantizers with dummy forward pass
    for module in model.modules():
        if hasattr(module, 'quantizers_input') and bits_key in module.quantizers_input:
            module.quantizers_input[bits_key].start_calibration()

    # Run dummy forward pass to collect input statistics
    with torch.no_grad():
        dummy_input = torch.randint(0, 50257, (1, 128))
        _ = model(dummy_input)

    # Finish input calibration
    for module in model.modules():
        if hasattr(module, 'quantizers_input') and bits_key in module.quantizers_input:
            module.quantizers_input[bits_key].finish_calibration()


def test_single_training_step():
    """Test one training step end-to-end"""
    print("Testing single training step...")

    # Create small model for testing
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=2,  # Small for testing
        n_head=12
    )
    config.bit_widths = [7, 32]
    config.lora_rank_per_bit = {7: 64, 32: 0}
    config.lora_alpha_per_bit = {7: 64, 32: 0}
    config.quantizer_per_bit = {7: 'log', 32: None}
    config.activation_bits_per_bit = {7: 7, 32: 32}

    model = SPQuestionAnsweringModel(config)
    model.train()

    # Create optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )

    # Create distillation manager
    distill_mgr = DistillationManagerQA(model, 32, MockConfig())

    # Create dummy batch
    batch = {
        'input_ids': torch.randint(0, 50257, (2, 128)),
        'attention_mask': torch.ones(2, 128),
        'start_positions': torch.randint(0, 128, (2,)),
        'end_positions': torch.randint(0, 128, (2,))
    }

    # Forward pass (teacher)
    loss_teacher = compute_loss_single_precision_qa(
        model, batch, precision=32, teacher_bits=32,
        distill_mgr=distill_mgr, config=MockConfig(), iteration=0
    )

    # Backward
    loss_teacher.backward()

    # Check gradients exist
    assert model.qa_start.weight.grad is not None, "Should have gradients"

    # Step optimizer
    optimizer.step()
    optimizer.zero_grad()

    print("✓ Single training step works")


def test_teacher_student_cycle():
    """Test teacher-student training cycle"""
    print("Testing teacher-student cycle...")

    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=2,
        n_head=12
    )
    config.bit_widths = [7, 32]
    config.lora_rank_per_bit = {7: 64, 32: 0}
    config.lora_alpha_per_bit = {7: 64, 32: 0}
    config.quantizer_per_bit = {7: 'log', 32: None}
    config.activation_bits_per_bit = {7: 7, 32: 32}

    model = SPQuestionAnsweringModel(config)
    model.train()

    # Calibrate 7-bit quantizers before using them
    calibrate_model_simple(model, 7)

    distill_mgr = DistillationManagerQA(model, 32, MockConfig())

    batch = {
        'input_ids': torch.randint(0, 50257, (2, 128)),
        'attention_mask': torch.ones(2, 128),
        'start_positions': torch.randint(0, 128, (2,)),
        'end_positions': torch.randint(0, 128, (2,))
    }

    # Teacher forward (32-bit)
    loss_teacher = compute_loss_single_precision_qa(
        model, batch, precision=32, teacher_bits=32,
        distill_mgr=distill_mgr, config=MockConfig(), iteration=0
    )

    assert loss_teacher.item() > 0, "Teacher loss should be positive"

    # Student forward (7-bit)
    loss_student = compute_loss_single_precision_qa(
        model, batch, precision=7, teacher_bits=32,
        distill_mgr=distill_mgr, config=MockConfig(), iteration=1
    )

    assert loss_student.item() > 0, "Student loss should be positive"

    # Student loss should include distillation (typically higher)
    # Note: This is not always true, but we can check it exists
    assert not torch.isnan(loss_student), "Student loss should be valid"

    print("✓ Teacher-student cycle works")


def test_loss_decreases():
    """Test that loss can decrease over a few steps"""
    print("Testing loss decrease...")

    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=2,
        n_head=12
    )
    config.bit_widths = [7, 32]
    config.lora_rank_per_bit = {7: 64, 32: 0}
    config.lora_alpha_per_bit = {7: 64, 32: 0}
    config.quantizer_per_bit = {7: 'log', 32: None}
    config.activation_bits_per_bit = {7: 7, 32: 32}

    model = SPQuestionAnsweringModel(config)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3  # Higher LR for faster convergence in test
    )

    distill_mgr = DistillationManagerQA(model, 32, MockConfig())

    # Use same batch for multiple steps
    batch = {
        'input_ids': torch.randint(0, 50257, (2, 128)),
        'attention_mask': torch.ones(2, 128),
        'start_positions': torch.tensor([10, 20]),  # Fixed positions
        'end_positions': torch.tensor([15, 25])
    }

    losses = []

    for i in range(10):
        # Teacher step
        loss = compute_loss_single_precision_qa(
            model, batch, precision=32, teacher_bits=32,
            distill_mgr=distill_mgr, config=MockConfig(), iteration=i
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

    # Loss should generally decrease (with some noise)
    # Check that final loss is lower than initial
    assert losses[-1] < losses[0], \
        f"Loss should decrease (initial: {losses[0]:.4f}, final: {losses[-1]:.4f})"

    print("✓ Loss decreases over training steps")


if __name__ == '__main__':
    test_single_training_step()
    test_teacher_student_cycle()
    test_loss_decreases()
    print("\n✅ All training step tests passed!")

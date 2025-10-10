"""
Test SPQuestionAnsweringModel architecture
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GPT2Config
from models_squad import SPQuestionAnsweringModel


def test_model_initialization():
    """Test SPQuestionAnsweringModel initializes correctly with separate QA heads"""
    print("Testing model initialization...")

    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        layer_norm_epsilon=1e-5
    )

    config.bit_widths = [7, 32]
    config.lora_rank_per_bit = {7: 64, 32: 0}
    config.lora_alpha_per_bit = {7: 64, 32: 0}
    config.quantizer_per_bit = {7: 'log', 32: None}
    config.activation_bits_per_bit = {7: 7, 32: 32}

    model = SPQuestionAnsweringModel(config)

    # Check separate QA heads exist (Option A)
    assert hasattr(model, 'qa_start'), "Model should have qa_start head"
    assert hasattr(model, 'qa_end'), "Model should have qa_end head"
    assert hasattr(model, 'qa_dropout'), "Model should have qa_dropout"

    # Check head output dimensions
    assert model.qa_start.out_features == 1, "qa_start should output 1 value per position"
    assert model.qa_end.out_features == 1, "qa_end should output 1 value per position"

    # Check transformer exists
    assert hasattr(model, 'transformer'), "Model should have transformer"

    print("✓ Model initialization works")


def test_forward_pass():
    """Test forward pass produces correct output shapes"""
    print("Testing forward pass...")

    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=2,  # Smaller for testing
        n_head=12
    )
    config.bit_widths = [7, 32]
    config.lora_rank_per_bit = {7: 64, 32: 0}
    config.lora_alpha_per_bit = {7: 64, 32: 0}
    config.quantizer_per_bit = {7: 'log', 32: None}
    config.activation_bits_per_bit = {7: 7, 32: 32}

    model = SPQuestionAnsweringModel(config)
    model.eval()

    # Set to 32-bit precision (no quantization/calibration needed)
    model.set_precision(32)

    # Create dummy input
    batch_size = 2
    seq_length = 384
    input_ids = torch.randint(0, 50257, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Check outputs
    assert 'start_logits' in outputs, "Output should have start_logits"
    assert 'end_logits' in outputs, "Output should have end_logits"

    # Check shapes
    assert outputs['start_logits'].shape == (batch_size, seq_length), \
        f"start_logits shape should be ({batch_size}, {seq_length})"
    assert outputs['end_logits'].shape == (batch_size, seq_length), \
        f"end_logits shape should be ({batch_size}, {seq_length})"

    print("✓ Forward pass works")


def test_precision_switching():
    """Test model switches precision correctly"""
    print("Testing precision switching...")

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

    # Test setting to 7-bit
    model.set_precision(7)
    assert model.get_current_precision() == 7, "Model should be at 7-bit precision"

    # Test setting to 32-bit
    model.set_precision(32)
    assert model.get_current_precision() == 32, "Model should be at 32-bit precision"

    print("✓ Precision switching works")


def test_loss_computation():
    """Test QA loss computation"""
    print("Testing loss computation...")

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
    model.eval()

    # Set to 32-bit precision (no quantization/calibration needed)
    model.set_precision(32)

    # Create dummy input
    batch_size = 2
    seq_length = 384
    input_ids = torch.randint(0, 50257, (batch_size, seq_length))
    start_positions = torch.randint(0, seq_length, (batch_size,))
    end_positions = torch.randint(0, seq_length, (batch_size,))

    # Forward pass with labels
    with torch.no_grad():
        outputs = model(
            input_ids,
            start_positions=start_positions,
            end_positions=end_positions
        )

    # Check loss
    assert 'loss' in outputs, "Output should have loss"
    assert outputs['loss'] is not None, "Loss should not be None"
    assert outputs['loss'].item() > 0, "Loss should be positive"
    assert not torch.isnan(outputs['loss']), "Loss should not be NaN"

    print("✓ Loss computation works")


if __name__ == '__main__':
    test_model_initialization()
    test_forward_pass()
    test_precision_switching()
    test_loss_computation()
    print("\n✅ All model tests passed!")

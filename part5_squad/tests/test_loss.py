"""
Test QA loss function and gradient flow
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import GPT2Config
from models_squad import SPQuestionAnsweringModel
from test_utils import freeze_weights_like_production


def test_qa_loss_computation():
    """Test QA loss computes correctly"""
    print("Testing QA loss computation...")

    batch_size = 4
    seq_length = 384

    # Create dummy logits
    start_logits = torch.randn(batch_size, seq_length)
    end_logits = torch.randn(batch_size, seq_length)

    # Create dummy positions
    start_positions = torch.randint(0, seq_length, (batch_size,))
    end_positions = torch.randint(0, seq_length, (batch_size,))

    # Compute loss
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2.0

    # Checks
    assert total_loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(total_loss), "Loss should not be NaN"
    assert not torch.isinf(total_loss), "Loss should not be inf"

    print("✓ QA loss computation works")


def test_gradient_flow():
    """Test gradients flow through model"""
    print("Testing gradient flow...")

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

    # Set to 32-bit precision (no quantization/calibration needed)
    model.set_precision(32)

    # Apply production-like freezing (matching main_squad.py load_pretrained_weights)
    freeze_weights_like_production(model)

    # Create dummy input
    batch_size = 2
    seq_length = 128  # Smaller for testing
    input_ids = torch.randint(0, 50257, (batch_size, seq_length))
    start_positions = torch.randint(0, seq_length, (batch_size,))
    end_positions = torch.randint(0, seq_length, (batch_size,))

    # Forward pass
    outputs = model(
        input_ids,
        start_positions=start_positions,
        end_positions=end_positions
    )
    loss = outputs['loss']

    # Backward pass
    loss.backward()

    # Check gradients exist for QA heads
    assert model.qa_start.weight.grad is not None, "qa_start should have gradients"
    assert model.qa_end.weight.grad is not None, "qa_end should have gradients"

    # Check embeddings are frozen (no gradients)
    assert model.transformer.wte.weight.grad is None, "Embeddings should be frozen"

    print("✓ Gradient flow works")


def test_loss_with_ignore_index():
    """Test loss handles ignore_index correctly"""
    print("Testing loss with ignore_index...")

    batch_size = 4
    seq_length = 384

    # Create dummy logits
    start_logits = torch.randn(batch_size, seq_length)
    end_logits = torch.randn(batch_size, seq_length)

    # Create positions with some -1 (to be ignored)
    start_positions = torch.randint(0, seq_length, (batch_size,))
    end_positions = torch.randint(0, seq_length, (batch_size,))
    start_positions[0] = -1  # Unanswerable
    end_positions[0] = -1

    # Compute loss with ignore_index
    loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2.0

    # Checks
    assert not torch.isnan(total_loss), "Loss should handle ignore_index"
    assert total_loss.item() > 0, "Loss should be positive"

    print("✓ Loss with ignore_index works")


if __name__ == '__main__':
    test_qa_loss_computation()
    test_gradient_flow()
    test_loss_with_ignore_index()
    print("\n✅ All loss tests passed!")

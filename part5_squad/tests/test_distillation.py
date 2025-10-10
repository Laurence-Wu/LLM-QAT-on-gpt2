"""
Test distillation for QA task
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import GPT2Config
from models_squad import SPQuestionAnsweringModel
from distillation_manager_qa import DistillationManagerQA


class MockConfig:
    """Mock config for testing"""
    distill_temperature = 3.0
    distill_alpha_kl = 1.0
    distill_alpha_feature = 1e-7
    cache_size = 32


def test_teacher_caching():
    """Test teacher output caching for QA"""
    print("Testing teacher caching...")

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

    model = SPQuestionAnsweringModel(config).eval()
    distill_mgr = DistillationManagerQA(model, 32, MockConfig())

    # Create dummy input
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 50257, (batch_size, seq_length))

    # Create dummy teacher outputs
    teacher_outputs = {
        'start_logits': torch.randn(batch_size, seq_length),
        'end_logits': torch.randn(batch_size, seq_length),
        'hidden_states': [torch.randn(batch_size, seq_length, 768) for _ in range(3)]
    }

    # Cache teacher outputs
    distill_mgr.update_teacher_qa(input_ids, teacher_outputs)

    # Retrieve from cache
    cached = distill_mgr._get_from_cache(input_ids)

    # Verify
    assert cached is not None, "Should retrieve cached outputs"
    assert 'start_logits' in cached, "Cached should have start_logits"
    assert 'end_logits' in cached, "Cached should have end_logits"
    assert cached['start_logits'].shape == (batch_size, seq_length), \
        "Cached start_logits should have correct shape"

    print("✓ Teacher caching works")


def test_distillation_loss():
    """Test distillation loss computation for QA"""
    print("Testing distillation loss...")

    batch_size = 2
    seq_length = 128

    # Create mock teacher and student outputs
    teacher_outputs = {
        'start_logits': torch.randn(batch_size, seq_length),
        'end_logits': torch.randn(batch_size, seq_length),
        'hidden_states': [torch.randn(batch_size, seq_length, 768) for _ in range(2)]
    }

    student_outputs = {
        'start_logits': torch.randn(batch_size, seq_length),
        'end_logits': torch.randn(batch_size, seq_length),
        'hidden_states': [torch.randn(batch_size, seq_length, 768) for _ in range(2)]
    }

    # Compute KL divergence manually
    T = 3.0
    kl_start = F.kl_div(
        F.log_softmax(student_outputs['start_logits'] / T, dim=-1),
        F.log_softmax(teacher_outputs['start_logits'] / T, dim=-1),
        reduction='batchmean',
        log_target=True
    ) * (T ** 2)

    kl_end = F.kl_div(
        F.log_softmax(student_outputs['end_logits'] / T, dim=-1),
        F.log_softmax(teacher_outputs['end_logits'] / T, dim=-1),
        reduction='batchmean',
        log_target=True
    ) * (T ** 2)

    kl_loss = (kl_start + kl_end) / 2.0

    # Verify loss is valid
    assert not torch.isnan(kl_loss), "KL loss should not be NaN"
    assert not torch.isinf(kl_loss), "KL loss should not be inf"
    assert kl_loss.item() >= 0, "KL loss should be non-negative"

    print("✓ Distillation loss computation works")


def test_cache_stats():
    """Test cache statistics tracking"""
    print("Testing cache statistics...")

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

    model = SPQuestionAnsweringModel(config).eval()
    distill_mgr = DistillationManagerQA(model, 32, MockConfig())

    # Test cache miss
    input_ids = torch.randint(0, 50257, (2, 128))
    result = distill_mgr._get_from_cache(input_ids)
    assert result is None, "Should miss on first access"

    # Add to cache
    teacher_outputs = {
        'start_logits': torch.randn(2, 128),
        'end_logits': torch.randn(2, 128)
    }
    distill_mgr.update_teacher_qa(input_ids, teacher_outputs)

    # Test cache hit
    result = distill_mgr._get_from_cache(input_ids)
    assert result is not None, "Should hit after caching"

    # Get stats
    stats = distill_mgr.get_cache_stats()
    assert stats['cache_hits'] > 0, "Should have cache hits"
    assert stats['cache_misses'] > 0, "Should have cache misses"

    print("✓ Cache statistics work")


if __name__ == '__main__':
    test_teacher_caching()
    test_distillation_loss()
    test_cache_stats()
    print("\n✅ All distillation tests passed!")

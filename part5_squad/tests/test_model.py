"""
Test SPQuestionAnsweringModel architecture
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GPT2Config
from models_squad import SPQuestionAnsweringModel
from test_utils import freeze_weights_like_production, get_trainable_param_count


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


def test_weight_freezing():
    """Test that only LoRA and QA heads are trainable (matching production setup)"""
    print("Testing weight freezing...")

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

    # Apply production freezing (matches main_squad.py load_pretrained_weights)
    lora_count = freeze_weights_like_production(model)

    # Check embeddings are frozen
    assert not model.transformer.wte.weight.requires_grad, "wte should be frozen"
    assert not model.transformer.wpe.weight.requires_grad, "wpe should be frozen"

    # Check QA heads are trainable
    assert model.qa_start.weight.requires_grad, "qa_start should be trainable"
    assert model.qa_end.weight.requires_grad, "qa_end should be trainable"
    assert model.qa_start.bias.requires_grad, "qa_start bias should be trainable"
    assert model.qa_end.bias.requires_grad, "qa_end bias should be trainable"

    # Check base transformer weights are frozen
    assert not model.transformer.h[0].attn.c_attn.linear.weight.requires_grad, \
        "Base attention weights should be frozen"
    assert not model.transformer.h[0].mlp.c_fc.linear.weight.requires_grad, \
        "Base MLP weights should be frozen"

    # Check LoRA adapters are trainable
    assert lora_count > 0, f"Should have LoRA adapters (found {lora_count})"

    # Get parameter counts
    trainable, frozen, total = get_trainable_param_count(model)

    print(f"✓ Weight freezing correct:")
    print(f"  - Total params: {total:,}")
    print(f"  - Trainable (LoRA + QA heads): {trainable:,} ({100*trainable/total:.1f}%)")
    print(f"  - Frozen (base model): {frozen:,} ({100*frozen/total:.1f}%)")
    print(f"  - LoRA adapter pairs: {lora_count}")


def test_lora_calibration_mode():
    """Test LoRA can be disabled/enabled for calibration (matching train_squad.py)"""
    print("Testing LoRA calibration mode...")

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

    # Disable LoRA for calibration (train_squad.py line 94)
    model.disable_lora_for_calibration()

    # Check calibration_mode is set on LoRA modules
    calibration_modules = 0
    for module in model.modules():
        if module.__class__.__name__ == 'SPLinearWithLoRA':
            assert hasattr(module, 'calibration_mode'), \
                "SPLinearWithLoRA should have calibration_mode attribute"
            assert module.calibration_mode == True, \
                "calibration_mode should be True after disable_lora_for_calibration"
            calibration_modules += 1

    assert calibration_modules > 0, "Should have LoRA modules"

    # Re-enable LoRA (train_squad.py line 108)
    model.enable_lora_after_calibration()

    for module in model.modules():
        if module.__class__.__name__ == 'SPLinearWithLoRA':
            assert module.calibration_mode == False, \
                "calibration_mode should be False after enable_lora_after_calibration"

    print(f"✓ LoRA calibration mode works ({calibration_modules} modules)")


def test_unfreeze_weights():
    """Test unfreeze_weights makes base model trainable (for full fine-tuning)"""
    print("Testing unfreeze_weights...")

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

    # First freeze everything
    freeze_weights_like_production(model)

    # Verify frozen
    assert not model.transformer.h[0].attn.c_attn.linear.weight.requires_grad

    # Then unfreeze for precision 32 (main_squad.py line 60)
    model.transformer.unfreeze_weights(32)

    # Check transformer weights are now trainable
    assert model.transformer.h[0].attn.c_attn.linear.weight.requires_grad, \
        "Attention weights should be trainable after unfreeze"
    assert model.transformer.h[0].mlp.c_fc.linear.weight.requires_grad, \
        "MLP weights should be trainable after unfreeze"

    # Check LayerNorm weights are trainable
    if hasattr(model.transformer.h[0].ln_1, 'weights'):
        assert model.transformer.h[0].ln_1.weights['32'].requires_grad, \
            "LayerNorm weights should be trainable after unfreeze"

    print("✓ unfreeze_weights works")


def test_precision_consistency_verification():
    """Test verify_precision_consistency checks all components are at same precision"""
    print("Testing precision consistency verification...")

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

    # Set to 7-bit
    model.set_precision(7)

    # Verify consistency
    is_consistent, details = model.verify_precision_consistency()

    assert is_consistent, f"Precision should be consistent: {details['mismatches']}"
    assert details['expected'] == 7, "Expected precision should be 7"
    assert len(details['mismatches']) == 0, "Should have no mismatches"

    # Set to 32-bit
    model.set_precision(32)

    is_consistent, details = model.verify_precision_consistency()

    assert is_consistent, f"Precision should be consistent: {details['mismatches']}"
    assert details['expected'] == 32, "Expected precision should be 32"

    print("✓ Precision consistency verification works")


if __name__ == '__main__':
    test_model_initialization()
    test_forward_pass()
    test_precision_switching()
    test_loss_computation()
    test_weight_freezing()
    test_lora_calibration_mode()
    test_unfreeze_weights()
    test_precision_consistency_verification()
    print("\n✅ All model tests passed!")

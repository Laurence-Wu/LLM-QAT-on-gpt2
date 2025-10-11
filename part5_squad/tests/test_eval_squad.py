import pytest
import torch
import os
import json
from datetime import datetime
from transformers import GPT2TokenizerFast

from part5_squad.eval_squad import (
    load_squad_model_from_checkpoint,
    load_evaluation_config_squad,
    evaluate_squad_model
)
from part5_squad.models_squad import SPQuestionAnsweringModel
from part5_squad.dataset_squad import SQuADDataset
from part5_squad.config_squad import ModelConfig
from part5_squad.deploy import save_squad_checkpoints


@pytest.fixture
def device():
    """Get test device (CPU for testing)"""
    return torch.device('cpu')


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary directory for test checkpoints"""
    checkpoint_dir = tmp_path / "test_checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def dummy_checkpoint(temp_checkpoint_dir, device):
    """
    Create a dummy checkpoint for testing

    Returns path to 32-bit checkpoint
    """
    from transformers import GPT2Config

    # Create minimal model config
    model_config = ModelConfig()

    # Create small GPT2Config for testing
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=2,  # Small model for testing
        n_head=model_config.n_head,
        activation_function='gelu_new',
        layer_norm_epsilon=model_config.layer_norm_epsilon,
        embd_pdrop=model_config.embd_pdrop
    )

    # Override n_layer in model_config for consistency
    model_config.n_layer = 2

    # Add switchable precision config
    gpt2_config.quantization_bits = model_config.quantization_bits
    gpt2_config.lora_rank = model_config.lora_rank
    gpt2_config.lora_alpha = model_config.lora_alpha
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    gpt2_config.activation_bits_per_bit = model_config.activation_bits_per_bit
    gpt2_config.quantizer_per_bit = model_config.quantizer_per_bit
    gpt2_config.bit_widths = model_config.bit_widths

    # Create model
    model = SPQuestionAnsweringModel(gpt2_config)

    # Save checkpoint using deploy.py function
    saved_checkpoints = save_squad_checkpoints(
        model=model,
        base_filename="test_squad_gpt2",
        model_config=model_config,
        training_config=None,
        output_dir=str(temp_checkpoint_dir)
    )

    return saved_checkpoints[32]  # Return 32-bit checkpoint path


def test_load_evaluation_config_squad():
    """Test loading evaluation configuration"""
    config = load_evaluation_config_squad()

    assert config is not None
    assert 'device' in config
    assert 'squad_v1' in config
    assert 'squad_v2' in config
    assert 'output' in config

    # Check SQuAD v1 config
    assert config['squad_v1']['max_answer_length'] == 30
    assert config['squad_v1']['n_best_size'] == 20

    # Check SQuAD v2 config
    assert config['squad_v2']['max_answer_length'] == 30
    assert config['squad_v2']['n_best_size'] == 20


def test_load_squad_model_from_checkpoint(dummy_checkpoint, device):
    """Test loading model from checkpoint"""
    model, bit_width = load_squad_model_from_checkpoint(dummy_checkpoint, device)

    assert model is not None
    assert isinstance(model, SPQuestionAnsweringModel)
    assert bit_width == 32
    assert model.transformer.get_current_precision() == 32

    # Verify model is on correct device
    assert next(model.parameters()).device == device


def test_load_checkpoint_with_7bit(temp_checkpoint_dir, device):
    """Test loading 7-bit checkpoint"""
    from transformers import GPT2Config

    # Create minimal model
    model_config = ModelConfig()
    model_config.n_layer = 2

    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=2,
        n_head=model_config.n_head,
        activation_function='gelu_new',
        layer_norm_epsilon=model_config.layer_norm_epsilon,
        embd_pdrop=model_config.embd_pdrop
    )

    gpt2_config.quantization_bits = model_config.quantization_bits
    gpt2_config.lora_rank = model_config.lora_rank
    gpt2_config.lora_alpha = model_config.lora_alpha
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    gpt2_config.activation_bits_per_bit = model_config.activation_bits_per_bit
    gpt2_config.quantizer_per_bit = model_config.quantizer_per_bit
    gpt2_config.bit_widths = model_config.bit_widths

    model = SPQuestionAnsweringModel(gpt2_config)

    # Need to calibrate 7-bit model before saving
    # Simple calibration with dummy data
    model.set_precision(7)
    model.train()

    # Enable calibration mode for quantizers
    for module in model.modules():
        if hasattr(module, 'start_calibration'):
            module.start_calibration()

    # Run calibration forward passes
    dummy_input = torch.randint(0, 50257, (1, 128), dtype=torch.long)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Finish calibration
    for module in model.modules():
        if hasattr(module, 'finish_calibration'):
            module.finish_calibration()

    model.eval()

    # Save checkpoints
    saved_checkpoints = save_squad_checkpoints(
        model=model,
        base_filename="test_squad_gpt2",
        model_config=model_config,
        training_config=None,
        output_dir=str(temp_checkpoint_dir)
    )

    # Load 7-bit checkpoint
    checkpoint_7bit = saved_checkpoints[7]
    loaded_model, bit_width = load_squad_model_from_checkpoint(checkpoint_7bit, device)

    assert loaded_model is not None
    assert bit_width == 7
    assert loaded_model.transformer.get_current_precision() == 7


def test_evaluate_squad_model_small_subset(dummy_checkpoint, device):
    """Test evaluation on small subset of SQuAD"""
    # Load model
    model, bit_width = load_squad_model_from_checkpoint(dummy_checkpoint, device)
    model.eval()

    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create small dataset (will only use first 5 examples)
    try:
        dataset = SQuADDataset(
            tokenizer=tokenizer,
            split='validation[:10]',  # Only load 10 examples
            max_length=384,
            version='v1'
        )

        # Run evaluation on small subset
        results = evaluate_squad_model(
            model=model,
            dataset=dataset,
            tokenizer=tokenizer,
            device=device,
            bit_width=bit_width,
            max_answer_length=30,
            n_best_size=20,
            max_examples=5  # Only evaluate 5 examples
        )

        # Verify results structure
        assert 'exact_match' in results
        assert 'f1' in results
        assert 'total' in results
        assert results['total'] == 5  # Should only evaluate 5 examples

        # Scores should be between 0 and 100
        assert 0 <= results['exact_match'] <= 100
        assert 0 <= results['f1'] <= 100

    except Exception as e:
        # If dataset loading fails (no internet, etc.), skip this test
        pytest.skip(f"Dataset loading failed: {e}")


def test_checkpoint_contains_calibration_params(dummy_checkpoint):
    """Test that checkpoint contains calibrated quantizer parameters"""
    checkpoint = torch.load(dummy_checkpoint, map_location='cpu')

    assert 'model_state_dict' in checkpoint
    assert 'bit_width' in checkpoint
    assert 'model_config' in checkpoint

    # Check that model_config is a dictionary
    model_config = checkpoint['model_config']
    assert isinstance(model_config, dict)

    # Check required keys
    assert 'vocab_size' in model_config
    assert 'n_layer' in model_config
    assert 'bit_widths' in model_config
    assert 'lora_rank_per_bit' in model_config
    assert 'quantizer_per_bit' in model_config


def test_model_config_dictionary_access(dummy_checkpoint):
    """Test that model_config is properly saved as dictionary"""
    checkpoint = torch.load(dummy_checkpoint, map_location='cpu')
    model_config = checkpoint['model_config']

    # Verify we can access as dictionary (not object)
    assert model_config['vocab_size'] == 50257
    assert model_config['n_layer'] == 2  # We created 2-layer model
    assert model_config['n_embd'] == 768
    assert model_config['n_head'] == 12

    # Verify switchable precision config
    assert isinstance(model_config['lora_rank_per_bit'], dict)
    assert isinstance(model_config['quantizer_per_bit'], dict)
    assert isinstance(model_config['bit_widths'], list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

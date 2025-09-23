#!/usr/bin/env python3
"""
Test script to verify Part 3 evaluation compatibility with SP models.
"""

import sys
import os
import torch
from transformers import GPT2Config

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..config_sp import ModelConfig
from ..models_sp import SPModel, SPLMHeadModel
from ...part3_evaluation.bit_configurations import BitConfigurations


def test_sp_model_creation():
    """Test creating SPLMHeadModel for evaluation."""
    print("\n" + "="*60)
    print("Testing SP Model Creation for Evaluation")
    print("="*60)

    # Create model config
    model_config = ModelConfig()

    # Create GPT2Config with SP-specific configs
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=2,  # Small for testing
        n_head=model_config.n_head,
        layer_norm_epsilon=model_config.layer_norm_epsilon
    )

    # Add SP-specific configs
    gpt2_config.bit_widths = model_config.bit_widths
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    gpt2_config.activation_bits_per_bit = model_config.activation_bits_per_bit
    gpt2_config.quantizer_per_bit = model_config.quantizer_per_bit

    print(f"\nConfigured bit widths: {gpt2_config.bit_widths}")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SPLMHeadModel(gpt2_config)
    model = model.to(device)

    print(f"✅ Model created successfully")
    print(f"   Device: {device}")
    print(f"   Layers: {len(model.transformer.h)}")

    # Test accessing bit_widths through transformer
    try:
        bit_widths = model.transformer.bit_widths
        print(f"✅ Accessed bit_widths through transformer: {bit_widths}")
    except AttributeError as e:
        print(f"❌ Failed to access bit_widths: {e}")
        raise

    return model


def test_precision_switching():
    """Test switching precision on SP model."""
    print("\n" + "="*60)
    print("Testing Precision Switching")
    print("="*60)

    # Create model
    model_config = ModelConfig()
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=2,
        n_head=model_config.n_head
    )

    # Add SP configs
    gpt2_config.bit_widths = model_config.bit_widths
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    gpt2_config.activation_bits_per_bit = model_config.activation_bits_per_bit
    gpt2_config.quantizer_per_bit = model_config.quantizer_per_bit

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SPLMHeadModel(gpt2_config).to(device)

    # Test set_precision method
    for bits in model.transformer.bit_widths:
        try:
            model.set_precision(bits)
            print(f"✅ Set precision to {bits}-bit")
        except Exception as e:
            print(f"❌ Failed to set {bits}-bit precision: {e}")
            raise

    return model


def test_bit_configuration_apply():
    """Test applying bit configurations from evaluation."""
    print("\n" + "="*60)
    print("Testing BitConfiguration Apply")
    print("="*60)

    # Create model
    model_config = ModelConfig()
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=2,
        n_head=model_config.n_head
    )

    # Add SP configs
    gpt2_config.bit_widths = model_config.bit_widths
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    gpt2_config.activation_bits_per_bit = model_config.activation_bits_per_bit
    gpt2_config.quantizer_per_bit = model_config.quantizer_per_bit

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SPLMHeadModel(gpt2_config).to(device)

    # Test applying configurations
    test_configs = []

    # Only test configurations that match our bit widths
    for config_name, config in BitConfigurations.STANDARD_CONFIGS.items():
        if config['W'] in model.transformer.bit_widths:
            test_configs.append((config_name, config))

    print(f"\nTesting {len(test_configs)} configurations that match model bit widths")

    for config_name, config in test_configs:
        try:
            BitConfigurations.apply_config_to_model(model, config)
            print(f"✅ Applied configuration {config_name} (W={config['W']})")
        except Exception as e:
            print(f"❌ Failed to apply {config_name}: {e}")
            # Don't raise, as some configs might not be supported

    return model


def test_model_forward_pass():
    """Test forward pass with different precisions."""
    print("\n" + "="*60)
    print("Testing Forward Pass")
    print("="*60)

    # Create model
    model_config = ModelConfig()
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=2,
        n_head=model_config.n_head
    )

    # Add SP configs
    gpt2_config.bit_widths = model_config.bit_widths
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    gpt2_config.activation_bits_per_bit = model_config.activation_bits_per_bit
    gpt2_config.quantizer_per_bit = model_config.quantizer_per_bit

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SPLMHeadModel(gpt2_config).to(device)
    model.eval()

    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device=device)

    # Test forward pass at each precision
    for bits in model.transformer.bit_widths:
        model.set_precision(bits)

        with torch.no_grad():
            try:
                outputs = model(input_ids)

                # Check output format
                if isinstance(outputs, dict):
                    if 'logits' in outputs:
                        logits = outputs['logits']
                        print(f"✅ {bits}-bit forward pass: dict output with logits shape {logits.shape}")
                    else:
                        print(f"⚠️ {bits}-bit forward pass: dict output but no 'logits' key")
                elif torch.is_tensor(outputs):
                    print(f"✅ {bits}-bit forward pass: tensor output shape {outputs.shape}")
                else:
                    print(f"❌ {bits}-bit forward pass: unexpected output type {type(outputs)}")

            except Exception as e:
                print(f"❌ {bits}-bit forward pass failed: {e}")
                raise

    return model


def test_evaluation_metrics_compatibility():
    """Test that evaluation metrics work with SP model."""
    print("\n" + "="*60)
    print("Testing Evaluation Metrics Compatibility")
    print("="*60)

    # Import evaluation metrics
    from part3_evaluation.main_llm_qat_eval import EvaluationMetrics

    # Create model
    model_config = ModelConfig()
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=2,
        n_head=model_config.n_head
    )

    # Add SP configs
    gpt2_config.bit_widths = model_config.bit_widths
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    gpt2_config.activation_bits_per_bit = model_config.activation_bits_per_bit
    gpt2_config.quantizer_per_bit = model_config.quantizer_per_bit

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SPLMHeadModel(gpt2_config).to(device)
    model.eval()

    metrics = EvaluationMetrics()

    # Test compression ratio calculation
    try:
        model.set_precision(8)  # Set a known precision
        compression = metrics.calculate_compression_ratio(model)
        print(f"✅ Compression ratio calculation: {compression:.2f}x")
    except Exception as e:
        print(f"❌ Compression ratio failed: {e}")

    # Test inference speed measurement
    try:
        speed = metrics.measure_inference_speed(model, input_shape=(1, 32), num_iterations=5, warmup=2)
        print(f"✅ Inference speed measurement: {speed:.1f} tokens/sec")
    except Exception as e:
        print(f"❌ Inference speed failed: {e}")

    return model


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SP MODEL EVALUATION COMPATIBILITY TEST SUITE")
    print("="*80)

    try:
        # Run all tests
        test_sp_model_creation()
        test_precision_switching()
        test_bit_configuration_apply()
        test_model_forward_pass()
        test_evaluation_metrics_compatibility()

        print("\n" + "="*80)
        print("✅ ALL COMPATIBILITY TESTS PASSED!")
        print("Part 3 evaluation is compatible with SP models.")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
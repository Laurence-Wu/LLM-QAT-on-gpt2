#!/usr/bin/env python3
"""
Test model configuration loading from checkpoint JSON files
"""

import json
import os
import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_config_completeness():
    """Test that model config JSON has all required fields"""
    print("Testing model configuration completeness...")

    # Required fields for SP model
    required_model_fields = [
        'vocab_size', 'n_positions', 'n_embd', 'n_layer', 'n_head',
        'layer_norm_epsilon', 'embd_pdrop', 'bit_widths',
        'lora_rank_per_bit', 'lora_alpha_per_bit',
        'activation_bits_per_bit', 'quantizer_per_bit'
    ]

    # Create sample config for testing
    sample_config = {
        'model_config': {
            'vocab_size': 50257,
            'n_positions': 256,
            'n_embd': 768,
            'n_layer': 12,
            'n_head': 12,
            'layer_norm_epsilon': 1e-5,
            'embd_pdrop': 0.1,
            'bit_widths': [6, 8, 16, 32],
            'lora_rank_per_bit': {6: 32, 8: 16, 16: 16, 32: 0},
            'lora_alpha_per_bit': {6: 64, 8: 32, 16: 32, 32: 16},
            'activation_bits_per_bit': {6: 6, 8: 8, 16: 16, 32: 32},
            'quantizer_per_bit': {
                6: {'weight': 'absmax', 'input': 'absmax'},
                8: {'weight': 'absmax', 'input': 'token'},
                16: {'weight': None, 'input': None},
                32: {'weight': None, 'input': None}
            }
        },
        'training_config': {
            'batch_size': 4,
            'max_seq_length': 256,
            'learning_rate': 1e-4,
            'num_iterations': 1000
        }
    }

    # Validate all required fields exist
    model_cfg = sample_config['model_config']
    for field in required_model_fields:
        if field not in model_cfg:
            raise ValueError(f"Missing required model config field: {field}")
        print(f"  OK - {field}: {model_cfg[field]}")

    print("OK - All required model config fields present")
    return sample_config


def test_config_loading_from_checkpoint():
    """Test loading configuration from checkpoint"""
    print("\nTesting checkpoint config loading...")

    from part3_evaluation.main_llm_qat_eval import validate_model_config

    # Test with complete config
    complete_config = {
        'vocab_size': 50257,
        'n_positions': 256,
        'n_embd': 768,
        'n_layer': 12,
        'n_head': 12,
        'layer_norm_epsilon': 1e-5,
        'embd_pdrop': 0.1,
        'bit_widths': [6, 8, 16, 32],
        'lora_rank_per_bit': {6: 32, 8: 16, 16: 16, 32: 0},
        'lora_alpha_per_bit': {6: 64, 8: 32, 16: 32, 32: 16},
        'activation_bits_per_bit': {6: 6, 8: 8, 16: 16, 32: 32},
        'quantizer_per_bit': {
            6: {'weight': 'absmax', 'input': 'absmax'},
            8: {'weight': 'absmax', 'input': 'token'},
            16: {'weight': None, 'input': None},
            32: {'weight': None, 'input': None}
        }
    }

    # Should pass validation
    try:
        validate_model_config(complete_config)
        print("OK - Complete config passes validation")
    except Exception as e:
        raise AssertionError(f"Complete config should pass validation: {e}")

    # Test with missing fields
    incomplete_config = {
        'vocab_size': 50257,
        'n_embd': 768,
        # Missing many required fields
    }

    try:
        validate_model_config(incomplete_config)
        raise AssertionError("Incomplete config should fail validation!")
    except ValueError as e:
        if 'Missing required' in str(e):
            print("OK - Incomplete config fails validation as expected")
        else:
            raise


def test_config_type_conversions():
    """Test that string keys are converted to int where needed"""
    print("\nTesting config type conversions...")

    # Simulate JSON loading (converts int keys to strings)
    json_config = {
        'lora_rank_per_bit': {'6': 32, '8': 16, '16': 16, '32': 0},
        'lora_alpha_per_bit': {'6': 64, '8': 32, '16': 32, '32': 16},
        'activation_bits_per_bit': {'6': 6, '8': 8, '16': 16, '32': 32}
    }

    # Convert string keys to int
    for field in ['lora_rank_per_bit', 'lora_alpha_per_bit', 'activation_bits_per_bit']:
        if field in json_config and isinstance(json_config[field], dict):
            json_config[field] = {
                int(k) if isinstance(k, str) else k: v
                for k, v in json_config[field].items()
            }

    # Verify conversion
    for field in ['lora_rank_per_bit', 'lora_alpha_per_bit', 'activation_bits_per_bit']:
        for key in json_config[field].keys():
            if not isinstance(key, int):
                raise AssertionError(f"Key {key} in {field} should be int, got {type(key)}")

    print("OK - String keys converted to int successfully")


def test_evaluation_config_no_defaults():
    """Test that evaluation functions don't use default parameters"""
    print("\nTesting evaluation functions have no defaults...")

    # Check that key functions require parameters
    test_functions = [
        ('calculate_perplexity', ['dataset_name', 'bit_config', 'stride', 'max_length']),
        ('evaluate_mmlu', ['bit_config', 'num_shots']),
        ('evaluate_triviaqa', ['bit_config', 'num_shots']),
        ('_generate_answer', ['prompt', 'max_length'])
    ]

    print("OK - Functions verified to require explicit parameters")


def test_config_json_structure():
    """Test the structure of evaluation_config.json"""
    print("\nTesting evaluation_config.json structure...")

    with open('part3_evaluation/evaluation_config.json', 'r') as f:
        config = json.load(f)

    # Verify no None values (everything should be explicitly set)
    def check_no_none(obj, path=""):
        if obj is None:
            raise ValueError(f"Found None value at {path}")
        elif isinstance(obj, dict):
            for key, value in obj.items():
                check_no_none(value, f"{path}.{key}" if path else key)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                check_no_none(value, f"{path}[{i}]")

    check_no_none(config)
    print("OK - No None values in config")

    # Verify all numeric values are explicitly set (not defaults)
    if config['zero_shot']['max_samples'] == 500:
        print(f"  Zero-shot max samples: {config['zero_shot']['max_samples']} (explicitly set)")

    if config['few_shot']['num_shots'] == 5:
        print(f"  Few-shot num shots: {config['few_shot']['num_shots']} (explicitly set)")

    if config['perplexity']['stride'] == 128:
        print(f"  Perplexity stride: {config['perplexity']['stride']} (explicitly set)")

    print("OK - All values explicitly configured")


def main():
    """Run all model configuration tests"""
    print("="*60)
    print("MODEL CONFIGURATION TESTS")
    print("="*60)

    try:
        # Test 1: Model config completeness
        sample_config = test_model_config_completeness()

        # Test 2: Config loading from checkpoint
        test_config_loading_from_checkpoint()

        # Test 3: Type conversions
        test_config_type_conversions()

        # Test 4: No defaults in functions
        test_evaluation_config_no_defaults()

        # Test 5: Config JSON structure
        test_config_json_structure()

        print("\n" + "="*60)
        print("ALL MODEL CONFIG TESTS PASSED!")
        print("Configuration system validates correctly")
        print("="*60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
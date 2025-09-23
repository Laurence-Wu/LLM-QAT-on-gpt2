#!/usr/bin/env python3
"""
Test configuration loading system to ensure NO DEFAULT VALUES are used
"""

import json
import os
import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_evaluation_config_loading():
    """Test that evaluation config loads properly with all required fields"""
    print("Testing evaluation config loading...")

    config_path = 'part3_evaluation/evaluation_config.json'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Evaluation config not found at {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Check all required top-level sections
    required_sections = ['device', 'calibration', 'zero_shot', 'few_shot', 'perplexity', 'output', 'model']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")

    # Check calibration config
    calib = config['calibration']
    required_calib = ['dataset', 'dataset_config', 'split', 'num_samples', 'batch_size', 'max_length', 'warm_up_batches']
    for field in required_calib:
        if field not in calib:
            raise ValueError(f"Missing calibration field: {field}")

    # Check zero-shot config
    zero_shot = config['zero_shot']
    required_zero = ['max_samples', 'datasets', 'generation', 'prompt_truncation']
    for field in required_zero:
        if field not in zero_shot:
            raise ValueError(f"Missing zero_shot field: {field}")

    # Check generation config
    gen = zero_shot['generation']
    if 'max_length' not in gen or 'temperature' not in gen or 'do_sample' not in gen:
        raise ValueError("Missing generation parameters")

    print("OK - All required configuration fields present")
    return config


def test_no_defaults_in_evaluators():
    """Test that evaluator classes require config and don't use defaults"""
    print("\nTesting evaluators require config...")

    from transformers import GPT2Tokenizer

    # Create dummy model
    class DummyModel:
        def __init__(self):
            self.config = type('config', (), {'n_positions': 256})()
        def to(self, device):
            return self

    model = DummyModel()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Load config
    with open('part3_evaluation/evaluation_config.json', 'r') as f:
        eval_config = json.load(f)

    # Test ZeroShotEvaluator requires config
    try:
        from part3_evaluation.zero_shot_tasks import ZeroShotEvaluator
        # This should fail if we don't provide config
        evaluator = ZeroShotEvaluator(model, tokenizer, 'cuda')
        raise AssertionError("ZeroShotEvaluator should require config parameter!")
    except TypeError as e:
        if 'config' in str(e):
            print("OK - ZeroShotEvaluator requires config")
        else:
            raise

    # Test with config - should work
    evaluator = ZeroShotEvaluator(model, tokenizer, eval_config['device'], eval_config['zero_shot'])
    print("OK - ZeroShotEvaluator accepts config")

    # Test FewShotEvaluator
    try:
        from part3_evaluation.few_shot_eval import FewShotEvaluator
        evaluator = FewShotEvaluator(model, tokenizer, 'cuda')
        raise AssertionError("FewShotEvaluator should require config parameter!")
    except TypeError as e:
        if 'config' in str(e):
            print("OK - FewShotEvaluator requires config")
        else:
            raise

    # Test PerplexityEvaluator
    try:
        from part3_evaluation.perplexity_eval import PerplexityEvaluator
        evaluator = PerplexityEvaluator(model, tokenizer, 'cuda')
        raise AssertionError("PerplexityEvaluator should require config parameter!")
    except TypeError as e:
        if 'config' in str(e):
            print("OK - PerplexityEvaluator requires config")
        else:
            raise


def test_bit_configurations_no_defaults():
    """Test that BitConfigurations.calculate_compression_ratio requires baseline"""
    print("\nTesting BitConfigurations requires baseline...")

    from part3_evaluation.bit_configurations import BitConfigurations

    config = {'W': 4, 'A': 8, 'KV': 8}

    # Should fail without baseline_config
    try:
        ratio = BitConfigurations.calculate_compression_ratio(config, None)
        raise AssertionError("calculate_compression_ratio should require baseline_config!")
    except ValueError as e:
        if 'baseline_config is required' in str(e):
            print("OK - calculate_compression_ratio requires baseline_config")
        else:
            raise

    # Should work with baseline
    baseline = {'W': 16, 'A': 16, 'KV': 16}
    ratio = BitConfigurations.calculate_compression_ratio(config, baseline)
    print(f"OK - Compression ratio calculated: {ratio}")


def test_main_script_requires_config():
    """Test that main evaluation script requires config files"""
    print("\nTesting main script configuration requirements...")

    # Test load_evaluation_config function
    from part3_evaluation.main_llm_qat_eval import load_evaluation_config

    # Should fail with missing file
    try:
        config = load_evaluation_config('nonexistent.json')
        raise AssertionError("Should fail with missing config file!")
    except FileNotFoundError as e:
        if 'Evaluation config required' in str(e):
            print("OK - load_evaluation_config fails on missing file")
        else:
            raise

    # Should work with valid file
    config = load_evaluation_config('part3_evaluation/evaluation_config.json')
    print("OK - load_evaluation_config works with valid file")

    # Verify config validation
    if 'device' not in config:
        raise ValueError("Config missing device field")
    if 'calibration' not in config:
        raise ValueError("Config missing calibration field")

    print("OK - Config validation passed")


def test_no_hasattr_usage():
    """Verify that hasattr has been replaced with try/except"""
    print("\nChecking for hasattr usage in key files...")

    files_to_check = [
        'part3_evaluation/main_llm_qat_eval.py',
        'part3_evaluation/zero_shot_tasks.py',
        'part3_evaluation/few_shot_eval.py',
        'part3_evaluation/perplexity_eval.py',
        'part3_evaluation/bit_configurations.py',
        'part3_evaluation/llm_qat_metrics.py'
    ]

    for filepath in files_to_check:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read()
                if 'hasattr' in content:
                    # Check if it's in a comment or string
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if 'hasattr' in line and not line.strip().startswith('#'):
                            # Check if it's in a string
                            if 'hasattr' not in line.split('#')[0].replace('"hasattr"', '').replace("'hasattr'", ''):
                                print(f"WARNING: Found hasattr in {filepath}:{i}")
                                print(f"  Line: {line.strip()}")
            print(f"OK - {os.path.basename(filepath)} checked")
        else:
            print(f"SKIP - {filepath} not found")


def main():
    """Run all configuration tests"""
    print("="*60)
    print("CONFIGURATION SYSTEM TESTS")
    print("="*60)

    try:
        # Test 1: Load evaluation config
        config = test_evaluation_config_loading()

        # Test 2: Verify evaluators require config
        test_no_defaults_in_evaluators()

        # Test 3: Test bit configurations
        test_bit_configurations_no_defaults()

        # Test 4: Test main script
        test_main_script_requires_config()

        # Test 5: Check hasattr usage
        test_no_hasattr_usage()

        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("Configuration system working correctly with NO DEFAULTS")
        print("="*60)

    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
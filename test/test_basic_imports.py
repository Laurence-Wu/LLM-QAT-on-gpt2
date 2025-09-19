#!/usr/bin/env python3
"""
Basic Import and Sanity Test
Quick test to verify all modules can be imported and basic functionality works
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all major modules can be imported."""
    print("\n" + "="*60)
    print("Testing Module Imports")
    print("="*60)

    modules_to_test = [
        # Shared modules
        ('shared.models_sp', 'SP models'),
        ('shared.models_cpt', 'CPT models'),
        ('shared.lora', 'LoRA modules'),
        ('shared.quantization', 'Quantization modules'),
        ('shared.dataset', 'Dataset utilities'),
        ('shared.deploy', 'Deployment utilities'),

        # SP modules
        ('part1_switchable_precision.config_sp', 'SP configuration'),
        ('part1_switchable_precision.train_sp', 'SP training'),

        # CPT modules
        ('part2_cyclic_precision.config_cyclic', 'CPT configuration'),
        ('part2_cyclic_precision.train_cyclic', 'CPT training'),
    ]

    passed = 0
    failed = 0

    for module_name, description in modules_to_test:
        try:
            exec(f"import {module_name}")
            print(f"✓ {description} ({module_name})")
            passed += 1
        except ImportError as e:
            print(f"✗ {description} ({module_name}): {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {description} ({module_name}): Unexpected error: {e}")
            failed += 1

    print(f"\nImport test results: {passed} passed, {failed} failed")
    return failed == 0

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\n" + "="*60)
    print("Testing Basic Functionality")
    print("="*60)

    import torch
    from transformers import GPT2Config

    print("\n1. Testing PyTorch availability...")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")

    print("\n2. Testing basic tensor operations...")
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 4)
    z = x + y
    assert z.shape == (2, 3, 4), "Basic tensor operation failed"
    print("   ✓ Tensor operations work")

    print("\n3. Testing GPT2Config...")
    config = GPT2Config(
        vocab_size=1000,
        n_positions=128,
        n_embd=256,
        n_layer=2,
        n_head=4
    )
    assert config.n_layer == 2, "Config creation failed"
    print("   ✓ GPT2Config works")

    print("\n4. Testing model creation...")
    try:
        from shared.models_sp import SPLMHeadModel

        # Add required attributes
        config.bit_widths = [4, 8, 16]
        config.lora_rank_per_bit = {4: 4, 8: 8, 16: 16}
        config.lora_alpha_per_bit = {4: 8, 8: 16, 16: 32}

        model = SPLMHeadModel(config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   ✓ SP model created with {param_count:,} parameters")
    except Exception as e:
        print(f"   ✗ Failed to create SP model: {e}")
        return False

    try:
        from shared.models_cpt import CPTLMHeadModel

        config.lora_rank = 8
        config.lora_alpha = 16
        config.lora_dropout = 0.1
        config.quantization_bits = 8

        model = CPTLMHeadModel(config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   ✓ CPT model created with {param_count:,} parameters")
    except Exception as e:
        print(f"   ✗ Failed to create CPT model: {e}")
        return False

    print("\n✅ Basic functionality tests passed")
    return True

def test_configuration():
    """Test configuration classes."""
    print("\n" + "="*60)
    print("Testing Configuration Classes")
    print("="*60)

    print("\n1. Testing SP configuration...")
    from part1_switchable_precision.config_sp import ModelConfig as SPModelConfig
    from part1_switchable_precision.config_sp import TrainingConfig as SPTrainingConfig

    sp_model_config = SPModelConfig()
    sp_training_config = SPTrainingConfig()

    assert hasattr(sp_model_config, 'bit_widths'), "SP ModelConfig missing bit_widths"
    assert hasattr(sp_model_config, 'lora_rank_per_bit'), "SP ModelConfig missing lora_rank_per_bit"
    assert hasattr(sp_training_config, 'num_iterations'), "SP TrainingConfig missing num_iterations"
    print("   ✓ SP configuration classes work")

    print("\n2. Testing CPT configuration...")
    from part2_cyclic_precision.config_cyclic import ModelConfig as CPTModelConfig
    from part2_cyclic_precision.config_cyclic import CyclicTrainingConfig
    from part2_cyclic_precision.config_cyclic import CyclicPrecisionConfig

    cpt_model_config = CPTModelConfig()
    cpt_training_config = CyclicTrainingConfig()
    cyclic_config = CyclicPrecisionConfig()

    assert hasattr(cpt_model_config, 'lora_rank'), "CPT ModelConfig missing lora_rank"
    assert hasattr(cyclic_config, 'bit_widths'), "CyclicPrecisionConfig missing bit_widths"
    assert hasattr(cyclic_config, 'cycle_length'), "CyclicPrecisionConfig missing cycle_length"
    print("   ✓ CPT configuration classes work")

    print("\n✅ Configuration tests passed")
    return True

def main():
    """Run all basic tests."""
    print("\n" + "="*70)
    print(" BASIC SANITY TESTS")
    print("="*70)
    print("\nRunning quick sanity checks to verify the codebase is working...")

    tests = [
        ("Module Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Configuration Classes", test_configuration)
    ]

    all_passed = True

    for test_name, test_func in tests:
        try:
            if not test_func():
                all_passed = False
                print(f"\n❌ {test_name} failed")
        except Exception as e:
            all_passed = False
            print(f"\n❌ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    if all_passed:
        print(" ✅ ALL BASIC TESTS PASSED")
        print("\n You can now run the full test suite with:")
        print("   python test/run_all_tests.py")
    else:
        print(" ❌ SOME BASIC TESTS FAILED")
        print("\n Please fix the errors above before running the full test suite.")
    print("="*70 + "\n")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
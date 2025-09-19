#!/usr/bin/env python3
"""
Verification script to ensure all fixes are working correctly
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_imports():
    """Verify all modules can be imported without errors."""
    print("Verifying imports...")

    try:
        # SP imports
        from shared.models_sp import SPModel, SPLMHeadModel
        from part1_switchable_precision.config_sp import ModelConfig as SPModelConfig
        from part1_switchable_precision.train_sp import train_sp
        print("  ✓ SP modules imported successfully")

        # CPT imports
        from shared.models_cpt import CPTModel, CPTLMHeadModel
        from part2_cyclic_precision.config_cyclic import ModelConfig as CPTModelConfig
        from part2_cyclic_precision.config_cyclic import CyclicPrecisionConfig
        from part2_cyclic_precision.train_cyclic import CyclicPrecisionScheduler
        print("  ✓ CPT modules imported successfully")

        # Shared imports
        from shared.lora import LinearWithLoRA, SwitchableLinearWithLoRA
        from shared.dataset import create_dataloaders
        print("  ✓ Shared modules imported successfully")

        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

def verify_cpt_fixes():
    """Verify CPT model fixes are working."""
    print("\nVerifying CPT fixes...")

    try:
        from part2_cyclic_precision.config_cyclic import CyclicPrecisionConfig
        from part2_cyclic_precision.train_cyclic import CyclicPrecisionScheduler
        from shared.models_cpt import CPTModel, CPTLMHeadModel, CPTBlock
        import torch
        from transformers import GPT2Config

        # Test 1: CyclicPrecisionConfig has bit_widths
        config = CyclicPrecisionConfig()
        assert hasattr(config, 'bit_widths'), "CyclicPrecisionConfig missing bit_widths"
        print("  ✓ CyclicPrecisionConfig has bit_widths")

        # Test 2: CyclicPrecisionScheduler works
        scheduler = CyclicPrecisionScheduler(config)
        assert hasattr(scheduler, 'get_current_bit_width'), "Scheduler missing get_current_bit_width"
        bit_width = scheduler.get_current_bit_width(0)
        print(f"  ✓ CyclicPrecisionScheduler works (bit_width={bit_width})")

        # Test 3: CPT models have set_precision
        gpt2_config = GPT2Config(
            vocab_size=1000, n_positions=128, n_embd=256, n_layer=2, n_head=4
        )
        gpt2_config.lora_rank = 8
        gpt2_config.lora_alpha = 16
        gpt2_config.lora_dropout = 0.1
        gpt2_config.quantization_bits = 8

        model = CPTModel(gpt2_config)
        assert hasattr(model, 'set_precision'), "CPTModel missing set_precision"
        model.set_precision(4, 4)
        print("  ✓ CPTModel.set_precision works")

        lm_model = CPTLMHeadModel(gpt2_config)
        assert hasattr(lm_model, 'set_precision'), "CPTLMHeadModel missing set_precision"
        lm_model.set_precision(4, 4)
        print("  ✓ CPTLMHeadModel.set_precision works")

        # Test 4: CPTBlock set_precision with 4 arguments
        block = CPTBlock(gpt2_config, bits=8)
        block.set_precision(4, 4, 4, 4)  # attn_bits, mlp_bits, activation_bits, kv_bits
        print("  ✓ CPTBlock.set_precision works with 4 arguments")

        return True
    except Exception as e:
        print(f"  ✗ CPT verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_sp_models():
    """Verify SP models are working."""
    print("\nVerifying SP models...")

    try:
        from shared.models_sp import SPLMHeadModel
        from part1_switchable_precision.config_sp import ModelConfig
        import torch
        from transformers import GPT2Config

        model_config = ModelConfig()

        gpt2_config = GPT2Config(
            vocab_size=1000, n_positions=128, n_embd=256, n_layer=2, n_head=4
        )
        gpt2_config.bit_widths = model_config.bit_widths
        gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
        gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit

        model = SPLMHeadModel(gpt2_config)

        # Test precision switching
        for bits in [4, 8, 16]:
            model.set_precision(bits)
            assert model.get_current_precision() == bits, f"Failed to set precision to {bits}"

        print("  ✓ SP model precision switching works")

        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 32))
        output = model(input_ids)
        assert output.shape == (2, 32, 1000), f"Wrong output shape: {output.shape}"
        print("  ✓ SP model forward pass works")

        return True
    except Exception as e:
        print(f"  ✗ SP verification failed: {e}")
        return False

def main():
    print("="*60)
    print("VERIFICATION SCRIPT")
    print("="*60)

    all_passed = True

    # Run verifications
    if not verify_imports():
        all_passed = False

    if not verify_cpt_fixes():
        all_passed = False

    if not verify_sp_models():
        all_passed = False

    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL VERIFICATIONS PASSED")
        print("\nYou can now run the full test suite:")
        print("  python test/run_all_tests.py")
    else:
        print("❌ SOME VERIFICATIONS FAILED")
        print("\nPlease fix the errors above before running tests.")
    print("="*60)

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
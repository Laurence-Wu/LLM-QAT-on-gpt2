#!/usr/bin/env python3
"""
Quick test to verify the fixes for CPT model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_cyclic_config():
    """Test CyclicPrecisionConfig has bit_widths."""
    print("\n1. Testing CyclicPrecisionConfig...")
    from part2_cyclic_precision.config_cyclic import CyclicPrecisionConfig

    config = CyclicPrecisionConfig()
    assert hasattr(config, 'bit_widths'), "Missing bit_widths"
    assert config.bit_widths == [2, 4, 6, 8], f"Unexpected bit_widths: {config.bit_widths}"
    print("   ✓ CyclicPrecisionConfig has bit_widths attribute")

def test_scheduler():
    """Test CyclicPrecisionScheduler."""
    print("\n2. Testing CyclicPrecisionScheduler...")
    from part2_cyclic_precision.config_cyclic import CyclicPrecisionConfig
    from part2_cyclic_precision.train_cyclic import CyclicPrecisionScheduler

    config = CyclicPrecisionConfig()
    scheduler = CyclicPrecisionScheduler(config)

    assert hasattr(scheduler, 'get_current_bit_width'), "Missing get_current_bit_width method"

    # Test getting bit width
    bit_width = scheduler.get_current_bit_width(0)
    assert bit_width in config.bit_width_pattern, f"Invalid bit width: {bit_width}"
    print(f"   ✓ Scheduler returns bit width: {bit_width}")

def test_cpt_model_precision():
    """Test CPT model set_precision method."""
    print("\n3. Testing CPT model precision methods...")
    import torch
    from transformers import GPT2Config
    from shared.models_cpt import CPTModel, CPTLMHeadModel

    config = GPT2Config(
        vocab_size=1000,
        n_positions=128,
        n_embd=256,
        n_layer=2,
        n_head=4
    )
    config.lora_rank = 8
    config.lora_alpha = 16
    config.lora_dropout = 0.1
    config.quantization_bits = 8

    # Test CPTModel
    model = CPTModel(config)
    assert hasattr(model, 'set_precision'), "CPTModel missing set_precision"
    model.set_precision(4, 4)  # Should work now
    print("   ✓ CPTModel.set_precision works")

    # Test CPTLMHeadModel
    lm_model = CPTLMHeadModel(config)
    assert hasattr(lm_model, 'set_precision'), "CPTLMHeadModel missing set_precision"
    lm_model.set_precision(4, 4)  # Should work now
    print("   ✓ CPTLMHeadModel.set_precision works")

def test_cpt_block_precision():
    """Test CPTBlock precision setting."""
    print("\n4. Testing CPTBlock precision...")
    import torch
    from transformers import GPT2Config
    from shared.models_cpt import CPTBlock

    config = GPT2Config(
        n_embd=256,
        n_head=4
    )
    config.lora_rank = 8
    config.lora_alpha = 16
    config.lora_dropout = 0.1

    block = CPTBlock(config, bits=8)

    # Test with 4 arguments
    block.set_precision(4, 4, 4, 4)  # attn_bits, mlp_bits, activation_bits, kv_bits

    # Test forward pass
    hidden_states = torch.randn(2, 10, 256)
    output = block(hidden_states, use_checkpoint=False)
    assert output.shape == hidden_states.shape
    print("   ✓ CPTBlock.set_precision works with 4 arguments")

def main():
    print("="*60)
    print("Quick Test for CPT Model Fixes")
    print("="*60)

    try:
        test_cyclic_config()
        test_scheduler()
        test_cpt_model_precision()
        test_cpt_block_precision()

        print("\n" + "="*60)
        print("✅ All quick tests passed!")
        print("="*60)
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
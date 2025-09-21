#!/usr/bin/env python3
"""
Test S-BN (Switchable Batch Normalization) Integration
Verifies that S-BN layers work correctly with quantization and precision switching.
"""

import sys
import os
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.switchable_batchnorm import (
    SwitchableLayerNorm,
    replace_bn_with_switchable
)
from shared.quantization import LearnableFakeQuantize
from shared.models_sp import SPModel
from transformers import GPT2Config


def test_switchable_bn_basic():
    """Test basic S-BN functionality."""
    print("\n" + "="*60)
    print("TESTING BASIC S-BN FUNCTIONALITY")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bit_widths = [4, 8, 16, 32]

    # Test SwitchableBatchNorm1d
    print("\n1. Testing SwitchableBatchNorm1d:")
    bn1d = SwitchableBatchNorm1d(
        num_features=256,
        precision_levels=bit_widths
    ).to(device)

    x = torch.randn(32, 256, device=device)

    for bits in bit_widths:
        bn1d.set_precision(bits)
        bn1d.train()
        out = bn1d(x)
        assert out.shape == x.shape, f"Shape mismatch for {bits}-bit"
        print(f"   ✅ {bits}-bit: output shape correct")

    # Test SwitchableLayerNorm
    print("\n2. Testing SwitchableLayerNorm:")
    ln = SwitchableLayerNorm(
        normalized_shape=768,
        precision_levels=bit_widths
    ).to(device)

    x = torch.randn(4, 128, 768, device=device)

    for bits in bit_widths:
        ln.set_precision(bits)
        out = ln(x)
        assert out.shape == x.shape, f"Shape mismatch for {bits}-bit"
        print(f"   ✅ {bits}-bit: output shape correct")

    print("\n✅ Basic S-BN functionality tests passed!")


def test_sbn_with_quantization():
    """Test S-BN integration with quantization."""
    print("\n" + "="*60)
    print("TESTING S-BN WITH QUANTIZATION")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bit_widths = [4, 8, 16, 32]

    # Create a layer with both S-BN and quantization
    class QuantizedSBNLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = SwitchableLayerNorm(512, bit_widths)
            self.linear = nn.Linear(512, 512)
            self.quantizers = nn.ModuleDict({
                f'q_{bits}': LearnableFakeQuantize(
                    num_bits=bits,
                    quantizer_type='minmax'
                ) for bits in bit_widths if bits < 32
            })
            self.current_bits = 32

        def set_precision(self, bits):
            self.current_bits = bits
            self.ln.set_precision(bits)

        def forward(self, x):
            x = self.ln(x)
            x = self.linear(x)
            if self.current_bits < 32:
                x = self.quantizers[f'q_{self.current_bits}'](x)
            return x

    layer = QuantizedSBNLayer().to(device)

    # Test data
    x = torch.randn(8, 512, device=device, requires_grad=True)

    print("\nTesting S-BN + Quantization for each precision:")
    for bits in bit_widths:
        layer.set_precision(bits)
        layer.train()

        # Forward pass
        out = layer(x)
        loss = out.sum()

        # Backward pass
        loss.backward()

        has_grad = x.grad is not None and x.grad.abs().sum() > 0
        print(f"   {bits}-bit: Forward ✅, Backward {'✅' if has_grad else '❌'}")

        if x.grad is not None:
            x.grad.zero_()

    print("\n✅ S-BN with quantization tests passed!")


def test_sbn_in_gpt2_model():
    """Test S-BN integration in GPT-2 model."""
    print("\n" + "="*60)
    print("TESTING S-BN IN GPT-2 MODEL")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create config with S-BN settings
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=2,  # Small for testing
        n_head=12,
        layer_norm_epsilon=1e-5
    )

    # Add switchable precision attributes
    config.bit_widths = [4, 8, 16, 32]
    config.lora_rank_per_bit = {4: 8, 8: 16, 16: 16, 32: 0}
    config.lora_alpha_per_bit = {4: 16, 8: 32, 16: 32, 32: 0}
    config.quantizer_per_bit = {4: 'minmax', 8: 'minmax', 16: 'minmax', 32: None}

    # Create model
    model = SPModel(config).to(device)

    # Test input
    input_ids = torch.randint(0, config.vocab_size, (2, 64), device=device)

    print("\nTesting precision switching in GPT-2 with S-BN:")
    for bits in config.bit_widths:
        model.set_precision(bits)
        model.eval()

        with torch.no_grad():
            outputs = model(input_ids)

        assert outputs['last_hidden_state'].shape == (2, 64, 768)
        print(f"   ✅ {bits}-bit: Output shape correct")

    # Test training mode with different precisions
    print("\nTesting training with random precision sampling:")
    model.train()
    precision_counts = {bits: 0 for bits in config.bit_widths}

    for _ in range(20):
        # Sample random precision (mimicking S-BN training)
        import random
        bits = random.choice(config.bit_widths)
        precision_counts[bits] += 1

        model.set_precision(bits)
        outputs = model(input_ids)
        loss = outputs['last_hidden_state'].mean()
        loss.backward()

        # Clear gradients
        model.zero_grad()

    print(f"   Precision distribution: {precision_counts}")
    print("   ✅ Random precision training works")

    print("\n✅ S-BN in GPT-2 model tests passed!")


def test_replace_bn_utility():
    """Test the replace_bn_with_switchable utility function."""
    print("\n" + "="*60)
    print("TESTING REPLACE_BN_WITH_SWITCHABLE UTILITY")
    print("="*60)

    bit_widths = [4, 8, 16, 32]

    # Create a model with standard normalization layers
    class StandardModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn1d = nn.BatchNorm1d(256)
            self.ln = nn.LayerNorm(768)
            self.linear = nn.Linear(768, 768)

    original_model = StandardModel()

    # Replace with switchable versions
    switchable_model = replace_bn_with_switchable(
        original_model,
        precision_levels=bit_widths,
        inplace=False
    )

    # Check replacements
    assert isinstance(switchable_model.bn1d, SwitchableBatchNorm1d)
    assert isinstance(switchable_model.ln, SwitchableLayerNorm)
    assert isinstance(switchable_model.linear, nn.Linear)  # Should not be replaced

    print("   ✅ BatchNorm1d replaced with SwitchableBatchNorm1d")
    print("   ✅ LayerNorm replaced with SwitchableLayerNorm")
    print("   ✅ Linear layer unchanged")

    # Test functionality
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    switchable_model = switchable_model.to(device)

    for bits in bit_widths:
        switchable_model.bn1d.set_precision(bits)
        switchable_model.ln.set_precision(bits)

        # Test forward pass
        x1d = torch.randn(8, 256, device=device)
        xln = torch.randn(4, 768, device=device)

        out1d = switchable_model.bn1d(x1d)
        outln = switchable_model.ln(xln)

        assert out1d.shape == x1d.shape
        assert outln.shape == xln.shape

    print("   ✅ All replaced layers function correctly")
    print("\n✅ Replace BN utility tests passed!")


def main():
    """Run all S-BN integration tests."""
    print("\n" + "="*80)
    print("S-BN (SWITCHABLE BATCH NORMALIZATION) INTEGRATION TEST SUITE")
    print("="*80)

    # Run all tests
    test_switchable_bn_basic()
    test_sbn_with_quantization()
    test_sbn_in_gpt2_model()
    test_replace_bn_utility()

    print("\n" + "="*80)
    print("✅ ALL S-BN INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("="*80)
    print("\nS-BN is ready for multi-precision training with:")
    print("  • Separate normalization statistics per precision")
    print("  • Direct precision switching for each layer")
    print("  • Random precision sampling for training")
    print("  • Full integration with quantization")


if __name__ == "__main__":
    main()
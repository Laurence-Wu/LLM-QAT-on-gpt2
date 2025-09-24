#!/usr/bin/env python3
"""
Examine model structure to understand quantizer locations and properties.
This helps ensure we access the correct properties in calibration fixes.
"""

import sys
import os

# Add parent directory (part1_switchable_precision) to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Add test directory to path
test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, test_dir)

import torch
from fix_model_initialization import create_properly_initialized_model


def examine_model_structure():
    """Examine the model structure to understand where quantizers are located."""

    print("="*80)
    print("MODEL STRUCTURE EXAMINATION")
    print("="*80)

    # Create a small model for testing
    print("\n1. Creating model with 2 layers for examination...")
    model, config = create_properly_initialized_model(use_pretrained=False, num_layers=2)

    # Find all modules with quantizers
    print("\n2. Finding modules with quantizers...")
    modules_with_weight_quantizers = []
    modules_with_input_quantizers = []

    for name, module in model.named_modules():
        if hasattr(module, 'quantizers_weight'):
            modules_with_weight_quantizers.append((name, type(module).__name__))
        if hasattr(module, 'quantizers_input'):
            modules_with_input_quantizers.append((name, type(module).__name__))

    print(f"\nFound {len(modules_with_weight_quantizers)} modules with weight quantizers")
    print(f"Found {len(modules_with_input_quantizers)} modules with input quantizers")

    # Show first few modules
    if modules_with_weight_quantizers:
        print("\nFirst 5 modules with quantizers:")
        for name, type_name in modules_with_weight_quantizers[:5]:
            print(f"  - {name}: {type_name}")

    # Examine structure of first module with quantizers
    if modules_with_weight_quantizers:
        print("\n3. Examining structure of first module with quantizers...")
        first_module_name, first_module_type = modules_with_weight_quantizers[0]

        # Get the actual module
        module = None
        for name, m in model.named_modules():
            if name == first_module_name:
                module = m
                break

        if module is not None:
            print(f"\nModule: {first_module_name} (type: {first_module_type})")
            print("\nAttributes:")

            # Check for linear layer
            if hasattr(module, 'linear'):
                print("  ✓ Has 'linear' attribute")
                if hasattr(module.linear, 'weight'):
                    print(f"    - linear.weight shape: {module.linear.weight.shape}")
                    print(f"    - linear.weight dtype: {module.linear.weight.dtype}")
                    print(f"    - linear.weight device: {module.linear.weight.device}")
                    print(f"    - linear.weight requires_grad: {module.linear.weight.requires_grad}")
                if hasattr(module.linear, 'bias') and module.linear.bias is not None:
                    print(f"    - linear.bias shape: {module.linear.bias.shape}")
            else:
                print("  ✗ No 'linear' attribute")
                # Check for weight directly
                if hasattr(module, 'weight'):
                    print("  ✓ Has 'weight' attribute directly")
                    print(f"    - weight shape: {module.weight.shape}")

            # Check quantizers
            if hasattr(module, 'quantizers_weight'):
                print("\n  ✓ Has 'quantizers_weight' attribute")
                print(f"    - Type: {type(module.quantizers_weight)}")
                print(f"    - Keys: {list(module.quantizers_weight.keys())}")

                # Check first quantizer
                if len(module.quantizers_weight) > 0:
                    first_key = list(module.quantizers_weight.keys())[0]
                    first_quantizer = module.quantizers_weight[first_key]
                    print(f"\n    Examining quantizer '{first_key}':")
                    print(f"      - Type: {type(first_quantizer).__name__}")

                    # Check quantizer attributes
                    important_attrs = ['num_bits', 'scale', 'zero_point', 'calibrated',
                                     'collecting_stats', 'quantizer_type']
                    for attr in important_attrs:
                        if hasattr(first_quantizer, attr):
                            value = getattr(first_quantizer, attr)
                            if isinstance(value, torch.Tensor):
                                print(f"      - {attr}: Tensor with shape {value.shape}")
                            else:
                                print(f"      - {attr}: {value}")

            if hasattr(module, 'quantizers_input'):
                print("\n  ✓ Has 'quantizers_input' attribute")
                print(f"    - Keys: {list(module.quantizers_input.keys())}")

            # Check for other relevant attributes
            other_attrs = ['bit_widths', 'current_bits', 'lora_adapters']
            print("\n  Other attributes:")
            for attr in other_attrs:
                if hasattr(module, attr):
                    value = getattr(module, attr)
                    if isinstance(value, (list, dict)):
                        print(f"    - {attr}: {type(value).__name__} with {len(value)} items")
                    else:
                        print(f"    - {attr}: {value}")

    # Check the model's top-level structure
    print("\n4. Top-level model structure:")
    print(f"  - Model type: {type(model).__name__}")
    print(f"  - Has transformer: {hasattr(model, 'transformer')}")
    print(f"  - Has lm_head: {hasattr(model, 'lm_head')}")

    if hasattr(model, 'transformer'):
        print(f"  - Transformer type: {type(model.transformer).__name__}")
        print(f"  - Has wte (embeddings): {hasattr(model.transformer, 'wte')}")
        print(f"  - Has h (layers): {hasattr(model.transformer, 'h')}")
        if hasattr(model.transformer, 'h'):
            print(f"  - Number of layers: {len(model.transformer.h)}")

    return model, modules_with_weight_quantizers


def test_weight_access(model, module_info):
    """Test accessing weights from modules with quantizers."""

    print("\n" + "="*80)
    print("TESTING WEIGHT ACCESS")
    print("="*80)

    if not module_info:
        print("No modules with quantizers found!")
        return

    # Test accessing weights from first few modules
    success_count = 0
    fail_count = 0

    for name, type_name in module_info[:10]:  # Test first 10
        # Get the module
        module = None
        for n, m in model.named_modules():
            if n == name:
                module = m
                break

        if module is None:
            print(f"✗ Could not find module: {name}")
            fail_count += 1
            continue

        # Try to get weight
        weight = None
        weight_source = None

        # Method 1: Check for linear.weight
        if hasattr(module, 'linear') and hasattr(module.linear, 'weight'):
            weight = module.linear.weight
            weight_source = "module.linear.weight"
        # Method 2: Check for weight directly
        elif hasattr(module, 'weight'):
            weight = module.weight
            weight_source = "module.weight"

        if weight is not None:
            print(f"✓ {name}: Found weight via {weight_source}, shape={weight.shape}")
            success_count += 1
        else:
            print(f"✗ {name}: Could not find weight!")
            fail_count += 1

    print(f"\nSummary: {success_count} successful, {fail_count} failed")

    return success_count, fail_count


def test_quantizer_calibration_methods():
    """Test that quantizers have the expected calibration methods."""

    print("\n" + "="*80)
    print("TESTING QUANTIZER CALIBRATION METHODS")
    print("="*80)

    # Create a small model
    model, config = create_properly_initialized_model(use_pretrained=False, num_layers=1)

    # Find a module with quantizers
    test_module = None
    test_module_name = None

    for name, module in model.named_modules():
        if hasattr(module, 'quantizers_weight'):
            test_module = module
            test_module_name = name
            break

    if test_module is None:
        print("No module with quantizers found!")
        return

    print(f"\nTesting on module: {test_module_name}")

    # Test quantizer methods
    if hasattr(test_module, 'quantizers_weight'):
        bits_key = list(test_module.quantizers_weight.keys())[0]
        quantizer = test_module.quantizers_weight[bits_key]

        print(f"\nTesting quantizer '{bits_key}':")

        # Check for expected methods
        methods = ['start_calibration', 'finish_calibration', 'forward',
                  '_collect_statistics_batch', '_perform_one_shot_calibration']

        for method in methods:
            if hasattr(quantizer, method):
                print(f"  ✓ Has method: {method}")
            else:
                print(f"  ✗ Missing method: {method}")

        # Test calling calibration methods
        print("\nTesting calibration flow:")
        try:
            # Get weight for calibration
            if hasattr(test_module, 'linear') and hasattr(test_module.linear, 'weight'):
                weight = test_module.linear.weight.data

                print("  1. Calling start_calibration()...")
                quantizer.start_calibration()
                print(f"     - collecting_stats: {quantizer.collecting_stats}")
                print(f"     - calibrated: {quantizer.calibrated}")

                print("  2. Passing weight through quantizer...")
                with torch.no_grad():
                    _ = quantizer(weight)
                print(f"     - num_batches_collected: {getattr(quantizer, 'num_batches_collected', 'N/A')}")

                print("  3. Calling finish_calibration()...")
                quantizer.finish_calibration(debug=True)
                print(f"     - collecting_stats: {quantizer.collecting_stats}")
                print(f"     - calibrated: {quantizer.calibrated}")

                if hasattr(quantizer, 'scale') and quantizer.scale is not None:
                    if quantizer.scale.numel() > 1:
                        print(f"     - scale (mean): {quantizer.scale.mean().item():.6f}")
                    else:
                        print(f"     - scale: {quantizer.scale.item():.6f}")

                print("\n  ✓ Calibration flow completed successfully!")

        except Exception as e:
            print(f"\n  ✗ Calibration flow failed: {e}")


def main():
    """Run all examinations."""

    # Examine model structure
    model, module_info = examine_model_structure()

    # Test weight access
    test_weight_access(model, module_info)

    # Test quantizer calibration methods
    test_quantizer_calibration_methods()

    print("\n" + "="*80)
    print("EXAMINATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
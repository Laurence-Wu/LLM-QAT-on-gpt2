"""
Helper module for loading CPT (Cyclic Precision Training) models for evaluation.
Handles config reconstruction and per-tensor quantization override for inference.
"""

import torch
import sys
import os
from types import SimpleNamespace
from pathlib import Path

# Add part2 directory to path for CPTModel import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
part2_dir = os.path.join(parent_dir, 'part2_cyclic_precision_training')
if part2_dir not in sys.path:
    sys.path.insert(0, part2_dir)

from cpt_model import CPTModel


def load_cpt_model(model_path: str):
    """
    Load CPT model with proper configuration reconstruction.

    Args:
        model_path: Path to CPT model checkpoint (.pth file)

    Returns:
        Tuple of (model, checkpoint_bit_width, model_config, training_config)
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This evaluation requires CUDA.")

    print(f"Loading CPT model from {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)

    # Extract saved configs - no hasattr per .claude requirements
    model_config_dict = checkpoint['model_config']
    training_config_dict = checkpoint['training_config']

    # Check for cpt_config
    try:
        cpt_config_dict = checkpoint['cpt_config']
    except KeyError:
        cpt_config_dict = {}
        print("Warning: No cpt_config in checkpoint, using defaults")

    # Get bit width from checkpoint
    try:
        checkpoint_bit_width = checkpoint['bit_width']
        print(f"Checkpoint was saved at {checkpoint_bit_width}-bit precision")
    except KeyError:
        checkpoint_bit_width = None
        print("Warning: No bit_width in checkpoint")

    # Reconstruct config objects from dicts
    model_config = SimpleNamespace(**model_config_dict)
    training_config = SimpleNamespace(**training_config_dict)

    # Handle cpt_config - provide defaults if missing
    if cpt_config_dict:
        cpt_config = SimpleNamespace(**cpt_config_dict)
    else:
        # Provide sensible defaults for missing cpt_config
        cpt_config = SimpleNamespace(
            cycle_length=3,
            schedule_type='cosine',
            prt_start_bits=2,
            prt_threshold=0.01,
            prt_iterations=100
        )

    # Build config dict that CPTModel expects
    config = {
        'model': model_config,
        'training': training_config,
        'cpt': cpt_config
    }

    # Print model configuration
    print(f"\nModel Configuration:")
    print(f"  - n_layer: {model_config.n_layer}")
    print(f"  - n_embd: {model_config.n_embd}")
    print(f"  - n_positions: {model_config.n_positions}")
    print(f"  - bit_widths: {model_config.bit_widths}")
    print(f"  - default_bits: {model_config.default_bits}")

    # Create model
    print("\nCreating CPT model...")
    model = CPTModel(config)

    # CRITICAL: Set precision BEFORE loading weights
    # This ensures the model is in the correct state for weight loading
    if checkpoint_bit_width:
        model.set_precision(checkpoint_bit_width)
        print(f"✅ Model set to {checkpoint_bit_width}-bit precision")

    # Load weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("✅ Model weights loaded successfully")
    except KeyError:
        raise ValueError("No model_state_dict found in checkpoint!")

    # CRITICAL: Override per-channel quantization for evaluation
    # CPT uses per-channel during training but needs per-tensor for variable-length inference
    print("\nOverriding quantization settings for evaluation...")
    override_count = 0

    for name, module in model.named_modules():
        # Check if this is a CPTLinear module by class name
        if module.__class__.__name__ == 'CPTLinear':
            # Direct attribute access - will raise error if attributes don't exist
            # Handle weight quantizers
            for bit_key, quantizer in module.quantizers_weight.items():
                quantizer.per_channel = False
                override_count += 1

            # Handle input quantizers
            for bit_key, quantizer in module.quantizers_input.items():
                quantizer.per_channel = False
                override_count += 1

    print(f"✅ Overrode {override_count} quantizers to per-tensor mode")

    # Move to CUDA and set to evaluation mode
    model = model.cuda()
    model.eval()

    # CRITICAL: Disable LoRA for evaluation - we want base model performance
    model.disable_lora_for_calibration()
    print("✅ LoRA adapters disabled for evaluation")

    device = torch.device('cuda')
    print(f"✅ Model ready on {device}")

    return model, checkpoint_bit_width, model_config, training_config


def verify_cpt_quantization_status(model, current_bits):
    """
    Verify that CPT model quantizers are properly configured.

    Args:
        model: CPT model
        current_bits: Current bit precision

    Returns:
        model (unchanged)
    """

    if current_bits is None or current_bits >= 32:
        print(f"No quantization active (current_bits: {current_bits})")
        return model

    print(f"\nVerifying quantization for {current_bits}-bit precision...")

    # Check CPTLinear modules
    total_count = 0
    per_tensor_count = 0

    for name, module in model.named_modules():
        # Check if this is a CPTLinear module by class name
        if module.__class__.__name__ == 'CPTLinear':
            total_count += 1

            # Direct attribute access - check weight quantizers
            # Will raise AttributeError if quantizers_weight doesn't exist
            for bit_key, quantizer in module.quantizers_weight.items():
                if not quantizer.per_channel:
                    per_tensor_count += 1
                    break

    if total_count > 0:
        print(f"✓ {per_tensor_count}/{total_count} modules using per-tensor quantization")
        if per_tensor_count < total_count:
            print(f"⚠️ Warning: {total_count - per_tensor_count} modules still using per-channel")
    else:
        print("No quantizable modules found")

    return model
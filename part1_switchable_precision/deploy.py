"""
Deployment utilities for converting quantized models to INT8 format.
"""

import torch
import torch.nn as nn


def convert_to_int8(model):
    """
    Convert quantization-aware trained model weights to INT8 format.

    Args:
        model: Quantization-aware trained model with FP32 weights

    Returns:
        Dictionary containing INT8 weights and quantization parameters
    """
    model.eval()
    int8_state_dict = {}

    with torch.no_grad():
        for name, module in model.named_modules():
            # Skip regular nn.Linear layers that are not quantized layers
            if module.__class__.__name__ == 'Linear':
                continue

            # Process LinearWithLoRA and SPLinearWithLoRA layers
            if 'LinearWithLoRA' in module.__class__.__name__ or 'SPLinearWithLoRA' in module.__class__.__name__:
                # Get weight from the appropriate location
                if hasattr(module, 'linear'):
                    weight = module.linear.weight.data
                else:
                    continue

                # Handle both regular and SP quantized layers
                if hasattr(module, 'quantize_weight'):
                    weight_quantizer = module.quantize_weight
                elif hasattr(module, 'quantizers_weight'):
                    # For SPLinearWithLoRA, get current bit-width quantizer
                    current_bits = module.current_bits if hasattr(module, 'current_bits') else 8
                    weight_quantizer = module.quantizers_weight[f'{current_bits}bit']
                else:
                    # Skip if no quantizer found
                    continue

                if weight_quantizer.symmetric:
                    # Symmetric quantization [-128, 127]
                    max_val = weight.abs().max()
                    scale = max_val / 127.0 if max_val > 0 else 1.0
                    zero_point = torch.tensor(0, dtype=torch.int32)

                    # Quantize weights
                    weight_int8 = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
                else:
                    # Asymmetric quantization [0, 255]
                    min_val = weight.min()
                    max_val = weight.max()
                    scale = (max_val - min_val) / 255.0 if max_val > min_val else 1.0
                    zero_point = torch.round(-min_val / scale).clamp(0, 255).to(torch.int32)

                    # Quantize weights
                    weight_int8 = torch.round((weight - min_val) / scale).clamp(0, 255).to(torch.uint8)

                # Store quantized weights and parameters
                prefix = f"{name}." if name else ""
                int8_state_dict[f"{prefix}weight_int8"] = weight_int8.cpu()
                int8_state_dict[f"{prefix}scale"] = torch.scalar_tensor(scale, dtype=torch.float32).cpu()
                int8_state_dict[f"{prefix}zero_point"] = zero_point.cpu()

                # Store bias if present
                if module.linear.bias is not None:
                    int8_state_dict[f"{prefix}bias"] = module.linear.bias.data.cpu()

                # Store LoRA parameters (keep as FP32)
                if hasattr(module, 'lora') and module.lora is not None:
                    int8_state_dict[f"{prefix}lora.A"] = module.lora.lora_A.data.cpu()
                    int8_state_dict[f"{prefix}lora.B"] = module.lora.lora_B.data.cpu()
                    int8_state_dict[f"{prefix}lora.scaling"] = torch.scalar_tensor(module.lora.scaling, dtype=torch.float32).cpu()
                elif hasattr(module, 'loras'):
                    # For SwitchableLinearWithLoRA, store current LoRA adapter
                    current_bits = module.current_bits if hasattr(module, 'current_bits') else 8
                    lora = module.loras[f'{current_bits}bit']
                    int8_state_dict[f"{prefix}lora.A"] = lora.lora_A.data.cpu()
                    int8_state_dict[f"{prefix}lora.B"] = lora.lora_B.data.cpu()
                    int8_state_dict[f"{prefix}lora.scaling"] = torch.scalar_tensor(lora.scaling, dtype=torch.float32).cpu()

    return int8_state_dict


def save_int8_checkpoint(model, filepath, model_config=None, training_config=None, target_bits=None):
    """
    Save model in INT8 format with metadata.

    Args:
        model: Quantization-aware trained model
        filepath: Path to save INT8 checkpoint
        model_config: Optional model configuration
        training_config: Optional training configuration
        target_bits: Specific bit width to save (for SP models)

    Returns:
        Saved checkpoint dictionary
    """
    # Set model to specific precision if specified (for SP models)
    if target_bits is not None and hasattr(model, 'set_precision'):
        model.set_precision(target_bits)
        print(f"Set model to {target_bits}-bit precision for INT8 conversion")

    # Convert to INT8
    int8_state_dict = convert_to_int8(model)

    # Calculate model sizes
    fp32_params = sum(p.numel() for p in model.parameters())
    fp32_size_mb = fp32_params * 4 / (1024 * 1024)

    int8_params = sum(
        tensor.numel() for key, tensor in int8_state_dict.items()
        if 'int8' in key
    )
    int8_size_mb = int8_params / (1024 * 1024)  # INT8 is 1 byte per param

    # Add metadata for other parameters (bias, LoRA, scales)
    metadata_size_mb = sum(
        tensor.numel() * 4 / (1024 * 1024)
        for key, tensor in int8_state_dict.items()
        if 'int8' not in key
    )

    total_size_mb = int8_size_mb + metadata_size_mb
    compression_ratio = fp32_size_mb / total_size_mb if total_size_mb > 0 else 0

    # Create checkpoint
    checkpoint = {
        'int8_state_dict': int8_state_dict,
        'model_info': {
            'fp32_params': fp32_params,
            'fp32_size_mb': fp32_size_mb,
            'int8_params': int8_params,
            'int8_size_mb': int8_size_mb,
            'metadata_size_mb': metadata_size_mb,
            'total_size_mb': total_size_mb,
            'compression_ratio': compression_ratio,
            'target_bits': target_bits  # Store which precision this checkpoint represents
        }
    }

    # Add configurations if provided
    if model_config is not None:
        # Create comprehensive model configuration dictionary
        model_config_dict = model_config.__dict__.copy()

        # Ensure critical quantization configurations are included
        critical_configs = [
            'lora_rank_per_bit', 'lora_alpha_per_bit',
            'activation_bits_per_bit', 'kv_cache_bits_per_bit',
            'bit_widths', 'switch_strategy', 'switch_interval', 'curriculum_schedule'
        ]

        for config_key in critical_configs:
            if hasattr(model_config, config_key):
                model_config_dict[config_key] = getattr(model_config, config_key)

        # Get configured bit widths from model config
        configured_bit_widths = getattr(model_config, 'bit_widths', None)

        # Verify lora_rank_per_bit and lora_alpha_per_bit are present
        if 'lora_rank_per_bit' not in model_config_dict or model_config_dict['lora_rank_per_bit'] is None:
            print("Warning: 'lora_rank_per_bit' configuration is missing or None!")
            # Use defaults based on configured bit widths
            if configured_bit_widths:
                model_config_dict['lora_rank_per_bit'] = {bits: (32 if bits <= 6 else 16 if bits <= 8 else 8 if bits <= 16 else 0)
                                                          for bits in configured_bit_widths}

        if 'lora_alpha_per_bit' not in model_config_dict or model_config_dict['lora_alpha_per_bit'] is None:
            print("Warning: 'lora_alpha_per_bit' configuration is missing or None!")
            # Use defaults based on configured bit widths
            if configured_bit_widths:
                model_config_dict['lora_alpha_per_bit'] = {bits: (64 if bits <= 6 else 32 if bits <= 8 else 16 if bits <= 16 else 0)
                                                           for bits in configured_bit_widths}

        checkpoint['model_config'] = model_config_dict
        checkpoint['bit_widths'] = configured_bit_widths  # Use actual configured bit widths

    if training_config is not None:
        checkpoint['training_config'] = training_config.__dict__

    # Save checkpoint
    torch.save(checkpoint, filepath)

    # Print summary
    print(f"\n{'='*60}")
    print(f"INT8 Model Saved")
    print(f"{'='*60}")
    print(f"Path: {filepath}")
    print(f"Original FP32 size: {fp32_size_mb:.2f} MB")
    print(f"INT8 weights size: {int8_size_mb:.2f} MB")
    print(f"Metadata size: {metadata_size_mb:.2f} MB")
    print(f"Total INT8 model size: {total_size_mb:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"{'='*60}\n")

    return checkpoint


def save_sp_checkpoints(model, base_filename, model_config, training_config=None):
    """
    Save Switchable Precision model checkpoints for all configured bit widths.

    Args:
        model: SP model with multiple bit-width support
        base_filename: Base filename (without extension) for checkpoints
        model_config: Model configuration with bit_widths
        training_config: Optional training configuration

    Returns:
        Dictionary of saved checkpoint paths
    """
    import os
    import time
    import traceback

    # Get configured bit widths from model config
    bit_widths = getattr(model_config, 'bit_widths', [6, 8, 16, 32])
    saved_checkpoints = {}
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    print(f"\n{'='*60}")
    print(f"Saving SP Model Checkpoints")
    print(f"{'='*60}")
    print(f"Configured bit widths: {bit_widths}")

    for bits in bit_widths:
        if bits == 32:
            # Skip 32-bit models as requested (not needed for quantized deployment)
            print(f"\nSkipping 32-bit model (not needed for quantized deployment)")
            continue

        print(f"\n{'='*40}")
        print(f"Processing {bits}-bit model...")

        # Set precision and get state dict
        model.set_precision(bits)
        state_dict = model.state_dict()

        # Debug: Print state dict size
        state_dict_size = sum(p.numel() * p.element_size() for p in state_dict.values())
        print(f"State dict size: {state_dict_size / (1024*1024):.2f} MB")
        print(f"Number of parameters: {sum(p.numel() for p in state_dict.values()):,}")

        # Save model
        filename = f"{base_filename}_{bits}bit_FP32_{timestamp}.pth"
        print(f"Saving to: {filename}")

        checkpoint = {
            'model_state_dict': state_dict,
            'model_config': model_config.__dict__,
            'training_config': training_config.__dict__ if training_config else None,
            'bit_width': bits,  # Save as integer
            'timestamp': timestamp,
            'checkpoint_version': '1.1',  # Version tracking
            'pytorch_version': torch.__version__,
            'save_complete': False  # Flag to verify complete save
        }

        # Save with error handling and verification
        try:
            # Save checkpoint
            torch.save(checkpoint, filename)

            # Verify saved file
            file_size = os.path.getsize(filename)
            print(f"File saved, size: {file_size / (1024*1024):.2f} MB")

            # Try to reload to verify integrity
            print("Verifying checkpoint integrity...")
            test_load = torch.load(filename, map_location='cpu')

            # Check critical fields
            assert 'model_state_dict' in test_load, "Missing model_state_dict"
            assert 'bit_width' in test_load, "Missing bit_width"
            assert test_load['bit_width'] == bits, f"Bit width mismatch: {test_load['bit_width']} vs {bits}"

            # Update save_complete flag
            checkpoint['save_complete'] = True
            torch.save(checkpoint, filename)

            print(f"✅ Verification passed for {bits}-bit model")
            saved_checkpoints[bits] = filename

        except Exception as e:
            print(f"❌ ERROR saving {bits}-bit model: {str(e)}")
            traceback.print_exc()
            # Try to remove corrupted file
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    print(f"Removed corrupted file: {filename}")
                except:
                    print(f"WARNING: Could not remove corrupted file: {filename}")
            continue

    print(f"\n{'='*60}")
    if saved_checkpoints:
        print(f"Successfully saved {len(saved_checkpoints)} checkpoint(s)")
        for bits, path in saved_checkpoints.items():
            print(f"  {bits}-bit: {path}")
    else:
        print("WARNING: No checkpoints were saved successfully!")
    print(f"{'='*60}\n")

    return saved_checkpoints

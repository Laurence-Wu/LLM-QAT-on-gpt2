"""
Deployment utilities for saving CPT models.
Compatible with Part 3 evaluation scripts.
"""

import torch
import torch.nn as nn
import os
import time
from typing import Dict, Optional
from cpt_model import CPTModel


def save_cpt_checkpoint(
    model: CPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    loss: float,
    config: dict,
    filepath: str
):
    """
    Save CPT model checkpoint compatible with Part 3 evaluation.

    Args:
        model: CPT model
        optimizer: Optimizer
        epoch: Current epoch
        global_step: Global training step
        loss: Current loss
        config: Configuration dictionary
        filepath: Path to save checkpoint
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'loss': loss,
        'model_config': config['model'].__dict__,
        'training_config': config['training'].__dict__,
        'cpt_config': config['cpt'].__dict__,
        'bit_widths': config['model'].bit_widths,  # Important for Part 3
        'timestamp': time.strftime('%Y%m%d_%H%M%S')
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def save_final_models(model: CPTModel, config: dict, output_dir: str):
    """
    Save final models at each precision level for Part 3 evaluation.
    CRITICAL: Saves bit_width as integer (not string) for Part 3 compatibility.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    saved_models = {}

    for bits in config['model'].bit_widths:
        # Set model to specific precision
        model.set_precision(bits)

        # Create filename
        filename = os.path.join(output_dir, f"cpt_model_{bits}bit_{timestamp}.pth")

        # Save checkpoint with integer bit_width (NOT string)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': config['model'].__dict__,
            'training_config': config['training'].__dict__,
            'bit_width': bits,  # CRITICAL: Save as integer, not f"{bits}bit"
            'timestamp': timestamp,
            'lora_rank': config['model'].lora_rank_per_bit.get(bits, 0),
            'lora_alpha': config['model'].lora_alpha_per_bit.get(bits, 0)
        }

        torch.save(checkpoint, filename)
        saved_models[bits] = filename
        print(f"Saved {bits}-bit model to {filename}")

    # Save model info summary
    summary_file = os.path.join(output_dir, f"model_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("CPT Model Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Bit widths: {config['model'].bit_widths}\n")
        f.write(f"Cyclic schedule: {config['cpt'].schedule_type}\n")
        f.write(f"LoRA ranks: {config['model'].lora_rank_per_bit}\n")
        f.write("\nSaved models:\n")
        for bits, path in saved_models.items():
            f.write(f"  {bits}-bit: {path}\n")

    print(f"Model summary saved to {summary_file}")
    return saved_models


def load_cpt_checkpoint(filepath: str, model: Optional[CPTModel] = None, device: str = 'cuda'):
    """
    Load CPT checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: Optional model to load weights into
        device: Device to load model to

    Returns:
        Dictionary with loaded checkpoint data
    """
    checkpoint = torch.load(filepath, map_location=device)

    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model weights loaded from {filepath}")

    return checkpoint


def convert_to_int8(model: CPTModel, target_bits: int = 8):
    """
    Convert CPT model to INT8 format for deployment.
    Compatible with Part 3 evaluation expectations.
    """
    model.eval()
    model.set_precision(target_bits)

    int8_state_dict = {}

    with torch.no_grad():
        for name, module in model.named_modules():
            # Process CPTLinear layers
            if hasattr(module, 'linear') and hasattr(module, 'weight_quantizers'):
                # Get weight from base linear layer
                weight = module.linear.weight.data

                # Get quantizer for target precision
                quantizer = module.weight_quantizers.get_quantizer(target_bits)

                # Quantize weight
                if hasattr(quantizer, 'calibrate_per_channel'):
                    scale, zero_point = quantizer.calibrate_per_channel(weight)
                    weight_quant = quantizer.quantize(weight, scale, zero_point)
                else:
                    weight_quant = quantizer(weight)

                # Convert to INT8
                weight_int8 = torch.round(weight_quant * 127 / weight.abs().max()).clamp(-128, 127).to(torch.int8)

                # Store quantized weights
                prefix = f"{name}." if name else ""
                int8_state_dict[f"{prefix}weight_int8"] = weight_int8.cpu()
                int8_state_dict[f"{prefix}scale"] = (weight.abs().max() / 127).cpu()
                int8_state_dict[f"{prefix}zero_point"] = torch.tensor(0, dtype=torch.int32)

                # Store bias if present
                if module.linear.bias is not None:
                    int8_state_dict[f"{prefix}bias"] = module.linear.bias.data.cpu()

                # Store LoRA parameters for target precision
                lora_key = f'lora_{target_bits}bit'
                if lora_key in module.lora_adapters:
                    lora = module.lora_adapters[lora_key]
                    if lora.lora_A is not None:
                        int8_state_dict[f"{prefix}lora.A"] = lora.lora_A.data.cpu()
                        int8_state_dict[f"{prefix}lora.B"] = lora.lora_B.data.cpu()
                        int8_state_dict[f"{prefix}lora.scaling"] = torch.tensor(lora.scaling)

    return int8_state_dict


def save_int8_checkpoint(model: CPTModel, filepath: str, target_bits: int = 8, config: Optional[dict] = None):
    """
    Save model in INT8 format for efficient deployment.
    """
    # Convert to INT8
    int8_state_dict = convert_to_int8(model, target_bits)

    # Calculate model sizes
    fp32_params = sum(p.numel() for p in model.parameters())
    fp32_size_mb = fp32_params * 4 / (1024 * 1024)

    int8_params = sum(
        tensor.numel() for key, tensor in int8_state_dict.items()
        if 'int8' in key
    )
    int8_size_mb = int8_params / (1024 * 1024)

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
            'target_bits': target_bits
        },
        'bit_width': target_bits  # For Part 3 compatibility
    }

    # Add configuration if provided
    if config is not None:
        checkpoint['model_config'] = config['model'].__dict__
        checkpoint['training_config'] = config['training'].__dict__
        checkpoint['bit_widths'] = config['model'].bit_widths

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


def export_for_inference(model: CPTModel, output_dir: str, config: dict):
    """
    Export CPT models for inference at different precisions.
    Creates optimized models for deployment.
    """
    os.makedirs(output_dir, exist_ok=True)
    exported_models = {}

    for bits in config['model'].bit_widths:
        # Skip FP32 (no quantization needed)
        if bits == 32:
            continue

        # Set precision
        model.set_precision(bits)
        model.eval()

        # Export INT8 version
        int8_path = os.path.join(output_dir, f"cpt_model_{bits}bit_int8.pth")
        save_int8_checkpoint(model, int8_path, bits, config)

        exported_models[bits] = int8_path

    print(f"Exported {len(exported_models)} models for inference")
    return exported_models
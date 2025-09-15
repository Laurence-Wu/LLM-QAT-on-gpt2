"""
Deployment utilities for converting QAT models to INT8 format.
"""

import torch
import torch.nn as nn


def convert_to_int8(model):
    """
    Convert QAT-trained model weights to INT8 format.

    Args:
        model: QAT-trained model with FP32 weights

    Returns:
        Dictionary containing INT8 weights and quantization parameters
    """
    model.eval()
    int8_state_dict = {}

    with torch.no_grad():
        for name, module in model.named_modules():
            # Process QATLinearWithLoRA layers
            if isinstance(module, nn.Module) and 'Linear' in module.__class__.__name__:
                weight = module.linear.weight.data
                weight_quantizer = module.quantize_weight

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
                int8_state_dict[f"{prefix}scale"] = torch.tensor(scale, dtype=torch.float32).cpu()
                int8_state_dict[f"{prefix}zero_point"] = zero_point.cpu()

                # Store bias if present
                if module.linear.bias is not None:
                    int8_state_dict[f"{prefix}bias"] = module.linear.bias.data.cpu()

                # Store LoRA parameters (keep as FP32)
                if module.lora is not None:
                    int8_state_dict[f"{prefix}lora.A"] = module.lora.lora_A.data.cpu()
                    int8_state_dict[f"{prefix}lora.B"] = module.lora.lora_B.data.cpu()
                    int8_state_dict[f"{prefix}lora.scaling"] = torch.tensor(module.lora.scaling).cpu()

    return int8_state_dict


def save_int8_checkpoint(model, filepath, model_config=None, training_config=None):
    """
    Save model in INT8 format with metadata.

    Args:
        model: QAT-trained model
        filepath: Path to save INT8 checkpoint
        model_config: Optional model configuration
        training_config: Optional training configuration

    Returns:
        Saved checkpoint dictionary
    """
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
            'compression_ratio': compression_ratio
        }
    }

    # Add configurations if provided
    if model_config is not None:
        checkpoint['model_config'] = model_config.__dict__

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


def load_int8_checkpoint(model, filepath):
    """
    Load INT8 weights into a model for inference.

    Args:
        model: Model instance to load weights into
        filepath: Path to INT8 checkpoint

    Returns:
        Model with loaded INT8 weights (dequantized to FP32 for inference)
    """
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location='cpu')
    int8_state_dict = checkpoint['int8_state_dict']

    # Load weights into model
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Module) and 'Linear' in module.__class__.__name__:
                prefix = f"{name}." if name else ""
                weight_key = f"{prefix}weight_int8"

                if weight_key in int8_state_dict:
                    # Get INT8 weights and quantization parameters
                    weight_int8 = int8_state_dict[weight_key]
                    scale = int8_state_dict[f"{prefix}scale"]
                    zero_point = int8_state_dict[f"{prefix}zero_point"]

                    # Dequantize to FP32
                    if zero_point.item() == 0:  # Symmetric
                        weight_fp32 = weight_int8.float() * scale
                    else:  # Asymmetric
                        weight_fp32 = (weight_int8.float() - zero_point.float()) * scale

                    # Load into model
                    module.linear.weight.data = weight_fp32.to(module.linear.weight.device)

                    # Load bias if present
                    bias_key = f"{prefix}bias"
                    if bias_key in int8_state_dict:
                        module.linear.bias.data = int8_state_dict[bias_key].to(module.linear.bias.device)

                    # Load LoRA parameters if present
                    if module.lora is not None:
                        lora_a_key = f"{prefix}lora.A"
                        if lora_a_key in int8_state_dict:
                            module.lora.lora_A.data = int8_state_dict[lora_a_key].to(module.lora.lora_A.device)
                            module.lora.lora_B.data = int8_state_dict[f"{prefix}lora.B"].to(module.lora.lora_B.device)
                            module.lora.scaling = int8_state_dict[f"{prefix}lora.scaling"].item()

    print(f"Loaded INT8 model from {filepath}")
    print(f"Model info: {checkpoint['model_info']}")

    return model
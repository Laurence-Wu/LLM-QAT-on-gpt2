"""
Deployment utilities for converting quantized models to INT8 format.
"""

import torch
import torch.nn as nn


def convert_to_int8(model):
    """Convert QAT model weights to INT8 format."""
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

                # Get quantizer
                if hasattr(module, 'quantize_weight'):
                    weight_quantizer = module.quantize_weight
                elif hasattr(module, 'quantizers_weight'):
                    current_bits = getattr(module, 'current_bits', 8)
                    weight_quantizer = module.quantizers_weight[f'{current_bits}bit']
                else:
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
    """Save model in INT8 format."""
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

    # Add configs
    if model_config:
        checkpoint['model_config'] = model_config.__dict__
        checkpoint['bit_widths'] = getattr(model_config, 'bit_widths', None)
    if training_config:
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
    """Save SP model checkpoints for all configured bit widths."""
    import os, time, traceback

    bit_widths = getattr(model_config, 'bit_widths', [6, 8, 16, 32])
    saved_checkpoints = {}
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    print(f"\nSaving SP checkpoints for bit widths: {bit_widths}")

    for bits in bit_widths:
        if bits == 32:
            # Skip 32-bit models as requested (not needed for quantized deployment)
            print(f"\nSkipping 32-bit model (not needed for quantized deployment)")
            continue

        print(f"\nProcessing {bits}-bit model...")
        model.set_precision(bits)
        state_dict = model.state_dict()
        filename = f"{base_filename}_{bits}bit_FP32_{timestamp}.pth"

        checkpoint = {
            'model_state_dict': state_dict,
            'model_config': model_config.__dict__,
            'training_config': training_config.__dict__ if training_config else None,
            'bit_width': bits,  # Integer for Part 3 compatibility
            'timestamp': timestamp
        }

        # Save with retry logic
        for attempt in range(3):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                torch.save(checkpoint, filename, pickle_protocol=4)
                time.sleep(0.5)  # Ensure write completion

                # Verify save
                test_load = torch.load(filename, map_location='cpu', weights_only=False)
                assert test_load['bit_width'] == bits

                file_size = os.path.getsize(filename) / (1024*1024)
                print(f"✅ Saved {bits}-bit model: {filename} ({file_size:.1f} MB)")
                saved_checkpoints[bits] = filename
                break
            except Exception as e:
                if attempt < 2:
                    print(f"Retry {attempt+1}: {str(e)}")
                    if os.path.exists(filename):
                        try: os.remove(filename)
                        except: pass
                    time.sleep(1.0)
                else:
                    print(f"❌ Failed to save {bits}-bit model: {str(e)}")
                    if os.path.exists(filename):
                        try: os.remove(filename)
                        except: pass

    print(f"\nSaved {len(saved_checkpoints)}/{len([b for b in bit_widths if b != 32])} checkpoints")
    return saved_checkpoints


def load_model_for_evaluation(checkpoint_path, config=None, target_bits=None, device='cuda'):
    """
    Load SP model for evaluation with per-tensor quantization.

    Args:
        checkpoint_path: Path to saved checkpoint
        config: Model config (will be loaded from checkpoint if not provided)
        target_bits: Target bit width for evaluation
        device: Device to load model on

    Returns:
        model: Loaded model ready for evaluation with per-tensor quantization
    """
    import torch
    from transformers import GPT2Config

    # Import model classes
    try:
        from models_sp import SPLMHeadModel
        from config_sp import ModelConfig
    except ImportError:
        from .models_sp import SPLMHeadModel
        from .config_sp import ModelConfig

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config from checkpoint if not provided
    if config is None:
        if 'model_config' in checkpoint:
            config = ModelConfig()
            # Load saved config attributes
            for key, value in checkpoint['model_config'].items():
                setattr(config, key, value)
        else:
            raise ValueError("No config provided and checkpoint doesn't contain model_config")

    # Override per_channel_quantization for evaluation
    config.per_channel_quantization = False  # Use per-tensor for evaluation

    # Determine target bits
    if target_bits is None:
        if 'bit_width' in checkpoint:
            target_bits = checkpoint['bit_width']
        elif 'target_bits' in checkpoint.get('metadata', {}):
            target_bits = checkpoint['metadata']['target_bits']
        else:
            target_bits = 16  # Default to 16-bit

    print(f"Setting up model for {target_bits}-bit evaluation with per-tensor quantization")

    # Create GPT2 config
    gpt2_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        layer_norm_epsilon=config.layer_norm_epsilon,
        use_cache=False,
        bos_token_id=50256,
        eos_token_id=50256,
    )

    # Add switchable precision configs
    gpt2_config.lora_rank_per_bit = config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = config.lora_alpha_per_bit
    gpt2_config.quantizer_per_bit = config.quantizer_per_bit
    gpt2_config.bit_widths = config.bit_widths
    gpt2_config.per_channel_quantization = False  # Ensure per-tensor for evaluation

    # Create model
    model = SPLMHeadModel(gpt2_config)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Set to target precision
    model.set_precision(target_bits)

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully for {target_bits}-bit evaluation with per-tensor quantization")
    return model

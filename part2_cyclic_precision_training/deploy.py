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
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    global_cycle: int,
    loss: float,
    config: dict,
    filepath: str
):
    """
    Save CPT model checkpoint compatible with Part 3 evaluation.

    Args:
        model: CPT model
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        epoch: Current epoch
        global_cycle: Global training cycle
        loss: Current loss
        config: Configuration dictionary
        filepath: Path to save checkpoint
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),  # Save LR scheduler state
        'epoch': epoch,
        'global_cycle': global_cycle,
        'loss': loss,
        'model_config': config['model'].__dict__,
        'training_config': config['training'].__dict__,
        'cpt_config': config['cpt'].__dict__,
        'bit_widths': config['model'].bit_widths,  # Important for Part 3
        'timestamp': time.strftime('%Y%m%d_%H%M%S')
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def save_target_model(model: CPTModel, config: dict, target_bits: int, output_dir: str):
    """
    Save CPT model at target precision only (filtered state_dict).

    Args:
        model: CPT model
        config: Configuration dictionary
        target_bits: Target precision to save (e.g., 6 for 6-bit)
        output_dir: Directory to save the model
    """
    import traceback
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    print(f"\n{'='*60}")
    print(f"Saving CPT Model at Target Precision")
    print(f"{'='*60}")
    print(f"Target precision: {target_bits}-bit")

    model.set_precision(target_bits)
    state_dict = model.state_dict()

    # Filter: Keep only target precision quantizers and LoRA
    filtered_state_dict = {}
    target_key = f'{target_bits}bit'

    for key, value in state_dict.items():
        filtered_state_dict[key] = value
        # # Keep base weights/biases
        # if 'linear.weight' in key or 'linear.bias' in key:
        #     filtered_state_dict[key] = value
        # # Keep embeddings, layer norms
        # elif any(x in key for x in ['wte', 'wpe', 'ln_', 'lm_head']):
        #     filtered_state_dict[key] = value
        # # Keep ONLY target precision quantizers and LoRA
        # elif target_key in key:
        #     filtered_state_dict[key] = value
        # # Keep gradient quantizers (8-bit BW)
        # elif 'grad_quantizer_8bit' in key:
        #     filtered_state_dict[key] = value

    print(f"Original: {len(state_dict)} tensors")
    print(f"Filtered: {len(filtered_state_dict)} tensors ({100*len(filtered_state_dict)/len(state_dict):.1f}%)")

    state_dict_size = sum(p.numel() * p.element_size() for p in filtered_state_dict.values())
    print(f"Size: {state_dict_size / (1024*1024):.2f} MB")

    # Create filename
    filename = os.path.join(output_dir, f"cpt_model_{target_bits}bit_target_{timestamp}.pth")
    print(f"Saving to: {filename}")

    # Save checkpoint with filtered state_dict
    checkpoint = {
        'model_state_dict': filtered_state_dict,  # Filtered!
        'model_config': config['model'].__dict__,
        'training_config': config['training'].__dict__,
        'cpt_config': config['cpt'].__dict__,
        'bit_width': target_bits,
        'target_precision': target_bits,
        'timestamp': timestamp,
        'lora_rank': config['model'].shared_lora_rank,
        'lora_alpha': config['model'].shared_lora_alpha,
        'checkpoint_version': '1.2',
        'pytorch_version': torch.__version__,
        'model_type': 'CPT_TARGET'
    }

    # Save with error handling
    max_retries = 3
    retry_count = 0
    save_successful = False

    while retry_count < max_retries and not save_successful:
        try:
            # Clear any GPU cache before saving
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Save checkpoint
            torch.save(checkpoint, filename, pickle_protocol=4)

            # Verify saved file
            time.sleep(0.5)
            file_size = os.path.getsize(filename)
            print(f"File saved, size: {file_size / (1024*1024):.2f} MB")

            # Verify integrity
            print("Verifying checkpoint integrity...")
            test_load = torch.load(filename, map_location='cpu', weights_only=False)

            # Check critical fields
            assert 'model_state_dict' in test_load, "Missing model_state_dict"
            assert 'bit_width' in test_load, "Missing bit_width"
            assert test_load['bit_width'] == target_bits, f"Bit width mismatch: {test_load['bit_width']} != {target_bits}"

            save_successful = True
            print(f"✅ Successfully saved {target_bits}-bit target model")

            del test_load

        except Exception as e:
            retry_count += 1
            print(f"⚠️ Attempt {retry_count} failed: {str(e)}")
            if retry_count < max_retries:
                print(f"Retrying... ({retry_count}/{max_retries})")
                time.sleep(1.0)
            else:
                print(f"❌ ERROR saving {target_bits}-bit model after {max_retries} attempts")
                traceback.print_exc()
                # Try to remove corrupted file
                if os.path.exists(filename):
                    try:
                        os.remove(filename)
                        print(f"Removed corrupted file: {filename}")
                    except:
                        print(f"WARNING: Could not remove corrupted file: {filename}")

    print(f"{'='*60}\n")

    if save_successful:
        # Save summary file
        summary_file = os.path.join(output_dir, f"cpt_target_model_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write("CPT Target Model Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Target Precision: {target_bits}-bit\n")
            f.write(f"Model Type: Cyclic Precision Training (CPT)\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model Path: {filename}\n")
            f.write(f"File Size: {file_size / (1024*1024):.2f} MB\n")
            f.write(f"Total Parameters: {sum(p.numel() for p in state_dict.values()):,}\n")
            f.write(f"LoRA Rank: {config['model'].shared_lora_rank}\n")
            f.write(f"LoRA Alpha: {config['model'].shared_lora_alpha}\n")
            f.write(f"Training Config:\n")
            f.write(f"  - Learning Rate: {config['training'].learning_rate}\n")
            f.write(f"  - Batch Size: {config['training'].batch_size}\n")
            f.write(f"  - Num Epochs: {config['training'].num_epochs}\n")
            f.write(f"CPT Config:\n")
            f.write(f"  - Cycle Length: {config['cpt'].total_cycles}\n")
            f.write(f"  - Schedule Type: {config['cpt'].schedule_type}\n")
            f.write("=" * 50 + "\n")
        print(f"Summary saved to: {summary_file}")

        return filename
    else:
        return None


def save_final_models(model: CPTModel, config: dict, output_dir: str):
    """
    Save final models at each precision level for Part 3 evaluation.
    CRITICAL: Saves bit_width as integer (not string) for Part 3 compatibility.
    """
    import traceback
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    saved_models = {}
    print(f"\n{'='*60}")
    print(f"Saving CPT Model Checkpoints")
    print(f"{'='*60}")
    print(f"Configured bit widths: {config['model'].bit_widths}")

    for bits in config['model'].bit_widths:
        if bits == 32:
            # Skip 32-bit models as not needed for quantized deployment
            print(f"\nSkipping 32-bit model (not needed for quantized deployment)")
            continue

        print(f"\n{'='*40}")
        print(f"Processing {bits}-bit model...")

        # Set model to specific precision
        model.set_precision(bits)
        state_dict = model.state_dict()

        # Debug: Print state dict size
        state_dict_size = sum(p.numel() * p.element_size() for p in state_dict.values())
        print(f"State dict size: {state_dict_size / (1024*1024):.2f} MB")
        print(f"Number of parameters: {sum(p.numel() for p in state_dict.values()):,}")

        # Create filename
        filename = os.path.join(output_dir, f"cpt_model_{bits}bit_{timestamp}.pth")
        print(f"Saving to: {filename}")

        # Save checkpoint with integer bit_width (NOT string)
        checkpoint = {
            'model_state_dict': state_dict,
            'model_config': config['model'].__dict__,
            'training_config': config['training'].__dict__,
            'bit_width': bits,  # CRITICAL: Save as integer, not f"{bits}bit"
            'timestamp': timestamp,
            'lora_rank': config['model'].shared_lora_rank,
            'lora_alpha': config['model'].shared_lora_alpha,
            'checkpoint_version': '1.1',  # Version tracking
            'pytorch_version': torch.__version__,
            'save_complete': False  # Flag to verify complete save
        }

        # Save with error handling and verification
        max_retries = 3
        retry_count = 0
        save_successful = False

        while retry_count < max_retries and not save_successful:
            try:
                # Clear any GPU cache before saving
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Save checkpoint with pickle protocol 4 for better compatibility
                torch.save(checkpoint, filename, pickle_protocol=4)

                # Small delay to ensure write completion
                time.sleep(0.5)

                # Verify saved file
                file_size = os.path.getsize(filename)
                print(f"File saved, size: {file_size / (1024*1024):.2f} MB")

                # Try to reload to verify integrity
                print("Verifying checkpoint integrity...")
                # Use weights_only=False for verification since we store torch.__version__
                test_load = torch.load(filename, map_location='cpu', weights_only=False)
                save_successful = True

                # Check critical fields
                assert 'model_state_dict' in test_load, "Missing model_state_dict"
                assert 'bit_width' in test_load, "Missing bit_width"
                assert test_load['bit_width'] == bits, f"Bit width mismatch: {test_load['bit_width']} vs {bits}"

                # Update save_complete flag
                checkpoint['save_complete'] = True
                torch.save(checkpoint, filename, pickle_protocol=4)

                print(f"✅ Verification passed for {bits}-bit model")
                saved_models[bits] = filename

            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"⚠️ Attempt {retry_count} failed: {str(e)}")
                    print(f"Retrying... ({retry_count}/{max_retries})")
                    # Try to remove potentially corrupted file
                    if os.path.exists(filename):
                        try:
                            os.remove(filename)
                        except:
                            pass
                    # Wait before retry
                    time.sleep(1.0)
                else:
                    print(f"❌ ERROR saving {bits}-bit model after {max_retries} attempts: {str(e)}")
                    traceback.print_exc()
                    # Try to remove corrupted file
                    if os.path.exists(filename):
                        try:
                            os.remove(filename)
                            print(f"Removed corrupted file: {filename}")
                        except:
                            print(f"WARNING: Could not remove corrupted file: {filename}")
                    break

        if not save_successful:
            continue

    print(f"\n{'='*60}")
    if saved_models:
        print(f"Successfully saved {len(saved_models)} checkpoint(s)")
        for bits, path in saved_models.items():
            print(f"  {bits}-bit: {path}")
    else:
        print("WARNING: No checkpoints were saved successfully!")
    print(f"{'='*60}\n")

    # Save model info summary
    summary_file = os.path.join(output_dir, f"model_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("CPT Model Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Bit widths: {config['model'].bit_widths}\n")
        f.write(f"Cyclic schedule: {config['cpt'].schedule_type}\n")
        f.write(f"Shared LoRA rank: {config['model'].shared_lora_rank}\n")
        f.write(f"Shared LoRA alpha: {config['model'].shared_lora_alpha}\n")
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

                # Store shared LoRA parameters
                if hasattr(module, 'shared_lora') and module.shared_lora is not None:
                    lora = module.shared_lora
                    if lora.lora_A is not None:
                        int8_state_dict[f"{prefix}lora.A"] = lora.lora_A.data.cpu()
                        int8_state_dict[f"{prefix}lora.B"] = lora.lora_B.data.cpu()
                        int8_state_dict[f"{prefix}lora.scaling"] = torch.tensor(lora.scaling)

    return int8_state_dict


def save_int8_checkpoint(model: CPTModel, filepath: str, target_bits: int = 8, config: Optional[dict] = None):
    """
    Save model in INT8 format for efficient deployment.
    """
    print(f"\nConverting {target_bits}-bit model to INT8 format...")

    # Convert to INT8
    int8_state_dict = convert_to_int8(model, target_bits)
    print(f"Converted {len(int8_state_dict)} tensors to INT8")

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

    # Save checkpoint with verification
    try:
        torch.save(checkpoint, filepath)

        # Verify saved file
        file_size = os.path.getsize(filepath)
        print(f"INT8 file saved, actual size: {file_size / (1024*1024):.2f} MB")

        # Verify integrity
        # Use weights_only=False for verification
        test_load = torch.load(filepath, map_location='cpu', weights_only=False)
        assert 'int8_state_dict' in test_load, "Missing int8_state_dict"
        assert 'bit_width' in test_load, "Missing bit_width"
        print("✅ INT8 checkpoint verification passed")

    except Exception as e:
        print(f"❌ ERROR saving INT8 checkpoint: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        raise

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

    print(f"\n{'='*60}")
    print(f"Exporting Models for Inference")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Bit widths to export: {[b for b in config['model'].bit_widths if b != 32]}")

    for bits in config['model'].bit_widths:
        # Skip FP32 (no quantization needed)
        if bits == 32:
            print(f"\nSkipping 32-bit model (no quantization needed)")
            continue

        print(f"\n{'='*40}")
        print(f"Exporting {bits}-bit model...")

        # Set precision
        model.set_precision(bits)
        model.eval()

        # Export INT8 version
        int8_path = os.path.join(output_dir, f"cpt_model_{bits}bit_int8.pth")
        try:
            save_int8_checkpoint(model, int8_path, bits, config)
            exported_models[bits] = int8_path
            print(f"✅ Successfully exported {bits}-bit model")
        except Exception as e:
            print(f"❌ Failed to export {bits}-bit model: {str(e)}")
            continue

    print(f"\n{'='*60}")
    if exported_models:
        print(f"Successfully exported {len(exported_models)} model(s) for inference:")
        for bits, path in exported_models.items():
            print(f"  {bits}-bit: {path}")
    else:
        print("WARNING: No models were exported successfully!")
    print(f"{'='*60}\n")

    return exported_models
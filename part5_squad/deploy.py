import torch
import os
from datetime import datetime
from typing import Dict, Optional


def save_squad_checkpoints(model, base_filename="squad_gpt2",
                           model_config=None, training_config=None,
                           output_dir="checkpoints"):
    """
    Save SQuAD QA model checkpoints at different bit-widths

    Creates separate checkpoint files for each precision level.
    Each checkpoint contains the full model but configured for a specific bit-width.

    Args:
        model: SPQuestionAnsweringModel
        base_filename: Base name for checkpoint files
        model_config: ModelConfig object
        training_config: TrainingConfig object
        output_dir: Directory to save checkpoints

    Returns:
        Dict mapping bit_width -> checkpoint_path
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    saved_checkpoints = {}

    # Get available bit-widths from model config
    bit_widths = model_config.bit_widths if model_config else [7, 32]

    print(f"\nSaving checkpoints to {output_dir}/")

    for bits in bit_widths:
        # Set model to this precision
        model.set_precision(bits)

        # Create checkpoint
        checkpoint = {
            'bit_width': bits,
            'model_state_dict': model.state_dict(),
            'timestamp': timestamp
        }

        # Add configs if provided
        if model_config:
            model_config_dict = {}
            for attr_name in dir(model_config):
                if not attr_name.startswith('_'):
                    attr_value = getattr(model_config, attr_name)
                    if not callable(attr_value):
                        model_config_dict[attr_name] = attr_value
            checkpoint['model_config'] = model_config_dict

        if training_config:
            training_config_dict = {}
            for attr_name in dir(training_config):
                if not attr_name.startswith('_'):
                    attr_value = getattr(training_config, attr_name)
                    if not callable(attr_value):
                        training_config_dict[attr_name] = attr_value
            checkpoint['training_config'] = training_config_dict

        # Save checkpoint
        checkpoint_path = os.path.join(
            output_dir,
            f"{base_filename}_{bits}bit_{timestamp}.pth"
        )

        torch.save(checkpoint, checkpoint_path)

        saved_checkpoints[bits] = checkpoint_path

        # Print checkpoint info
        checkpoint_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"  ✓ Saved {bits}-bit checkpoint ({checkpoint_size_mb:.2f} MB): {checkpoint_path}")

    return saved_checkpoints


def load_squad_checkpoint(checkpoint_path, model, device='cuda'):
    """
    Load SQuAD QA model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        model: SPQuestionAnsweringModel (initialized)
        device: Device to load model on

    Returns:
        (model, checkpoint_info)
    """
    print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to checkpoint's precision
    if 'bit_width' in checkpoint:
        model.set_precision(checkpoint['bit_width'])
        print(f"  Model set to {checkpoint['bit_width']}-bit precision")

    # Return checkpoint info
    checkpoint_info = {
        'bit_width': checkpoint.get('bit_width'),
        'timestamp': checkpoint.get('timestamp'),
        'model_config': checkpoint.get('model_config'),
        'training_config': checkpoint.get('training_config')
    }

    print("  ✓ Checkpoint loaded successfully")

    return model, checkpoint_info

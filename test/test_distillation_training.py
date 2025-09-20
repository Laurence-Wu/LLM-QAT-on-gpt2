"""
Test distillation-based training with 32-bit teacher and multiple student precisions.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from part1_switchable_precision.config_sp import ModelConfig, TrainingConfig
from shared.models_sp import SPLMHeadModel
from transformers import GPT2Config, GPT2TokenizerFast
from shared.dataset import create_dataloaders


def test_distillation_training():
    """Test that distillation-based training works correctly."""
    print("Testing distillation-based training...")

    # Setup configs
    model_config = ModelConfig()
    training_config = TrainingConfig()

    # Reduce model size for testing
    model_config.n_layer = 2
    model_config.n_embd = 256
    model_config.n_head = 4

    # Reduce training iterations
    training_config.num_iterations = 5
    training_config.gradient_accumulation_steps = 2
    training_config.batch_size = 2

    # Print configuration
    print(f"Teacher bits: {model_config.teacher_bits}")
    print(f"Student bits: {model_config.bit_widths}")
    print(f"Using distillation: {training_config.use_distillation}")

    # Create model
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        layer_norm_epsilon=model_config.layer_norm_epsilon,
        embd_pdrop=model_config.embd_pdrop,
        bit_widths=model_config.bit_widths,
        lora_rank_per_bit=model_config.lora_rank_per_bit,
        lora_alpha_per_bit=model_config.lora_alpha_per_bit,
        activation_bits_per_bit=model_config.activation_bits_per_bit
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SPLMHeadModel(gpt2_config).to(device)

    # Test setting different precisions
    print("\nTesting precision switching...")

    # Test teacher precision (32-bit)
    try:
        model.set_precision(32)
        print("✓ Successfully set 32-bit (FP32) teacher precision")
    except Exception as e:
        print(f"✗ Failed to set 32-bit precision: {e}")
        return False

    # Test student precisions
    for bits in model_config.bit_widths:
        try:
            model.set_precision(bits)
            print(f"✓ Successfully set {bits}-bit student precision")
        except Exception as e:
            print(f"✗ Failed to set {bits}-bit precision: {e}")
            return False

    # Create tokenizer and dataloaders
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader = create_dataloaders(
        tokenizer,
        train_split='train[:100]',  # Small dataset for testing
        val_split='validation[:20]',
        batch_size=training_config.batch_size,
        max_length=128  # Shorter sequences for testing
    )

    # Import training function
    from part1_switchable_precision.train_sp import (
        train_step, CalibrationManager, DistillationManager,
        setup_optimizer, compute_loss_for_all_students
    )
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # Setup training components
    calib_mgr = CalibrationManager(model, train_loader, device)
    calib_mgr.calibrate_all_precisions(model_config.bit_widths)

    # Initialize distillation manager
    distill_mgr = DistillationManager(
        model=model,
        full_precision_bits=model_config.teacher_bits,
        config=training_config
    )

    # Setup optimizer
    optimizer = setup_optimizer(model, training_config)
    scheduler = CosineAnnealingLR(optimizer, T_max=training_config.num_iterations)
    scaler = torch.amp.GradScaler('cuda') if training_config.use_amp else None

    # Test a single training step
    print("\nTesting training step...")
    train_iter = iter(train_loader)

    try:
        # Get a batch
        batch = next(train_iter)

        # Test loss computation for all students
        print("Testing loss computation for teacher and students...")
        loss = compute_loss_for_all_students(
            model, batch,
            model_config.teacher_bits,
            model_config.bit_widths,
            distill_mgr, training_config, 0
        )
        print(f"✓ Loss computed successfully: {loss.item():.4f}")

        # Test full training step
        print("Testing full training step...")
        total_loss = train_step(
            model, train_iter, train_loader, optimizer, scaler,
            model_config.teacher_bits, model_config.bit_widths,
            distill_mgr, training_config, 0
        )
        print(f"✓ Training step completed successfully. Total loss: {total_loss:.4f}")

    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    success = test_distillation_training()
    exit(0 if success else 1)
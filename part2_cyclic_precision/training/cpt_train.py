"""
CPT Training Script
Main training loop for Cyclic Precision Training.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from typing import Dict, Optional, Any
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import CPT components
from cpt_scheduler import CPTScheduler
from cpt_calibrate import calibrate, selective_calibrate
from cpt_prt import CPT_PRT


def train_cpt(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: Optional[torch.utils.data.DataLoader],
    config: Dict[str, Any],
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Main training function for CPT.

    Args:
        model: The CPT model to train
        train_dataloader: DataLoader for training data
        val_dataloader: Optional DataLoader for validation
        config: Configuration dictionary
        device: Device to train on

    Returns:
        Dictionary containing training metrics and history
    """
    print("\n" + "="*60)
    print("Starting Cyclic Precision Training (CPT)")
    print("="*60)

    # Extract config parameters
    training_config = config['training']
    if training_config['Bmin'] is None or training_config['Bmax'] is None:
        raise ValueError("Bmin and Bmax must be specified in config (no defaults allowed)")

    Bmin = training_config['Bmin']
    Bmax = training_config['Bmax']
    total_epochs = training_config['total_epochs']
    num_cycles = training_config['num_cycles']
    learning_rate = training_config['learning_rate']
    gradient_bits = training_config.get('gradient_bits', 8)
    batch_size = training_config['batch_size']

    # PRT configuration
    prt_config = config.get('prt', {})
    use_prt = prt_config.get('use_prt', False)

    # Calibration configuration
    calib_config = config.get('calibration', {})
    calib_batches = calib_config.get('calib_batches', 10)
    recalib_threshold = calib_config.get('recalib_threshold', 1)

    # Run PRT if enabled to find optimal Bmin
    if use_prt:
        print("\nRunning Precision Range Test (PRT)...")
        Bmin_prt, _ = CPT_PRT(
            model=model,
            dataloader=train_dataloader,
            calibrate_fn=lambda m, d: calibrate(m, d, calib_batches, device),
            threshold=prt_config.get('threshold', 0.01),
            test_iterations=prt_config.get('test_iterations', 100),
            min_bits=2,
            max_bits=Bmax,
            device=device
        )
        print(f"PRT recommends Bmin={Bmin_prt}")
        if Bmin_prt > Bmin:
            print(f"Updating Bmin from {Bmin} to {Bmin_prt}")
            Bmin = Bmin_prt

    # Initialize scheduler
    scheduler = CPTScheduler(Bmin, Bmax, total_epochs, num_cycles)

    # Print schedule statistics
    stats = scheduler.get_stats()
    print(f"\nSchedule Statistics:")
    print(f"  Unique precisions: {stats['unique_precisions']}")
    print(f"  Average precision: {stats['avg_precision']:.2f} bits")

    # Move model to device
    model = model.to(device)

    # Initialize optimizer (single optimizer for all precisions)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Learning rate scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=learning_rate * 0.1)

    # Training history
    history = {
        'train_loss': [],
        'train_ppl': [],
        'val_loss': [],
        'val_ppl': [],
        'precision': [],
        'calibrations': [],
        'epoch_times': []
    }

    # Track last precision for selective calibration
    last_precision = -1

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {total_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Gradient bits: {gradient_bits}")
    print(f"  Calibration batches: {calib_batches}")
    print(f"  Recalibration threshold: {recalib_threshold} bits")

    # Training loop
    for epoch in range(total_epochs):
        epoch_start = time.time()

        # Get precision for current epoch
        precision = scheduler.get_precision(epoch)
        history['precision'].append(precision)

        # Set precision and selectively calibrate
        if precision != last_precision:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{total_epochs} - Precision: {precision} bits")
            print(f"{'='*60}")

            # Set model precision (weights/activations cycle, gradients fixed)
            model.set_precision(precision, precision, gradient_bits)

            # Selective calibration based on precision change
            if abs(precision - last_precision) >= recalib_threshold:
                print(f"Recalibrating (precision changed by {abs(precision - last_precision)} bits)")
                calibrate(model, train_dataloader, calib_batches, device, verbose=False)
                history['calibrations'].append(epoch)
            else:
                print(f"Skipping calibration (change < {recalib_threshold} bits)")

            last_precision = precision
        else:
            print(f"\nEpoch {epoch + 1}/{total_epochs} - Precision: {precision} bits (unchanged)")

        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0

        progress_bar = tqdm(train_dataloader, desc=f"Training (P={precision})")
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            # Track metrics
            train_loss += loss.item()
            train_steps += 1

            # Update progress bar
            avg_loss = train_loss / train_steps
            ppl = torch.exp(torch.tensor(avg_loss)).item()
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'ppl': f'{ppl:.2f}'})

        # Calculate epoch metrics
        avg_train_loss = train_loss / train_steps
        train_ppl = torch.exp(torch.tensor(avg_train_loss)).item()
        history['train_loss'].append(avg_train_loss)
        history['train_ppl'].append(train_ppl)

        # Validation
        if val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            val_steps = 0

            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Validation (P={precision})"):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs

                    val_loss += loss.item()
                    val_steps += 1

            avg_val_loss = val_loss / val_steps
            val_ppl = torch.exp(torch.tensor(avg_val_loss)).item()
            history['val_loss'].append(avg_val_loss)
            history['val_ppl'].append(val_ppl)

            print(f"Epoch {epoch + 1} Results:")
            print(f"  Train Loss: {avg_train_loss:.4f}, PPL: {train_ppl:.2f}")
            print(f"  Val Loss: {avg_val_loss:.4f}, PPL: {val_ppl:.2f}")
        else:
            print(f"Epoch {epoch + 1} Results:")
            print(f"  Train Loss: {avg_train_loss:.4f}, PPL: {train_ppl:.2f}")

        # Step learning rate scheduler
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.6f}")

        # Track epoch time
        epoch_time = time.time() - epoch_start
        history['epoch_times'].append(epoch_time)
        print(f"  Epoch Time: {epoch_time:.2f}s")

        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'history': history,
                'config': config
            }
            checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

    # Final statistics
    print(f"\nFinal Statistics:")
    print(f"  Best Train PPL: {min(history['train_ppl']):.2f}")
    if history['val_ppl']:
        print(f"  Best Val PPL: {min(history['val_ppl']):.2f}")
    print(f"  Total Calibrations: {len(history['calibrations'])}")
    print(f"  Average Epoch Time: {sum(history['epoch_times'])/len(history['epoch_times']):.2f}s")

    return history


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        config_path: Path to config JSON file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Validate required fields
    required_fields = {
        'training': ['Bmin', 'Bmax', 'total_epochs', 'num_cycles',
                    'learning_rate', 'batch_size']
    }

    for section, fields in required_fields.items():
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
        for field in fields:
            if field not in config[section]:
                raise ValueError(f"Missing config field: {section}.{field}")
            if config[section][field] is None:
                raise ValueError(f"Config field cannot be None: {section}.{field}")

    return config


def main():
    """Main entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description='CPT Training')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config JSON file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create dummy model and data for testing
    print("Creating model and data...")
    # This would be replaced with actual model/data loading
    from models.cpt_model import CPTLMHeadModel
    from utils.dataset import create_dataloaders

    # Initialize model (would load from config)
    # model = CPTLMHeadModel(config['model'])

    # Create dataloaders (would load from config)
    # train_dataloader, val_dataloader = create_dataloaders(config['data'])

    # Train
    # history = train_cpt(model, train_dataloader, val_dataloader, config, args.device)

    print("Training script ready. Uncomment model/data loading to run.")


if __name__ == "__main__":
    main()
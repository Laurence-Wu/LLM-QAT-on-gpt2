"""
Calibration utilities for CPT
Handles calibration of quantizers when precision changes.
"""

import torch
import torch.nn as nn
from typing import Optional, List
from tqdm import tqdm


def start_calibration(model: nn.Module) -> None:
    """Start calibration mode for all quantizers in the model.

    Args:
        model: The model to calibrate
    """
    for name, module in model.named_modules():
        if hasattr(module, 'start_calibration'):
            module.start_calibration()
            print(f"Started calibration for {name}")


def finish_calibration(model: nn.Module) -> None:
    """Finish calibration and compute quantization parameters.

    Args:
        model: The model being calibrated
    """
    for name, module in model.named_modules():
        if hasattr(module, 'finish_calibration'):
            module.finish_calibration()
            print(f"Finished calibration for {name}")


def calibrate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_batches: int = 10,
    device: str = 'cuda',
    verbose: bool = True
) -> None:
    """Calibrate model after precision change.

    This function collects statistics on activations and weights,
    then computes optimal quantization parameters (scale and zero_point).

    Args:
        model: The model to calibrate
        dataloader: DataLoader containing calibration data
        num_batches: Number of batches to use for calibration
        device: Device to run calibration on
        verbose: Whether to show progress bar
    """
    if verbose:
        print(f"Starting calibration with {num_batches} batches...")

    model = model.to(device)
    model.eval()

    # Start calibration mode
    start_calibration(model)

    # Run forward passes to collect statistics
    with torch.no_grad():
        iterator = tqdm(dataloader, total=num_batches, desc="Calibrating") if verbose else dataloader

        for i, batch in enumerate(iterator):
            if i >= num_batches:
                break

            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                # Forward pass
                _ = model(**batch)
            else:
                # Handle tuple/list batches
                batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
                _ = model(*batch)

    # Finish calibration and compute quantization parameters
    finish_calibration(model)

    model.train()
    if verbose:
        print("Calibration complete!")


def selective_calibrate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    current_precision: int,
    last_precision: int,
    num_batches: int = 10,
    recalibration_threshold: int = 1,
    device: str = 'cuda'
) -> bool:
    """Selectively calibrate based on precision change magnitude.

    Only recalibrates if the precision change is significant
    (>= recalibration_threshold bits).

    Args:
        model: The model to calibrate
        dataloader: DataLoader containing calibration data
        current_precision: Current precision setting
        last_precision: Previous precision setting
        num_batches: Number of batches for calibration
        recalibration_threshold: Minimum bit change to trigger recalibration
        device: Device to run on

    Returns:
        True if calibration was performed, False otherwise
    """
    precision_change = abs(current_precision - last_precision)

    if precision_change >= recalibration_threshold:
        print(f"Precision changed by {precision_change} bits ({last_precision} -> {current_precision})")
        print(f"Triggering recalibration (threshold: {recalibration_threshold} bits)")
        calibrate(model, dataloader, num_batches, device)
        return True
    else:
        print(f"Precision change too small ({precision_change} < {recalibration_threshold}), skipping calibration")
        return False


def calibrate_layers(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer_names: Optional[List[str]] = None,
    num_batches: int = 10,
    device: str = 'cuda'
) -> None:
    """Calibrate specific layers of the model.

    Args:
        model: The model to calibrate
        dataloader: DataLoader containing calibration data
        layer_names: Optional list of layer names to calibrate (None = all layers)
        num_batches: Number of batches for calibration
        device: Device to run on
    """
    model = model.to(device)
    model.eval()

    # Start calibration for specified layers
    calibrated_layers = []
    for name, module in model.named_modules():
        if layer_names is None or name in layer_names:
            if hasattr(module, 'start_calibration'):
                module.start_calibration()
                calibrated_layers.append(name)

    if not calibrated_layers:
        print("No layers to calibrate!")
        return

    print(f"Calibrating {len(calibrated_layers)} layers: {calibrated_layers[:3]}...")

    # Run forward passes
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                _ = model(**batch)
            else:
                batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
                _ = model(*batch)

    # Finish calibration for specified layers
    for name, module in model.named_modules():
        if name in calibrated_layers:
            if hasattr(module, 'finish_calibration'):
                module.finish_calibration()

    model.train()
    print(f"Calibration complete for {len(calibrated_layers)} layers")


def get_calibration_stats(model: nn.Module) -> dict:
    """Get calibration statistics from the model.

    Args:
        model: The model to analyze

    Returns:
        Dictionary containing calibration statistics
    """
    stats = {
        'calibrated_layers': [],
        'uncalibrated_layers': [],
        'total_quantizers': 0,
        'calibrated_quantizers': 0
    }

    for name, module in model.named_modules():
        if hasattr(module, 'calibrated'):
            stats['total_quantizers'] += 1
            if getattr(module, 'calibrated', False):
                stats['calibrated_quantizers'] += 1
                stats['calibrated_layers'].append(name)
            else:
                stats['uncalibrated_layers'].append(name)

    stats['calibration_rate'] = stats['calibrated_quantizers'] / max(stats['total_quantizers'], 1)

    return stats


def reset_calibration(model: nn.Module) -> None:
    """Reset calibration state for all quantizers.

    Args:
        model: The model to reset
    """
    for name, module in model.named_modules():
        if hasattr(module, 'reset'):
            module.reset()
            print(f"Reset calibration for {name}")


def test_calibration():
    """Test calibration with dummy model and data."""
    import torch.nn as nn

    class DummyQuantizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.calibrated = False

        def start_calibration(self):
            self.calibrated = False
            print("Starting calibration...")

        def finish_calibration(self):
            self.calibrated = True
            print("Finished calibration!")

        def forward(self, x):
            return x

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)
            self.quantizer = DummyQuantizer()

        def forward(self, x):
            x = self.fc(x)
            x = self.quantizer(x)
            return x

    # Create dummy data
    data = [{'x': torch.randn(4, 10)} for _ in range(20)]
    dataloader = torch.utils.data.DataLoader(data, batch_size=1)

    # Create model
    model = DummyModel()

    # Test calibration
    print("Testing calibration...")
    calibrate(model, dataloader, num_batches=5, device='cpu')

    # Check stats
    stats = get_calibration_stats(model)
    print(f"Calibration stats: {stats}")

    # Test selective calibration
    print("\nTesting selective calibration...")
    calibrated = selective_calibrate(model, dataloader,
                                    current_precision=4,
                                    last_precision=2,
                                    device='cpu')
    print(f"Calibration performed: {calibrated}")


if __name__ == "__main__":
    test_calibration()
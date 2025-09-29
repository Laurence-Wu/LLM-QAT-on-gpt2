"""
Calibration manager for Cyclic Precision Training.
Follows the same calibration strategy as part1.
"""

import torch
import torch.nn as nn
import gc
from typing import List, Set
from tqdm import tqdm


class CalibrationManager:
    """
    Manages calibration of quantizers for all precision levels.
    Based on part1's calibration strategy.
    """

    def __init__(self, model, train_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.calibrated_bits = set()

    def calibrate_all_precisions(self, bit_widths: List[int], num_batches: int = 10):
        """Calibrate all precision levels."""
        for bits in bit_widths:
            if bits < 32 and bits not in self.calibrated_bits:
                print(f"\nCalibrating {bits}-bit precision...")
                self.model.set_precision(bits)
                self._calibrate_precision(bits, num_batches)
                self.calibrated_bits.add(bits)

    def _calibrate_precision(self, bits: int, num_batches: int):
        """
        Calibrate quantizers for a specific precision level.
        Follows part1's approach: weight calibration, then input calibration.
        """
        bits_key = f'{bits}bit'

        if bits >= 32:
            print(f"  Skipping calibration for {bits}-bit (no quantization needed)")
            return

        # Step 1: Calibrate weight quantizers
        weight_calibrated = 0
        weight_errors = []

        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizers_weight') and bits_key in module.quantizers_weight:
                weight_quantizer = module.quantizers_weight[bits_key]

                # Get the weight tensor
                if hasattr(module, 'linear') and hasattr(module.linear, 'weight'):
                    weight = module.linear.weight.data
                elif hasattr(module, 'weight'):
                    weight = module.weight.data
                else:
                    weight_errors.append(f"{name}: No weight tensor found")
                    continue

                try:
                    # Calibrate weight quantizer
                    weight_quantizer.start_calibration()
                    with torch.no_grad():
                        _ = weight_quantizer(weight)
                    weight_quantizer.finish_calibration(debug=False)
                    weight_calibrated += 1
                except Exception as e:
                    weight_errors.append(f"{name}: {str(e)}")

        print(f"  ✓ Calibrated {weight_calibrated} weight quantizers")

        if weight_errors:
            print(f"    ⚠️ {len(weight_errors)} warnings (showing first 3):")
            for err in weight_errors[:3]:
                print(f"      - {err}")

        # Step 2: Start calibration for input quantizers
        input_started = 0
        for name, module in self.model.named_modules():
            # Use consistent naming with part1
            if hasattr(module, 'quantizers_input') and bits_key in module.quantizers_input:
                module.quantizers_input[bits_key].start_calibration()
                input_started += 1

        print(f"    Started calibration for {input_started} input quantizers")

        # Step 3: Disable LoRA during input calibration (like part1)
        self.model.disable_lora_for_calibration()

        # Step 4: Run forward passes to collect statistics
        train_iter = iter(self.train_loader)
        with torch.no_grad():
            for i in range(num_batches):
                try:
                    batch = next(train_iter)
                    input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                    _ = self.model(input_ids)
                    del batch, input_ids
                except StopIteration:
                    print(f"    Only {i} batches available")
                    break

        # Step 5: Re-enable LoRA after calibration
        self.model.enable_lora_after_calibration()

        # Step 6: Finish calibration for input quantizers
        input_calibrated = 0
        for name, module in self.model.named_modules():
            # Use consistent naming with part1
            if hasattr(module, 'quantizers_input') and bits_key in module.quantizers_input:
                module.quantizers_input[bits_key].finish_calibration(debug=False)
                input_calibrated += 1

        print(f"    ✓ Calibrated {input_calibrated} input quantizers")

        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()

    def calibrate_lora_only(self, bits: int, num_batches: int = 10):
        """
        Calibrate only LoRA adapter quantizers.
        Called during training to recalibrate LoRA after updates.
        """
        if bits >= 32:
            return

        bits_key = f'{bits}bit'
        # Use same naming as part1 (without 'lora_' prefix)
        lora_key = f'{bits}bit'

        lora_calibrated = 0
        for name, module in self.model.named_modules():
            if not hasattr(module, 'lora_adapters'):
                continue
            if lora_key not in module.lora_adapters:
                continue

            lora_layer = module.lora_adapters[lora_key]

            # Calibrate LoRA A quantizer if it exists
            if hasattr(lora_layer, 'quantize_A') and hasattr(lora_layer, 'lora_A'):
                try:
                    lora_layer.quantize_A.start_calibration()
                    with torch.no_grad():
                        _ = lora_layer.quantize_A(lora_layer.lora_A)
                    lora_layer.quantize_A.finish_calibration(debug=False)
                    lora_calibrated += 1
                except Exception as e:
                    print(f"    Warning calibrating {name} quantize_A: {e}")

            # Calibrate LoRA B quantizer if it exists
            if hasattr(lora_layer, 'quantize_B') and hasattr(lora_layer, 'lora_B'):
                try:
                    lora_layer.quantize_B.start_calibration()
                    with torch.no_grad():
                        _ = lora_layer.quantize_B(lora_layer.lora_B)
                    lora_layer.quantize_B.finish_calibration(debug=False)
                    lora_calibrated += 1
                except Exception as e:
                    print(f"    Warning calibrating {name} quantize_B: {e}")

        if lora_calibrated > 0:
            torch.cuda.empty_cache()
            gc.collect()

    def ensure_calibrated(self, bits: int):
        """
        Ensure a specific bit width is calibrated.
        If not, calibrate it now.
        """
        if bits >= 32:
            return

        if bits not in self.calibrated_bits:
            print(f"  ⚠️ {bits}-bit not calibrated, calibrating now...")
            self.model.set_precision(bits)
            self._calibrate_precision(bits, num_batches=10)
            self.calibrated_bits.add(bits)

        self._print_calibration_stats(bits)

    def _print_calibration_stats(self, bits: int):
        """Print calibration statistics for debugging."""
        bits_key = f'{bits}bit'

        weight_stats = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizers_weight') and bits_key in module.quantizers_weight:
                q = module.quantizers_weight[bits_key]
                if q.calibrated:
                    weight_stats.append({
                        'name': f"{name}.weight",
                        'scale': q.scale.mean().item() if q.scale.numel() > 1 else q.scale.item(),
                        'zero_point': q.zero_point.mean().item() if q.zero_point.numel() > 1 else q.zero_point.item(),
                        'min': q.running_min.min().item() if hasattr(q, 'running_min') else 0,
                        'max': q.running_max.max().item() if hasattr(q, 'running_max') else 0,
                        'type': q.quantizer_type
                    })

        if weight_stats:
            print(f"  Weight Quantizers ({len(weight_stats)} total):")
            scales = [s['scale'] for s in weight_stats]
            unique_scales = len(set(scales))

            if unique_scales < len(scales):
                print(f"    ⚠️ WARNING: Only {unique_scales} unique scale values out of {len(scales)} quantizers!")

            scale_min = min(scales)
            scale_max = max(scales)
            scale_mean = sum(scales) / len(scales)
            print(f"    Scale range: [{scale_min:.6f}, {scale_max:.6f}], mean: {scale_mean:.6f}")

            # Check for duplicate scales (potential issue)
            from collections import Counter
            scale_counts = Counter(scales)
            duplicates = [(scale, count) for scale, count in scale_counts.items() if count > 1]
            if duplicates:
                print(f"    Duplicate scales found:")
                for scale, count in duplicates[:3]:
                    print(f"      Scale {scale:.6f}: {count} quantizers")
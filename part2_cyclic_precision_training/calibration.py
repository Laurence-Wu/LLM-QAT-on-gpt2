"""
Calibration manager for Cyclic Precision Training.
Follows the same calibration strategy as part1.
"""

import torch
import torch.nn as nn
import gc
from typing import List, Set
from tqdm import tqdm
from cpt_model import LoRAAdapter


class CalibrationManager:
    """Manages calibration of quantizers for all precision levels."""

    def __init__(self, model, train_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.calibrated_bits = set()
        self.gradient_calibrated = False

    def calibrate_all_precisions(self, bit_widths: List[int], num_batches: int = 10):
        """Calibrate all precision levels."""
        for bits in bit_widths:
            if bits < 32 and bits not in self.calibrated_bits:
                print(f"\nCalibrating {bits}-bit precision...")
                self.model.set_precision(bits)
                self._calibrate_precision(bits, num_batches)
                self.calibrated_bits.add(bits)

        if not self.gradient_calibrated:
            self.calibrate_gradient_quantizers()
            self.gradient_calibrated = True

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
            if hasattr(module, 'quantizer_weight'):
                weight_quantizer = module.quantizer_weight

                if hasattr(module, 'linear') and hasattr(module.linear, 'weight'):
                    weight = module.linear.weight.data
                elif hasattr(module, 'weight'):
                    weight = module.weight.data
                else:
                    weight_errors.append(f"{name}: No weight tensor found")
                    continue

                try:
                    weight_quantizer.set_num_bits(bits)
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

        if weight_calibrated > 0:
            scales = []
            zero_points = []
            weight_mins = []
            weight_maxs = []
            for name, module in self.model.named_modules():
                if hasattr(module, 'quantizer_weight'):
                    q = module.quantizer_weight
                    if q.calibrated and q.num_bits == bits:
                        scale_val = q.scale.mean().item() if q.scale.numel() > 1 else q.scale.item()
                        zp_val = q.zero_point.mean().item() if q.zero_point.numel() > 1 else q.zero_point.item()
                        scales.append(scale_val)
                        zero_points.append(zp_val)
                        if hasattr(q, 'running_min'):
                            weight_mins.append(q.running_min.min().item())
                            weight_maxs.append(q.running_max.max().item())

            if scales:
                import numpy as np
                print(f"    Weight Quantizer Statistics:")
                print(f"      Scales: mean={np.mean(scales):.6f}, std={np.std(scales):.6f}, min={np.min(scales):.6f}, max={np.max(scales):.6f}")
                print(f"      Zero Points: mean={np.mean(zero_points):.6f}, std={np.std(zero_points):.6f}")
                if weight_mins:
                    print(f"      Weight Range: [{np.min(weight_mins):.4f}, {np.max(weight_maxs):.4f}]")
                print(f"      Quant Range: [{-(2**(bits-1))}, {2**(bits-1)-1}]")

        # Step 2: Start calibration for input quantizers
        input_started = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizer_input'):
                module.quantizer_input.set_num_bits(bits)
                module.quantizer_input.start_calibration()
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
        input_issues = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizer_input'):
                q = module.quantizer_input
                q.finish_calibration(debug=False)

                if not q.calibrated:
                    input_issues.append(f"{name}: not calibrated")
                elif q.scale.abs().max().item() < 1e-9:
                    input_issues.append(f"{name}: zero scale ({q.scale.mean().item():.2e})")

                input_calibrated += 1

        print(f"    ✓ Calibrated {input_calibrated} input quantizers")
        if input_issues:
            print(f"    ⚠️  {len(input_issues)} input quantizers have issues (showing first 3):")
            for issue in input_issues[:3]:
                print(f"      - {issue}")

        if input_calibrated > 0:
            inp_scales = []
            inp_zero_points = []
            inp_mins = []
            inp_maxs = []
            for name, module in self.model.named_modules():
                if hasattr(module, 'quantizer_input'):
                    q = module.quantizer_input
                    if q.calibrated and q.num_bits == bits:
                        scale_val = q.scale.mean().item() if q.scale.numel() > 1 else q.scale.item()
                        zp_val = q.zero_point.mean().item() if q.zero_point.numel() > 1 else q.zero_point.item()
                        inp_scales.append(scale_val)
                        inp_zero_points.append(zp_val)
                        if hasattr(q, 'running_min'):
                            inp_mins.append(q.running_min.min().item())
                            inp_maxs.append(q.running_max.max().item())

            if inp_scales:
                import numpy as np
                print(f"    Input Quantizer Statistics:")
                print(f"      Scales: mean={np.mean(inp_scales):.6f}, std={np.std(inp_scales):.6f}, min={np.min(inp_scales):.6f}, max={np.max(inp_scales):.6f}")
                print(f"      Zero Points: mean={np.mean(inp_zero_points):.6f}, std={np.std(inp_zero_points):.6f}")
                if inp_mins:
                    print(f"      Activation Range: [{np.min(inp_mins):.4f}, {np.max(inp_maxs):.4f}]")

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

        lora_key = f'{bits}bit'

        lora_calibrated = 0
        for name, module in self.model.named_modules():
            if not hasattr(module, 'lora_adapters'):
                continue
            if lora_key not in module.lora_adapters:
                continue

            lora_layer = module.lora_adapters[lora_key]

            if hasattr(lora_layer, 'quantize_A') and hasattr(lora_layer, 'lora_A'):
                try:
                    lora_layer.quantize_A.set_num_bits(bits)
                    lora_layer.quantize_A.start_calibration()
                    with torch.no_grad():
                        _ = lora_layer.quantize_A(lora_layer.lora_A)
                    lora_layer.quantize_A.finish_calibration(debug=False)
                    lora_calibrated += 1
                except Exception as e:
                    print(f"    Warning calibrating {name} quantize_A: {e}")

            if hasattr(lora_layer, 'quantize_B') and hasattr(lora_layer, 'lora_B'):
                try:
                    lora_layer.quantize_B.set_num_bits(bits)
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

    def calibrate_gradient_quantizers(self):
        """Calibrate 8-bit gradient quantizers on LoRA adapters (BW8)."""
        print("\nCalibrating gradient quantizers...")

        grad_quantizers = []
        for name, module in self.model.named_modules():
            if isinstance(module, LoRAAdapter) and hasattr(module, 'grad_quantizer_8bit') and module.grad_quantizer_8bit is not None:
                module.grad_quantizer_8bit.start_calibration()
                grad_quantizers.append((name, module.grad_quantizer_8bit))

        if not grad_quantizers:
            print("  No gradient quantizers found")
            return

        original_mode = self.model.training
        self.model.train()

        train_iter = iter(self.train_loader)
        try:
            batch = next(train_iter)
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()

            self.model.zero_grad()

        except StopIteration:
            print("  Warning: No batches available for gradient calibration")

        if not original_mode:
            self.model.eval()

        for name, quantizer in grad_quantizers:
            quantizer.finish_calibration(debug=False)

        print(f"  ✓ Calibrated {len(grad_quantizers)} gradient quantizers")

        torch.cuda.empty_cache()
        gc.collect()

    def calibrate_current_precision(self, bits: int, num_batches: int = 10):
        """Calibrate quantizers for current precision only (per-epoch calibration)."""
        if bits >= 32:
            return

        print(f"  Calibrating {bits}-bit precision...")
        self._calibrate_precision(bits, num_batches)
        self.calibrated_bits.add(bits)

    def _print_calibration_stats(self, bits: int):
        """Print calibration statistics for debugging."""
        weight_stats = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizer_weight'):
                q = module.quantizer_weight
                if q.calibrated and q.num_bits == bits:
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
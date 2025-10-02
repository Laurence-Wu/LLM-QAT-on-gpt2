import torch
import torch.nn as nn
import gc
from typing import List, Set
from tqdm import tqdm
from cpt_model import LoRAAdapter

class CalibrationManager:
    def __init__(self, model, train_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.calibrated_bits = set()
        self.gradient_calibrated = False

    def calibrate_all_precisions(self, bit_widths: List[int], num_batches: int = 10):
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
        if bits >= 32:
            print(f"  Skipping calibration for {bits}-bit")
            return

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
                    weight_errors.append(f"{name}: No weight tensor")
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
            print(f"    ⚠️ {len(weight_errors)} warnings (first 3):")
            for err in weight_errors[:3]:
                print(f"      - {err}")

        input_started = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizer_input'):
                module.quantizer_input.set_num_bits(bits)
                module.quantizer_input.start_calibration()
                input_started += 1

        print(f"    Started {input_started} input quantizers")

        self.model.disable_lora_for_calibration()

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

        self.model.enable_lora_after_calibration()

        input_calibrated = 0
        input_issues = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizer_input'):
                q = module.quantizer_input
                q.finish_calibration(debug=False)

                if not q.calibrated:
                    input_issues.append(f"{name}: not calibrated")
                elif q.scale.abs().max().item() < 1e-9:
                    input_issues.append(f"{name}: zero scale")

                input_calibrated += 1

        print(f"    ✓ Calibrated {input_calibrated} input quantizers")
        if input_issues:
            print(f"    ⚠️ {len(input_issues)} issues (first 3):")
            for issue in input_issues[:3]:
                print(f"      - {issue}")

        torch.cuda.empty_cache()
        gc.collect()

    def calibrate_lora_only(self, bits: int, num_batches: int = 10):
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
                    print(f"    Warning {name} quantize_A: {e}")

            if hasattr(lora_layer, 'quantize_B') and hasattr(lora_layer, 'lora_B'):
                try:
                    lora_layer.quantize_B.set_num_bits(bits)
                    lora_layer.quantize_B.start_calibration()
                    with torch.no_grad():
                        _ = lora_layer.quantize_B(lora_layer.lora_B)
                    lora_layer.quantize_B.finish_calibration(debug=False)
                    lora_calibrated += 1
                except Exception as e:
                    print(f"    Warning {name} quantize_B: {e}")

        if lora_calibrated > 0:
            torch.cuda.empty_cache()
            gc.collect()

    def ensure_calibrated(self, bits: int):
        if bits >= 32:
            return

        if bits not in self.calibrated_bits:
            print(f"  ⚠️ {bits}-bit not calibrated, calibrating now...")
            self.model.set_precision(bits)
            self._calibrate_precision(bits, num_batches=10)
            self.calibrated_bits.add(bits)

    def calibrate_gradient_quantizers(self):
        print("\nCalibrating gradient quantizers...")

        grad_quantizers = []
        for name, module in self.model.named_modules():
            if isinstance(module, LoRAAdapter) and hasattr(module, 'grad_quantizer_8bit') and module.grad_quantizer_8bit is not None:
                module.grad_quantizer_8bit.start_calibration()
                grad_quantizers.append((name, module.grad_quantizer_8bit))

        if not grad_quantizers:
            print("  No gradient quantizers")
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
            print("  Warning: No batches for gradient calibration")

        if not original_mode:
            self.model.eval()

        for name, quantizer in grad_quantizers:
            quantizer.finish_calibration(debug=False)

        print(f"  ✓ Calibrated {len(grad_quantizers)} gradient quantizers")

        torch.cuda.empty_cache()
        gc.collect()

    def calibrate_current_precision(self, bits: int, num_batches: int = 10):
        if bits >= 32:
            return
        print(f"  Calibrating {bits}-bit precision...")
        self._calibrate_precision(bits, num_batches)
        self.calibrated_bits.add(bits)
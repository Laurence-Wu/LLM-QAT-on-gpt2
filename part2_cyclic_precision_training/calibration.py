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
        self.lora_calibrated_bits = set()

    def calibrate_all_precisions(self, bit_widths: List[int], num_batches: int = 10):
        for bits in bit_widths:
            if bits < 32 and bits not in self.calibrated_bits:
                print(f"\nCalibrating {bits}-bit precision...")
                self.model.set_precision(bits)
                self._calibrate_precision(bits, num_batches)
                self.calibrated_bits.add(bits)

        # Calibrate LoRA weight quantizers for all precisions
        self.calibrate_lora_weight_quantizers(bit_widths)

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
        grad_quantizer_modes = []
        for name, module in self.model.named_modules():
            if isinstance(module, LoRAAdapter):
                # Calibrate grad_quantizer_A
                if hasattr(module, 'grad_quantizer_A') and module.grad_quantizer_A is not None:
                    grad_quantizer_modes.append(module.grad_quantizer_A.training)
                    module.grad_quantizer_A.eval()
                    module.grad_quantizer_A.start_calibration()
                    grad_quantizers.append((f"{name}.grad_A", module.grad_quantizer_A))

                # Calibrate grad_quantizer_B
                if hasattr(module, 'grad_quantizer_B') and module.grad_quantizer_B is not None:
                    grad_quantizer_modes.append(module.grad_quantizer_B.training)
                    module.grad_quantizer_B.eval()
                    module.grad_quantizer_B.start_calibration()
                    grad_quantizers.append((f"{name}.grad_B", module.grad_quantizer_B))

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

        calibrated_count = 0
        for i, (name, quantizer) in enumerate(grad_quantizers):
            quantizer.finish_calibration(debug=False)
            if quantizer.calibrated:
                calibrated_count += 1
            if i < len(grad_quantizer_modes):
                quantizer.train(grad_quantizer_modes[i])

        print(f"  ✓ Calibrated {calibrated_count}/{len(grad_quantizers)} gradient quantizers")
        if calibrated_count == 0:
            print(f"  ⚠️ WARNING: No gradient quantizers were successfully calibrated!")

        torch.cuda.empty_cache()
        gc.collect()

    def calibrate_lora_weight_quantizers(self, bit_widths: List[int]):
        print("\nCalibrating LoRA weight quantizers for all precisions...")

        from cpt_model import CPTLinear

        # Collect all LoRA weight quantizers across all precisions
        lora_quantizers_by_bits = {}
        for bits in bit_widths:
            if bits >= 32:
                continue
            lora_quantizers_by_bits[bits] = []

        # Find all CPTLinear modules and their LoRA weight quantizers
        for name, module in self.model.named_modules():
            if isinstance(module, CPTLinear):
                if not hasattr(module, 'shared_lora') or module.shared_lora is None:
                    continue

                # Get the shared LoRA adapter
                shared_lora = module.shared_lora
                if not hasattr(shared_lora, 'lora_A') or shared_lora.lora_A is None:
                    continue

                # Calibrate quantizers for each precision
                for bits in bit_widths:
                    if bits >= 32:
                        continue

                    lora_key = f'{bits}bit'
                    if lora_key not in module.lora_weight_quantizers:
                        continue

                    quantizer = module.lora_weight_quantizers[lora_key]
                    lora_quantizers_by_bits[bits].append((name, quantizer, shared_lora))

        # Calibrate each precision's quantizers
        for bits in bit_widths:
            if bits >= 32 or bits not in lora_quantizers_by_bits:
                continue

            quantizers = lora_quantizers_by_bits[bits]
            if not quantizers:
                continue

            print(f"  Calibrating {bits}-bit LoRA weight quantizers...")
            calibrated_count = 0

            for name, quantizer, shared_lora in quantizers:
                try:
                    quantizer.set_num_bits(bits)
                    quantizer.start_calibration()

                    with torch.no_grad():
                        # Calibrate on lora_A
                        _ = quantizer(shared_lora.lora_A)
                        # Calibrate on lora_B
                        _ = quantizer(shared_lora.lora_B)

                    quantizer.finish_calibration(debug=False)

                    if quantizer.calibrated:
                        calibrated_count += 1
                except Exception as e:
                    print(f"    Warning {name}: {str(e)}")

            print(f"    ✓ Calibrated {calibrated_count}/{len(quantizers)} LoRA weight quantizers for {bits}-bit")

        torch.cuda.empty_cache()
        gc.collect()

    def calibrate_current_precision(self, bits: int, num_batches: int = 10):
        if bits >= 32:
            return
        print(f"  Calibrating {bits}-bit precision...")
        self._calibrate_precision(bits, num_batches)
        self.calibrated_bits.add(bits)
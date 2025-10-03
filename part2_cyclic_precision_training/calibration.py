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

    def _calibrate_precision(self, bits: int, num_batches: int):
        if bits >= 32:
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

        input_started = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizer_input'):
                module.quantizer_input.set_num_bits(bits)
                module.quantizer_input.start_calibration()
                input_started += 1

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
                    break

        self.model.enable_lora_after_calibration()

        input_calibrated = 0
        input_issues = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizer_input'):
                q = module.quantizer_input
                q.finish_calibration(debug=False)

                if bits not in q.calibrated_bits:
                    input_issues.append(f"{name}: not calibrated at {bits}-bit")
                elif bits in q.scales and q.scales[bits].abs().max().item() < 1e-9:
                    input_issues.append(f"{name}: zero scale at {bits}-bit")

                input_calibrated += 1

        torch.cuda.empty_cache()
        gc.collect()

    def ensure_calibrated(self, bits: int):
        if bits >= 32:
            return

        if bits not in self.calibrated_bits:
            self.model.set_precision(bits)
            self._calibrate_precision(bits, num_batches=10)
            self.calibrated_bits.add(bits)

        if bits not in self.lora_calibrated_bits:
            self.calibrate_lora_weight_quantizers([bits])
            self.lora_calibrated_bits.add(bits)

    def calibrate_gradient_quantizers(self):
        grad_quantizers = []
        grad_quantizer_modes = []
        for name, module in self.model.named_modules():
            if isinstance(module, LoRAAdapter):
                if hasattr(module, 'grad_quantizer_A') and module.grad_quantizer_A is not None:
                    grad_quantizer_modes.append(module.grad_quantizer_A.training)
                    module.grad_quantizer_A.eval()
                    module.grad_quantizer_A.start_calibration()
                    grad_quantizers.append((f"{name}.grad_A", module.grad_quantizer_A))

                if hasattr(module, 'grad_quantizer_B') and module.grad_quantizer_B is not None:
                    grad_quantizer_modes.append(module.grad_quantizer_B.training)
                    module.grad_quantizer_B.eval()
                    module.grad_quantizer_B.start_calibration()
                    grad_quantizers.append((f"{name}.grad_B", module.grad_quantizer_B))

        if not grad_quantizers:
            return

        original_mode = self.model.training
        original_precision = self.model.current_precision

        self.model.set_precision(32)
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
            pass

        self.model.set_precision(original_precision)
        if not original_mode:
            self.model.eval()

        calibrated_count = 0
        for i, (name, quantizer) in enumerate(grad_quantizers):
            quantizer.finish_calibration(debug=False)
            if quantizer.num_bits in quantizer.calibrated_bits:
                calibrated_count += 1
            if i < len(grad_quantizer_modes):
                quantizer.train(grad_quantizer_modes[i])

        torch.cuda.empty_cache()
        gc.collect()

    def calibrate_lora_weight_quantizers(self, bit_widths: List[int]):
        from cpt_model import CPTLinear

        lora_quantizers_by_bits = {}
        for bits in bit_widths:
            if bits >= 32:
                continue
            lora_quantizers_by_bits[bits] = []

        for name, module in self.model.named_modules():
            if isinstance(module, CPTLinear):
                if not hasattr(module, 'shared_lora') or module.shared_lora is None:
                    continue

                shared_lora = module.shared_lora
                if not hasattr(shared_lora, 'lora_A') or shared_lora.lora_A is None:
                    continue

                for bits in bit_widths:
                    if bits >= 32:
                        continue

                    lora_key = f'{bits}bit'
                    if lora_key not in module.lora_weight_quantizers:
                        continue

                    quantizer = module.lora_weight_quantizers[lora_key]
                    lora_quantizers_by_bits[bits].append((name, quantizer, shared_lora))

        for bits in bit_widths:
            if bits >= 32 or bits not in lora_quantizers_by_bits:
                continue

            quantizers = lora_quantizers_by_bits[bits]
            if not quantizers:
                continue

            calibrated_count = 0

            for name, quantizer, shared_lora in quantizers:
                try:
                    quantizer.set_num_bits(bits)
                    quantizer.start_calibration()

                    with torch.no_grad():
                        _ = quantizer(shared_lora.lora_A)
                        _ = quantizer(shared_lora.lora_B)

                    quantizer.finish_calibration(debug=False)

                    if bits in quantizer.calibrated_bits:
                        calibrated_count += 1
                except Exception as e:
                    pass

        torch.cuda.empty_cache()
        gc.collect()

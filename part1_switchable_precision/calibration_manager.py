
class CalibrationManager:
    """Manages quantizer calibration for different bit-widths."""

    def __init__(self, model, train_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.calibrated_bits = set()

    def calibrate_all_precisions(self, bit_widths, num_batches=10):
        for bits in bit_widths:
            if bits < 32 and bits not in self.calibrated_bits:
                self.model.set_precision(bits)
                self._calibrate_precision(bits, num_batches)
                self.calibrated_bits.add(bits)

    def _calibrate_precision(self, bits, num_batches):
        """Internal calibration logic with separate weight and input calibration."""
        self.model.train()
        bits_key = f'{bits}bit'

        # CRITICAL: Skip 32-bit as it doesn't need calibration
        if bits >= 32:
            print(f"  Skipping calibration for {bits}-bit (no quantization needed)")
            return

        # Step 1: Calibrate WEIGHT quantizers on actual weight tensors
        print(f"  Step 1: Calibrating weight quantizers for {bits}-bit...")
        weight_calibrated = 0
        weight_errors = []

        for name, module in self.model.named_modules():
            if not hasattr(module, 'quantizers_weight'):
                continue

            if bits_key not in module.quantizers_weight:
                print(f"{name}: Missing {bits_key} in quantizers_weight")
                continue

            weight_quantizer = module.quantizers_weight[bits_key]

            # Get the weight tensor
            if hasattr(module, 'linear') and hasattr(module.linear, 'weight'):
                weight = module.linear.weight.data
            elif hasattr(module, 'weight'):
                weight = module.weight.data
            else:
                weight_errors.append(f"{name}: No weight tensor found")
                continue

            # Calibrate weight quantizer
            try:
                weight_quantizer.start_calibration()
                with torch.no_grad():
                    _ = weight_quantizer(weight)
                weight_quantizer.finish_calibration(debug=False)
                weight_calibrated += 1
            except Exception as e:
                weight_errors.append(f"{name}: {str(e)}")

        print(f"    ✓ Calibrated {weight_calibrated} weight quantizers")
        if weight_errors:
            print(f"    ⚠️ {len(weight_errors)} warnings (showing first 3):")
            for err in weight_errors[:3]:
                print(f"      - {err}")

        # Step 2: Calibrate INPUT and LoRA quantizers via forward passes
        print(f"  Step 2: Calibrating input and LoRA quantizers for {bits}-bit...")

        # Start input quantizer calibration
        input_started = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizers_input') and bits_key in module.quantizers_input:
                module.quantizers_input[bits_key].start_calibration()
                input_started += 1

        # Start LoRA quantizer calibration (they need forward passes too)
        lora_started = 0
        for name, module in self.model.named_modules():
            if not hasattr(module, 'lora_adapters'):
                continue
            if bits_key not in module.lora_adapters:
                continue

            lora_layer = module.lora_adapters[bits_key]
            if hasattr(lora_layer, 'quantize_A') and lora_layer.enabled:
                lora_layer.quantize_A.start_calibration()
                lora_started += 1
            if hasattr(lora_layer, 'quantize_B') and lora_layer.enabled:
                lora_layer.quantize_B.start_calibration()
                lora_started += 1

        print(f"    Started calibration for {input_started} input quantizers and {lora_started} LoRA quantizers")

        # Collect statistics via forward passes
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

        # Finish input quantizer calibration
        input_calibrated = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizers_input') and bits_key in module.quantizers_input:
                module.quantizers_input[bits_key].finish_calibration(debug=False)
                input_calibrated += 1

        # Finish LoRA quantizer calibration
        lora_calibrated = 0
        for name, module in self.model.named_modules():
            if not hasattr(module, 'lora_adapters'):
                continue
            if bits_key not in module.lora_adapters:
                continue

            lora_layer = module.lora_adapters[bits_key]
            if hasattr(lora_layer, 'quantize_A') and lora_layer.enabled:
                lora_layer.quantize_A.finish_calibration(debug=False)
                lora_calibrated += 1
            if hasattr(lora_layer, 'quantize_B') and lora_layer.enabled:
                lora_layer.quantize_B.finish_calibration(debug=False)
                lora_calibrated += 1

        print(f"    ✓ Calibrated {input_calibrated} input quantizers and {lora_calibrated} LoRA quantizers")

        cleanup_memory()

    def ensure_calibrated(self, bits):
        """Ensure the given bit-width is calibrated, calibrate if not."""
        if bits >= 32:
            # 32-bit doesn't need calibration
            return

        if bits not in self.calibrated_bits:
            print(f"  ⚠️ {bits}-bit not calibrated, calibrating now...")
            self.model.set_precision(bits)
            self._calibrate_precision(bits, num_batches=10)
            self.calibrated_bits.add(bits)

        # Print calibration statistics for debugging
        self._print_calibration_stats(bits)

    def _print_calibration_stats(self, bits):
        """Print detailed calibration statistics for debugging."""
        bits_key = f'{bits}bit'
        print(f"\n  📊 Calibration Statistics for {bits}-bit:")

        # Collect statistics for weight quantizers
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

        # Collect statistics for LoRA quantizers
        lora_stats = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_adapters') and bits_key in module.lora_adapters:
                lora_layer = module.lora_adapters[bits_key]
                if hasattr(lora_layer, 'quantize_A') and lora_layer.enabled and lora_layer.quantize_A.calibrated:
                    q = lora_layer.quantize_A
                    lora_stats.append({
                        'name': f"{name}.lora_A",
                        'scale': q.scale.mean().item() if q.scale.numel() > 1 else q.scale.item(),
                        'zero_point': q.zero_point.mean().item() if q.zero_point.numel() > 1 else q.zero_point.item(),
                        'min': q.running_min.min().item() if hasattr(q, 'running_min') else 0,
                        'max': q.running_max.max().item() if hasattr(q, 'running_max') else 0,
                        'type': q.quantizer_type
                    })
                if hasattr(lora_layer, 'quantize_B') and lora_layer.enabled and lora_layer.quantize_B.calibrated:
                    q = lora_layer.quantize_B
                    lora_stats.append({
                        'name': f"{name}.lora_B",
                        'scale': q.scale.mean().item() if q.scale.numel() > 1 else q.scale.item(),
                        'zero_point': q.zero_point.mean().item() if q.zero_point.numel() > 1 else q.zero_point.item(),
                        'min': q.running_min.min().item() if hasattr(q, 'running_min') else 0,
                        'max': q.running_max.max().item() if hasattr(q, 'running_max') else 0,
                        'type': q.quantizer_type
                    })

        # Print weight quantizer statistics
        if weight_stats:
            print(f"  Weight Quantizers ({len(weight_stats)} total):")
            # Check for duplicate scales
            scales = [s['scale'] for s in weight_stats]
            unique_scales = len(set(scales))
            if unique_scales < len(scales):
                print(f"    ⚠️ WARNING: Only {unique_scales} unique scale values out of {len(scales)} quantizers!")

            # Show distribution
            scale_min = min(scales)
            scale_max = max(scales)
            scale_mean = sum(scales) / len(scales)
            print(f"    Scale range: [{scale_min:.6f}, {scale_max:.6f}], mean: {scale_mean:.6f}")

            # Show sample of quantizers with same scale (if any)
            from collections import Counter
            scale_counts = Counter(scales)
            duplicates = [(scale, count) for scale, count in scale_counts.items() if count > 1]
            if duplicates:
                print(f"    Duplicate scales found:")
                for scale, count in duplicates[:3]:  # Show first 3
                    print(f"      Scale {scale:.6f}: {count} quantizers")

        # Print LoRA quantizer statistics
        if lora_stats:
            print(f"  LoRA Quantizers ({len(lora_stats)} total):")
            scales = [s['scale'] for s in lora_stats]
            unique_scales = len(set(scales))
            if unique_scales < len(scales):
                print(f"    ⚠️ WARNING: Only {unique_scales} unique scale values out of {len(scales)} quantizers!")

            scale_min = min(scales)
            scale_max = max(scales)
            scale_mean = sum(scales) / len(scales)
            print(f"    Scale range: [{scale_min:.6f}, {scale_max:.6f}], mean: {scale_mean:.6f}")

            # Group by type (A vs B)
            a_scales = [s['scale'] for s in lora_stats if 'lora_A' in s['name']]
            b_scales = [s['scale'] for s in lora_stats if 'lora_B' in s['name']]
            if a_scales:
                print(f"    LoRA A scales - min: {min(a_scales):.6f}, max: {max(a_scales):.6f}, unique: {len(set(a_scales))}")
            if b_scales:
                print(f"    LoRA B scales - min: {min(b_scales):.6f}, max: {max(b_scales):.6f}, unique: {len(set(b_scales))}")

        print()  # Empty line for readability


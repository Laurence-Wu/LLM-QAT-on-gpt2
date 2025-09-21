#!/usr/bin/env python3
"""
Calibration Manager for Switchable Precision Models
Manages quantizer calibration for all bit-widths to ensure proper quantization behavior
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any

try:
    from .quant_methods import LearnableFakeQuantize
except ImportError:
    from quant_methods import LearnableFakeQuantize


class CalibrationManager:
    """Manages quantizer calibration for all bit-widths"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Diverse calibration texts to ensure good quantizer initialization
        self.calibration_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming modern technology rapidly.",
            "Python is a versatile programming language for data science.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models require significant computational resources for training.",
            "The weather today is sunny with clear blue skies and gentle breeze.",
            "Books are a great source of knowledge and entertainment for everyone.",
            "Technology continues to advance at an unprecedented pace in our society.",
            "Scientific research drives innovation and discovery across multiple fields.",
            "Education plays a crucial role in shaping future generations.",
            "Climate change poses significant challenges to global sustainability efforts.",
            "Artificial intelligence systems are becoming increasingly sophisticated and capable."
        ]

    def calibrate_all_precisions(self):
        """Calibrate quantizers for all bit-widths"""
        print("🔧 Calibrating quantization scales for all precisions...")

        # MUST be in eval mode for one-shot calibration
        self.model.eval()

        # Calibrate each bit-width separately
        for bits in [4, 8]:  # Skip 16-bit entirely
            print(f"  📊 Calibrating {bits}-bit mode...")
            self.model.set_precision(bits)

            # Reset calibration state for quantizers matching this precision
            reset_count = self._reset_calibration_for_precision(bits)
            print(f"    Reset {reset_count} quantizers")

            # Run forward passes to calibrate quantizers
            bits_key = f'{bits}bit'
            calibrated_after_first = 0

            with torch.no_grad():
                for i, text in enumerate(self.calibration_texts):
                    tokens = self.tokenizer(
                        text,
                        return_tensors='pt',
                        max_length=64,
                        truncation=True,
                        padding=False
                    )['input_ids'].to(self.device)

                    # This triggers calibration in LearnableFakeQuantize
                    outputs = self.model(tokens)

                    # Check calibration status after first sample
                    if i == 0:
                        for name, module in self.model.named_modules():
                            if isinstance(module, LearnableFakeQuantize) and bits_key in name:
                                if module.calibrated:
                                    calibrated_after_first += 1
                        print(f"    After first sample: {calibrated_after_first}/{reset_count} calibrated")

                    if (i + 1) % 4 == 0:
                        print(f"    Processed {i + 1}/{len(self.calibration_texts)} samples")

            print(f"  ✅ {bits}-bit calibration complete")

        # Reset to 16-bit after calibration
        self.model.set_precision(16)
        print("✅ Calibration complete for all precisions")

    def _reset_calibration_for_precision(self, bits: int):
        """Reset calibration state for quantizers at specific precision"""
        if bits >= 16:
            raise ValueError("16-bit mode doesn't need calibration")

        bits_key = f'{bits}bit'
        reset_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, LearnableFakeQuantize) and bits_key in name:
                # Reset calibration to allow fresh calibration
                module.calibrated = False
                # Reset statistics for clean calibration
                if hasattr(module, 'running_min'):
                    module.running_min.zero_()
                if hasattr(module, 'running_max'):
                    module.running_max.zero_()
                reset_count += 1

        return reset_count

    def validate_calibration(self) -> bool:
        """Check if each layer has properly calibrated quantizers for each precision"""
        print("\n🔍 Validating quantizer calibration...")

        all_valid = True
        for bits in [4, 8]:  # Only check 4 and 8 bit (16-bit bypasses)
            bits_key = f'{bits}bit'
            total = 0
            calibrated = 0
            issues = []

            # Check quantizers for this specific bit-width
            for name, module in self.model.named_modules():
                if isinstance(module, LearnableFakeQuantize) and bits_key in name:
                    total += 1
                    if not hasattr(module, 'calibrated'):
                        issues.append(f"{name[:40]}... (no calibrated attr)")
                    elif not module.calibrated:
                        issues.append(f"{name[:40]}... (not calibrated)")
                    elif hasattr(module, 'scale') and torch.all(module.scale == 1.0):
                        issues.append(f"{name[:40]}... (default scale)")
                    else:
                        calibrated += 1

            if total == 0:
                print(f"  ❌ {bits}-bit: No quantizers found!")
                return False
            elif calibrated == total:
                print(f"  ✅ {bits}-bit: {calibrated}/{total} quantizers calibrated")
            else:
                print(f"  ❌ {bits}-bit: Only {calibrated}/{total} quantizers calibrated")
                if len(issues) <= 3:
                    for issue in issues:
                        print(f"      - {issue}")
                else:
                    for issue in issues[:2]:
                        print(f"      - {issue}")
                    print(f"      ... and {len(issues)-2} more")
                all_valid = False

        if all_valid:
            print("✅ All precision modes properly calibrated")
        return all_valid

    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration statistics for analysis"""
        stats = {
            'quantizers': [],
            'scales': {},
            'zero_points': {},
            'ranges': {}
        }

        for name, module in self.model.named_modules():
            if isinstance(module, LearnableFakeQuantize):
                stats['quantizers'].append(name)
                try:
                    stats['scales'][name] = module.scale.detach().cpu()
                    stats['zero_points'][name] = module.zero_point.detach().cpu()

                    min_val = module.running_min.detach().cpu()
                    max_val = module.running_max.detach().cpu()
                    stats['ranges'][name] = (min_val, max_val)
                except AttributeError as e:
                    print(f"    Stats collection error for {name}: {e}")
                    stats['scales'][name] = None
                    stats['zero_points'][name] = None
                    stats['ranges'][name] = None

        return stats

    def quick_calibration_check(self) -> None:
        """Quick check of calibration quality with test input"""
        print("\n🧪 Quick calibration quality check...")

        test_text = "Machine learning and artificial intelligence are revolutionizing technology."
        tokens = self.tokenizer(test_text, return_tensors='pt')['input_ids'].to(self.device)

        self.model.eval()
        results = {}

        with torch.no_grad():
            for bits in [16, 8, 4]:
                self.model.set_precision(bits)
                outputs = self.model(tokens, labels=tokens)
                loss = outputs['loss'].item()
                ppl = torch.exp(torch.tensor(loss)).item()
                results[bits] = {'loss': loss, 'ppl': ppl}
                print(f"  {bits:2d}-bit: Loss = {loss:.4f}, PPL = {ppl:.2f}")

        # Analyze degradation
        baseline_ppl = results[16]['ppl']
        for bits in [8, 4]:
            ppl = results[bits]['ppl']
            degradation = ((ppl - baseline_ppl) / baseline_ppl) * 100
            status = "✅" if degradation < 20 else "⚠️" if degradation < 100 else "❌"
            print(f"  {bits}-bit degradation: {degradation:.1f}% {status}")

        self.model.set_precision(16)
        return results


def calibrate_sp_model(model, tokenizer, device='cuda'):
    """Convenience function to calibrate an SP model"""
    calibration_mgr = CalibrationManager(model, tokenizer, device)
    calibration_mgr.calibrate_all_precisions()

    if not calibration_mgr.validate_calibration():
        raise RuntimeError("Model calibration failed!")

    calibration_mgr.quick_calibration_check()
    return True


def debug_quantizer_routing(model, tokenizer, device='cuda'):
    """Debug function to check if quantizers are being called correctly"""
    print("\n🔍 Debugging quantizer routing...")

    model.eval()
    test_text = "Test"
    tokens = tokenizer(test_text, return_tensors='pt')['input_ids'].to(device)

    for bits in [4, 8]:
        print(f"\n  Testing {bits}-bit mode:")
        model.set_precision(bits)
        bits_key = f'{bits}bit'

        # Add hooks to track calls
        hooks = []
        called = {}

        def make_hook(name):
            def hook(module, input, output):
                called[name] = True
                print(f"    ✓ Called: {name[:60]}...")
            return hook

        # Register hooks on quantizers for this bit-width
        for name, module in model.named_modules():
            if isinstance(module, LearnableFakeQuantize) and bits_key in name:
                h = module.register_forward_hook(make_hook(name))
                hooks.append(h)

        # Run forward pass
        with torch.no_grad():
            _ = model(tokens)

        # Remove hooks
        for h in hooks:
            h.remove()

        print(f"    Total quantizers called: {len(called)}")

    model.set_precision(16)
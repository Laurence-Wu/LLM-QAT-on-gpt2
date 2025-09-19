#!/usr/bin/env python3
"""
Calibration Manager for Switchable Precision Models
Manages quantizer calibration for all bit-widths to ensure proper quantization behavior
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any

try:
    from .quantization import LearnableFakeQuantize
except ImportError:
    from quantization import LearnableFakeQuantize


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

        # Calibrate each bit-width separately to avoid cross-contamination
        for bits in [4, 8, 16]:
            print(f"  📊 Calibrating {bits}-bit mode...")
            self.model.set_precision(bits)

            # Reset calibration state for all quantizers at this precision
            self._reset_calibration_for_precision(bits)

            # Run forward passes to calibrate quantizers
            with torch.no_grad():
                for i, text in enumerate(self.calibration_texts):
                    tokens = self.tokenizer(
                        text,
                        return_tensors='pt',
                        max_length=64,  # Reasonable length for calibration
                        truncation=True,
                        padding=False
                    )['input_ids'].to(self.device)

                    # This triggers calibration in LearnableFakeQuantize
                    try:
                        outputs = self.model(tokens)

                        # Progress indicator
                        if (i + 1) % 4 == 0:
                            print(f"    🔄 Processed {i + 1}/{len(self.calibration_texts)} calibration samples")

                    except Exception as e:
                        print(f"    ⚠️ Warning: Calibration error on sample {i+1}: {e}")
                        continue

            print(f"    ✅ {bits}-bit calibration complete")

        # Reset to 16-bit after calibration
        self.model.set_precision(16)
        print("✅ Calibration complete for all precisions")

    def _reset_calibration_for_precision(self, bits: int):
        """Reset calibration state for quantizers at specific precision"""
        for name, module in self.model.named_modules():
            if isinstance(module, LearnableFakeQuantize):
                # Reset calibration to allow fresh calibration
                module.calibrated = False
                module.running_min.fill_(float('inf'))
                module.running_max.fill_(float('-inf'))
                print(f"    Reset calibration for LearnableFakeQuantize: {name}")

    def validate_calibration(self) -> bool:
        """Check if quantizers are properly calibrated for each precision"""
        print("\n🔍 Validating quantizer calibration...")

        # Test each precision separately
        validation_results = {}

        for bits in [4, 8, 16]:
            print(f"    Checking {bits}-bit precision...")
            issues = []
            calibrated_quantizers = 0
            total_quantizers = 0

            # Set model to specific precision for validation
            self.model.set_precision(bits)

            for name, module in self.model.named_modules():
                if isinstance(module, LearnableFakeQuantize):
                    # Skip quantizers that shouldn't be active for this precision
                    if bits >= 16 and module.num_bits >= 16:
                        # 16-bit quantizers are expected to be uncalibrated (they pass through)
                        continue
                    elif bits < 16 and module.num_bits != bits:
                        # Only check quantizers that match current precision
                        continue

                    total_quantizers += 1

                    try:
                        if not module.calibrated:
                            issues.append(f"{name} not calibrated")
                        elif torch.all(module.scale == 1.0) and torch.all(module.zero_point == 0.0):
                            issues.append(f"{name} has default parameters")
                        elif torch.any(torch.isinf(module.running_min)) or torch.any(torch.isinf(module.running_max)):
                            issues.append(f"{name} has invalid running statistics")
                        else:
                            calibrated_quantizers += 1
                    except AttributeError as e:
                        issues.append(f"{name} missing calibration attributes: {e}")
                        print(f"      Calibration validation error for {name}: {e}")

            validation_results[bits] = {
                'issues': issues,
                'calibrated': calibrated_quantizers,
                'total': total_quantizers
            }

            if issues:
                print(f"      ⚠️ {len(issues)} issues found for {bits}-bit")
                print(f"      📊 Calibrated: {calibrated_quantizers}/{total_quantizers} quantizers")
            else:
                print(f"      ✅ All {total_quantizers} quantizers calibrated for {bits}-bit")

        # Overall assessment
        total_issues = sum(len(result['issues']) for result in validation_results.values())

        if total_issues > 0:
            print(f"\n⚠️ Total calibration issues: {total_issues}")
            for bits, result in validation_results.items():
                if result['issues']:
                    print(f"  {bits}-bit precision: {len(result['issues'])} issues")
            return False

        print(f"\n✅ All precision modes properly calibrated")
        return True

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

                try:
                    outputs = self.model(tokens, labels=tokens)
                    loss = outputs['loss'].item()
                    ppl = torch.exp(torch.tensor(loss)).item()

                    results[bits] = {'loss': loss, 'ppl': ppl}
                    print(f"  {bits:2d}-bit: Loss = {loss:.4f}, PPL = {ppl:.2f}")

                except Exception as e:
                    print(f"  {bits:2d}-bit: Error - {e}")
                    results[bits] = {'error': str(e)}

        # Analyze degradation
        if 16 in results and 8 in results and 'loss' in results[16] and 'loss' in results[8]:
            baseline_ppl = results[16]['ppl']

            for bits in [8, 4]:
                if bits in results and 'ppl' in results[bits]:
                    ppl = results[bits]['ppl']
                    degradation = ((ppl - baseline_ppl) / baseline_ppl) * 100

                    if degradation < 20:
                        status = "✅ Excellent"
                    elif degradation < 100:
                        status = "⚠️ Acceptable"
                    else:
                        status = "❌ Poor"

                    print(f"  {bits}-bit degradation: {degradation:.1f}% ({status})")

        # Reset to 16-bit
        self.model.set_precision(16)

        return results


def calibrate_sp_model(model, tokenizer, device='cuda'):
    """Convenience function to calibrate an SP model"""
    calibration_mgr = CalibrationManager(model, tokenizer, device)
    calibration_mgr.calibrate_all_precisions()

    # Validate calibration worked
    if calibration_mgr.validate_calibration():
        print("🎉 Model calibration successful!")

        # Quick quality check
        calibration_mgr.quick_calibration_check()

        return True
    else:
        print("❌ Model calibration failed!")
        return False
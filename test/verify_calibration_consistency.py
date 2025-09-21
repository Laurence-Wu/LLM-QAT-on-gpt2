#!/usr/bin/env python3
"""
Verify that calibration produces consistent statistics across multiple runs.
This ensures that the running min/max values are stable and reproducible.
"""

import torch
import numpy as np
from test.dataset_utils import get_calibration_texts


def verify_calibration_consistency(sp_model, tokenizer, device, precision=8, num_runs=3):
    """
    Verify that calibration statistics are consistent across multiple runs.

    Args:
        sp_model: The model to test
        tokenizer: Tokenizer for text processing
        device: Device to run on
        precision: Bit-width to test (4 or 8)
        num_runs: Number of calibration runs to compare

    Returns:
        Dictionary with consistency analysis
    """
    print(f"\n{'='*60}")
    print(f"CALIBRATION CONSISTENCY VERIFICATION FOR {precision}-BIT")
    print(f"{'='*60}")

    # Get fixed calibration texts for reproducibility
    calibration_texts = get_calibration_texts(num_texts=10)
    bits_key = f'{precision}bit'

    # Store statistics from each run
    all_stats = []

    for run in range(num_runs):
        print(f"\n📊 Calibration Run {run + 1}/{num_runs}:")

        # Set precision and start calibration
        sp_model.set_precision(precision)
        sp_model.train()

        # Collect quantizer references before calibration
        quantizers_to_check = []

        for name, module in sp_model.named_modules():
            if hasattr(module, 'quantizers_weight') and hasattr(module, 'quantizers_input'):
                if bits_key in module.quantizers_weight:
                    quantizers_to_check.append((f"{name}.weight", module.quantizers_weight[bits_key]))
                if bits_key in module.quantizers_input:
                    quantizers_to_check.append((f"{name}.input", module.quantizers_input[bits_key]))

        # Start calibration for all quantizers
        for name, quantizer in quantizers_to_check:
            quantizer.start_calibration()

        # Collect statistics with fixed data
        with torch.no_grad():
            for i, text in enumerate(calibration_texts):
                tokens = tokenizer(text, return_tensors='pt',
                                 max_length=128, truncation=True)['input_ids'].to(device)
                _ = sp_model(tokens)

        # Finish calibration and collect statistics
        run_stats = {}

        for idx, (name, quantizer) in enumerate(quantizers_to_check[:4]):  # Check first 4 quantizers
            # Show debug for first quantizer of each run
            show_debug = (idx == 0)
            quantizer.finish_calibration(debug=show_debug)

            # Collect statistics after calibration
            if quantizer.calibrated:
                stats = {
                    'name': name,
                    'min': quantizer.running_min.min().item() if quantizer.per_channel else quantizer.running_min.item(),
                    'max': quantizer.running_max.max().item() if quantizer.per_channel else quantizer.running_max.item(),
                    'scale': quantizer.scale.mean().item(),
                    'scale_min': quantizer.scale.min().item(),
                    'scale_max': quantizer.scale.max().item(),
                }
                run_stats[name] = stats

        all_stats.append(run_stats)

        # Finish remaining quantizers without debug
        for name, quantizer in quantizers_to_check[4:]:
            quantizer.finish_calibration(debug=False)

        print(f"   Collected stats for {len(run_stats)} quantizers")

    # Analyze consistency across runs
    print(f"\n{'='*40}")
    print("CONSISTENCY ANALYSIS")
    print(f"{'='*40}")

    consistency_results = {}

    for quantizer_name in all_stats[0].keys():
        print(f"\n📊 {quantizer_name}:")

        # Collect values across runs
        mins = [all_stats[run][quantizer_name]['min'] for run in range(num_runs)]
        maxs = [all_stats[run][quantizer_name]['max'] for run in range(num_runs)]
        scales = [all_stats[run][quantizer_name]['scale'] for run in range(num_runs)]

        # Calculate variations
        min_std = np.std(mins)
        max_std = np.std(maxs)
        scale_std = np.std(scales)

        print(f"   Running min across runs: {mins}")
        print(f"      Mean: {np.mean(mins):.6f}, Std: {min_std:.6f}")
        print(f"   Running max across runs: {maxs}")
        print(f"      Mean: {np.mean(maxs):.6f}, Std: {max_std:.6f}")
        print(f"   Scale across runs: {scales}")
        print(f"      Mean: {np.mean(scales):.6f}, Std: {scale_std:.6f}")

        # Check consistency
        is_consistent = (min_std < 1e-5) and (max_std < 1e-5) and (scale_std < 1e-5)
        status = "✅ CONSISTENT" if is_consistent else "⚠️ INCONSISTENT"
        print(f"   Status: {status}")

        consistency_results[quantizer_name] = {
            'min_std': min_std,
            'max_std': max_std,
            'scale_std': scale_std,
            'consistent': is_consistent
        }

    # Overall summary
    all_consistent = all(v['consistent'] for v in consistency_results.values())

    print(f"\n{'='*40}")
    print("OVERALL SUMMARY")
    print(f"{'='*40}")

    if all_consistent:
        print("✅ All quantizers show CONSISTENT calibration across runs")
        print("   Running min/max values are stable and reproducible")
    else:
        inconsistent = [k for k, v in consistency_results.items() if not v['consistent']]
        print(f"⚠️ {len(inconsistent)} quantizers show INCONSISTENT calibration")
        print(f"   Inconsistent quantizers: {inconsistent[:3]}...")

    return consistency_results


def compare_4bit_vs_8bit_calibration(sp_model, tokenizer, device):
    """
    Compare calibration statistics between 4-bit and 8-bit quantization.

    This helps verify that lower bit-widths are capturing appropriate ranges.
    """
    print(f"\n{'='*60}")
    print("4-BIT vs 8-BIT CALIBRATION COMPARISON")
    print(f"{'='*60}")

    calibration_texts = get_calibration_texts(num_texts=12)

    stats_by_precision = {}

    for precision in [4, 8]:
        print(f"\n📊 Calibrating {precision}-bit precision:")

        sp_model.set_precision(precision)
        sp_model.train()

        bits_key = f'{precision}bit'

        # Start calibration
        for name, module in sp_model.named_modules():
            if hasattr(module, 'quantizers_weight'):
                if bits_key in module.quantizers_weight:
                    module.quantizers_weight[bits_key].start_calibration()
                if bits_key in module.quantizers_input:
                    module.quantizers_input[bits_key].start_calibration()

        # Collect statistics
        with torch.no_grad():
            for text in calibration_texts:
                tokens = tokenizer(text, return_tensors='pt',
                                 max_length=128, truncation=True)['input_ids'].to(device)
                _ = sp_model(tokens)

        # Finish calibration and collect stats
        precision_stats = []

        for name, module in sp_model.named_modules():
            if hasattr(module, 'quantizers_weight'):
                if bits_key in module.quantizers_weight:
                    quantizer = module.quantizers_weight[bits_key]
                    # Show debug for first quantizer
                    show_debug = len(precision_stats) == 0
                    quantizer.finish_calibration(debug=show_debug)

                    if quantizer.calibrated:
                        stats = {
                            'name': f"{name}.weight",
                            'min': quantizer.running_min.min().item() if quantizer.per_channel else quantizer.running_min.item(),
                            'max': quantizer.running_max.max().item() if quantizer.per_channel else quantizer.running_max.item(),
                            'range': None,  # Will calculate
                            'scale': quantizer.scale.mean().item(),
                        }
                        stats['range'] = stats['max'] - stats['min']
                        precision_stats.append(stats)

                    if len(precision_stats) >= 3:  # Just check first 3
                        break

        stats_by_precision[precision] = precision_stats

    # Compare statistics
    print(f"\n{'='*40}")
    print("COMPARISON RESULTS")
    print(f"{'='*40}")

    for i in range(min(3, len(stats_by_precision[4]))):
        stats_4 = stats_by_precision[4][i]
        stats_8 = stats_by_precision[8][i]

        print(f"\n{stats_4['name']}:")
        print(f"   4-bit: range=[{stats_4['min']:.6f}, {stats_4['max']:.6f}], scale={stats_4['scale']:.6f}")
        print(f"   8-bit: range=[{stats_8['min']:.6f}, {stats_8['max']:.6f}], scale={stats_8['scale']:.6f}")

        # The ranges should be similar (same data), but scales will differ
        range_diff = abs(stats_4['range'] - stats_8['range'])
        scale_ratio = stats_4['scale'] / stats_8['scale']

        print(f"   Range difference: {range_diff:.6f}")
        print(f"   Scale ratio (4-bit/8-bit): {scale_ratio:.2f}")

        # Expected scale ratio should be approximately 2^(8-4) = 16
        expected_ratio = 2 ** (8 - 4)
        actual_vs_expected = scale_ratio / expected_ratio

        if 0.5 < actual_vs_expected < 2.0:
            print(f"   ✅ Scale ratio is reasonable (expected ~{expected_ratio}x)")
        else:
            print(f"   ⚠️ Scale ratio differs from expected (~{expected_ratio}x)")

    return stats_by_precision


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from part1_switchable_precision.config_sp import ModelConfig
    from part1_switchable_precision.main_sp import initialize_model
    from transformers import GPT2TokenizerFast

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model and tokenizer
    config = ModelConfig()
    model = initialize_model(config, device)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Run verification tests
    print("Testing calibration consistency...")
    consistency_results = verify_calibration_consistency(model, tokenizer, device, precision=8)

    print("\n" + "="*60)
    print("\nComparing 4-bit vs 8-bit calibration...")
    comparison_results = compare_4bit_vs_8bit_calibration(model, tokenizer, device)
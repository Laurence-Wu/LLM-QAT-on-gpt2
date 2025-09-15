import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import pandas as pd

class BaselineComparison:
    def __init__(self, your_results: Dict):
        self.your_results = your_results

        self.baseline_results = {
            "RTN": {
                "4-8-8": {"zero_shot_avg": 41.2, "wiki2_ppl": 42.3, "c4_ppl": 48.5},
                "8-8-8": {"zero_shot_avg": 65.8, "wiki2_ppl": 10.5, "c4_ppl": 8.2},
                "4-16-16": {"zero_shot_avg": 45.3, "wiki2_ppl": 35.6, "c4_ppl": 40.1}
            },
            "GPTQ": {
                "4-8-8": {"zero_shot_avg": 61.7, "wiki2_ppl": 11.5, "c4_ppl": 9.1},
                "8-8-8": {"zero_shot_avg": 66.1, "wiki2_ppl": 10.3, "c4_ppl": 8.0},
                "4-16-16": {"zero_shot_avg": 62.5, "wiki2_ppl": 11.2, "c4_ppl": 8.8}
            },
            "SmoothQuant": {
                "8-8-8": {"zero_shot_avg": 65.0, "wiki2_ppl": 10.7, "c4_ppl": 8.4},
                "4-8-8": {"zero_shot_avg": 58.2, "wiki2_ppl": 12.8, "c4_ppl": 10.2}
            },
            "AWQ": {
                "4-16-16": {"zero_shot_avg": 63.8, "wiki2_ppl": 10.9, "c4_ppl": 8.5},
                "4-8-8": {"zero_shot_avg": 60.4, "wiki2_ppl": 11.8, "c4_ppl": 9.3}
            },
            "LLM-QAT (Paper)": {
                "4-8-8": {"zero_shot_avg": 68.2, "wiki2_ppl": 10.8, "c4_ppl": 8.3},
                "8-8-8": {"zero_shot_avg": 69.5, "wiki2_ppl": 10.2, "c4_ppl": 7.9},
                "4-8-4": {"zero_shot_avg": 66.8, "wiki2_ppl": 11.2, "c4_ppl": 8.6},
                "4-16-16": {"zero_shot_avg": 68.9, "wiki2_ppl": 10.5, "c4_ppl": 8.1}
            }
        }

    def compare_with_baselines(self) -> Dict:
        """
        Compare your method with RTN, GPTQ, SmoothQuant, AWQ, and paper results
        Calculate relative improvement
        """
        comparison_results = {}

        for config_name, result in self.your_results.items():
            bits = result.get('bits', 'N/A')

            if bits == 'N/A':
                continue

            comparison_results[config_name] = {
                'Your Method': {
                    'zero_shot': result.get('zero_shot', {}).get('Average', 0),
                    'wiki2_ppl': result.get('perplexity', {}).get('WikiText2', float('inf')),
                    'c4_ppl': result.get('perplexity', {}).get('C4', float('inf'))
                }
            }

            for baseline_name, baseline_data in self.baseline_results.items():
                if bits in baseline_data:
                    comparison_results[config_name][baseline_name] = baseline_data[bits]

        self._print_comparison_table(comparison_results)

        return comparison_results

    def _print_comparison_table(self, comparison_results: Dict):
        """Print formatted comparison table"""
        print("\n" + "="*80)
        print("Baseline Comparison Results")
        print("="*80)

        for config_name, methods in comparison_results.items():
            print(f"\nConfiguration: {config_name}")
            print("-"*60)

            headers = ['Method', 'Zero-shot Avg↑', 'WikiText2 PPL↓', 'C4 PPL↓']
            rows = []

            for method_name, scores in methods.items():
                row = [
                    method_name,
                    f"{scores.get('zero_shot', 0):.1f}",
                    f"{scores.get('wiki2_ppl', float('inf')):.1f}",
                    f"{scores.get('c4_ppl', float('inf')):.1f}"
                ]
                rows.append(row)

            df = pd.DataFrame(rows, columns=headers)
            print(df.to_string(index=False))

            your_score = methods.get('Your Method', {}).get('zero_shot', 0)
            if 'LLM-QAT (Paper)' in methods:
                paper_score = methods['LLM-QAT (Paper)'].get('zero_shot_avg', 0)
                if paper_score > 0:
                    improvement = ((your_score - paper_score) / paper_score) * 100
                    print(f"\nRelative to LLM-QAT paper: {improvement:+.1f}%")

    def plot_accuracy_vs_bits(self):
        """Plot accuracy vs model size for all methods"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        bit_configs = {
            '4-8-8': 4.67,
            '4-8-4': 4.0,
            '4-16-16': 6.0,
            '8-8-8': 8.0,
            '16-16-16': 16.0
        }

        colors = {
            'Your Method': 'red',
            'LLM-QAT (Paper)': 'blue',
            'GPTQ': 'green',
            'RTN': 'orange',
            'SmoothQuant': 'purple',
            'AWQ': 'brown'
        }

        markers = {
            'Your Method': 'o',
            'LLM-QAT (Paper)': 's',
            'GPTQ': '^',
            'RTN': 'v',
            'SmoothQuant': 'D',
            'AWQ': 'p'
        }

        for method_name in ['Your Method', 'LLM-QAT (Paper)', 'GPTQ', 'RTN', 'SmoothQuant', 'AWQ']:
            x_vals = []
            y_acc = []
            y_ppl = []

            if method_name == 'Your Method':
                for config_name, result in self.your_results.items():
                    bits = result.get('bits', '')
                    if bits in bit_configs:
                        x_vals.append(bit_configs[bits])
                        y_acc.append(result.get('zero_shot', {}).get('Average', 0))
                        y_ppl.append(result.get('perplexity', {}).get('WikiText2', float('inf')))
            else:
                if method_name in self.baseline_results:
                    for config, scores in self.baseline_results[method_name].items():
                        if config in bit_configs:
                            x_vals.append(bit_configs[config])
                            y_acc.append(scores.get('zero_shot_avg', 0))
                            y_ppl.append(scores.get('wiki2_ppl', float('inf')))

            if x_vals:
                ax1.plot(x_vals, y_acc, marker=markers[method_name], color=colors[method_name],
                        label=method_name, markersize=8, linewidth=2, alpha=0.7)

                valid_ppl = [(x, y) for x, y in zip(x_vals, y_ppl) if y != float('inf')]
                if valid_ppl:
                    x_ppl, y_ppl_valid = zip(*valid_ppl)
                    ax2.plot(x_ppl, y_ppl_valid, marker=markers[method_name], color=colors[method_name],
                            label=method_name, markersize=8, linewidth=2, alpha=0.7)

        ax1.set_xlabel('Average Bits', fontsize=12)
        ax1.set_ylabel('Zero-shot Accuracy (%)', fontsize=12)
        ax1.set_title('Zero-shot Performance vs Model Size', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Average Bits', fontsize=12)
        ax2.set_ylabel('WikiText2 Perplexity', fontsize=12)
        ax2.set_title('Perplexity vs Model Size', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        plt.tight_layout()
        plt.savefig('part3_evaluation/results/baseline_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def calculate_degradation_from_fp16(self) -> Dict:
        """Calculate performance drop from FP16 baseline"""
        degradation_results = {}

        fp16_result = None
        for config_name, result in self.your_results.items():
            if '16-16-16' in result.get('bits', '') or 'FP16' in config_name:
                fp16_result = result
                break

        if not fp16_result:
            print("Warning: No FP16 baseline found")
            return degradation_results

        fp16_zero_shot = fp16_result.get('zero_shot', {}).get('Average', 100)
        fp16_wiki2 = fp16_result.get('perplexity', {}).get('WikiText2', 1)

        for config_name, result in self.your_results.items():
            if config_name == 'FP16' or '16-16-16' in result.get('bits', ''):
                continue

            zero_shot = result.get('zero_shot', {}).get('Average', 0)
            wiki2 = result.get('perplexity', {}).get('WikiText2', float('inf'))

            degradation_results[config_name] = {
                'zero_shot_drop': ((fp16_zero_shot - zero_shot) / fp16_zero_shot) * 100,
                'ppl_increase': ((wiki2 - fp16_wiki2) / fp16_wiki2) * 100 if wiki2 != float('inf') else float('inf')
            }

        print("\n" + "="*60)
        print("Degradation from FP16 Baseline")
        print("="*60)

        for config_name, degradation in degradation_results.items():
            print(f"\n{config_name}:")
            print(f"  Zero-shot accuracy drop: {degradation['zero_shot_drop']:.1f}%")
            if degradation['ppl_increase'] != float('inf'):
                print(f"  Perplexity increase: {degradation['ppl_increase']:.1f}%")

        return degradation_results
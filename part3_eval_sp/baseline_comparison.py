from typing import Dict
import numpy as np

class BaselineComparison:

    def __init__(self, results: Dict):
        self.results = results

    def compare_with_baselines(self):
        print('\n=== Baseline Comparison ===')
        print('Comparing LLM-QAT with standard quantization approaches...')
        for config_name, result in self.results.items():
            if 'zero_shot' in result and result['zero_shot']:
                avg_score = result['zero_shot'].get('Average', 0)
                print(f'{config_name}: {avg_score:.1f}% (zero-shot avg)')

    def plot_accuracy_vs_bits(self):
        print('Plotting accuracy vs bits tradeoff...')

    def calculate_degradation_from_fp16(self):
        if 'FP16' not in self.results:
            print('No FP16 baseline found')
            return
        fp16_result = self.results['FP16']
        print('\n=== Degradation from FP16 ===')
        for config_name, result in self.results.items():
            if config_name == 'FP16':
                continue
            degradation = {}
            if 'zero_shot' in result and 'zero_shot' in fp16_result:
                fp16_avg = fp16_result['zero_shot'].get('Average', 0)
                config_avg = result['zero_shot'].get('Average', 0)
                degradation['zero_shot'] = fp16_avg - config_avg
            if 'perplexity' in result and 'perplexity' in fp16_result:
                fp16_ppl = fp16_result['perplexity'].get('WikiText2', float('inf'))
                config_ppl = result['perplexity'].get('WikiText2', float('inf'))
                degradation['perplexity'] = config_ppl - fp16_ppl
            print(f"\n{config_name} ({result.get('bits', 'N/A')}):")
            if 'zero_shot' in degradation:
                print(f"  Zero-shot degradation: {degradation['zero_shot']:.1f}%")
            if 'perplexity' in degradation:
                print(f"  Perplexity increase: {degradation['perplexity']:.1f}")
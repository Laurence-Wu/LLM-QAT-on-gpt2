import torch
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import json
from pathlib import Path

class CPTEvaluation:
    """Evaluation suite for Cyclic Precision Training models"""

    def __init__(self, model, tokenizer, model_size='GPT2', device='cuda'):
        # Force CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This evaluation requires CUDA.")

        self.device = torch.device('cuda')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.model_size = model_size

        # Verify model is on CUDA
        if not next(self.model.parameters()).is_cuda:
            raise RuntimeError("Model failed to move to CUDA")

        self.model_params = self._count_parameters()

    def _count_parameters(self):
        """Count model parameters in millions"""
        return sum(p.numel() for p in self.model.parameters()) / 1e6

    def evaluate_zero_shot_common_sense(self, bit_config: Dict) -> Dict:
        """
        Evaluate on available tasks: BoolQ, HellaSwag,
        WinoGrande, ARC-e, ARC-c, OBQA
        Return individual scores and average
        """
        from zero_shot_tasks import ZeroShotEvaluator

        self._apply_bit_config(bit_config)

        evaluator = ZeroShotEvaluator(self.model, self.tokenizer)
        results = evaluator.evaluate_all_tasks(bit_config)

        # Return the results as-is from the evaluator
        return results

    def evaluate_perplexity(self, bit_config: Dict) -> Dict:
        """
        Calculate perplexity on WikiText2 and OpenWebText
        Return both values
        """
        from perplexity_eval import PerplexityEvaluator

        self._apply_bit_config(bit_config)

        evaluator = PerplexityEvaluator(self.model, self.tokenizer)
        results = evaluator.evaluate_all_datasets(bit_config)

        return {
            'WikiText2': results.get('WikiText2', float('inf')),
            'OpenWebText': results.get('OpenWebText', float('inf'))
        }

    # Few-shot QA evaluation removed - focusing on language modeling metrics

    def calculate_model_size(self, bit_config: Dict) -> float:
        """
        Calculate model size in GB based on bit configuration
        Account for weights and KV cache
        """
        try:
            weight_bits = bit_config['W']
        except KeyError:
            weight_bits = 16

        try:
            kv_bits = bit_config['KV']
        except KeyError:
            kv_bits = 16

        weight_size_gb = (self.model_params * weight_bits) / (8 * 1024)

        # Handle CPT model config structure (config is a dict)
        try:
            # CPT model has config as dictionary
            model_config = self.model.config['model']
            n_layers = model_config.n_layer
            n_heads = model_config.n_head
            n_embd = model_config.n_embd
        except (TypeError, KeyError, AttributeError) as e:
            print(f"Warning: Cannot access CPT model config: {e}, using defaults")
            n_layers = 12
            n_heads = 12
            n_embd = 768

        d_head = n_embd // n_heads
        max_seq_len = 2048
        batch_size = 1

        kv_cache_size_gb = (2 * n_layers * n_heads * d_head * max_seq_len * batch_size * kv_bits) / (8 * 1024**3)

        total_size_gb = weight_size_gb + kv_cache_size_gb

        return round(total_size_gb, 2)

    def _apply_bit_config(self, bit_config: Dict):
        """Apply W-A-KV configuration to CPT model"""
        try:
            weight_bits = bit_config['W']
        except KeyError:
            weight_bits = 8

        # CPT models use set_precision method
        try:
            self.model.set_precision(weight_bits)
            print(f"Set CPT model precision to {weight_bits}-bit")
        except AttributeError as e:
            print(f"Warning: Model does not support set_precision: {e}")
            pass

        # CPT doesn't have KV cache bit setting, but keep for compatibility
        try:
            kv_bits = bit_config['KV']
            # CPT doesn't support this, but no error needed
        except KeyError:
            pass

    def run_complete_evaluation(self, configs: List[str] = None, skip_few_shot: bool = True) -> Dict:
        """
        Run all evaluations for standard configurations
        Generate paper-style tables
        """
        from bit_configurations import BitConfigurations
        from generate_tables import ResultTableGenerator

        if configs is None:
            configs = ['FP16', 'INT8', 'W4A8KV8', 'W4A8KV4']

        all_results = {}

        for config_name in configs:
            print(f"\n{'='*50}")
            print(f"Evaluating configuration: {config_name}")
            print('='*50)

            config = BitConfigurations.STANDARD_CONFIGS.get(config_name)
            if not config:
                print(f"Warning: Configuration {config_name} not found")
                continue

            results = {
                'config_name': config['name'],
                'bits': f"{config['W']}-{config['A']}-{config['KV']}",
                'model_size_gb': self.calculate_model_size(config)
            }

            print(f"Model size: {results['model_size_gb']} GB")

            print("\n1. Zero-shot common sense evaluation...")
            zero_shot_results = self.evaluate_zero_shot_common_sense(config)
            results['zero_shot'] = zero_shot_results
            print(f"   Average score: {zero_shot_results['Average']:.1f}%")

            print("\n2. Perplexity evaluation...")
            perplexity_results = self.evaluate_perplexity(config)
            results['perplexity'] = perplexity_results
            print(f"   WikiText2: {perplexity_results['WikiText2']:.1f}")
            print(f"   OpenWebText: {perplexity_results.get('OpenWebText', float('inf')):.1f}")

            # Few-shot evaluation removed

            all_results[config_name] = results

        print("\n" + "="*50)
        print("Generating result tables...")
        print("="*50)

        table_gen = ResultTableGenerator(all_results)
        table_gen.generate_table_1_zero_shot()
        table_gen.generate_table_2_perplexity()

        # Few-shot table generation removed

        output_dir = Path('part3_eval_cpt/results')
        output_dir.mkdir(exist_ok=True, parents=True)

        with open(output_dir / 'cpt_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nResults saved to {output_dir / 'cpt_results.json'}")

        return all_results
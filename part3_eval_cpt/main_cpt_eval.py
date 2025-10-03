import json
import argparse
from pathlib import Path
import torch
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_cpt_model import load_cpt_model
from transformers import GPT2Tokenizer
from cpt_metrics import CPTEvaluation
from zero_shot_tasks import ZeroShotEvaluator
from perplexity_eval import PerplexityEvaluator

def load_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_evaluation_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='CPT Model Evaluation Suite')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained CPT model checkpoint (.pth file)')
    parser.add_argument('--eval_config', type=str, default='evaluation_config.json', help='Path to evaluation configuration JSON file')
    args = parser.parse_args()
    eval_config = load_evaluation_config(args.eval_config)
    print(f'Loaded config from: {args.eval_config}')
    model, checkpoint_bit_width, model_config, training_config = load_cpt_model(args.model_path)
    tokenizer = load_tokenizer()
    device = eval_config['device']
    evaluator = CPTEvaluation(model, tokenizer)
    zero_shot_evaluator = ZeroShotEvaluator(model, tokenizer, device=device, config=eval_config['zero_shot'])
    perplexity_evaluator = PerplexityEvaluator(model, tokenizer, device=device, config=eval_config['perplexity'])
    current_bits = checkpoint_bit_width or model_config.default_bits
    print(f'\nRunning CPT Evaluation at {current_bits}-bit precision')
    results = {'model_type': 'CPT', 'bit_width': current_bits, 'model_size_gb': evaluator.calculate_model_size({'W': current_bits}), 'compression_ratio': 32 / current_bits}
    bit_config = {'W': current_bits, 'A': current_bits, 'KV': current_bits}
    print('\n1. Perplexity evaluation...')
    perplexity_results = perplexity_evaluator.evaluate_all_datasets(bit_config)
    results['perplexity'] = perplexity_results
    for dataset, ppl in perplexity_results.items():
        print(f'   {dataset}: {ppl:.1f}')
    print('\n2. Zero-shot evaluation...')
    zero_shot_results = zero_shot_evaluator.evaluate_all_tasks(bit_config)
    results['zero_shot'] = zero_shot_results
    for task, score in zero_shot_results.items():
        if task != 'Average':
            print(f'   {task}: {score:.1f}%')
    if 'Average' in zero_shot_results:
        print(f"   Average: {zero_shot_results['Average']:.1f}%")
    output_dir = Path(eval_config['output']['directory'])
    output_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'cpt_results_{current_bits}bit_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {results_file}')
if __name__ == '__main__':
    main()
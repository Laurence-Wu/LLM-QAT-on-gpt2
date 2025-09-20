#!/usr/bin/env python3
"""
Test script to evaluate GPT-2 model perplexity using the correct sliding window approach.
Based on the standard evaluation methodology for language models.
"""

import os
import sys
import torch
import math
from tqdm import tqdm
import argparse
import json
import time
from typing import Dict, Optional, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset


def evaluate_perplexity_correctly(model, tokenizer, dataset_name='wikitext', split='test',
                                 max_length=1024, stride=512, device='cuda'):
    """
    Correctly evaluate perplexity using sliding window approach.

    Args:
        model: The language model to evaluate
        tokenizer: The tokenizer for the model
        dataset_name: Name of the dataset ('wikitext', 'c4', 'ptb')
        split: Dataset split to use
        max_length: Maximum length of each window
        stride: Stride for sliding window
        device: Device to use for evaluation

    Returns:
        Dictionary containing perplexity and other metrics
    """
    model = model.to(device)
    model.eval()

    # Load dataset
    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        # Concatenate all text (preserving document boundaries)
        text = '\n\n'.join([t for t in dataset['text'] if t.strip()])
    elif dataset_name == 'c4':
        # For C4, we'll use a subset
        dataset = load_dataset('c4', 'en', split=split, streaming=True)
        texts = []
        for i, sample in enumerate(dataset):
            if i >= 100:  # Limit to 100 documents for C4
                break
            texts.append(sample['text'])
        text = '\n\n'.join(texts)
    elif dataset_name == 'ptb':
        dataset = load_dataset('ptb_text_only', split=split)
        text = ' '.join([sample['sentence'] for sample in dataset])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Tokenize as one long sequence
    encodings = tokenizer(text, return_tensors='pt')

    # Evaluate in sliding windows
    nlls = []
    prev_end_loc = 0

    print(f"Total tokens: {encodings.input_ids.size(1):,}")
    print(f"Window size: {max_length}, Stride: {stride}")

    # Create progress bar
    num_windows = (encodings.input_ids.size(1) - 1) // stride + 1
    pbar = tqdm(total=num_windows, desc=f"Evaluating {dataset_name}")

    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # Only compute loss on the last trg_len tokens

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        pbar.update(1)

        if i == 0:
            print(f"First window loss: {outputs.loss.item():.4f}")

    pbar.close()

    # Calculate perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

    return {
        'perplexity': ppl.item(),
        'total_tokens': end_loc,
        'num_windows': len(nlls),
        'avg_loss': (torch.stack(nlls).sum() / end_loc).item()
    }


def test_standard_gpt2(model_name='gpt2', datasets=['wikitext']):
    """
    Test standard GPT-2 model perplexity using correct evaluation.

    Args:
        model_name: Name of the GPT-2 model variant
        datasets: List of datasets to evaluate on

    Returns:
        Dictionary of results for each dataset
    """
    print("=" * 60)
    print(f"Testing Standard GPT-2 Model: {model_name}")
    print("=" * 60)

    # Load model and tokenizer
    print(f"Loading {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: {num_params * 4 / 1e9:.2f} GB (FP32)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Evaluate on each dataset
    results = {}
    for dataset_name in datasets:
        print(f"\n{'='*40}")
        print(f"Evaluating on {dataset_name}")
        print('='*40)

        result = evaluate_perplexity_correctly(
            model=model,
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            split='test' if dataset_name != 'c4' else 'validation',
            device=device
        )

        results[dataset_name] = result

        print(f"\nResults for {dataset_name}:")
        print(f"  Perplexity: {result['perplexity']:.2f}")
        print(f"  Average loss: {result['avg_loss']:.4f}")
        print(f"  Total tokens: {result['total_tokens']:,}")
        print(f"  Number of windows: {result['num_windows']}")

    return results


def test_quantized_model(model_path, datasets=['wikitext']):
    """
    Test quantized/switchable precision model perplexity.

    Args:
        model_path: Path to the quantized model checkpoint
        datasets: List of datasets to evaluate on

    Returns:
        Dictionary of results for each bit-width and dataset
    """
    print("=" * 60)
    print(f"Testing Quantized Model: {model_path}")
    print("=" * 60)

    # Import the switchable model
    from shared.models import SwitchableQATGPT2
    from transformers import GPT2Config

    # Load checkpoint
    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cuda')

    # Extract configuration
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        bit_widths = model_config.get('bit_widths', [4, 8, 16])

        # Create GPT2Config
        config = GPT2Config(
            vocab_size=model_config.get('vocab_size', 50257),
            n_positions=model_config.get('n_positions', 1024),
            n_embd=model_config['n_embd'],
            n_layer=model_config['n_layer'],
            n_head=model_config['n_head']
        )

        # Create model
        model = SwitchableQATGPT2(config, bit_widths=bit_widths)

        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        print(f"Model supports bit-widths: {bit_widths}")
    else:
        raise ValueError("Invalid checkpoint format")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test each bit-width configuration
    all_results = {}

    for bits in bit_widths:
        print(f"\n{'='*40}")
        print(f"Testing {bits}-bit precision")
        print('='*40)

        # Set model precision
        model.set_precision(bits)

        # Evaluate on each dataset
        bit_results = {}
        for dataset_name in datasets:
            print(f"\nEvaluating {dataset_name} at {bits}-bit...")

            result = evaluate_perplexity_correctly(
                model=model,
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                split='test' if dataset_name != 'c4' else 'validation',
                device=device
            )

            bit_results[dataset_name] = result

            print(f"Results for {dataset_name} ({bits}-bit):")
            print(f"  Perplexity: {result['perplexity']:.2f}")
            print(f"  Average loss: {result['avg_loss']:.4f}")

        all_results[f'{bits}bit'] = bit_results

    return all_results


def compare_results(standard_results: Dict, quantized_results: Optional[Dict] = None):
    """
    Compare standard and quantized model results.

    Args:
        standard_results: Results from standard GPT-2 model
        quantized_results: Results from quantized model (optional)
    """
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    # Print standard model results
    print("\nStandard GPT-2 (FP32):")
    for dataset, results in standard_results.items():
        print(f"  {dataset}: PPL = {results['perplexity']:.2f}")

    # Print quantized model results if available
    if quantized_results:
        print("\nQuantized Model:")
        for bit_config, datasets in quantized_results.items():
            print(f"\n  {bit_config}:")
            for dataset, results in datasets.items():
                ppl = results['perplexity']
                standard_ppl = standard_results.get(dataset, {}).get('perplexity', 0)
                if standard_ppl > 0:
                    degradation = ((ppl - standard_ppl) / standard_ppl) * 100
                    print(f"    {dataset}: PPL = {ppl:.2f} (degradation: {degradation:+.1f}%)")
                else:
                    print(f"    {dataset}: PPL = {ppl:.2f}")


def quick_test():
    """
    Quick test function to verify the implementation works correctly.
    This matches the provided reference implementation.
    """
    print("Running quick verification test...")

    model = GPT2LMHeadModel.from_pretrained('gpt2').cuda().eval()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Load WikiText-2 properly
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Concatenate all text (preserving document boundaries)
    text = '\n\n'.join([t for t in dataset['text'] if t.strip()])

    # Tokenize as one long sequence
    encodings = tokenizer(text, return_tensors='pt')

    # Evaluate in sliding windows
    max_length = 1024
    stride = 512
    nlls = []

    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop

        input_ids = encodings.input_ids[:, begin_loc:end_loc].cuda()

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        if i == 0:
            print(f"First window loss: {outputs.loss.item():.4f}")

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    print(f"Correct GPT-2 perplexity: {ppl.item():.2f}")
    return ppl.item()


def main():
    parser = argparse.ArgumentParser(description='Test GPT-2 Model Perplexity (Correct Implementation)')
    parser.add_argument('--model', type=str, default='gpt2',
                       help='Model to test (gpt2, gpt2-medium, gpt2-large, or path to checkpoint)')
    parser.add_argument('--quantized_model', type=str, default=None,
                       help='Path to quantized/switchable model checkpoint')
    parser.add_argument('--datasets', nargs='+', default=['wikitext'],
                       choices=['wikitext', 'c4', 'ptb'],
                       help='Datasets to evaluate on')
    parser.add_argument('--output', type=str, default='perplexity_results.json',
                       help='Output file for results')
    parser.add_argument('--skip_standard', action='store_true',
                       help='Skip standard model evaluation')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick verification test')
    parser.add_argument('--max_length', type=int, default=1024,
                       help='Maximum length for sliding window')
    parser.add_argument('--stride', type=int, default=512,
                       help='Stride for sliding window')
    args = parser.parse_args()

    # Run quick test if requested
    if args.quick_test:
        quick_test()
        return

    # Test standard GPT-2
    standard_results = None
    if not args.skip_standard:
        standard_results = test_standard_gpt2(
            model_name=args.model,
            datasets=args.datasets
        )

    # Test quantized model if provided
    quantized_results = None
    if args.quantized_model:
        quantized_results = test_quantized_model(
            model_path=args.quantized_model,
            datasets=args.datasets
        )

    # Compare results
    if standard_results:
        compare_results(standard_results, quantized_results)

    # Save results
    all_results = {
        'timestamp': time.strftime('%Y%m%d_%H%M%S'),
        'standard_model': args.model,
        'standard_results': standard_results,
        'quantized_model': args.quantized_model,
        'quantized_results': quantized_results,
        'datasets': args.datasets,
        'window_config': {
            'max_length': args.max_length,
            'stride': args.stride
        }
    }

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {args.output}")

    # Print final summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    if standard_results and 'wikitext' in standard_results:
        print(f"Standard GPT-2 WikiText-2 Perplexity: {standard_results['wikitext']['perplexity']:.2f}")

    if quantized_results:
        print("\nQuantized Model Results:")
        for bit_config, datasets in quantized_results.items():
            if 'wikitext' in datasets:
                print(f"  {bit_config}: WikiText-2 PPL = {datasets['wikitext']['perplexity']:.2f}")


if __name__ == "__main__":
    main()
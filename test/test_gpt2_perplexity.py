#!/usr/bin/env python3
"""
Test script to evaluate GPT-2 model perplexity on various datasets.
This script tests both standard GPT-2 and quantized models.
"""

import os
import sys
import torch
import torch.nn.functional as F
import math
from tqdm import tqdm
import argparse
from typing import Dict, Optional
import json
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    """Simple dataset for text data."""

    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


class PerplexityEvaluator:
    """Evaluator for computing perplexity of language models."""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    def calculate_perplexity(self, data_loader, max_samples=None, desc="Calculating perplexity"):
        """
        Calculate perplexity on a dataset.

        Args:
            data_loader: DataLoader for the dataset
            max_samples: Maximum number of samples to evaluate (None for all)
            desc: Description for progress bar

        Returns:
            Dictionary containing perplexity and other metrics
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        num_samples = 0

        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc=desc)
            for batch_idx, batch in enumerate(progress_bar):
                if max_samples and num_samples >= max_samples:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )

                # Get loss
                loss = outputs.loss

                # Count actual tokens (excluding padding)
                num_tokens = attention_mask.sum().item()

                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                num_samples += input_ids.shape[0]

                # Update progress bar
                if total_tokens > 0:
                    current_ppl = math.exp(total_loss / total_tokens) if total_loss / total_tokens < 20 else float('inf')
                    progress_bar.set_postfix({'ppl': f'{current_ppl:.2f}', 'samples': num_samples})

        # Calculate final perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')

        return {
            'perplexity': perplexity,
            'loss': avg_loss,
            'total_tokens': total_tokens,
            'num_samples': num_samples
        }

    def evaluate_on_dataset(self, dataset_name='wikitext', split='test', max_samples=100):
        """
        Evaluate perplexity on a specific dataset.

        Args:
            dataset_name: Name of the dataset ('wikitext', 'c4', 'ptb')
            split: Dataset split to use
            max_samples: Maximum number of samples to evaluate

        Returns:
            Dictionary containing evaluation results
        """
        print(f"\nEvaluating on {dataset_name} ({split} split)...")

        # Load dataset
        if dataset_name == 'wikitext':
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
            texts = [sample['text'] for sample in dataset if len(sample['text'].strip()) > 0]
        elif dataset_name == 'c4':
            dataset = load_dataset('c4', 'en', split=split, streaming=True)
            texts = []
            for i, sample in enumerate(dataset):
                if i >= max_samples:
                    break
                texts.append(sample['text'])
        elif dataset_name == 'ptb':
            dataset = load_dataset('ptb_text_only', split=split)
            texts = [sample['sentence'] for sample in dataset]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Limit samples if needed
        if max_samples:
            texts = texts[:max_samples]

        print(f"Loaded {len(texts)} text samples")

        # Create dataset and dataloader
        text_dataset = TextDataset(texts, self.tokenizer, max_length=512)
        data_loader = DataLoader(text_dataset, batch_size=4, shuffle=False)

        # Calculate perplexity
        results = self.calculate_perplexity(
            data_loader,
            max_samples=max_samples,
            desc=f"Evaluating {dataset_name}"
        )

        return results


def test_standard_gpt2(model_name='gpt2', datasets=['wikitext'], max_samples=100):
    """Test standard GPT-2 model perplexity."""

    print("="*60)
    print(f"Testing Standard GPT-2 Model: {model_name}")
    print("="*60)

    # Load model and tokenizer
    print(f"Loading {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: {num_params * 4 / 1e9:.2f} GB (FP32)")

    # Initialize evaluator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    evaluator = PerplexityEvaluator(model, tokenizer, device)

    # Evaluate on each dataset
    all_results = {}
    for dataset_name in datasets:
        results = evaluator.evaluate_on_dataset(
            dataset_name=dataset_name,
            split='test' if dataset_name != 'c4' else 'validation',
            max_samples=max_samples
        )

        all_results[dataset_name] = results

        print(f"\nResults for {dataset_name}:")
        print(f"  Perplexity: {results['perplexity']:.2f}")
        print(f"  Loss: {results['loss']:.4f}")
        print(f"  Total tokens: {results['total_tokens']:,}")
        print(f"  Samples evaluated: {results['num_samples']}")

    return all_results


def test_quantized_model(model_path, config_path=None, datasets=['wikitext'], max_samples=100):
    """Test quantized/switchable precision model perplexity."""

    print("="*60)
    print(f"Testing Quantized Model: {model_path}")
    print("="*60)

    # Import the switchable model
    from shared.models import SwitchableQATGPT2

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
            n_positions=model_config.get('n_positions', 256),
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
    tokenizer.pad_token = tokenizer.eos_token

    # Test each bit-width configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = {}

    for bits in bit_widths:
        print(f"\n{'='*40}")
        print(f"Testing {bits}-bit precision")
        print(f"{'='*40}")

        # Set model precision
        model.set_precision(bits)

        # Initialize evaluator
        evaluator = PerplexityEvaluator(model, tokenizer, device)

        # Evaluate on each dataset
        bit_results = {}
        for dataset_name in datasets:
            results = evaluator.evaluate_on_dataset(
                dataset_name=dataset_name,
                split='test' if dataset_name != 'c4' else 'validation',
                max_samples=max_samples
            )

            bit_results[dataset_name] = results

            print(f"\nResults for {dataset_name} ({bits}-bit):")
            print(f"  Perplexity: {results['perplexity']:.2f}")
            print(f"  Loss: {results['loss']:.4f}")
            print(f"  Total tokens: {results['total_tokens']:,}")

        all_results[f'{bits}bit'] = bit_results

    return all_results


def compare_models(standard_results, quantized_results=None):
    """Compare standard and quantized model results."""

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

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


def main():
    parser = argparse.ArgumentParser(description='Test GPT-2 Model Perplexity')
    parser.add_argument('--model', type=str, default='gpt2',
                       help='Model to test (gpt2, gpt2-medium, gpt2-large, or path to checkpoint)')
    parser.add_argument('--quantized_model', type=str, default=None,
                       help='Path to quantized/switchable model checkpoint')
    parser.add_argument('--datasets', nargs='+', default=['wikitext'],
                       choices=['wikitext', 'c4', 'ptb'],
                       help='Datasets to evaluate on')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Maximum samples to evaluate per dataset')
    parser.add_argument('--output', type=str, default='perplexity_results.json',
                       help='Output file for results')
    parser.add_argument('--skip_standard', action='store_true',
                       help='Skip standard model evaluation')
    args = parser.parse_args()

    # Test standard GPT-2
    standard_results = None
    if not args.skip_standard:
        standard_results = test_standard_gpt2(
            model_name=args.model,
            datasets=args.datasets,
            max_samples=args.max_samples
        )

    # Test quantized model if provided
    quantized_results = None
    if args.quantized_model:
        quantized_results = test_quantized_model(
            model_path=args.quantized_model,
            datasets=args.datasets,
            max_samples=args.max_samples
        )

    # Compare results
    if standard_results:
        compare_models(standard_results, quantized_results)

    # Save results
    all_results = {
        'timestamp': time.strftime('%Y%m%d_%H%M%S'),
        'standard_model': args.model,
        'standard_results': standard_results,
        'quantized_model': args.quantized_model,
        'quantized_results': quantized_results,
        'datasets': args.datasets,
        'max_samples': args.max_samples
    }

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {args.output}")

    # Print final summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

    if standard_results and 'wikitext' in standard_results:
        print(f"Standard GPT-2 WikiText-2 Perplexity: {standard_results['wikitext']['perplexity']:.2f}")

    if quantized_results:
        print("\nQuantized Model Best Results:")
        for bit_config, datasets in quantized_results.items():
            if 'wikitext' in datasets:
                print(f"  {bit_config}: {datasets['wikitext']['perplexity']:.2f}")


if __name__ == "__main__":
    main()
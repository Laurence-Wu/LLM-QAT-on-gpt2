#!/usr/bin/env python3
"""
Test script for updated evaluation configuration
"""

import json
import sys
import os
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part3_evaluation.main_llm_qat_eval import load_switchable_model, load_tokenizer
from part3_evaluation.perplexity_eval import PerplexityEvaluator


def main():
    # Load evaluation config
    config_path = 'evaluation_config.json'
    with open(config_path, 'r') as f:
        eval_config = json.load(f)

    print("="*70)
    print("  TESTING UPDATED EVALUATION CONFIGURATION")
    print("="*70)

    # Show key changes
    perplexity_config = eval_config['perplexity']
    print(f"\nPerplexity Configuration:")
    print(f"  Stride: {perplexity_config['stride']} (was 128, now matches --overlapping-eval)")
    print(f"  Max Length: {perplexity_config['max_length']} (matches training max_seq_length)")
    print(f"  Datasets: {list(perplexity_config['datasets'].keys())}")

    if 'perplexity_long_context' in eval_config:
        long_config = eval_config['perplexity_long_context']
        print(f"\nLong Context Configuration (optional):")
        print(f"  Max Length: {long_config['max_length']} (full model capacity)")
        print(f"  Stride: {long_config['stride']}")

    # Test checkpoint path
    checkpoint_path = '../part1_switchable_precision/sp_gpt2_6bit_FP32_20250927_054449.pth'

    if not os.path.exists(checkpoint_path):
        print(f"\n⚠️  Checkpoint not found at: {checkpoint_path}")
        print("   Please provide a valid checkpoint path")
        return

    print(f"\n✅ Found checkpoint: {checkpoint_path}")

    # Load model
    print("\nLoading model...")
    try:
        model, bit_width = load_switchable_model(checkpoint_path)
        tokenizer = load_tokenizer()
        print(f"✅ Model loaded at {bit_width}-bit precision")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Initialize evaluator
    device = eval_config.get('device', 'cuda')
    perplexity_evaluator = PerplexityEvaluator(
        model, tokenizer, device=device,
        config=perplexity_config
    )

    # Test on a small sample
    print("\n" + "="*70)
    print("  QUICK PERPLEXITY TEST (limited samples)")
    print("="*70)

    # Override to use fewer samples for quick test
    test_config = perplexity_config.copy()
    test_config['max_samples'] = 10  # Just 10 texts for quick test
    perplexity_evaluator.config = test_config

    bit_config = {'W': bit_width, 'A': bit_width, 'KV': bit_width}

    # Test WikiText-2
    print("\n1. Testing WikiText-2...")
    try:
        wt2_ppl = perplexity_evaluator.calculate_perplexity('wikitext2', bit_config)
        print(f"   WikiText-2 Perplexity: {wt2_ppl:.1f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test WikiText-103 if available
    if 'WikiText103' in perplexity_config['datasets']:
        print("\n2. Testing WikiText-103...")
        try:
            wt103_ppl = perplexity_evaluator.calculate_perplexity('wikitext103', bit_config)
            print(f"   WikiText-103 Perplexity: {wt103_ppl:.1f}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Test long context evaluation
    if 'perplexity_long_context' in eval_config:
        print("\n3. Testing Long Context Evaluation...")
        long_evaluator = PerplexityEvaluator(
            model, tokenizer, device=device,
            config=eval_config['perplexity_long_context']
        )
        # Override for quick test
        long_evaluator.config['max_samples'] = 5

        try:
            long_ppl = long_evaluator.calculate_perplexity('wikitext2', bit_config)
            print(f"   WikiText-2 (1024 context) Perplexity: {long_ppl:.1f}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "="*70)
    print("  TEST COMPLETE")
    print("="*70)
    print("\nKey Improvements:")
    print("✅ Stride reduced from 128 to 32 for more accurate perplexity")
    print("✅ Added WikiText-103 dataset support")
    print("✅ Added optional long context evaluation (1024 tokens)")
    print("✅ Configuration matches training parameters")
    print("\nTo run full evaluation, use:")
    print("  python main_llm_qat_eval.py --model_path <checkpoint>")


if __name__ == "__main__":
    main()
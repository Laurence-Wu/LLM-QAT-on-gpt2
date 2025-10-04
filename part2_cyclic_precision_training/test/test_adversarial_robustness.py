"""
Test script for adversarial robustness evaluation
Tests TextFooler and BERT-Attack implementations
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part4_randomSwitching.adversarial_attacks import TextFoolerAttack, BERTAttack, AttackEvaluator
from part4_randomSwitching.simplified_random_switching import load_sp_model_with_bit_config
from transformers import GPT2Tokenizer


def test_textfooler():
    """Test TextFooler attack implementation"""
    print("="*60)
    print("TEST 1: TextFooler Attack")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a simple GPT-2 model for testing
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Test TextFooler
    textfooler = TextFoolerAttack(model, tokenizer, device)

    test_text = "The quick brown fox jumps over the lazy dog in the forest"
    print(f"\nOriginal text: {test_text}")

    result = textfooler.generate_adversarial(test_text, max_perturb_ratio=0.3)

    print(f"\nAdversarial text: {result['adversarial_text']}")
    print(f"Success: {result['success']}")
    print(f"Number of changes: {result['num_changes']}")
    print(f"Perturbation ratio: {result['perturb_ratio']:.2%}")
    print(f"Original perplexity: {result['original_perplexity']:.2f}")
    print(f"Adversarial perplexity: {result['adversarial_perplexity']:.2f}")
    print(f"Perplexity increase: {result['perplexity_increase']:.2%}")

    print("\n✓ TextFooler test passed")
    return True


def test_bert_attack():
    """Test BERT-Attack implementation"""
    print("\n" + "="*60)
    print("TEST 2: BERT-Attack")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a simple GPT-2 model for testing
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Test BERT-Attack
    bert_attack = BERTAttack(model, tokenizer, device)

    test_text = "The quick brown fox jumps over the lazy dog in the forest"
    print(f"\nOriginal text: {test_text}")

    result = bert_attack.generate_adversarial(test_text, max_perturb_ratio=0.3)

    print(f"\nAdversarial text: {result['adversarial_text']}")
    print(f"Success: {result['success']}")
    print(f"Number of changes: {result['num_changes']}")
    print(f"Perturbation ratio: {result['perturb_ratio']:.2%}")
    print(f"Original perplexity: {result['original_perplexity']:.2f}")
    print(f"Adversarial perplexity: {result['adversarial_perplexity']:.2f}")
    print(f"Perplexity increase: {result['perplexity_increase']:.2%}")

    print("\n✓ BERT-Attack test passed")
    return True


def test_attack_evaluator():
    """Test AttackEvaluator with both attacks"""
    print("\n" + "="*60)
    print("TEST 3: AttackEvaluator")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a simple GPT-2 model for testing
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create test samples
    test_samples = []
    test_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language for data science"
    ]

    for text in test_texts:
        input_ids = tokenizer.encode(text, return_tensors='pt').squeeze(0)
        test_samples.append({
            'text': text,
            'input_ids': input_ids,
            'labels': input_ids.clone()
        })

    # Create evaluator
    evaluator = AttackEvaluator(model, tokenizer, device)

    # Test TextFooler evaluation
    print("\nEvaluating TextFooler...")
    tf_results = evaluator.evaluate_textfooler(test_samples, max_samples=3)
    print(f"Attack success rate: {tf_results['attack_success_rate']:.2%}")
    print(f"Avg perturbation ratio: {tf_results['avg_perturb_ratio']:.2%}")

    # Test BERT-Attack evaluation
    print("\nEvaluating BERT-Attack...")
    bert_results = evaluator.evaluate_bert_attack(test_samples, max_samples=3)
    print(f"Attack success rate: {bert_results['attack_success_rate']:.2%}")
    print(f"Avg perturbation ratio: {bert_results['avg_perturb_ratio']:.2%}")
    print(f"Avg perplexity increase: {bert_results['avg_perplexity_increase']:.2%}")

    print("\n✓ AttackEvaluator test passed")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ADVERSARIAL ROBUSTNESS TESTS")
    print("="*60)

    try:
        # Run tests
        test_textfooler()
        test_bert_attack()
        test_attack_evaluator()

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

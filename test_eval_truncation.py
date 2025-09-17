#!/usr/bin/env python3
"""
Test script to verify evaluation works with n_positions=256 models
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import GPT2Config, GPT2Tokenizer
from shared.models import SwitchableQATGPT2
from part3_evaluation.zero_shot_tasks import ZeroShotEvaluator
from part3_evaluation.few_shot_eval import FewShotEvaluator

def test_truncation_handling():
    """Test that evaluation handles limited context models correctly"""
    print("Testing evaluation with n_positions=256 model...")

    # Create small model with limited context
    config = GPT2Config(
        n_positions=256,  # Limited context
        n_embd=768,
        n_layer=2,  # Small for testing
        n_head=12
    )

    model = SwitchableQATGPT2(config, bit_widths=[8], initialize_weights=True)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"Model n_positions: {model.config.n_positions}")
    print(f"Device: {device}")

    # Test zero-shot evaluation
    print("\n1. Testing zero-shot evaluation with truncation...")
    zero_shot_eval = ZeroShotEvaluator(model, tokenizer, device)

    # Test with a mock example that would normally be too long
    test_example_arc = {
        'question': 'This is a very long question that would normally exceed the context window of a small model. ' * 10,
        'choices': {
            'text': ['Choice A is also quite long ' * 5,
                     'Choice B is similarly verbose ' * 5,
                     'Choice C has many words ' * 5,
                     'Choice D is the longest of all ' * 5],
            'label': ['A', 'B', 'C', 'D']
        },
        'answerKey': 'A'
    }

    try:
        score = zero_shot_eval._evaluate_single_example('ARC-e', test_example_arc)
        print(f"  ✓ Zero-shot ARC evaluation completed without errors (score: {score})")
    except Exception as e:
        print(f"  ✗ Error in zero-shot evaluation: {e}")
        return False

    # Test few-shot evaluation
    print("\n2. Testing few-shot evaluation with truncation...")
    few_shot_eval = FewShotEvaluator(model, tokenizer, device)

    # Test MMLU prompt generation
    test_example_mmlu = {
        'subject': 'mathematics',
        'question': 'What is the solution to this extremely complicated mathematical problem that has a very long description? ' * 5,
        'choices': [
            'Answer A with lots of detail ' * 3,
            'Answer B with even more detail ' * 3,
            'Answer C is the most detailed ' * 3,
            'Answer D has unnecessary verbosity ' * 3
        ],
        'answer': 0
    }

    try:
        prompt = few_shot_eval._create_mmlu_prompt(test_example_mmlu, num_shots=5)
        print(f"  Generated MMLU prompt length: {len(prompt)} chars")

        # Try to generate answer
        answer = few_shot_eval._generate_answer(prompt, max_length=5)
        print(f"  ✓ Few-shot MMLU evaluation completed (answer: '{answer}')")
    except Exception as e:
        print(f"  ✗ Error in few-shot evaluation: {e}")
        return False

    # Test TriviaQA prompt
    test_question = "This is an extremely long trivia question that would definitely exceed the context window of our small model. " * 5

    try:
        prompt = few_shot_eval._create_triviaqa_prompt(test_question, num_shots=5)
        print(f"\n3. Generated TriviaQA prompt length: {len(prompt)} chars")

        answer = few_shot_eval._generate_answer(prompt, max_length=10)
        print(f"  ✓ TriviaQA evaluation completed (answer: '{answer}')")
    except Exception as e:
        print(f"  ✗ Error in TriviaQA evaluation: {e}")
        return False

    print("\n" + "="*50)
    print("✅ All truncation tests passed successfully!")
    print("The evaluation code properly handles models with n_positions=256")
    print("="*50)
    return True

if __name__ == "__main__":
    success = test_truncation_handling()
    sys.exit(0 if success else 1)
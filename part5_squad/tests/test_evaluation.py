"""
Test evaluation pipeline and answer extraction
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GPT2TokenizerFast
from eval_squad import extract_answer, extract_answer_batch


def test_answer_extraction():
    """Test extracting answer from logits"""
    print("Testing answer extraction...")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    seq_length = 100

    # Create logits with clear best span
    start_logits = torch.randn(seq_length)
    end_logits = torch.randn(seq_length)

    # Make positions 10-15 clearly the best
    start_logits[10] = 10.0  # Much higher than others
    end_logits[15] = 10.0

    # Create dummy input_ids
    input_ids = torch.randint(0, 50257, (seq_length,))

    # Extract answer
    answer = extract_answer(
        start_logits,
        end_logits,
        input_ids,
        tokenizer,
        max_answer_length=30,
        n_best_size=20
    )

    # Verify
    assert 'text' in answer, "Answer should have text"
    assert 'start' in answer, "Answer should have start"
    assert 'end' in answer, "Answer should have end"
    assert 'score' in answer, "Answer should have score"

    assert answer['start'] == 10, "Should find correct start position"
    assert answer['end'] == 15, "Should find correct end position"
    assert answer['start'] <= answer['end'], "Start should be <= end"

    print("✓ Answer extraction works")


def test_beam_search():
    """Test beam search finds better spans than simple argmax"""
    print("Testing beam search...")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    seq_length = 100

    # Create logits where best span requires search
    start_logits = torch.randn(seq_length)
    end_logits = torch.randn(seq_length)

    # Best individual positions
    start_logits[5] = 6.0  # High start (but no good valid end after it)
    end_logits[3] = 7.0    # High end (but before start!)

    # Best valid span
    start_logits[10] = 7.0  # Highest valid start
    end_logits[15] = 8.0    # Highest valid end

    input_ids = torch.randint(0, 50257, (seq_length,))

    # Extract with beam search
    answer = extract_answer(
        start_logits,
        end_logits,
        input_ids,
        tokenizer,
        max_answer_length=30,
        n_best_size=20
    )

    # Should find (10, 15) not (5, 3)
    assert answer['start'] == 10, "Beam search should find valid span"
    assert answer['end'] == 15, "Beam search should find valid span"

    print("✓ Beam search works")


def test_max_answer_length_constraint():
    """Test maximum answer length constraint"""
    print("Testing max answer length constraint...")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    seq_length = 100

    # Create logits with long span
    start_logits = torch.randn(seq_length)
    end_logits = torch.randn(seq_length)

    start_logits[10] = 10.0
    end_logits[60] = 10.0  # 50 tokens away (too long)

    # Also create a shorter valid span
    start_logits[20] = 8.0
    end_logits[25] = 8.0   # 5 tokens away (valid)

    input_ids = torch.randint(0, 50257, (seq_length,))

    # Extract with max_answer_length=10
    answer = extract_answer(
        start_logits,
        end_logits,
        input_ids,
        tokenizer,
        max_answer_length=10,
        n_best_size=20
    )

    # Should find shorter span
    span_length = answer['end'] - answer['start'] + 1
    assert span_length <= 10, "Should respect max_answer_length constraint"

    print("✓ Max answer length constraint works")


def test_batch_extraction():
    """Test batch answer extraction"""
    print("Testing batch extraction...")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    batch_size = 3
    seq_length = 100

    # Create batch logits
    start_logits = torch.randn(batch_size, seq_length)
    end_logits = torch.randn(batch_size, seq_length)

    # Make clear best spans for each example
    start_logits[0, 10] = 10.0
    end_logits[0, 15] = 10.0

    start_logits[1, 20] = 10.0
    end_logits[1, 25] = 10.0

    start_logits[2, 30] = 10.0
    end_logits[2, 35] = 10.0

    # Create batch input_ids
    input_ids = torch.randint(0, 50257, (batch_size, seq_length))

    # Extract answers for batch
    answers = extract_answer_batch(
        start_logits,
        end_logits,
        input_ids,
        tokenizer
    )

    # Verify
    assert len(answers) == batch_size, "Should extract answer for each example"

    assert answers[0]['start'] == 10 and answers[0]['end'] == 15
    assert answers[1]['start'] == 20 and answers[1]['end'] == 25
    assert answers[2]['start'] == 30 and answers[2]['end'] == 35

    print("✓ Batch extraction works")


if __name__ == '__main__':
    test_answer_extraction()
    test_beam_search()
    test_max_answer_length_constraint()
    test_batch_extraction()
    print("\n✅ All evaluation tests passed!")

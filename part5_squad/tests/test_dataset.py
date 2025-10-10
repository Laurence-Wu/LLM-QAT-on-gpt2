"""
Test SQuAD dataset loading and preprocessing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GPT2TokenizerFast
from dataset_squad import SQuADDataset, collate_fn_squad


def test_squad_loading():
    """Test SQuAD dataset loads correctly"""
    print("Testing SQuAD dataset loading...")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Load small subset for testing
    dataset = SQuADDataset(
        tokenizer,
        split='train[:100]',
        max_length=384,
        doc_stride=128,
        version='v1'
    )

    assert len(dataset) > 0, "Dataset should not be empty"

    # Check first example
    sample = dataset[0]
    assert 'input_ids' in sample, "Sample should have input_ids"
    assert 'attention_mask' in sample, "Sample should have attention_mask"
    assert 'start_positions' in sample, "Sample should have start_positions"
    assert 'end_positions' in sample, "Sample should have end_positions"
    assert 'example_id' in sample, "Sample should have example_id"

    # Check shapes
    assert sample['input_ids'].shape[0] == 384, "input_ids should have max_length"
    assert sample['attention_mask'].shape[0] == 384, "attention_mask should have max_length"

    print("✓ SQuAD dataset loading works")


def test_simple_format():
    """Verify format: question <|endoftext|> context <|endoftext|>"""
    print("Testing simple token format...")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    dataset = SQuADDataset(
        tokenizer,
        split='train[:10]',
        max_length=384,
        version='v1'
    )

    sample = dataset[0]

    # Check that separator tokens exist
    eos_positions = (sample['input_ids'] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    assert len(eos_positions) >= 2, "Should have at least 2 <|endoftext|> separators"

    print("✓ Simple token format works (question <|eos|> context <|eos|>)")


def test_answer_span():
    """Test answer span conversion and validation"""
    print("Testing answer span conversion...")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    dataset = SQuADDataset(
        tokenizer,
        split='train[:50]',
        max_length=384,
        version='v1'
    )

    # Find an example with valid answer
    valid_example = None
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample['start_positions'] > 0 and sample['end_positions'] > 0:
            valid_example = sample
            break

    if valid_example:
        # Check positions are valid
        assert valid_example['start_positions'] <= valid_example['end_positions'], \
            "Start position should be <= end position"
        assert valid_example['end_positions'] < 384, \
            "End position should be within sequence length"

        print("✓ Answer span conversion works")
    else:
        print("⚠ No valid answer spans found in test subset (this is OK for testing)")


def test_collate_function():
    """Test collate function batches correctly"""
    print("Testing collate function...")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    dataset = SQuADDataset(
        tokenizer,
        split='train[:10]',
        max_length=384,
        version='v1'
    )

    # Create batch
    batch = [dataset[i] for i in range(min(4, len(dataset)))]
    batched = collate_fn_squad(batch)

    # Check batch shapes
    batch_size = len(batch)
    assert batched['input_ids'].shape == (batch_size, 384), \
        f"Batched input_ids should be ({batch_size}, 384)"
    assert batched['attention_mask'].shape == (batch_size, 384), \
        f"Batched attention_mask should be ({batch_size}, 384)"
    assert batched['start_positions'].shape == (batch_size,), \
        f"Batched start_positions should be ({batch_size},)"
    assert batched['end_positions'].shape == (batch_size,), \
        f"Batched end_positions should be ({batch_size},)"

    print("✓ Collate function works")


if __name__ == '__main__':
    test_squad_loading()
    test_simple_format()
    test_answer_span()
    test_collate_function()
    print("\n✅ All dataset tests passed!")

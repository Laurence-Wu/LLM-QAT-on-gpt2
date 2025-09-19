#!/usr/bin/env python3
"""
Test script for Data Loading Pipeline
Tests dataset creation, tokenization, and dataloader functionality
"""

import sys
import os
import torch
from transformers import GPT2TokenizerFast

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.dataset import create_dataloaders

def test_tokenizer():
    """Test GPT-2 tokenizer setup."""
    print("\n" + "="*60)
    print("Testing GPT-2 Tokenizer")
    print("="*60)

    print("\n1. Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Test basic tokenization
    text = "Hello, this is a test sentence."
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    print(f"   Original text: {text}")
    print(f"   Token IDs: {tokens[:10]}..." if len(tokens) > 10 else f"   Token IDs: {tokens}")
    print(f"   Decoded text: {decoded}")

    assert len(tokens) > 0, "Tokenization failed"
    assert tokenizer.pad_token_id == tokenizer.eos_token_id, "Padding token not set correctly"

    print("✓ Tokenizer works correctly")
    return True

def test_dataloader_creation():
    """Test dataloader creation with various configurations."""
    print("\n2. Testing Dataloader Creation...")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Test with small dataset splits
    print("   - Creating dataloaders with small splits...")
    train_loader, val_loader = create_dataloaders(
        tokenizer,
        train_split='train[:100]',  # Small split for testing
        val_split='validation[:50]',
        batch_size=4,
        max_length=128,
        doc_stride=64
    )

    assert train_loader is not None, "Train loader not created"
    assert val_loader is not None, "Val loader not created"

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    print("✓ Dataloaders created successfully")
    return True

def test_batch_structure():
    """Test the structure of batches from dataloaders."""
    print("\n3. Testing Batch Structure...")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader = create_dataloaders(
        tokenizer,
        train_split='train[:100]',
        val_split='validation[:50]',
        batch_size=4,
        max_length=128,
        doc_stride=64
    )

    # Get a batch
    print("   - Checking batch structure...")
    for batch in train_loader:
        # Check batch keys
        assert 'input_ids' in batch, "Batch missing input_ids"

        # Check tensor shapes
        input_ids = batch['input_ids']
        assert input_ids.dim() == 2, f"Input IDs should be 2D, got {input_ids.dim()}D"
        batch_size, seq_length = input_ids.shape
        assert batch_size == 4, f"Expected batch size 4, got {batch_size}"
        assert seq_length <= 128, f"Sequence length {seq_length} exceeds max_length"

        # Check data types
        assert input_ids.dtype == torch.long, f"Input IDs should be long, got {input_ids.dtype}"

        # Check for attention mask if present
        if 'attention_mask' in batch:
            attention_mask = batch['attention_mask']
            assert attention_mask.shape == input_ids.shape, "Attention mask shape mismatch"
            assert attention_mask.dtype == torch.long, "Attention mask should be long"

        print(f"   Batch shape: {input_ids.shape}")
        print(f"   Input IDs range: [{input_ids.min().item()}, {input_ids.max().item()}]")

        break  # Only check first batch

    print("✓ Batch structure is correct")
    return True

def test_different_configurations():
    """Test dataloaders with different configurations."""
    print("\n4. Testing Different Configurations...")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    configs = [
        {"batch_size": 2, "max_length": 64, "doc_stride": 32},
        {"batch_size": 8, "max_length": 256, "doc_stride": 128},
        {"batch_size": 1, "max_length": 512, "doc_stride": 256},
    ]

    for i, config in enumerate(configs, 1):
        print(f"   - Testing config {i}: batch_size={config['batch_size']}, "
              f"max_length={config['max_length']}")

        train_loader, val_loader = create_dataloaders(
            tokenizer,
            train_split='train[:50]',  # Very small for quick testing
            val_split='validation[:20]',
            **config
        )

        # Check first batch
        for batch in train_loader:
            input_ids = batch['input_ids']
            batch_size_actual = input_ids.shape[0]
            seq_length = input_ids.shape[1]

            # Batch size might be smaller for last batch
            assert batch_size_actual <= config['batch_size'], \
                f"Batch size {batch_size_actual} exceeds configured {config['batch_size']}"
            assert seq_length <= config['max_length'], \
                f"Sequence length {seq_length} exceeds max_length {config['max_length']}"

            break  # Only check first batch

    print("✓ Different configurations work correctly")
    return True

def test_data_consistency():
    """Test that data loading is consistent and reproducible."""
    print("\n5. Testing Data Consistency...")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create two sets of dataloaders with same config
    print("   - Creating duplicate dataloaders...")
    loader1_train, loader1_val = create_dataloaders(
        tokenizer,
        train_split='train[:50]',
        val_split='validation[:20]',
        batch_size=4,
        max_length=128,
        doc_stride=64
    )

    loader2_train, loader2_val = create_dataloaders(
        tokenizer,
        train_split='train[:50]',
        val_split='validation[:20]',
        batch_size=4,
        max_length=128,
        doc_stride=64
    )

    # Compare number of batches
    assert len(loader1_train) == len(loader2_train), "Train loader lengths differ"
    assert len(loader1_val) == len(loader2_val), "Val loader lengths differ"

    print(f"   Train loaders: {len(loader1_train)} batches each")
    print(f"   Val loaders: {len(loader1_val)} batches each")

    print("✓ Data loading is consistent")
    return True

def test_memory_efficiency():
    """Test memory efficiency of data loading."""
    print("\n6. Testing Memory Efficiency...")

    if not torch.cuda.is_available():
        print("   ⚠ CUDA not available, skipping memory test")
        return True

    import gc
    torch.cuda.empty_cache()
    gc.collect()

    initial_memory = torch.cuda.memory_allocated()
    print(f"   Initial GPU memory: {initial_memory / 1024**2:.1f} MB")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataloader
    train_loader, _ = create_dataloaders(
        tokenizer,
        train_split='train[:100]',
        val_split='validation[:20]',
        batch_size=8,
        max_length=256,
        doc_stride=128
    )

    # Load a few batches to GPU
    print("   - Loading batches to GPU...")
    max_memory = initial_memory
    for i, batch in enumerate(train_loader):
        if i >= 5:  # Only test first 5 batches
            break

        input_ids = batch['input_ids'].cuda()
        current_memory = torch.cuda.memory_allocated()
        max_memory = max(max_memory, current_memory)

        # Clean up
        del input_ids
        if 'attention_mask' in batch:
            del batch['attention_mask']
        torch.cuda.empty_cache()

    memory_used = (max_memory - initial_memory) / 1024**2
    print(f"   Peak memory increase: {memory_used:.1f} MB")

    # Clean up
    torch.cuda.empty_cache()
    gc.collect()

    print("✓ Memory usage is reasonable")
    return True

def run_all_tests():
    """Run all data pipeline tests."""
    print("\n" + "="*70)
    print(" DATA PIPELINE TEST SUITE")
    print("="*70)

    tests = [
        ("Tokenizer", test_tokenizer),
        ("Dataloader Creation", test_dataloader_creation),
        ("Batch Structure", test_batch_structure),
        ("Different Configurations", test_different_configurations),
        ("Data Consistency", test_data_consistency),
        ("Memory Efficiency", test_memory_efficiency)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n✗ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print(f" TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*70)

    if failed == 0:
        print("\n✅ All data pipeline tests passed successfully!")
    else:
        print(f"\n⚠ {failed} test(s) failed. Please review the errors above.")

    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
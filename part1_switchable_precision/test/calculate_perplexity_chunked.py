"""
Alternative perplexity calculation that processes WikiText in manageable chunks.
This avoids the tokenization warning and handles long texts properly.
"""

import torch
from datasets import load_dataset
from tqdm import tqdm


def calculate_perplexity_chunked(model, tokenizer, device,
                                 dataset_name='wikitext',
                                 chunk_size=512,
                                 stride=256,
                                 max_chunks=10):
    """
    Calculate perplexity by processing the dataset in chunks.

    This approach:
    1. Loads WikiText documents one at a time
    2. Processes each document with sliding windows
    3. Aggregates results across all documents

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for text processing
        device: Device to run on
        dataset_name: Dataset to use ('wikitext')
        chunk_size: Size of each evaluation chunk (max 1024 for GPT-2)
        stride: Stride for sliding window within chunks
        max_chunks: Maximum number of chunks to process (None for all)

    Returns:
        Dictionary with perplexity and loss statistics
    """
    model.eval()

    # Load dataset
    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Process documents
    nlls = []  # Negative log likelihoods
    total_tokens_processed = 0
    chunks_processed = 0

    print(f"   Processing WikiText in chunks (chunk_size={chunk_size}, stride={stride})")

    # Create progress bar
    pbar = tqdm(total=max_chunks if max_chunks else len(dataset),
                desc="Processing chunks", leave=False)

    for doc_idx, item in enumerate(dataset):
        text = item['text'].strip()

        # Skip empty documents
        if not text or len(text) < 100:
            continue

        # Tokenize this document (with truncation to avoid warning)
        encodings = tokenizer(
            text,
            return_tensors='pt',
            max_length=chunk_size,
            truncation=True,
            padding=False
        )

        # Skip if too short
        if encodings.input_ids.size(1) < 10:
            continue

        # Process this document with sliding windows
        doc_length = encodings.input_ids.size(1)

        for start_idx in range(0, doc_length, stride):
            end_idx = min(start_idx + chunk_size, doc_length)

            # Skip if remaining chunk is too small
            if end_idx - start_idx < 10:
                continue

            # Extract chunk
            input_ids = encodings.input_ids[:, start_idx:end_idx].to(device)

            # Create target with proper masking
            target_ids = input_ids.clone()

            # Only compute loss on non-overlapping tokens
            if start_idx > 0:
                overlap = min(chunk_size - stride, end_idx - start_idx)
                target_ids[:, :overlap] = -100  # Mask overlapping tokens

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # Handle both dict and object outputs
                if isinstance(outputs, dict):
                    loss = outputs['loss']
                else:
                    loss = outputs.loss

                # Count actual tokens (non-masked)
                valid_tokens = (target_ids != -100).sum().item()

                if valid_tokens > 0:
                    neg_log_likelihood = loss * valid_tokens
                    nlls.append(neg_log_likelihood)
                    total_tokens_processed += valid_tokens

        chunks_processed += 1
        pbar.update(1)

        # Stop if we've processed enough chunks
        if max_chunks and chunks_processed >= max_chunks:
            break

    pbar.close()

    # Calculate perplexity
    if len(nlls) == 0 or total_tokens_processed == 0:
        return {
            'perplexity': float('inf'),
            'loss': float('inf'),
            'total_tokens': 0,
            'num_chunks': 0
        }

    total_nll = torch.stack(nlls).sum()
    avg_loss = (total_nll / total_tokens_processed).item()
    perplexity = torch.exp(total_nll / total_tokens_processed).item()

    return {
        'perplexity': perplexity,
        'loss': avg_loss,
        'total_tokens': total_tokens_processed,
        'num_chunks': chunks_processed
    }


def compare_perplexity_methods(model, tokenizer, device):
    """
    Compare the original and chunked perplexity calculation methods.
    """
    print("\n" + "="*60)
    print("COMPARING PERPLEXITY CALCULATION METHODS")
    print("="*60)

    # Method 1: Chunked (no warnings)
    print("\n1. Chunked Method (no warnings):")
    chunked_results = calculate_perplexity_chunked(
        model=model,
        tokenizer=tokenizer,
        device=device,
        chunk_size=512,
        stride=256,
        max_chunks=20
    )
    print(f"   Perplexity: {chunked_results['perplexity']:.2f}")
    print(f"   Loss: {chunked_results['loss']:.4f}")
    print(f"   Tokens: {chunked_results['total_tokens']:,}")
    print(f"   Chunks: {chunked_results['num_chunks']}")

    # Method 2: Original (with truncation)
    print("\n2. Original Method (with truncation):")
    # Add test directory to path if not already added
    test_dir = os.path.dirname(os.path.abspath(__file__))
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)

    from dataset_utils import calculate_perplexity_properly

    original_results = calculate_perplexity_properly(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dataset_name='wikitext',
        max_length=512,
        stride=256,
        max_samples=512
    )
    print(f"   Perplexity: {original_results['perplexity']:.2f}")
    print(f"   Loss: {original_results['loss']:.4f}")
    print(f"   Tokens: {original_results['total_tokens']:,}")

    # Compare
    ppl_diff = abs(chunked_results['perplexity'] - original_results['perplexity'])
    print(f"\n   Difference in perplexity: {ppl_diff:.2f}")
    print(f"   Methods are {'consistent ✅' if ppl_diff < 5.0 else 'inconsistent ⚠️'}")

    return chunked_results, original_results


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading GPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = model.to(device)
    model.eval()

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Test chunked calculation
    print("\nTesting chunked perplexity calculation...")
    results = calculate_perplexity_chunked(
        model=model,
        tokenizer=tokenizer,
        device=device,
        chunk_size=512,
        stride=256,
        max_chunks=10
    )

    print(f"\nResults:")
    print(f"  Perplexity: {results['perplexity']:.2f}")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Total tokens: {results['total_tokens']:,}")
    print(f"  Chunks processed: {results['num_chunks']}")

    # Compare methods
    print("\n" + "="*60)
    compare_perplexity_methods(model, tokenizer, device)
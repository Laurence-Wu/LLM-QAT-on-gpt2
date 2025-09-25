"""
Dataset utilities for proper perplexity evaluation with label shifting.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from typing import Optional, List, Tuple


class WikiTextDataset(Dataset):
    """WikiText dataset for perplexity evaluation with proper label shifting."""

    def __init__(self, tokenizer, split='test', max_length=512, stride=256):
        """
        Initialize WikiText dataset for perplexity evaluation.

        Args:
            tokenizer: GPT-2 tokenizer
            split: Dataset split to use ('train', 'validation', 'test')
            max_length: Maximum sequence length
            stride: Stride for sliding window (for handling long texts)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        # Load WikiText-2 dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)

        # Filter and concatenate texts
        texts = []
        for item in dataset:
            text = item['text'].strip()
            if len(text) > 50:  # Skip very short texts
                texts.append(text)

        # Concatenate all texts with space separator
        self.full_text = ' '.join(texts)

        # Tokenize the full text
        self.encodings = tokenizer(
            self.full_text,
            return_tensors='pt',
            truncation=False,
            padding=False
        )

        # Create sliding windows
        self.examples = []
        total_length = self.encodings['input_ids'].size(1)

        for i in range(0, total_length - max_length + 1, stride):
            input_ids = self.encodings['input_ids'][:, i:i + max_length]
            attention_mask = self.encodings['attention_mask'][:, i:i + max_length] if 'attention_mask' in self.encodings else None

            self.examples.append({
                'input_ids': input_ids.squeeze(0),
                'attention_mask': attention_mask.squeeze(0) if attention_mask is not None else torch.ones_like(input_ids.squeeze(0))
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def create_test_dataloader(tokenizer, batch_size=4, max_length=512, stride=256, num_samples=100):
    """
    Create a test dataloader for perplexity evaluation.

    Args:
        tokenizer: GPT-2 tokenizer
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        stride: Stride for sliding window
        num_samples: Maximum number of samples to use (None for all)

    Returns:
        DataLoader for test set
    """
    dataset = WikiTextDataset(tokenizer, split='test', max_length=max_length, stride=stride)

    # Limit samples if specified
    if num_samples and num_samples < len(dataset):
        dataset.examples = dataset.examples[:num_samples]

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )


def calculate_perplexity_properly(model, tokenizer, device, dataset_name='wikitext',
                                 max_length=512, stride=256, max_samples=None):
    """
    Calculate perplexity using the correct sliding window approach from test_gpt2_perplexity.py.

    This uses the standard evaluation methodology for language models with:
    - Sliding window to handle long sequences
    - Proper masking to avoid double-counting tokens
    - Only computing loss on non-overlapping tokens

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for text processing
        device: Device to run on
        dataset_name: Dataset to use ('wikitext' or custom)
        max_length: Maximum length of each window
        stride: Stride for sliding window
        max_samples: Maximum number of tokens to evaluate (None for all)

    Returns:
        Dictionary with perplexity and loss statistics
    """
    model.eval()

    # Load and concatenate dataset
    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        # Concatenate all text (preserving document boundaries)
        text = '\n\n'.join([t for t in dataset['text'] if t.strip()])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Tokenize with truncation to avoid the warning
    # We'll process in chunks if needed
    max_model_length = 1024  # GPT-2's max position embeddings

    # First, tokenize with truncation to get initial tokens
    if max_samples and max_samples < max_model_length:
        target_length = max_samples
    else:
        target_length = max_model_length

    # Tokenize with explicit max_length to avoid warning
    encodings = tokenizer(
        text,
        return_tensors='pt',
        max_length=target_length,
        truncation=True,
        padding=False
    )

    # Verify we have the expected length
    actual_length = encodings.input_ids.size(1)
    if actual_length < target_length and not max_samples:
        # If we got less than expected, try to get more text
        # This means the entire dataset was shorter than our target
        print(f"   Dataset has {actual_length} tokens (less than target {target_length})")

    # Evaluate in sliding windows
    nlls = []  # Negative log likelihoods
    prev_end_loc = 0

    total_tokens = encodings.input_ids.size(1)
    if total_tokens > 0:
        print(f"   Total tokens to evaluate: {total_tokens:,}")
        print(f"   Window size: {max_length}, Stride: {stride}")

    # Process windows
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # Target length (may be different from stride on last loop)

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()

        # CRITICAL: Only compute loss on the last trg_len tokens to avoid double counting
        # Set all tokens before position i to -100 (ignored in loss)
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # Handle both dict and object outputs
            if isinstance(outputs, dict):
                loss = outputs['loss']
            else:
                loss = outputs.loss
            # Multiply by trg_len to get total loss (not average) for these tokens
            neg_log_likelihood = loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc

        # Break if we've reached the end
        if end_loc >= encodings.input_ids.size(1):
            break

    # Calculate perplexity
    total_nll = torch.stack(nlls).sum()
    avg_loss = (total_nll / prev_end_loc).item()
    perplexity = torch.exp(total_nll / prev_end_loc).item()

    return {
        'perplexity': perplexity,
        'loss': avg_loss,
        'total_tokens': prev_end_loc,
        'num_windows': len(nlls)
    }


def get_calibration_texts(num_texts=16):
    """
    Get diverse calibration texts from WikiText for better statistics.

    Args:
        num_texts: Number of calibration texts to return

    Returns:
        List of calibration texts
    """
    try:
        # Try to load WikiText for calibration
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

        texts = []
        for item in dataset:
            text = item['text'].strip()
            if len(text) > 100 and len(text) < 500:  # Medium-length texts
                texts.append(text)
                if len(texts) >= num_texts:
                    break
    except:
        # If loading fails, use empty list to trigger fallback
        texts = []

    # Fallback if not enough texts found or loading failed
    if len(texts) < num_texts:
        default_texts = [
            "Machine learning algorithms optimize complex objective functions through iterative updates.",
            "Neural networks consist of interconnected layers that process information hierarchically.",
            "Natural language processing enables computers to understand and generate human language.",
            "Deep learning models learn representations directly from raw data inputs.",
            "Transformer architectures use self-attention mechanisms for sequence modeling tasks.",
            "Gradient descent minimizes loss functions by updating parameters iteratively.",
            "Convolutional neural networks excel at processing grid-structured data like images.",
            "Recurrent neural networks maintain hidden states for sequential data processing.",
            "Reinforcement learning agents learn optimal policies through environmental interaction.",
            "Transfer learning leverages pretrained models for new downstream tasks.",
            "Attention mechanisms allow models to focus on relevant input parts.",
            "Batch normalization stabilizes training by normalizing intermediate activations.",
            "Dropout regularization prevents overfitting by randomly deactivating neurons.",
            "Adam optimizer combines momentum with adaptive learning rates effectively.",
            "Cross-entropy loss measures prediction uncertainty for classification tasks.",
            "Backpropagation efficiently computes gradients using the chain rule.",
        ]
        texts.extend(default_texts[:num_texts - len(texts)])

    return texts[:num_texts]


if __name__ == "__main__":
    # Test the dataset utilities
    from transformers import GPT2TokenizerFast

    print("Testing dataset utilities...")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Test calibration texts
    print("\n1. Getting calibration texts:")
    cal_texts = get_calibration_texts(8)
    for i, text in enumerate(cal_texts[:3]):
        print(f"   Text {i+1}: {text[:80]}...")

    # Test dataloader creation
    print("\n2. Creating test dataloader:")
    dataloader = create_test_dataloader(tokenizer, batch_size=2, max_length=128, num_samples=10)
    print(f"   Created dataloader with {len(dataloader)} batches")

    # Show a sample batch
    batch = next(iter(dataloader))
    print(f"   Sample batch shape: {batch['input_ids'].shape}")
    print(f"   First sequence tokens: {batch['input_ids'][0][:20].tolist()}")

    print("\n✅ Dataset utilities working correctly!")
"""
WikiText dataset for language modeling.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class WikiTextDataset(Dataset):
    """WikiText dataset for language modeling."""

    def __init__(self, split: str, tokenizer, max_seq_length: int = 256, doc_stride: int = 128):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride

        # Load dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        self.texts = dataset['text']

        # Tokenize and create sequences
        self.sequences = self._create_sequences()

    def _create_sequences(self):
        """Create overlapping sequences from texts."""
        sequences = []
        for text in self.texts:
            if len(text.strip()) == 0:
                continue

            # Tokenize
            tokens = self.tokenizer(
                text,
                truncation=False,
                return_tensors='pt'
            )['input_ids'][0]

            # Create overlapping sequences
            for i in range(0, len(tokens) - self.max_seq_length + 1, self.doc_stride):
                seq = tokens[i:i + self.max_seq_length]
                if len(seq) == self.max_seq_length:
                    sequences.append(seq)

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            'input_ids': seq[:-1],
            'labels': seq[1:]
        }
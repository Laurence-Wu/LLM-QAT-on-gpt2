"""
WikiText-2 Dataset Preparation and Evaluation Utilities
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm


class WikiText2Dataset(Dataset):
    """
    WikiText-2 dataset for language modeling evaluation.
    """

    def __init__(self, tokenizer, split: str = 'test',
                 max_length: int = 128,
                 num_samples: Optional[int] = None):
        """
        Initialize WikiText-2 dataset.

        Args:
            tokenizer: Tokenizer for the model
            split: Dataset split ('train', 'validation', 'test')
            max_length: Maximum sequence length
            num_samples: Limit number of samples (None for all)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        self.samples = self._prepare_samples(dataset, num_samples)

        print(f"Loaded {len(self.samples)} samples from WikiText-2 {split} split")

    def _prepare_samples(self, dataset, num_samples: Optional[int]) -> List[Dict]:
        """
        Prepare samples from raw WikiText data.

        Args:
            dataset: Raw WikiText dataset
            num_samples: Number of samples to prepare

        Returns:
            List of prepared samples
        """
        samples = []
        count = 0

        for item in tqdm(dataset, desc="Preparing WikiText-2 samples"):
            text = item['text'].strip()

            if len(text) < 50:
                continue

            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            sample = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'text': text[:500]
            }

            samples.append(sample)
            count += 1

            if num_samples and count >= num_samples:
                break

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def prepare_wikitext2_samples(tokenizer, num_samples: int = 100,
                             split: str = 'test',
                             max_length: int = 128) -> List[Dict]:
    dataset = WikiText2Dataset(
        tokenizer,
        split=split,
        max_length=max_length,
        num_samples=num_samples
    )
    return dataset.samples
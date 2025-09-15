import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
from abc import ABC, abstractmethod


class BaseDataset(Dataset, ABC):
    """Base class for all datasets."""

    def __init__(self, tokenizer, max_length=384):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

    @abstractmethod
    def preprocess_dataset(self):
        """Preprocess the dataset - to be implemented by subclasses."""
        pass

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class SQuADDataset(BaseDataset):
    """SQuAD dataset for generative question answering."""

    def __init__(self, tokenizer, split='train', max_length=384, doc_stride=128):
        super().__init__(tokenizer, max_length)
        self.doc_stride = doc_stride
        self.dataset = load_dataset('squad', split=split)
        self.examples = self.preprocess_dataset()

    def preprocess_dataset(self):
        """Preprocess SQuAD for generative training."""
        processed = []
        for example in tqdm(self.dataset, desc="Preprocessing SQuAD"):
            processed_example = self._process_example(example)
            if processed_example is not None:
                processed.append(processed_example)
        return processed

    def _process_example(self, example):
        """Process a single SQuAD example for generative training."""
        context = example['context']
        question = example['question']
        answers = example['answers']

        # Skip examples without answers
        if len(answers['answer_start']) == 0:
            return None

        answer_text = answers['text'][0]

        # Format for generative training: "Context: ... Question: ... Answer: ..."
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        full_text = f"{prompt} {answer_text}"

        # Tokenize the full text
        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Tokenize just the prompt to find where answer starts
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt'
        )

        input_ids = full_encoding['input_ids'].squeeze(0)
        attention_mask = full_encoding['attention_mask'].squeeze(0)

        # Create labels for language modeling
        labels = input_ids.clone()
        prompt_length = prompt_encoding['input_ids'].shape[1]

        # Mask out the prompt part - we only want to compute loss on the answer
        labels[:prompt_length] = -100

        # Also mask out padding tokens
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class WikiTextDataset(BaseDataset):
    """WikiText dataset for language modeling."""

    def __init__(self, tokenizer, split='train', max_length=384,
                 stride=128, dataset_name='wikitext-103-raw-v1'):
        super().__init__(tokenizer, max_length)
        self.stride = stride
        self.dataset_name = dataset_name

        # Load WikiText dataset
        self.dataset = load_dataset('wikitext', dataset_name, split=split)
        self.examples = self.preprocess_dataset()

    def preprocess_dataset(self):
        """Preprocess WikiText for language modeling with sliding window."""
        processed = []

        for article in tqdm(self.dataset, desc=f"Preprocessing WikiText-{self.dataset_name}"):
            text = article['text'].strip()

            # Skip empty articles
            if not text or len(text) < 10:
                continue

            # Tokenize the entire article
            tokenized = self.tokenizer(
                text,
                truncation=False,
                padding=False,
                return_tensors='pt'
            )

            input_ids = tokenized['input_ids'].squeeze(0)

            # Create sliding window chunks
            for i in range(0, len(input_ids), self.stride):
                chunk_ids = input_ids[i:i + self.max_length]

                # Skip if chunk is too small
                if len(chunk_ids) < 50:
                    continue

                # Pad if necessary
                if len(chunk_ids) < self.max_length:
                    padding_length = self.max_length - len(chunk_ids)
                    chunk_ids = torch.cat([
                        chunk_ids,
                        torch.full((padding_length,), self.tokenizer.pad_token_id)
                    ])

                # Create attention mask
                attention_mask = (chunk_ids != self.tokenizer.pad_token_id).long()

                # For language modeling, labels are the same as input_ids
                # but we mask out padding tokens
                labels = chunk_ids.clone()
                labels[attention_mask == 0] = -100

                processed.append({
                    'input_ids': chunk_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                })

                # Stop if we've processed enough chunks (for memory efficiency)
                if len(processed) >= 50000:  # Limit dataset size
                    return processed

        return processed


def collate_fn(batch):
    """Custom collate function for generative training."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def create_dataloaders(tokenizer, dataset_type='squad', train_split='train',
                      val_split='validation', batch_size=8, max_length=384,
                      doc_stride=128, num_workers=0):
    """
    Create dataloaders for different dataset types.

    Args:
        tokenizer: The tokenizer to use
        dataset_type: 'squad' or 'wikitext' or 'wikitext-2'
        train_split: Split name for training data
        val_split: Split name for validation data
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        doc_stride: Stride for sliding window (used by both datasets)
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader
    """

    # Select appropriate dataset class
    if dataset_type == 'squad':
        train_dataset = SQuADDataset(
            tokenizer,
            split=train_split,
            max_length=max_length,
            doc_stride=doc_stride
        )
        val_dataset = SQuADDataset(
            tokenizer,
            split=val_split,
            max_length=max_length,
            doc_stride=doc_stride
        )
    elif dataset_type == 'wikitext' or dataset_type == 'wikitext-103':
        train_dataset = WikiTextDataset(
            tokenizer,
            split=train_split,
            max_length=max_length,
            stride=doc_stride,
            dataset_name='wikitext-103-raw-v1'
        )
        val_dataset = WikiTextDataset(
            tokenizer,
            split=val_split,
            max_length=max_length,
            stride=doc_stride,
            dataset_name='wikitext-103-raw-v1'
        )
    elif dataset_type == 'wikitext-2':
        train_dataset = WikiTextDataset(
            tokenizer,
            split=train_split,
            max_length=max_length,
            stride=doc_stride,
            dataset_name='wikitext-2-raw-v1'
        )
        val_dataset = WikiTextDataset(
            tokenizer,
            split=val_split,
            max_length=max_length,
            stride=doc_stride,
            dataset_name='wikitext-2-raw-v1'
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                        f"Choose from: 'squad', 'wikitext', 'wikitext-103', 'wikitext-2'")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn
    )

    return train_loader, val_loader
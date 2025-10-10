import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

class SQuADDataset(Dataset):
    """
    SQuAD Dataset for Question Answering

    Format: question <|endoftext|> context <|endoftext|>
    Uses existing GPT-2 tokens, no embedding resizing needed
    """

    def __init__(self, tokenizer, split='train', max_length=384,
                 doc_stride=128, max_query_length=64, version='v1',
                 verify_spans=False):
        """
        Args:
            tokenizer: GPT2TokenizerFast
            split: Dataset split ('train', 'validation')
            max_length: Maximum sequence length
            doc_stride: Sliding window stride for long contexts
            max_query_length: Maximum question length
            version: 'v1' for SQuAD 1.1, 'v2' for SQuAD 2.0
            verify_spans: Whether to verify answer span conversion (for debugging)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.verify_spans = verify_spans

        # Load SQuAD dataset
        dataset_name = 'squad' if version == 'v1' else 'squad_v2'
        print(f"Loading {dataset_name} dataset...")
        self.dataset = load_dataset(dataset_name, split=split)

        # Preprocess dataset
        self.examples = self.preprocess_dataset()

        print(f"Preprocessed {len(self.examples)} examples from {len(self.dataset)} original examples")

    def preprocess_dataset(self):
        """
        Preprocess SQuAD examples into model inputs

        Steps:
        1. Tokenize question and context separately
        2. Build input: question <|eos|> context <|eos|>
        3. Apply sliding window for long contexts
        4. Convert character-based answer spans to token indices
        """
        processed = []

        for example in tqdm(self.dataset, desc="Preprocessing SQuAD"):
            question = example['question']
            context = example['context']
            answers = example['answers']
            example_id = example['id']

            # Tokenize question (truncate if too long)
            question_tokens = self.tokenizer(
                question,
                max_length=self.max_query_length,
                truncation=True,
                add_special_tokens=False
            )

            # Tokenize context with character-to-token mapping
            context_tokens = self.tokenizer(
                context,
                truncation=False,
                add_special_tokens=False,
                return_offsets_mapping=True
            )

            # Calculate maximum context length
            # Format: question + <|eos|> + context + <|eos|>
            max_context_length = self.max_length - len(question_tokens['input_ids']) - 2

            if max_context_length <= 0:
                # Question too long, skip
                continue

            # Apply sliding window for long contexts
            context_input_ids = context_tokens['input_ids']
            context_offset_mapping = context_tokens['offset_mapping']

            # Determine sliding windows
            num_windows = 0
            for chunk_start in range(0, len(context_input_ids), self.doc_stride):
                chunk_end = min(chunk_start + max_context_length, len(context_input_ids))

                # Build input sequence: question + <|eos|> + context_chunk + <|eos|>
                input_ids = (
                    question_tokens['input_ids'] +
                    [self.tokenizer.eos_token_id] +
                    context_input_ids[chunk_start:chunk_end] +
                    [self.tokenizer.eos_token_id]
                )

                # Pad to max_length
                padding_length = self.max_length - len(input_ids)
                if padding_length > 0:
                    input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                else:
                    input_ids = input_ids[:self.max_length]

                # Create attention mask (1 for real tokens, 0 for padding)
                attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0
                                 for token_id in input_ids]

                # Find answer span in this chunk
                # context_offset is where context starts in input_ids
                context_offset = len(question_tokens['input_ids']) + 1  # +1 for <|eos|>

                start_position, end_position = self._find_answer_span(
                    answers=answers,
                    context_offset_mapping=context_offset_mapping,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    context_offset=context_offset,
                    context_text=context
                )

                # Add processed example
                processed.append({
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                    'start_positions': start_position,
                    'end_positions': end_position,
                    'example_id': example_id
                })

                num_windows += 1

                # If chunk covers entire context, no need for more windows
                if chunk_end >= len(context_input_ids):
                    break

        return processed

    def _find_answer_span(self, answers: Dict, context_offset_mapping: List[Tuple[int, int]],
                          chunk_start: int, chunk_end: int, context_offset: int,
                          context_text: str) -> Tuple[int, int]:
        """
        Convert character-based answer position to token position

        Args:
            answers: Dict with 'text' and 'answer_start' lists
            context_offset_mapping: List of (start_char, end_char) for each context token
            chunk_start: Start index of context chunk (in tokens)
            chunk_end: End index of context chunk (in tokens)
            context_offset: Offset where context starts in input_ids
            context_text: Original context text (for verification)

        Returns:
            (start_token_idx, end_token_idx) in the input sequence
            Returns (0, 0) for unanswerable or out-of-chunk answers
        """
        # Check if unanswerable (SQuAD v2.0)
        if len(answers['text']) == 0 or len(answers['answer_start']) == 0:
            return 0, 0

        # Use first answer
        answer_start_char = answers['answer_start'][0]
        answer_text = answers['text'][0]
        answer_end_char = answer_start_char + len(answer_text)

        # Find token indices using offset_mapping
        start_token = None
        end_token = None

        for i in range(len(context_offset_mapping)):
            token_start_char, token_end_char = context_offset_mapping[i]

            # Find start token (token that contains answer start)
            if token_start_char <= answer_start_char < token_end_char:
                start_token = i

            # Find end token (token that contains answer end)
            if token_start_char < answer_end_char <= token_end_char:
                end_token = i

        # If answer span not found, return (0, 0)
        if start_token is None or end_token is None:
            return 0, 0

        # Check if answer is in this chunk
        if start_token < chunk_start or end_token >= chunk_end:
            return 0, 0

        # Adjust to chunk coordinates and add context offset
        start_position = start_token - chunk_start + context_offset
        end_position = end_token - chunk_start + context_offset

        # Ensure positions are within sequence
        if start_position >= self.max_length or end_position >= self.max_length:
            return 0, 0

        return start_position, end_position

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn_squad(batch):
    """
    Collate function for SQuAD dataset

    Args:
        batch: List of examples from SQuADDataset

    Returns:
        Dict with batched tensors
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    start_positions = torch.tensor([item['start_positions'] for item in batch], dtype=torch.long)
    end_positions = torch.tensor([item['end_positions'] for item in batch], dtype=torch.long)
    example_ids = [item['example_id'] for item in batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'start_positions': start_positions,
        'end_positions': end_positions,
        'example_ids': example_ids
    }


def create_squad_dataloaders(tokenizer, train_split='train', val_split='validation',
                             batch_size=16, max_length=384, doc_stride=128,
                             max_query_length=64, version='v1', num_workers=0):
    """
    Create SQuAD dataloaders for training and validation

    Args:
        tokenizer: GPT2TokenizerFast
        train_split: Training split specification
        val_split: Validation split specification
        batch_size: Batch size
        max_length: Maximum sequence length
        doc_stride: Sliding window stride
        max_query_length: Maximum question length
        version: 'v1' or 'v2' for SQuAD version
        num_workers: Number of dataloader workers

    Returns:
        (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = SQuADDataset(
        tokenizer,
        split=train_split,
        max_length=max_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        version=version
    )

    val_dataset = SQuADDataset(
        tokenizer,
        split=val_split,
        max_length=max_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        version=version
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn_squad
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn_squad
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    return train_loader, val_loader

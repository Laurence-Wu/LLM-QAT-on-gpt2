import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm

class SQuADDataset(Dataset):
    def __init__(self, tokenizer, split='train', max_length=384, doc_stride=128, preprocess_all=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.preprocess_all = preprocess_all

        # Load dataset but don't preprocess everything at once
        self.dataset = load_dataset('squad', split=split)

        # Only preprocess if explicitly requested (for backward compatibility)
        if preprocess_all:
            self.examples = self.preprocess_dataset()
        else:
            self.examples = None

    def preprocess_dataset(self):
        """Legacy preprocessing - loads all data into memory."""
        processed = []
        for example in tqdm(self.dataset, desc="Preprocessing SQuAD"):
            processed_example = self._process_example(example)
            if processed_example is not None:
                processed.append(processed_example)
        return processed

    def _process_example(self, example):
        """Process a single example on-demand."""
        context = example['context']
        question = example['question']
        answers = example['answers']

        encoding = self.tokenizer(
            question,
            context,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_offsets_mapping=True,
            return_tensors='pt'  # Return PyTorch tensors directly
        )

        if len(answers['answer_start']) > 0:
            start_char = answers['answer_start'][0]
            end_char = start_char + len(answers['text'][0])

            start_token = 0
            end_token = 0
            offset_mapping = encoding['offset_mapping'][0]  # Get first element since return_tensors='pt'

            for i, (offset_start, offset_end) in enumerate(offset_mapping):
                if offset_start <= start_char < offset_end:
                    start_token = i
                if offset_start < end_char <= offset_end:
                    end_token = i
                    break

            # Return squeezed tensors to remove batch dimension
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'start_positions': torch.tensor(start_token),
                'end_positions': torch.tensor(end_token)
            }

        return None

    def __len__(self):
        if self.examples is not None:
            return len(self.examples)
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.examples is not None:
            # Use preprocessed data if available
            return self.examples[idx]
        else:
            # Process on-demand to save memory
            example = self.dataset[idx]
            processed = self._process_example(example)

            # If processing failed, return a dummy example
            if processed is None:
                return {
                    'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                    'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                    'start_positions': torch.tensor(0),
                    'end_positions': torch.tensor(0)
                }

            return processed

def create_dataloaders(tokenizer, train_split, val_split,
                       batch_size, max_length, doc_stride, num_workers=0):
    # Create datasets with on-demand processing to save memory
    train_dataset = SQuADDataset(tokenizer, split=train_split,
                                 max_length=max_length, doc_stride=doc_stride,
                                 preprocess_all=False)  # Process on-demand
    val_dataset = SQuADDataset(tokenizer, split=val_split,
                               max_length=max_length, doc_stride=doc_stride,
                               preprocess_all=False)  # Process on-demand

    # Use pin_memory for faster GPU transfer and num_workers for parallel loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)  # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader
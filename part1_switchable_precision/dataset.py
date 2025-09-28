import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm

class WikiTextDataset(Dataset):

    def __init__(self, tokenizer, split='train', max_length=384, stride=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        self.dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
        self.examples = self.preprocess_dataset()

    def preprocess_dataset(self):

        processed = []

        for article in tqdm(self.dataset, desc=f"Preprocessing WikiText-103"):
            text = article['text'].strip()

            if not text or len(text) < 10:
                continue

            tokenized = self.tokenizer(
                text,
                truncation=False,
                padding=False,
                return_tensors='pt'
            )

            input_ids = tokenized['input_ids'].squeeze(0)

            for i in range(0, len(input_ids), self.stride):
                chunk_ids = input_ids[i:i + self.max_length]

                if len(chunk_ids) < 50:
                    continue

                original_length = len(chunk_ids)

                if len(chunk_ids) < self.max_length:
                    padding_length = self.max_length - len(chunk_ids)
                    pad_id = self.tokenizer.pad_token_id
                    chunk_ids = torch.cat([
                        chunk_ids,
                        torch.full((padding_length,), pad_id)
                    ])

                attention_mask = torch.zeros(self.max_length, dtype=torch.long)
                attention_mask[:original_length] = 1

                labels = chunk_ids.clone()
                labels[attention_mask == 0] = -100

                processed.append({
                    'input_ids': chunk_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                })

        return processed

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):

    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def create_dataloaders(tokenizer, train_split='train', val_split='validation',
                      test_split=None, batch_size=8, max_length=384,
                      doc_stride=128, num_workers=0):

    train_dataset = WikiTextDataset(
        tokenizer,
        split=train_split,
        max_length=max_length,
        stride=doc_stride
    )

    val_dataset = WikiTextDataset(
        tokenizer,
        split=val_split,
        max_length=max_length,
        stride=doc_stride
    )

    test_loader = None
    if test_split:
        test_dataset = WikiTextDataset(
            tokenizer,
            split=test_split,
            max_length=max_length,
            stride=doc_stride
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            collate_fn=collate_fn
        )

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

    if test_loader is not None:
        return train_loader, val_loader, test_loader
    return train_loader, val_loader
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm

class SQuADDataset(Dataset):
    def __init__(self, tokenizer, split='train', max_length=384, doc_stride=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        
        self.dataset = load_dataset('squad', split=split)
        self.examples = self.preprocess_dataset()
        
    def preprocess_dataset(self):
        processed = []
        for example in tqdm(self.dataset, desc="Preprocessing SQuAD"):
            context = example['context']
            question = example['question']
            answers = example['answers']
            
            encoding = self.tokenizer(
                question,
                context,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_offsets_mapping=True
            )
            
            if len(answers['answer_start']) > 0:
                start_char = answers['answer_start'][0]
                end_char = start_char + len(answers['text'][0])
                
                start_token = 0
                end_token = 0
                for i, (offset_start, offset_end) in enumerate(encoding['offset_mapping']):
                    if offset_start <= start_char < offset_end:
                        start_token = i
                    if offset_start < end_char <= offset_end:
                        end_token = i
                        break
                
                processed.append({
                    'input_ids': torch.tensor(encoding['input_ids']),
                    'attention_mask': torch.tensor(encoding['attention_mask']),
                    'start_positions': torch.tensor(start_token),
                    'end_positions': torch.tensor(end_token)
                })
        
        return processed
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def create_dataloaders(tokenizer, train_split, val_split, 
                       batch_size, max_length, doc_stride):
    train_dataset = SQuADDataset(tokenizer, split=train_split, 
                                 max_length=max_length, doc_stride=doc_stride)
    val_dataset = SQuADDataset(tokenizer, split=val_split, 
                               max_length=max_length, doc_stride=doc_stride)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader
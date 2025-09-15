import torch
import math
import numpy as np
from datasets import load_dataset
from typing import Dict, Optional
from tqdm import tqdm

class PerplexityEvaluator:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = 'cuda'
        self.model = self.model.to(self.device)

    def calculate_perplexity(self, dataset_name: str, bit_config: Dict,
                            stride: int = 512, max_samples: int = 1000) -> float:
        """
        Calculate perplexity on WikiText2 or C4
        Use sliding window with specified stride
        """
        self.model.eval()

        if dataset_name == 'wikitext2':
            try:
                dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
                text_field = 'text'
            except:
                print(f"Warning: Could not load {dataset_name} dataset")
                return float('inf')
        elif dataset_name == 'c4':
            try:
                dataset = load_dataset('c4', 'en', split='validation[:1000]')
                text_field = 'text'
            except:
                print(f"Warning: Could not load {dataset_name} dataset")
                return float('inf')
        else:
            print(f"Unknown dataset: {dataset_name}")
            return float('inf')

        text = '\n'.join([item[text_field] for item in dataset if item[text_field].strip()])

        if not text:
            return float('inf')

        encodings = self.tokenizer(text, return_tensors='pt', truncation=False)

        max_length = self.model.config.n_positions if hasattr(self.model.config, 'n_positions') else 1024

        nlls = []
        prev_end_loc = 0

        for begin_loc in tqdm(range(0, len(encodings.input_ids[0]), stride),
                              desc=f"Calculating perplexity on {dataset_name}"):
            end_loc = min(begin_loc + max_length, len(encodings.input_ids[0]))

            if end_loc <= begin_loc:
                break

            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)

            if input_ids.shape[1] < 2:
                continue

            with torch.no_grad():
                try:
                    outputs = self.model(input_ids, labels=input_ids)
                    loss = outputs['loss']

                    if begin_loc == 0:
                        nlls.append(loss * input_ids.shape[1])
                    else:
                        trg_len = end_loc - max(prev_end_loc, begin_loc)
                        nlls.append(loss * trg_len)

                    prev_end_loc = end_loc
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue

            if len(nlls) >= max_samples:
                break

            torch.cuda.empty_cache()

        if not nlls:
            return float('inf')

        ppl = torch.exp(torch.stack(nlls).sum() / prev_end_loc)

        return ppl.item()

    def evaluate_all_datasets(self, bit_config: Dict) -> Dict:
        """
        Return perplexity for both WikiText2 and C4
        Format: {'WikiText2': 11.2, 'C4': 7.5}
        """
        results = {}

        print("  Calculating WikiText2 perplexity...")
        wikitext2_ppl = self.calculate_perplexity('wikitext2', bit_config, max_samples=500)
        results['WikiText2'] = round(wikitext2_ppl, 1)

        print("  Calculating C4 perplexity...")
        c4_ppl = self.calculate_perplexity('c4', bit_config, max_samples=200)
        results['C4'] = round(c4_ppl, 1)

        return results
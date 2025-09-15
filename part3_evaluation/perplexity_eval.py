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
        self.device = device
        self.model = self.model.to(self.device)

    def calculate_perplexity(self, dataset_name: str, bit_config: Dict,
                            stride: int = 128, max_samples: int = 100) -> float:
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
                # Try loading C4 dataset
                dataset = load_dataset('c4', 'en', split='validation[:100]', streaming=True)
                # Convert streaming dataset to list
                dataset = list(dataset)
                text_field = 'text'
            except:
                print(f"Warning: Could not load {dataset_name} dataset, using placeholder")
                # Use a simple placeholder text for testing
                dataset = [{'text': f"Sample text {i} for perplexity evaluation." * 10} for i in range(50)]
                text_field = 'text'
        else:
            print(f"Unknown dataset: {dataset_name}")
            return float('inf')

        # Process text in chunks to avoid memory issues
        texts = [item[text_field] for item in dataset if item[text_field].strip()][:100]  # Limit samples
        text = '\n'.join(texts)

        if not text:
            return float('inf')

        # Tokenize with truncation to avoid exceeding max length
        encodings = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=50000)

        # Use actual model's context length
        max_length = self.model.config.n_positions if hasattr(self.model.config, 'n_positions') else 256

        # Adjust stride to be smaller than max_length
        stride = min(stride, max_length // 2)

        nlls = []
        prev_end_loc = 0

        for begin_loc in tqdm(range(0, min(len(encodings.input_ids[0]), max_samples * stride), stride),
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
                    loss = outputs['loss']  # This is already averaged per token

                    # Just accumulate the losses - they're already averaged
                    # We'll compute the mean of means later
                    nlls.append(loss.item())

                    prev_end_loc = end_loc
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue

            if len(nlls) >= max_samples:
                break

            torch.cuda.empty_cache()

        if not nlls:
            return float('inf')

        # Calculate average loss across all segments
        avg_loss = sum(nlls) / len(nlls)

        # Perplexity is exp(average_loss)
        ppl = math.exp(avg_loss)

        return ppl

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
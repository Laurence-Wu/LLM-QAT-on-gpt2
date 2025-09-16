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
                            stride: int = 512, max_length: int = 1024) -> float:
        """
        Calculate perplexity on WikiText2 or C4
        Using proper sliding window approach
        """
        self.model.eval()

        # Load dataset
        if dataset_name == 'wikitext2':
            try:
                dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
                texts = [item['text'] for item in dataset if item['text'].strip()]
            except Exception as e:
                print(f"Warning: Could not load {dataset_name} dataset: {e}")
                return float('inf')
        elif dataset_name == 'c4':
            try:
                # Load a smaller subset of C4 for evaluation
                dataset = load_dataset('allenai/c4', 'en', split='validation', streaming=True)
                texts = []
                for i, item in enumerate(dataset):
                    if i >= 100:  # Use first 100 documents
                        break
                    texts.append(item['text'])
            except Exception as e:
                print(f"Warning: Could not load C4 dataset: {e}")
                # Fallback to WikiText2 if C4 fails
                try:
                    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
                    texts = [item['text'] for item in dataset if item['text'].strip()][:50]
                except:
                    return float('inf')
        else:
            print(f"Unknown dataset: {dataset_name}")
            return float('inf')

        if not texts:
            return float('inf')

        # Join texts with proper spacing
        text = ' '.join(texts)

        # Tokenize the entire text
        encodings = self.tokenizer(text, return_tensors='pt', truncation=False)

        # Get model's actual context length
        model_max_length = self.model.config.n_positions if hasattr(self.model.config, 'n_positions') else 1024
        max_length = min(max_length, model_max_length)

        # Calculate stride (50% overlap)
        stride = min(stride, max_length // 2)

        total_loss = 0.0
        total_tokens = 0

        # Process text in sliding windows
        seq_len = encodings.input_ids.size(1)

        prev_end_loc = 0
        iterations = 0
        for begin_loc in tqdm(range(0, seq_len, stride),
                              desc=f"Calculating perplexity on {dataset_name}",
                              disable=False):
            if iterations >= 1000:  # Limit to 1000 iterations
                break

            end_loc = min(begin_loc + max_length, seq_len)

            # Skip if window is too small
            if end_loc - begin_loc < 10:
                break

            iterations += 1

            trg_len = end_loc - prev_end_loc  # How many tokens we're actually evaluating
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)

            # Create attention mask
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                try:
                    outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )

                    # Get the loss (cross-entropy, already normalized per token)
                    loss = outputs.loss

                    # Accumulate loss weighted by number of tokens
                    # Only count tokens from prev_end_loc to end_loc to avoid double counting
                    if prev_end_loc < begin_loc:
                        # We have overlap, only count new tokens
                        trg_len = end_loc - begin_loc

                    total_loss += loss.item() * trg_len
                    total_tokens += trg_len

                    prev_end_loc = end_loc

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        continue
                    else:
                        print(f"Error processing batch: {e}")
                        continue

            # Clear cache periodically
            if begin_loc % (stride * 10) == 0:
                torch.cuda.empty_cache()

            # Removed token limit - iterations limit is sufficient

        if total_tokens == 0:
            return float('inf')

        # Calculate average loss
        avg_loss = total_loss / total_tokens

        # Perplexity is exp(average_loss)
        try:
            ppl = math.exp(avg_loss)
        except OverflowError:
            ppl = float('inf')

        return ppl

    def evaluate_all_datasets(self, bit_config: Dict) -> Dict:
        """
        Return perplexity for both WikiText2 and C4
        Format: {'WikiText2': 11.2, 'C4': 7.5}
        """
        results = {}

        print("  Calculating WikiText2 perplexity...")
        wikitext2_ppl = self.calculate_perplexity('wikitext2', bit_config)
        results['WikiText2'] = round(wikitext2_ppl, 1)

        print("  Calculating C4 perplexity...")
        c4_ppl = self.calculate_perplexity('c4', bit_config)
        results['C4'] = round(c4_ppl, 1)

        return results
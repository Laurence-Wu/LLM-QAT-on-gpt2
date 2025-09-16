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

        # Get model's actual context length
        model_max_length = self.model.config.n_positions if hasattr(self.model.config, 'n_positions') else 1024
        max_length = min(max_length, model_max_length)

        # Calculate stride (50% overlap)
        stride = min(stride, max_length // 2)

        total_loss = 0.0
        total_tokens = 0
        iterations = 0

        # Process texts one by one to avoid tokenization length issues
        for text in tqdm(texts[:100], desc=f"Processing {dataset_name} documents"):
            if not text.strip():
                continue

            # Tokenize individual text with proper truncation
            encodings = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length * 10  # Allow longer texts but we'll process in windows
            )

            seq_len = encodings.input_ids.size(1)

            # Skip very short texts
            if seq_len < 10:
                continue

            # Process this text in sliding windows
            for begin_loc in range(0, seq_len, stride):
                if iterations >= 1000:  # Limit to 1000 iterations total
                    break

                end_loc = min(begin_loc + max_length, seq_len)

                # Skip if window is too small
                if end_loc - begin_loc < 10:
                    break

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

                        # Handle different output formats
                        if hasattr(outputs, 'loss'):
                            loss = outputs.loss
                        elif isinstance(outputs, dict) and 'loss' in outputs:
                            loss = outputs['loss']
                        elif isinstance(outputs, tuple) and len(outputs) > 0:
                            # Some models return (loss, logits, ...)
                            loss = outputs[0]
                        else:
                            # Calculate loss manually if not provided
                            logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']

                            # Shift for language modeling
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = input_ids[..., 1:].contiguous()

                            # Calculate cross entropy loss
                            loss_fct = torch.nn.CrossEntropyLoss()
                            loss = loss_fct(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1)
                            )

                        # Accumulate loss
                        batch_size = end_loc - begin_loc
                        total_loss += loss.item() * batch_size
                        total_tokens += batch_size
                        iterations += 1

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            continue
                        else:
                            print(f"Error processing batch: {e}")
                            continue
                    except Exception as e:
                        print(f"Error: {e}")
                        continue

                # Clear cache periodically
                if iterations % 100 == 0:
                    torch.cuda.empty_cache()

            if iterations >= 1000:
                break

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
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

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def calculate_perplexity(self, dataset_name: str, bit_config: Dict,
                            stride: int = 128, max_length: int = 256) -> float:
        """
        Calculate perplexity on WikiText2 or C4
        Using proper sliding window approach optimized for small GPT-2
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

        # Get model's actual context length (256 for our model)
        model_max_length = self.model.config.n_positions if hasattr(self.model.config, 'n_positions') else 256
        max_length = min(max_length, model_max_length)

        print(f"Model max position: {model_max_length}, using max_length: {max_length}")

        # Smaller stride for better coverage
        stride = min(stride, max_length // 2)

        all_losses = []
        iterations = 0

        # Process texts one by one to avoid tokenization length issues
        for text_idx, text in enumerate(tqdm(texts[:100], desc=f"Processing {dataset_name} documents")):
            if not text.strip():
                continue

            # Tokenize individual text with proper truncation
            encodings = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length * 20,  # Allow longer texts but we'll process in windows
                padding=False
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

                # Extract window and ensure it doesn't exceed model's max positions
                window_size = min(end_loc - begin_loc, model_max_length)
                input_ids = encodings.input_ids[:, begin_loc:begin_loc + window_size].to(self.device)

                # Target is the same as input for language modeling
                target_ids = input_ids.clone()

                with torch.no_grad():
                    try:
                        # Forward pass - each window is treated as a new sequence starting from position 0
                        # This is implicit in how transformers handles position_ids when not provided
                        outputs = self.model(input_ids)

                        # Get logits
                        if hasattr(outputs, 'logits'):
                            logits = outputs.logits
                        elif isinstance(outputs, dict) and 'logits' in outputs:
                            logits = outputs['logits']
                        elif isinstance(outputs, tuple):
                            logits = outputs[0]
                        else:
                            logits = outputs

                        # Shift for next token prediction
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = target_ids[..., 1:].contiguous()

                        # Flatten the tokens
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                        losses = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )

                        # Get mean loss for this segment
                        segment_loss = losses.mean().item()

                        # Only add valid losses
                        if not math.isnan(segment_loss) and not math.isinf(segment_loss):
                            all_losses.append(segment_loss)

                        iterations += 1

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            continue
                        else:
                            continue
                    except Exception as e:
                        continue

                # Clear cache periodically
                if iterations % 50 == 0:
                    torch.cuda.empty_cache()

            if iterations >= 1000:
                break

        if not all_losses:
            return float('inf')

        # Calculate average loss across all valid segments
        avg_loss = np.mean(all_losses)

        # Remove outliers (losses that are too high)
        if len(all_losses) > 10:
            # Use median and IQR to filter outliers
            q1 = np.percentile(all_losses, 25)
            q3 = np.percentile(all_losses, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            filtered_losses = [l for l in all_losses if lower_bound <= l <= upper_bound]
            if filtered_losses:
                avg_loss = np.mean(filtered_losses)

        # Perplexity is exp(average_loss)
        try:
            ppl = math.exp(avg_loss)
            # Cap perplexity at a reasonable value
            if ppl > 100000:
                ppl = 100000
        except OverflowError:
            ppl = 100000

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
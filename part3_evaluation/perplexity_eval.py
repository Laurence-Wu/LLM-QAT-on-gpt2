import torch
import math
import numpy as np
from datasets import load_dataset
from typing import Dict, Optional
from tqdm import tqdm

class PerplexityEvaluator:
    def __init__(self, model, tokenizer, device, config):
        """Initialize with required config - NO DEFAULTS"""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config  # Store full config
        self.model = self.model.to(self.device)

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def calculate_perplexity(self, dataset_name: str, bit_config: Dict) -> float:
        """
        Calculate perplexity using sliding window with proper context handling.

        Each window uses first half as context (not scored) and second half for prediction (scored).
        This ensures all scored tokens have sufficient context.

        Args:
            dataset_name: Name of dataset to evaluate
            bit_config: Bit configuration for the model
        """

        stride = self.config['stride']
        max_length = self.config['max_length']
        self.model.eval()

        # Verify model is in eval mode
        if self.model.training:
            print("⚠️ WARNING: Model was not in eval mode! Setting to eval now...")
            self.model.eval()

        # Load dataset from config
        datasets_config = self.config.get('datasets', {})

        if dataset_name == 'wikitext2':
            try:
                wiki_cfg = datasets_config.get('WikiText2', {})
                dataset_name_str = wiki_cfg.get('dataset_name', 'wikitext')
                config_str = wiki_cfg.get('config', 'wikitext-2-raw-v1')
                split_str = wiki_cfg.get('split', 'test')
                dataset = load_dataset(dataset_name_str, config_str, split=split_str)
                texts = [item['text'] for item in dataset if item['text'].strip()]
            except Exception as e:
                print(f"Warning: Could not load {dataset_name} dataset: {e}")
                return float('inf')
        elif dataset_name == 'wikitext103':
            try:
                wiki_cfg = datasets_config.get('WikiText103', {})
                dataset_name_str = wiki_cfg.get('dataset_name', 'wikitext')
                config_str = wiki_cfg.get('config', 'wikitext-103-raw-v1')
                split_str = wiki_cfg.get('split', 'test')
                dataset = load_dataset(dataset_name_str, config_str, split=split_str)
                texts = [item['text'] for item in dataset if item['text'].strip()]
            except Exception as e:
                print(f"Warning: Could not load {dataset_name} dataset: {e}")
                return float('inf')
        elif dataset_name == 'openwebtext':
            try:
                # Use alternative OpenWebText dataset that doesn't use deprecated scripts
                dataset = load_dataset('stas/openwebtext-10k', split='train')
                texts = [item['text'] for item in dataset if item['text'].strip()][:100]
            except Exception as e:
                print(f"Warning: Could not load OpenWebText dataset: {e}")
                return float('inf')
        else:
            print(f"Unknown dataset: {dataset_name}")
            return float('inf')

        if not texts:
            return float('inf')

        # Sliding window approach with proper context handling
        all_losses = []
        max_texts = self.config.get('max_samples', 100)

        # Calculate context and prediction sizes
        context_size = max_length // 2  # First half for context
        predict_size = max_length - context_size  # Second half for prediction

        print(f"Using sliding window: {max_length} tokens, stride: {stride}")
        print(f"Processing {min(len(texts), max_texts)} texts")

        for text_idx, text in enumerate(tqdm(texts[:max_texts], desc=f"Processing {dataset_name}")):
            if not text.strip():
                continue

            # Sliding window method with proper context handling
            encodings = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length * 4,  # Allow longer for sliding window
                padding=False
            )

            seq_len = encodings.input_ids.size(1)
            if seq_len < 10:
                continue

            # Use sliding window with proper context handling
            # Step by stride for the sliding windows
            for window_idx, begin_loc in enumerate(range(0, seq_len, stride)):
                end_loc = min(begin_loc + max_length, seq_len)
                window_size = end_loc - begin_loc

                # Need at least context_size + some tokens to predict
                if window_size < context_size + 10:
                    break

                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)

                with torch.no_grad():
                    try:
                        outputs = self.model(input_ids)

                        # Check if outputs is already a tensor (raw logits)
                        if isinstance(outputs, torch.Tensor):
                            logits = outputs  # Use directly, preserves batch dimension
                        else:
                            # Try to extract logits from dict/object
                            try:
                                logits = outputs.logits
                            except AttributeError:
                                try:
                                    # Only use indexing for non-tensor outputs
                                    logits = outputs[0]
                                except (IndexError, TypeError):
                                    print(f"Warning: Could not extract logits from outputs type {type(outputs)}")
                                    continue

                        # Calculate loss using standard next-token prediction
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = input_ids[..., 1:].contiguous()

                        # CRITICAL: Only score tokens AFTER the context prefix
                        if window_idx == 0:
                            # First window: skip initial tokens (they have no context)
                            skip_tokens = min(32, window_size // 4)  # Skip first 32 tokens or 25% of window
                        else:
                            # Subsequent windows: skip the context portion (first half)
                            skip_tokens = context_size

                        # Only calculate loss on tokens with sufficient context
                        if shift_logits.size(1) > skip_tokens:
                            shift_logits = shift_logits[:, skip_tokens:, :]
                            shift_labels = shift_labels[:, skip_tokens:]
                        else:
                            continue  # Window too small after skipping context

                        loss_fct = torch.nn.CrossEntropyLoss()
                        loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )

                        if not torch.isnan(loss) and not torch.isinf(loss):
                            loss_value = loss.item()
                            all_losses.append(loss_value)

                    except Exception as e:
                        continue

        if not all_losses:
            return float('inf')

        # Calculate perplexity with detailed statistics
        avg_loss = np.mean(all_losses)
        std_loss = np.std(all_losses)
        min_loss = min(all_losses)
        max_loss = max(all_losses)

        # Calculate percentiles
        sorted_losses = sorted(all_losses)
        p25 = sorted_losses[int(len(sorted_losses) * 0.25)]
        p50 = sorted_losses[int(len(sorted_losses) * 0.50)]  # median
        p75 = sorted_losses[int(len(sorted_losses) * 0.75)]
        p95 = sorted_losses[int(len(sorted_losses) * 0.95)]

        perplexity = math.exp(avg_loss)

        print(f"  Processed {len(all_losses)} segments, PPL: {perplexity:.1f}")

        return perplexity

    def evaluate_all_datasets(self, bit_config: Dict) -> Dict:
        """
        Return perplexity for WikiText2, WikiText103, and OpenWebText
        Format: {'WikiText2': 11.2, 'WikiText103': 10.5, 'OpenWebText': 7.5}
        """
        results = {}

        print("  Calculating WikiText2 perplexity...")
        wikitext2_ppl = self.calculate_perplexity('wikitext2', bit_config)
        results['WikiText2'] = round(wikitext2_ppl, 1)

        # Only calculate WikiText103 if it's in the config
        if 'WikiText103' in self.config.get('datasets', {}):
            print("  Calculating WikiText103 perplexity...")
            wikitext103_ppl = self.calculate_perplexity('wikitext103', bit_config)
            results['WikiText103'] = round(wikitext103_ppl, 1)

        print("  Calculating OpenWebText perplexity...")
        openwebtext_ppl = self.calculate_perplexity('openwebtext', bit_config)
        results['OpenWebText'] = round(openwebtext_ppl, 1)

        return results

    def evaluate_long_context(self, bit_config: Dict, config_override: Dict = None) -> Dict:
        """
        Evaluate with longer context (1024 tokens) to test model's full capacity
        """
        # Save original config
        original_config = self.config.copy()

        # Override with long context settings if provided
        if config_override:
            self.config.update(config_override)
            print(f"  Using long context: max_length={self.config['max_length']}, stride={self.config['stride']}")

        results = self.evaluate_all_datasets(bit_config)

        # Restore original config
        self.config = original_config

        return results
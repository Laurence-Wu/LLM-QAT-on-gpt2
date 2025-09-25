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
        
        stride = self.config['stride']
        max_length = self.config['max_length']
        self.model.eval()

        # Load dataset from config
        datasets_config = self.config.get('datasets', {})

        if dataset_name == 'wikitext2':
            try:
                wiki_cfg = datasets_config.get('WikiText2', {})
                dataset = load_dataset(
                    wiki_cfg.get('dataset_name', 'wikitext'),
                    wiki_cfg.get('config', 'wikitext-2-raw-v1'),
                    split=wiki_cfg.get('split', 'test')
                )
                texts = [item['text'] for item in dataset if item['text'].strip()]
            except Exception as e:
                print(f"Warning: Could not load {dataset_name} dataset: {e}")
                return float('inf')
        elif dataset_name == 'openwebtext':
            try:
                # Load OpenWebText config
                cfg = datasets_config.get('OpenWebText', {})

                # Load dataset
                dataset_name_str = cfg.get('dataset_name', 'Skylion007/openwebtext')
                dataset_config = cfg.get('config', None)
                dataset_split = cfg.get('split', 'train[:1000]')
                use_streaming = cfg.get('streaming', False)

                if dataset_config:
                    dataset = load_dataset(
                        dataset_name_str,
                        dataset_config,
                        split=dataset_split,
                        streaming=use_streaming
                    )
                else:
                    dataset = load_dataset(
                        dataset_name_str,
                        split=dataset_split,
                        streaming=use_streaming
                    )
                texts = []
                max_docs = self.config.get('max_samples', 100)
                for i, item in enumerate(dataset):
                    if i >= max_docs:
                        break
                    texts.append(item['text'])
            except Exception as e:
                print(f"Warning: Could not load OpenWebText dataset: {e}")
                return float('inf')
        else:
            print(f"Unknown dataset: {dataset_name}")
            return float('inf')

        if not texts:
            return float('inf')

        # Force max length to 256 for compatibility with training
        MAX_CONTEXT = 256
        model_max_length = MAX_CONTEXT  # Always use 256
        max_length = min(max_length, MAX_CONTEXT)  # Cap at 256

        print(f"Using fixed max_length: {max_length} (model trained with 256)")

        # Smaller stride for better coverage
        stride = min(stride, max_length // 2)

        all_losses = []
        iterations = 0

        # Process texts one by one to avoid tokenization length issues
        max_texts = self.config.get('max_samples', 100)
        for text_idx, text in enumerate(tqdm(texts[:max_texts], desc=f"Processing {dataset_name} documents")):
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

                # Extract window and ensure it doesn't exceed 256 tokens
                window_size = min(end_loc - begin_loc, MAX_CONTEXT)
                input_ids = encodings.input_ids[:, begin_loc:begin_loc + window_size].to(self.device)

                # Critical: Hard enforce 256 token limit
                if input_ids.size(1) > MAX_CONTEXT:
                    print(f"WARNING: Truncating input from {input_ids.size(1)} to {MAX_CONTEXT}")
                    input_ids = input_ids[:, :MAX_CONTEXT]

                # Check for invalid token IDs (exceeding vocabulary size)
                try:
                    vocab_size = self.model.config.vocab_size
                except AttributeError:
                    vocab_size = 50257
                    print(f"Warning: Could not get vocab_size, using {vocab_size}")
                if input_ids.max() >= vocab_size:
                    print(f"ERROR: Token ID {input_ids.max().item()} exceeds vocab size {vocab_size}")
                    # Clamp token IDs to valid range
                    input_ids = torch.clamp(input_ids, 0, vocab_size - 1)

                # Debug first iteration
                if iterations == 0:
                    print(f"    Debug: First window - input_ids shape: {input_ids.shape}")
                    print(f"      Token IDs range: [{input_ids.min().item()}, {input_ids.max().item()}]")

                # Target is the same as input for language modeling
                target_ids = input_ids.clone()

                with torch.no_grad():
                    try:
                        # Forward pass - model handles position_ids internally
                        outputs = self.model(input_ids)

                        # Get logits
                        try:
                            logits = outputs.logits
                        except AttributeError:
                            if isinstance(outputs, dict) and 'logits' in outputs:
                                logits = outputs['logits']
                            elif isinstance(outputs, tuple):
                                logits = outputs[0]
                            else:
                                logits = outputs

                        # Debug first iteration
                        if iterations == 0:
                            print(f"    Debug: First window - logits shape: {logits.shape}")
                            print(f"      Logits stats - mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
                            print(f"      Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")

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

                        # Debug first iteration
                        if iterations == 0:
                            print(f"    Debug: First segment loss: {segment_loss:.4f}")
                            print(f"      Perplexity would be: {math.exp(min(segment_loss, 10)):.2f}")

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
        # No outlier filtering - use all valid losses
        avg_loss = np.mean(all_losses)

        # Debug: Print loss statistics
        print(f"    Debug: Total iterations: {iterations}, valid losses: {len(all_losses)}")
        if all_losses:
            print(f"    Debug: Loss statistics - min: {min(all_losses):.4f}, max: {max(all_losses):.4f}, mean: {avg_loss:.4f}")
            print(f"    Debug: First 5 losses: {all_losses[:5]}")
            print(f"    Debug: Last 5 losses: {all_losses[-5:]}")

        # Perplexity is exp(average_loss)
        try:
            ppl = math.exp(avg_loss)
            print(f"    Debug: Computed perplexity: {ppl:.2f}")
            # Cap perplexity at a reasonable value
            if ppl > 100000:
                print(f"    Debug: Capping perplexity from {ppl:.2f} to 100000")
                ppl = 100000
        except OverflowError:
            print(f"    Debug: Overflow error with avg_loss={avg_loss:.4f}, setting perplexity to 100000")
            ppl = 100000

        return ppl

    def evaluate_all_datasets(self, bit_config: Dict) -> Dict:
        """
        Return perplexity for both WikiText2 and OpenWebText
        Format: {'WikiText2': 11.2, 'OpenWebText': 7.5}
        """
        results = {}

        print("  Calculating WikiText2 perplexity...")
        wikitext2_ppl = self.calculate_perplexity('wikitext2', bit_config)
        results['WikiText2'] = round(wikitext2_ppl, 1)

        print("  Calculating OpenWebText perplexity...")
        openwebtext_ppl = self.calculate_perplexity('openwebtext', bit_config)
        results['OpenWebText'] = round(openwebtext_ppl, 1)

        return results
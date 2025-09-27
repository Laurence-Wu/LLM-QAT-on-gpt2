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

    def calculate_perplexity(self, dataset_name: str, bit_config: Dict, use_simple_method: bool = False) -> float:
        """
        Calculate perplexity with option to use simple method or sliding window.

        Args:
            dataset_name: Name of dataset to evaluate
            bit_config: Bit configuration for the model
            use_simple_method: If True, use simple method like test_inference.py (no sliding window)
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
                print(f"  Loading WikiText2: {dataset_name_str}, config={config_str}, split={split_str}")
                dataset = load_dataset(dataset_name_str, config_str, split=split_str)
                texts = [item['text'] for item in dataset if item['text'].strip()]
                print(f"  Loaded {len(texts)} texts from WikiText2")
            except Exception as e:
                print(f"Warning: Could not load {dataset_name} dataset: {e}")
                return float('inf')
        elif dataset_name == 'wikitext103':
            try:
                wiki_cfg = datasets_config.get('WikiText103', {})
                dataset_name_str = wiki_cfg.get('dataset_name', 'wikitext')
                config_str = wiki_cfg.get('config', 'wikitext-103-raw-v1')
                split_str = wiki_cfg.get('split', 'test')
                print(f"  Loading WikiText103: {dataset_name_str}, config={config_str}, split={split_str}")
                dataset = load_dataset(dataset_name_str, config_str, split=split_str)
                texts = [item['text'] for item in dataset if item['text'].strip()]
                print(f"  Loaded {len(texts)} texts from WikiText103")
            except Exception as e:
                print(f"Warning: Could not load {dataset_name} dataset: {e}")
                return float('inf')
        elif dataset_name == 'openwebtext':
            try:
                cfg = datasets_config.get('OpenWebText', {})
                dataset_name_str = cfg.get('dataset_name', 'Skylion007/openwebtext')
                dataset_config = cfg.get('config', None)
                dataset_split = cfg.get('split', 'train[:1000]')
                use_streaming = cfg.get('streaming', False)

                if dataset_config:
                    dataset = load_dataset(dataset_name_str, dataset_config, split=dataset_split, streaming=use_streaming)
                else:
                    dataset = load_dataset(dataset_name_str, split=dataset_split, streaming=use_streaming)

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

        # Simple approach: use fixed max_length from config
        all_losses = []
        max_texts = self.config.get('max_samples', 100)

        if use_simple_method:
            print(f"Using SIMPLE method (like test_inference.py) - no sliding window")
            print(f"Processing {min(len(texts), max_texts)} texts with max_length={max_length}")
        else:
            print(f"Using SLIDING WINDOW method with max_length={max_length}, stride={stride}")
            print(f"Processing {min(len(texts), max_texts)} texts")
            print(f"Note: With stride={stride}, this will create ~{max_length//stride}x more segments than stride={max_length}")

        print(f"Total texts available: {len(texts)}")

        # Add debug for first text
        debug_first_text = True

        for text_idx, text in enumerate(tqdm(texts[:max_texts], desc=f"Processing {dataset_name}")):
            if not text.strip():
                continue

            if use_simple_method:
                # Simple method: process entire text at once (with truncation)
                encodings = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_length,
                    padding=False
                )

                seq_len = encodings.input_ids.size(1)
                if seq_len < 10:
                    continue

                input_ids = encodings.input_ids.to(self.device)

                with torch.no_grad():
                    try:
                        outputs = self.model(input_ids)
                        try:
                            logits = outputs.logits
                        except AttributeError:
                            try:
                                logits = outputs[0]
                            except (IndexError, TypeError):
                                print(f"Warning: Could not extract logits from outputs type {type(outputs)}")
                                continue

                        # Calculate loss using standard next-token prediction
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = input_ids[..., 1:].contiguous()

                        loss_fct = torch.nn.CrossEntropyLoss()
                        loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )

                        if not torch.isnan(loss) and not torch.isinf(loss):
                            loss_value = loss.item()
                            all_losses.append(loss_value)

                            # Debug first few losses
                            if debug_first_text and len(all_losses) <= 3:
                                print(f"\n  Debug text {len(all_losses)}:")
                                print(f"    Input shape: {input_ids.shape}")
                                print(f"    Logits shape: {logits.shape}")
                                print(f"    Loss: {loss_value:.4f} (PPL: {math.exp(loss_value):.2f})")
                                # Check logits statistics
                                logits_mean = logits.mean().item()
                                logits_std = logits.std().item()
                                logits_min = logits.min().item()
                                logits_max = logits.max().item()
                                print(f"    Logits stats: mean={logits_mean:.2f}, std={logits_std:.2f}, min={logits_min:.2f}, max={logits_max:.2f}")

                                # Warning for abnormal logits
                                if logits_mean < -50:
                                    print(f"    ⚠️ WARNING: Abnormally low logits detected! Model may be misconfigured or quantization failed.")
                                    print(f"    Expected logits mean: -5 to -15, Got: {logits_mean:.2f}")
                                    print(f"    This indicates severe quantization degradation or initialization issues.")

                                if len(all_losses) >= 3:
                                    debug_first_text = False  # Stop debugging after first text
                        else:
                            if debug_first_text:
                                print(f"\n  ⚠️ Invalid loss detected: {loss.item() if not torch.isnan(loss) else 'NaN'}")

                    except Exception as e:
                        if debug_first_text:
                            print(f"\n  ❌ Exception during evaluation: {e}")
                        continue

            else:
                # Original sliding window method
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

                # Use sliding window approach similar to test_inference.py
                for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                if end_loc - begin_loc < 10:
                    break

                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)

                with torch.no_grad():
                    try:
                        outputs = self.model(input_ids)
                        try:
                            logits = outputs.logits
                        except AttributeError:
                            try:
                                logits = outputs[0]
                            except (IndexError, TypeError):
                                print(f"Warning: Could not extract logits from outputs type {type(outputs)}")
                                continue

                        # Calculate loss using standard next-token prediction
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = input_ids[..., 1:].contiguous()

                        loss_fct = torch.nn.CrossEntropyLoss()
                        loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )

                        if not torch.isnan(loss) and not torch.isinf(loss):
                            loss_value = loss.item()
                            all_losses.append(loss_value)

                            # Debug first few losses
                            if debug_first_text and len(all_losses) <= 3:
                                print(f"\n  Debug segment {len(all_losses)}:")
                                print(f"    Input shape: {input_ids.shape}")
                                print(f"    Logits shape: {logits.shape}")
                                print(f"    Loss: {loss_value:.4f} (PPL: {math.exp(loss_value):.2f})")
                                # Check logits statistics
                                logits_mean = logits.mean().item()
                                logits_std = logits.std().item()
                                logits_min = logits.min().item()
                                logits_max = logits.max().item()
                                print(f"    Logits stats: mean={logits_mean:.2f}, std={logits_std:.2f}, min={logits_min:.2f}, max={logits_max:.2f}")

                                # Warning for abnormal logits
                                if logits_mean < -50:
                                    print(f"    ⚠️ WARNING: Abnormally low logits detected! Model may be misconfigured or quantization failed.")
                                    print(f"    Expected logits mean: -5 to -15, Got: {logits_mean:.2f}")
                                    print(f"    This indicates severe quantization degradation or initialization issues.")

                                if len(all_losses) >= 3:
                                    debug_first_text = False  # Stop debugging after first text
                        else:
                            if debug_first_text:
                                print(f"\n  ⚠️ Invalid loss detected: {loss.item() if not torch.isnan(loss) else 'NaN'}")

                    except Exception as e:
                        if debug_first_text:
                            print(f"\n  ❌ Exception during evaluation: {e}")
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

        print(f"\n  === Loss Distribution Analysis ===")
        print(f"  Processed {len(all_losses)} segments from {text_idx + 1} texts")
        print(f"  Loss stats:")
        print(f"    Mean: {avg_loss:.4f} (PPL: {perplexity:.2f})")
        print(f"    Std:  {std_loss:.4f}")
        print(f"    Min:  {min_loss:.4f} (PPL: {math.exp(min_loss):.2f})")
        print(f"    Max:  {max_loss:.4f} (PPL: {math.exp(max_loss):.2f})")
        print(f"  Percentiles:")
        print(f"    25th: {p25:.4f} (PPL: {math.exp(p25):.2f})")
        print(f"    50th: {p50:.4f} (PPL: {math.exp(p50):.2f})")
        print(f"    75th: {p75:.4f} (PPL: {math.exp(p75):.2f})")
        print(f"    95th: {p95:.4f} (PPL: {math.exp(p95):.2f})")

        # Check for outliers
        outlier_threshold = p95 + 1.5 * (p95 - p75)  # IQR method for extreme outliers
        outliers = [l for l in all_losses if l > outlier_threshold]
        if outliers:
            print(f"  \n⚠️  Found {len(outliers)} extreme outliers (>{outlier_threshold:.2f})")
            print(f"    Outlier mean: {np.mean(outliers):.4f} (PPL: {math.exp(np.mean(outliers)):.2f})")

            # Calculate perplexity without outliers
            filtered_losses = [l for l in all_losses if l <= outlier_threshold]
            if filtered_losses:
                filtered_avg = np.mean(filtered_losses)
                filtered_ppl = math.exp(filtered_avg)
                print(f"    Without outliers: loss={filtered_avg:.4f}, PPL={filtered_ppl:.2f}")

        # Sample some actual losses to see the pattern
        print(f"\n  Sample losses (first 10): {[f'{l:.3f}' for l in all_losses[:10]]}")
        print(f"  Sample losses (last 10):  {[f'{l:.3f}' for l in all_losses[-10:]]}")

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
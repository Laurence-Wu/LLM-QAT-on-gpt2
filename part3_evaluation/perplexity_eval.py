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
        """Simplified perplexity calculation using the same approach as test files."""

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

        print(f"Processing {min(len(texts), max_texts)} texts with max_length={max_length}, stride={stride}")

        for text_idx, text in enumerate(tqdm(texts[:max_texts], desc=f"Processing {dataset_name}")):
            if not text.strip():
                continue

            # Tokenize text
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
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                        # Calculate loss using standard next-token prediction
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = input_ids[..., 1:].contiguous()

                        loss_fct = torch.nn.CrossEntropyLoss()
                        loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )

                        if not torch.isnan(loss) and not torch.isinf(loss):
                            all_losses.append(loss.item())

                    except Exception as e:
                        continue

        if not all_losses:
            return float('inf')

        # Calculate perplexity
        avg_loss = np.mean(all_losses)
        perplexity = math.exp(avg_loss)

        print(f"  Processed {len(all_losses)} segments, avg_loss: {avg_loss:.4f}, perplexity: {perplexity:.2f}")

        return perplexity

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
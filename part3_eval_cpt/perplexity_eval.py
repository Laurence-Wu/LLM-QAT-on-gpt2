import torch
import math
import numpy as np
from datasets import load_dataset
from typing import Dict
from tqdm import tqdm

class PerplexityEvaluator:

    def __init__(self, model, tokenizer, device, config):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_dataset(self, dataset_name: str) -> list:
        datasets_config = self.config.get('datasets', {})
        dataset_map = {'wikitext2': ('WikiText2', 'wikitext', 'wikitext-2-raw-v1', 'validation', False), 'wikitext103': ('WikiText103', 'wikitext', 'wikitext-103-raw-v1', 'validation', False), 'c4': ('C4', 'allenai/c4', 'en', 'validation', True)}
        if dataset_name not in dataset_map:
            return []
        cfg_key, default_name, default_config, default_split, streaming = dataset_map[dataset_name]
        cfg = datasets_config.get(cfg_key, {})
        try:
            dataset = load_dataset(cfg.get('dataset_name', default_name), cfg.get('config', default_config), split=cfg.get('split', default_split), streaming=streaming)
            if streaming:
                return [item['text'] for i, item in enumerate(dataset) if i < 5000]
            else:
                return [item['text'] for item in dataset if item['text'].strip()]
        except Exception as e:
            print(f'Warning: Could not load {dataset_name}: {e}')
            return []

    def calculate_perplexity(self, dataset_name: str, bit_config: Dict) -> float:
        texts = self._load_dataset(dataset_name)
        if not texts:
            return float('inf')
        print(f'  Processing {len(texts)} texts')
        stride = self.config['stride']
        max_length = self.config['max_length']
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        window_count = 0
        loss_values = []
        for text_idx, text in enumerate(tqdm(texts, desc=f'Processing {dataset_name}')):
            if not text.strip():
                continue
            input_ids = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length * 10, padding=False).input_ids.to(self.device)
            if input_ids.size(1) < 32:
                continue
            prev_end_loc = 0
            for begin_loc in range(0, input_ids.size(1), stride):
                end_loc = min(begin_loc + max_length, input_ids.size(1))
                if end_loc - begin_loc < 32:
                    break
                target_start = max(prev_end_loc, begin_loc)
                target_end = end_loc
                if target_end <= target_start:
                    continue
                window_ids = input_ids[:, begin_loc:end_loc]
                with torch.no_grad():
                    outputs = self.model(window_ids)
                    logits = outputs.logits
                    if window_count == 0:
                        print(f'\n  DEBUG First window:')
                        print(f'    Logits shape: {logits.shape}')
                        print(f'    Logits mean: {logits.mean().item():.4f}')
                        print(f'    Logits std: {logits.std().item():.4f}')
                        print(f'    Logits min: {logits.min().item():.4f}')
                        print(f'    Logits max: {logits.max().item():.4f}')
                        positive_count = (logits > 0).sum().item()
                        total_logits = logits.numel()
                        print(f'    Positive logits: {positive_count}/{total_logits} ({100 * positive_count / total_logits:.1f}%)')
                        last_token_logits = logits[0, -1, :]
                        probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
                        top5 = probs.topk(5)
                        print(f"    Top-5 probs: {[f'{p:.4f}' for p in top5.values.tolist()]}")
                        top5_tokens = self.tokenizer.convert_ids_to_tokens(top5.indices.tolist())
                        print(f'    Top-5 tokens: {top5_tokens}')
                        print(f'    Window size: {window_ids.size(1)} tokens')
                        preview_text = self.tokenizer.decode(window_ids[0, :min(20, window_ids.size(1))])
                        print(f'    Window text preview: {preview_text}')
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = window_ids[..., 1:].contiguous()
                    target_start_in_window = target_start - begin_loc
                    target_end_in_window = target_end - begin_loc - 1
                    if target_end_in_window > target_start_in_window:
                        loss_logits = shift_logits[:, target_start_in_window:target_end_in_window, :]
                        loss_labels = shift_labels[:, target_start_in_window:target_end_in_window]
                        loss = torch.nn.functional.cross_entropy(loss_logits.reshape(-1, loss_logits.size(-1)), loss_labels.reshape(-1), reduction='sum')
                        if not (torch.isnan(loss) or torch.isinf(loss)):
                            loss_value = loss.item()
                            num_tokens = loss_labels.numel()
                            per_token_loss = loss_value / num_tokens
                            total_loss += loss_value
                            total_tokens += num_tokens
                            window_count += 1
                            loss_values.append(per_token_loss)
                            if window_count == 1:
                                print(f'    Loss (sum): {loss_value:.4f}')
                                print(f'    Tokens: {num_tokens}')
                                print(f'    Loss/token: {per_token_loss:.4f}')
                                print(f'    PPL (this window): {math.exp(per_token_loss):.2f}\n')
                prev_end_loc = target_end
        if total_tokens == 0:
            return float('inf')
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        print(f'\n  DEBUG Summary:')
        print(f'    Total windows: {window_count}')
        print(f'    Total tokens: {total_tokens}')
        print(f'    Avg loss/token: {avg_loss:.4f}')
        print(f'    Loss min/max: {min(loss_values):.4f}/{max(loss_values):.4f}')
        print(f'    Final PPL: {perplexity:.1f}\n')
        return perplexity

    def evaluate_all_datasets(self, bit_config: Dict) -> Dict:
        results = {}
        print('  Calculating WikiText2 perplexity...')
        wikitext2_ppl = self.calculate_perplexity('wikitext2', bit_config)
        results['WikiText2'] = round(wikitext2_ppl, 1)
        if 'WikiText103' in self.config.get('datasets', {}):
            print('  Calculating WikiText103 perplexity...')
            wikitext103_ppl = self.calculate_perplexity('wikitext103', bit_config)
            results['WikiText103'] = round(wikitext103_ppl, 1)
        print('  Calculating C4 perplexity...')
        c4_ppl = self.calculate_perplexity('c4', bit_config)
        results['C4'] = round(c4_ppl, 1)
        return results
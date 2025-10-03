import torch
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import json
from pathlib import Path

class CPTEvaluation:

    def __init__(self, model, tokenizer, model_size='GPT2', device='cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')
        self.device = torch.device('cuda')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.model_size = model_size
        self.model_params = sum((p.numel() for p in self.model.parameters())) / 1000000.0

    def calculate_model_size(self, bit_config: Dict) -> float:
        weight_bits = bit_config.get('W', 16)
        kv_bits = bit_config.get('KV', 16)
        weight_size_gb = self.model_params * weight_bits / (8 * 1024)
        try:
            model_config = self.model.config['model']
            n_layers = model_config.n_layer
            n_heads = model_config.n_head
            n_embd = model_config.n_embd
        except:
            n_layers = 12
            n_heads = 12
            n_embd = 768
        d_head = n_embd // n_heads
        max_seq_len = 2048
        batch_size = 1
        kv_cache_size_gb = 2 * n_layers * n_heads * d_head * max_seq_len * batch_size * kv_bits / (8 * 1024 ** 3)
        total_size_gb = weight_size_gb + kv_cache_size_gb
        return round(total_size_gb, 2)
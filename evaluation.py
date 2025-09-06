import torch
import torch.nn as nn
import math
import time
import random
from typing import Dict, List, Any
from config_h100 import AdversarialConfig

def evaluate_model(model, eval_loader):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            device = next(model.parameters()).device
            input_ids = batch['input_ids'].to(device)
            outputs = model(input_ids, labels=input_ids)
            total_loss += outputs['loss'].item()
            num_batches += 1
            
            if num_batches >= 10:
                break
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return avg_loss

def evaluate_quantization_configs(model, eval_loader, n_layers: int = 12):
    # Use model's bit widths from config
    model_bit_widths = getattr(model.config, 'bit_widths', [8, 16, 32])
    low_bits = min(model_bit_widths)
    mid_bits = model_bit_widths[len(model_bit_widths)//2] if len(model_bit_widths) > 1 else low_bits
    high_bits = max(model_bit_widths)
    
    configs = {
        'FP32': [{'attn_bits': 32, 'mlp_bits': 32} for _ in range(n_layers)],
        f'{high_bits}-bit': [{'attn_bits': high_bits, 'mlp_bits': high_bits} for _ in range(n_layers)],
        f'{low_bits}-bit': [{'attn_bits': low_bits, 'mlp_bits': low_bits} for _ in range(n_layers)],
        'Mixed': [{'attn_bits': high_bits, 'mlp_bits': low_bits} for _ in range(n_layers)],
        'Progressive': [{'attn_bits': high_bits if i < n_layers//3 else low_bits, 
                        'mlp_bits': high_bits if i < n_layers//2 else low_bits} 
                       for i in range(n_layers)]
    }
    
    results = {}
    for config_name, config in configs.items():
        model.set_layer_precision(config)
        
        perplexity = calculate_perplexity(model, eval_loader)
        model_size = calculate_model_size(config, n_embd=getattr(model.config, 'n_embd', 1024))
        throughput = measure_throughput(model, eval_loader)
        
        results[config_name] = {
            'perplexity': perplexity,
            'model_size_mb': model_size,
            'throughput_tokens_per_sec': throughput,
            'efficiency_score': throughput / (model_size * max(perplexity, 1.0))
        }
    
    return results

def calculate_perplexity(model, eval_loader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            device = next(model.parameters()).device
            input_ids = batch['input_ids'].to(device)
            outputs = model(input_ids, labels=input_ids)
            total_loss += outputs['loss'].item() * input_ids.numel()
            total_tokens += input_ids.numel()
            
            if total_tokens > 10000:
                break
    
    if total_tokens == 0:
        return float('inf')
    avg_loss = total_loss / total_tokens
    avg_loss = min(avg_loss, 20.0)
    return math.exp(avg_loss)

def calculate_model_size(layer_configs, n_embd=1024):
    """Calculate model size based on actual model dimensions"""
    total_bits = 0
    # Estimate parameters per layer based on embedding dimension
    # Attention: 4 * n_embd^2 (qkv projection + output)
    # MLP: 8 * n_embd^2 (fc + proj, assuming 4x expansion)
    params_per_layer = 12 * n_embd * n_embd
    
    for config in layer_configs:
        attn_bits = config.get('attn_bits', 32)
        mlp_bits = config.get('mlp_bits', 32)
        avg_bits = (attn_bits + mlp_bits) / 2
        total_bits += params_per_layer * avg_bits
    
    return total_bits / (8 * 1024 * 1024)

def measure_throughput(model, eval_loader):
    model.eval()
    
    total_tokens = 0
    start_time = time.time()
    
    with torch.no_grad():
        for batch in eval_loader:
            device = next(model.parameters()).device
            input_ids = batch['input_ids'].to(device)
            outputs = model(input_ids)
            total_tokens += input_ids.numel()
            
            if total_tokens > 10000:
                break
    
    elapsed_time = time.time() - start_time
    return total_tokens / max(elapsed_time, 0.001)

class AdversarialRobustnessTester:
    def __init__(self, model, epsilon=None, config=None):
        self.model = model
        if config is None:
            config = AdversarialConfig()
        self.config = config
        self.epsilon = epsilon if epsilon is not None else config.epsilon
        
    def fgsm_attack(self, inputs, labels):
        self.model.eval()
        
        with torch.no_grad():
            embeddings = self.model.wte(inputs)
        embeddings = embeddings.clone().detach().requires_grad_(True)
        
        outputs = self.model.forward_from_embeddings(embeddings, labels=labels)
        loss = outputs['loss']
        
        self.model.zero_grad()
        loss.backward()
        
        data_grad = embeddings.grad.data
        perturbed_embeddings = embeddings + self.epsilon * data_grad.sign()
        
        return perturbed_embeddings.detach()
    
    def evaluate_robustness(self, test_loader, use_random_precision=None):
        self.model.eval()
        
        clean_correct = 0
        adv_correct = 0
        total = 0
        
        if use_random_precision is None:
            use_random_precision = self.config.use_random_precision
        
        bit_widths = self.config.bit_widths if use_random_precision else [self.config.bit_widths[-1]]
        
        for batch in test_loader:
            device = next(self.model.parameters()).device
            inputs = batch['input_ids'].to(device)
            labels = batch['input_ids'].to(device)
            
            batch_size = inputs.size(0)
            
            for i in range(batch_size):
                if use_random_precision:
                    random_bits = random.choice(bit_widths)
                    n_layers = len(self.model.h) if hasattr(self.model, 'h') else 12
                    config = [{'attn_bits': random_bits, 'mlp_bits': random_bits} 
                             for _ in range(n_layers)]
                    self.model.set_layer_precision(config)
                
                with torch.no_grad():
                    clean_outputs = self.model(inputs[i:i+1])
                    clean_pred = clean_outputs['logits'].argmax(dim=-1)
                    clean_correct += (clean_pred == labels[i:i+1]).float().mean().item()
                
                adv_embeds = self.fgsm_attack(inputs[i:i+1], labels[i:i+1])
                
                with torch.no_grad():
                    adv_outputs = self.model.forward_from_embeddings(adv_embeds)
                    adv_pred = adv_outputs['logits'].argmax(dim=-1)
                    adv_correct += (adv_pred == labels[i:i+1]).float().mean().item()
                
                total += 1
                
                if total >= self.config.test_samples:
                    break
            
            if total >= self.config.test_samples:
                break
        
        clean_accuracy = clean_correct / max(total, 1)
        robust_accuracy = adv_correct / max(total, 1)
        
        return {
            'clean_accuracy': clean_accuracy,
            'robust_accuracy': robust_accuracy,
            'robustness_gap': clean_accuracy - robust_accuracy,
            'robustness_ratio': robust_accuracy / clean_accuracy if clean_accuracy > 0 else 0
        }
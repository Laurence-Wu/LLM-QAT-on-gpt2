
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
from datasets import load_dataset
import numpy as np
import random
import math
import json
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from dataclasses import dataclass
import os


@dataclass
class QuantizationConfig:
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_iterations: int = 1000
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 384
    doc_stride: int = 128


class QuantizationFunction(torch.autograd.Function):
    Enhanced with LLM-QAT paper insights for symmetric MinMax quantization
    def __init__(self, in_features, out_features, bias=True, 
                 weight_bits=8, activation_bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.weight_quantizer = LearnableFakeQuantize(weight_bits, symmetric=True, per_channel=True)
        self.activation_quantizer = LearnableFakeQuantize(activation_bits, symmetric=False)
        
    def forward(self, input):
        input_q = self.activation_quantizer(input)
        weight_q = self.weight_quantizer(self.weight)
        
        return F.linear(input_q, weight_q, self.bias)


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank if rank > 0 else 1.0
        
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.empty(rank, out_features))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling

class MultiPrecisionLoRA(nn.Module):
    def __init__(self, in_features, out_features, bit_widths=[4, 8, 16]):
        super().__init__()
        self.bit_widths = bit_widths
        
        self.lora_modules = nn.ModuleDict()
        for bits in bit_widths:
            if bits <= 4:
                rank = max(1, min(8, in_features // 64))  
            elif bits <= 8:
                rank = max(2, min(16, in_features // 32))  
            else:
                rank = max(4, min(32, in_features // 16))  
            
            alpha = rank * bits // 2
            
            self.lora_modules[f'lora_{bits}bit'] = LoRALayer(
                in_features, out_features, rank=rank, alpha=alpha
            )
        
        self.current_bits = 8
        
    def forward(self, x, bits=None):
        if bits is None:
            bits = self.current_bits
        
        key = f'lora_{bits}bit'
        if key in self.lora_modules:
            return self.lora_modules[key](x)
        else:
            nearest_bits = min(self.bit_widths, key=lambda b: abs(b - bits))
            return self.lora_modules[f'lora_{nearest_bits}bit'](x)
    
    def set_bits(self, bits):
        self.current_bits = bits

class QuantizedLinearWithLoRA(nn.Module):
    def __init__(self, config: GPT2Config, bit_widths=[4, 8, 16]):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        self.c_attn = QuantizedLinearWithLoRA(config.n_embd, 3 * config.n_embd, bit_widths=bit_widths)
        self.c_proj = QuantizedLinearWithLoRA(config.n_embd, config.n_embd, bit_widths=bit_widths)
        
        self.kv_quantizer = LearnableFakeQuantize(num_bits=8, symmetric=False)
        
        self.register_buffer("bias", torch.tril(torch.ones(config.n_positions, config.n_positions)))
        
    def forward(self, hidden_states, attention_mask=None):
        B, T, C = hidden_states.shape
        
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        k = self.kv_quantizer(k)
        v = self.kv_quantizer(v)
        
        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights.masked_fill(self.bias[:T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        
        attn_output = self.c_proj(attn_output)
        
        return attn_output
    
    def set_precision(self, weight_bits, activation_bits, kv_bits=8):
        self.c_attn.set_precision(weight_bits, activation_bits)
        self.c_proj.set_precision(weight_bits, activation_bits)
        self.kv_quantizer.num_bits = kv_bits

class QuantizedGPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config, bit_widths=[4, 8, 16]):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = QuantizedGPT2Attention(config, bit_widths)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = QuantizedGPT2MLP(config, bit_widths)
        
    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, attention_mask)
        hidden_states = residual + attn_output
        
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        return hidden_states
    
    def set_precision(self, attn_bits, mlp_bits, activation_bits=8, kv_bits=8):
        self.attn.set_precision(attn_bits, activation_bits, kv_bits)
        self.mlp.set_precision(mlp_bits, activation_bits)

class SwitchableQuantizedGPT2(nn.Module):
        for i, config in enumerate(layer_configs):
            if i < len(self.h):
                self.h[i].set_precision(**config)


class CyclicPrecisionScheduler:
        if self.current_step < self.warmup_steps:
            return self.max_bits
        
        adjusted_step = self.current_step - self.warmup_steps
        cycle_position = (adjusted_step % self.cycle_length) / self.cycle_length
        cos_value = (1 + math.cos(math.pi * cycle_position)) / 2
        current_bits = self.min_bits + (self.max_bits - self.min_bits) * cos_value
        
        return int(round(current_bits))
    
    def step(self):
        self.current_step += 1
        return self.get_precision()


class SQuADDataset(Dataset):
        processed = []
        for example in tqdm(self.dataset, desc="Preprocessing SQuAD"):
            context = example['context']
            question = example['question']
            answers = example['answers']
            
            encoding = self.tokenizer(
                question,
                context,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_offsets_mapping=True
            )
            
            if len(answers['answer_start']) > 0:
                start_char = answers['answer_start'][0]
                end_char = start_char + len(answers['text'][0])
                
                start_token = 0
                end_token = 0
                for i, (offset_start, offset_end) in enumerate(encoding['offset_mapping']):
                    if offset_start <= start_char < offset_end:
                        start_token = i
                    if offset_start < end_char <= offset_end:
                        end_token = i
                        break
                
                processed.append({
                    'input_ids': torch.tensor(encoding['input_ids']),
                    'attention_mask': torch.tensor(encoding['attention_mask']),
                    'start_positions': torch.tensor(start_token),
                    'end_positions': torch.tensor(end_token)
                })
        
        return processed
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def knowledge_distillation_loss(student_logits, teacher_logits, temperature=4.0, alpha=0.5):
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    
    distillation_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    
    return distillation_loss

def data_free_distillation_loss(model, teacher_model, input_ids, attention_mask):
    with torch.no_grad():
        teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask)
        teacher_logits = teacher_outputs.logits
    
    student_outputs = model(input_ids, attention_mask=attention_mask)
    student_logits = student_outputs.logits
    
    return knowledge_distillation_loss(student_logits, teacher_logits)


def train_switchable_quantization(model, train_loader, val_loader, config: TrainingConfig):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    teacher_model = None
    try:
        from transformers import GPT2LMHeadModel
        teacher_model = GPT2LMHeadModel.from_pretrained('gpt2')
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        pass  
    except Exception as e:
        pass  
    
    bit_configs = [
        [{'attn_bits': 8, 'mlp_bits': 8} for _ in range(12)],
        [{'attn_bits': 4, 'mlp_bits': 4} for _ in range(12)],
        [{'attn_bits': 8, 'mlp_bits': 4} for _ in range(12)],
        [{'attn_bits': 8 if i < 4 else 4, 'mlp_bits': 8 if i < 8 else 4} 
         for i in range(12)]
    ]
    
    best_val_loss = float('inf')
    
    for iteration in tqdm(range(config.num_iterations), desc="Training"):
        model.train()
        
        config_idx = random.randint(0, len(bit_configs) - 1)
        current_config = bit_configs[config_idx]
        model.set_layer_precision(current_config)
        
        total_loss = 0
        total_ce_loss = 0
        total_kd_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= config.gradient_accumulation_steps:
                break
            
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask', None)
            
            outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
            ce_loss = outputs['loss']
            
            kd_loss = 0
            if teacher_model is not None:
                try:
                    kd_loss = data_free_distillation_loss(model, teacher_model, input_ids, attention_mask)
                    loss = 0.7 * ce_loss + 0.3 * kd_loss
                except Exception:
                    loss = ce_loss  
            else:
                loss = ce_loss
            
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item()
            total_ce_loss += ce_loss.item() / config.gradient_accumulation_steps
            total_kd_loss += kd_loss.item() / config.gradient_accumulation_steps if isinstance(kd_loss, torch.Tensor) else 0
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        if iteration % 100 == 0:
            val_loss = evaluate_model(model, val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, iteration, 'best_model.pt')
    
    return model

def train_with_cpt(model, train_loader, val_loader, config: TrainingConfig):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            outputs = model(batch['input_ids'], labels=batch['input_ids'])
            total_loss += outputs['loss'].item()
            num_batches += 1
            
            if num_batches >= 10:  
                break
    
    return total_loss / num_batches

def evaluate_quantization_configs(model, eval_loader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            outputs = model(batch['input_ids'], labels=batch['input_ids'])
            total_loss += outputs['loss'].item() * batch['input_ids'].numel()
            total_tokens += batch['input_ids'].numel()
            
            if total_tokens > 10000:  
                break
    
    return math.exp(total_loss / total_tokens)

def calculate_model_size(layer_configs):
    model.eval()
    
    import time
    total_tokens = 0
    start_time = time.time()
    
    with torch.no_grad():
        for batch in eval_loader:
            outputs = model(batch['input_ids'])
            total_tokens += batch['input_ids'].numel()
            
            if total_tokens > 10000:
                break
    
    elapsed_time = time.time() - start_time
    return total_tokens / elapsed_time


class AdversarialRobustnessTester:
        inputs.requires_grad = True
        
        outputs = self.model(inputs, labels=labels)
        loss = outputs['loss']
        
        self.model.zero_grad()
        loss.backward()
        
        data_grad = inputs.grad.data
        perturbed_inputs = inputs + self.epsilon * data_grad.sign()
        
        return perturbed_inputs
    
    def evaluate_robustness(self, test_loader, use_random_precision=True):
    torch.save({
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)
    pass  

def load_checkpoint(model, optimizer, filename):
    report = {
        'quantization_evaluation': results['quantization_configs'],
        'adversarial_robustness': results['robustness'],
        'training_metrics': results['training'],
        'insights': {
            'best_config': max(results['quantization_configs'].items(), 
                             key=lambda x: x[1]['efficiency_score'])[0],
            'robustness_improvement': results['robustness']['dynamic']['robustness_ratio'] / 
                                     results['robustness']['static']['robustness_ratio'] - 1
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    pass  
    return report


def main():

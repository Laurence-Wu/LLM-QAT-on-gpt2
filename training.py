import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm
from typing import Optional
import gc

from config_h100 import TrainingConfig, CyclicPrecisionConfig
from quantization import CyclicPrecisionScheduler
# Temporarily disable checkpoint import due to disk quota issues
# from utils import save_checkpoint
from evaluation import evaluate_model

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

def train_switchable_quantization(model, train_loader, val_loader, config: TrainingConfig, n_layers: int = 12):
    # Aggressive H100 memory optimizations
    torch.cuda.empty_cache()
    gc.collect()
    
    # Set model to use minimal memory
    model.use_gradient_checkpointing = True
    
    # Use memory-efficient optimizer settings with reduced precision
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, 
                                  eps=1e-6, weight_decay=0.01, amsgrad=False)
    
    # Enable mixed precision training for memory efficiency
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    teacher_model = None
    try:
        from transformers import GPT2LMHeadModel
        teacher_model = GPT2LMHeadModel.from_pretrained('gpt2')
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
    except:
        teacher_model = None
    
    # Use config bit widths instead of hard-coded values
    model_bit_widths = getattr(model.config, 'bit_widths', [4, 8, 16])
    bit_configs = [
        [{'attn_bits': model_bit_widths[0], 'mlp_bits': model_bit_widths[0]} for _ in range(n_layers)],
        [{'attn_bits': model_bit_widths[1], 'mlp_bits': model_bit_widths[1]} for _ in range(n_layers)],
        [{'attn_bits': model_bit_widths[2], 'mlp_bits': model_bit_widths[2]} for _ in range(n_layers)],
        [{'attn_bits': model_bit_widths[0] if i > n_layers//2 else model_bit_widths[1], 
          'mlp_bits': model_bit_widths[0] if i > n_layers//3 else model_bit_widths[1]} for i in range(n_layers)]
    ]
    
    best_val_loss = float('inf')
    
    for iteration in tqdm(range(config.num_iterations), desc="Training"):
        model.train()
        config_idx = iteration % len(bit_configs)
        current_config = bit_configs[config_idx]
        model.set_layer_precision(current_config)
        
        total_loss = 0
        total_ce_loss = 0
        total_kd_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= config.gradient_accumulation_steps:
                break
            
            device = next(model.parameters()).device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Use mixed precision for memory efficiency
            if scaler is not None:
                with torch.cuda.amp.autocast():
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
                
                scaler.scale(loss).backward()
            else:
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
        
        # Use scaler for mixed precision if available
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        optimizer.zero_grad()
        
        # Memory cleanup every 10 iterations for H100 efficiency
        if iteration % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            val_loss = evaluate_model(model, val_loader)
            print(f"Iter {iteration}: Loss={total_loss:.4f}, Val Loss={val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best validation loss: {val_loss:.4f} (checkpoint saving disabled due to disk quota)")
    
    return model

def train_with_cpt(model, train_loader, val_loader, config: TrainingConfig, n_layers: int = 12):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Use CyclicPrecisionConfig if available
    cpt_config = CyclicPrecisionConfig()
    scheduler = CyclicPrecisionScheduler(
        min_bits=cpt_config.min_bits, 
        max_bits=cpt_config.max_bits, 
        cycle_length=cpt_config.cycle_length, 
        warmup_steps=cpt_config.warmup_steps
    )
    
    for iteration in tqdm(range(config.num_iterations), desc="CPT Training"):
        model.train()
        
        current_bits = scheduler.step()
        layer_config = [{'attn_bits': current_bits, 'mlp_bits': current_bits} 
                       for _ in range(n_layers)]
        model.set_layer_precision(layer_config)
        
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= config.gradient_accumulation_steps:
                break
                
            device = next(model.parameters()).device
            input_ids = batch['input_ids'].to(device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs['loss'] / config.gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    return model
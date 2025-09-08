"""
Training module for Switchable Precision GPT-2
Implements training loops with dynamic bit-width switching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm
import gc
import psutil
import os
from typing import Optional, Dict, List


def log_memory_usage(step: str = ""):
    """Log current memory usage for debugging OOM issues."""
    # CPU Memory
    cpu_mem = psutil.virtual_memory()
    cpu_used_gb = cpu_mem.used / (1024**3)
    cpu_total_gb = cpu_mem.total / (1024**3)
    cpu_percent = cpu_mem.percent
    
    print(f"[{step}] CPU Memory: {cpu_used_gb:.2f}/{cpu_total_gb:.2f} GB ({cpu_percent:.1f}%)")
    
    # GPU Memory
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
        gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"[{step}] GPU Memory: {gpu_allocated:.2f} GB allocated, {gpu_reserved:.2f} GB reserved")
    
    # Process-specific memory
    process = psutil.Process(os.getpid())
    process_mem = process.memory_info().rss / (1024**3)
    print(f"[{step}] Process Memory: {process_mem:.2f} GB")


def knowledge_distillation_loss(student_logits, teacher_logits, temperature=4.0):
    """
    Calculate knowledge distillation loss between student and teacher models.
    
    Args:
        student_logits: Logits from the student model
        teacher_logits: Logits from the teacher model
        temperature: Temperature for softening probability distributions
    
    Returns:
        KL divergence loss scaled by temperature squared
    """
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    
    distillation_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    
    return distillation_loss


def get_bit_width_schedule(strategy: str, num_iterations: int, bit_widths: List[int]) -> List[int]:
    if strategy == 'cyclic':
        # Cycle through bit widths in order
        schedule = []
        for i in range(num_iterations):
            schedule.append(bit_widths[i % len(bit_widths)])
        return schedule
    
    elif strategy == 'random':
        # Random bit width selection
        return [random.choice(bit_widths) for _ in range(num_iterations)]
    
    elif strategy == 'progressive':
        # Start with high precision, gradually decrease
        segments = num_iterations // len(bit_widths)
        schedule = []
        for bit_width in reversed(sorted(bit_widths)):
            schedule.extend([bit_width] * segments)
        # Fill remaining iterations
        while len(schedule) < num_iterations:
            schedule.append(min(bit_widths))
        return schedule[:num_iterations]
    
    else:
        return get_bit_width_schedule('cyclic', num_iterations, bit_widths)


def create_layer_precision_config(bit_width: int, n_layers: int, 
                                 strategy: str = 'uniform') -> List[Dict]:
    if strategy == 'uniform':
        # All layers use the same bit width
        return [{'attn_bits': bit_width, 'mlp_bits': bit_width} for _ in range(n_layers)]
    
    elif strategy == 'progressive':
        # Higher precision for early and late layers
        config = []
        for i in range(n_layers):
            if i < 2 or i >= n_layers - 2:
                # First 2 and last 2 layers get higher precision
                config.append({'attn_bits': min(bit_width * 2, 16), 
                             'mlp_bits': min(bit_width * 2, 16)})
            else:
                config.append({'attn_bits': bit_width, 'mlp_bits': bit_width})
        return config
    
    elif strategy == 'mixed':
        # Alternate between attention and MLP precision
        config = []
        for i in range(n_layers):
            if i % 2 == 0:
                # Even layers: higher precision attention
                config.append({'attn_bits': min(bit_width * 2, 16), 
                             'mlp_bits': bit_width})
            else:
                # Odd layers: higher precision MLP
                config.append({'attn_bits': bit_width, 
                             'mlp_bits': min(bit_width * 2, 16)})
        return config
    
    else:
        return create_layer_precision_config(bit_width, n_layers, 'uniform')


def train_switchable_quantization(model, train_loader, val_loader, config, model_config, n_layers: int = 12):
    """
    Main training loop for switchable precision training.
    
    Args:
        model: SwitchableQuantizedGPT2 model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: TrainingConfig object
        n_layers: Number of layers in the model
    
    Returns:
        Trained model
    """
    # Clear GPU cache for optimal memory usage
    torch.cuda.empty_cache()
    gc.collect()
    model.use_gradient_checkpointing = True
    
    # Log initial memory usage
    log_memory_usage("Training Start")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay
    )
    
    # Setup learning rate scheduler
    scheduler = None
    if config.warmup_steps > 0:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=config.warmup_steps,
            eta_min=config.learning_rate*0.1
        )
    
    # Setup mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    # Distilation step
    teacher_model = None
    try:
        from transformers import GPT2LMHeadModel
        print("Loading teacher model")
        teacher_model = GPT2LMHeadModel.from_pretrained('gpt2')
        device = next(model.parameters()).device
        teacher_model = teacher_model.to(device)
        teacher_model.eval() # USE THE buffer information of running mean and variance
        for param in teacher_model.parameters():
            param.requires_grad = False
        print("Teacher model loaded")
        log_memory_usage("After Teacher Model Load")
    except Exception as e:
        print(f"teacher model error: {e}")
        teacher_model = None
    
    bit_widths = model_config.bit_widths

    # Generate bit-width schedule
    bit_width_schedule = get_bit_width_schedule(
        config.switch_strategy, 
        config.num_iterations,
        bit_widths
    )
    
    # Training metrics tracking
    training_stats = {
        'iteration_losses': [],
        'validation_losses': [],
        'bit_width_usage': {bit: 0 for bit in bit_widths},
        'best_val_loss': float('inf'),
        'best_iteration': 0
    }
    
    # Main training loop
    print("\nStarting switchable precision training...")
    log_memory_usage("Before Training Loop")
    
    try:
        for iteration in tqdm(range(config.num_iterations), desc="switchableP"):
            
            model.train() # my little flag ~~~~~
        
        # Get current bit width from schedule
        current_bit_width = bit_width_schedule[iteration]
        training_stats['bit_width_usage'][current_bit_width] += 1
        
        # Create layer precision configuration
        layer_config = create_layer_precision_config(
            current_bit_width, 
            n_layers,
            strategy=config.switch_strategy
        )
        
        # Set model precision
        model.set_layer_precision(layer_config)
        
        # Log memory at first iteration to catch early OOM
        if iteration == 0:
            log_memory_usage("Before First Forward Pass")
        
        # Training step
        total_loss = 0
        total_ce_loss = 0
        total_kd_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= config.gradient_accumulation_steps:
                break
            
            # Log memory on first batch to catch loading issues
            if iteration == 0 and batch_idx == 0:
                log_memory_usage("After Loading First Batch")
            
            device = next(model.parameters()).device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward pass with mixed precision
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
                    ce_loss = outputs['loss']
                    
                    # Knowledge distillation if teacher is available
                    kd_loss = torch.tensor(0.0, device=device)
                    if teacher_model is not None:
                        with torch.no_grad():
                            teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask)
                            teacher_logits = teacher_outputs.logits
                        
                        student_logits = outputs['logits']
                        kd_loss = knowledge_distillation_loss(student_logits, teacher_logits)
                        loss = 0.7 * ce_loss + 0.3 * kd_loss
                    else:
                        loss = ce_loss
                    
                    loss = loss / config.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
                ce_loss = outputs['loss']
                
                kd_loss = torch.tensor(0.0, device=device)
                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask)
                        teacher_logits = teacher_outputs.logits
                    
                    student_logits = outputs['logits']
                    kd_loss = knowledge_distillation_loss(student_logits, teacher_logits)
                    loss = 0.7 * ce_loss + 0.3 * kd_loss
                else:
                    loss = ce_loss
                
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_kd_loss += kd_loss.item() if teacher_model is not None else 0
        
        # Optimizer step
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
        
        optimizer.zero_grad()
        
        if scheduler is not None and iteration < config.warmup_steps:
            scheduler.step()
        
        # Record training loss
        avg_loss = total_loss / config.gradient_accumulation_steps
        training_stats['iteration_losses'].append(avg_loss)
        
        # Logging
        if iteration % config.log_interval == 0:
            print(f"\nIteration {iteration}/{config.num_iterations}")
            print(f"Bit width: {current_bit_width}")
            print(f"Loss: {avg_loss:.4f}")
            print(f"CE Loss: {total_ce_loss / config.gradient_accumulation_steps:.4f}")
            if teacher_model is not None:
                print(f"KD Loss: {total_kd_loss / config.gradient_accumulation_steps:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Validation
        if iteration % config.eval_interval == 0 and iteration > 0:
            model.eval()
            val_loss = 0
            val_steps = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= 10:  # Limit validation steps
                        break
                    
                    device = next(model.parameters()).device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch.get('attention_mask', None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    
                    outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
                    val_loss += outputs['loss'].item()
                    val_steps += 1
            
            avg_val_loss = val_loss / max(val_steps, 1)
            training_stats['validation_losses'].append(avg_val_loss)
            
            print(f"  Validation Loss: {avg_val_loss:.4f}")
            
            # Track best model
            if avg_val_loss < training_stats['best_val_loss']:
                training_stats['best_val_loss'] = avg_val_loss
                training_stats['best_iteration'] = iteration
                print(f"  New best validation loss!")
        
        # Clear cache periodically
        if iteration % config.empty_cache_interval == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
        log_memory_usage("Training Loop Completed")
        print("Training loop finished successfully!")
    
    except Exception as e:
        print(f"\n!!! TRAINING ERROR CAUGHT !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        log_memory_usage("At Error")
        import traceback
        traceback.print_exc()
        raise e
    
    # Save training statistics
    log_memory_usage("Before Saving Stats")
    try:
        import json
        with open('training_stats.json', 'w') as f:
            json.dump(training_stats, f, indent=2)
        log_memory_usage("After Saving Stats")
        print("Training stats saved to training_stats.json")
    except Exception as e:
        print(f"Error saving stats: {e}")
    
    log_memory_usage("Before Final Message")
    print(f"\nTraining completed. Best validation loss: {training_stats['best_val_loss']:.4f}")
    
    log_memory_usage("Before Return Model")
    return model
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


def log_memory_usage(step: str = "", detailed=False):
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
        gpu_cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        gpu_cached_gb = gpu_cached / (1024**3)
        
        print(f"[{step}] GPU Memory: {gpu_allocated:.2f} GB allocated, {gpu_reserved:.2f} GB reserved, {gpu_cached_gb:.2f} GB cached")
        
        if detailed:
            # More detailed GPU memory stats
            print(f"[{step}] GPU Max Memory Allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
            print(f"[{step}] GPU Max Memory Reserved: {torch.cuda.max_memory_reserved() / (1024**3):.2f} GB")
            
            # Check for memory fragmentation
            try:
                mem_stats = torch.cuda.memory_stats()
                active_bytes = mem_stats.get('active_bytes.all.current', 0)
                reserved_bytes = mem_stats.get('reserved_bytes.all.current', 0)
                allocated_bytes = mem_stats.get('allocated_bytes.all.current', 0)
                print(f"[{step}] Active: {active_bytes/(1024**3):.2f} GB, Allocated: {allocated_bytes/(1024**3):.2f} GB")
            except:
                pass
    
    # Process-specific memory
    process = psutil.Process(os.getpid())
    process_mem = process.memory_info().rss / (1024**3)
    print(f"[{step}] Process Memory: {process_mem:.2f} GB")
    
    # Return memory values for tracking
    return {
        'cpu_used_gb': cpu_used_gb,
        'gpu_allocated_gb': gpu_allocated if torch.cuda.is_available() else 0,
        'gpu_reserved_gb': gpu_reserved if torch.cuda.is_available() else 0,
        'process_gb': process_mem
    }


def knowledge_distillation_loss(student_logits, teacher_logits, temperature=4.0):
    teacher_logits = teacher_logits.detach() #detach the teacher node from the bp graph
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    
    distillation_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    del soft_teacher, soft_student # delete these intermediate value for memory saving
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
    # Clear GPU cache for optimal memory usage
    torch.cuda.empty_cache()
    gc.collect()
    
    # Enable gradient checkpointing to save memory
    model.use_gradient_checkpointing = True
    model.train()  # Ensure model is in training mode
    
    # Log initial memory usage
    log_memory_usage("Training Start")
    
    # Initialize CPU gradient storage for accumulation
    cpu_gradients = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            cpu_gradients[name] = torch.zeros_like(param, dtype=torch.float32, device='cpu')
    
    # favorite RMSprop and momentum optimizer ~
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay
    )
    
    # Setup learning rate scheduler
    scheduler = None
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
            
            # Training step - reset accumulators  
            total_loss = 0
            total_ce_loss = 0
            total_kd_loss = 0
            
            # Clear GPU gradients and reset CPU accumulator for this iteration
            optimizer.zero_grad(set_to_none=True)  # More aggressive gradient clearing
            for name in cpu_gradients:
                cpu_gradients[name].zero_()
            
            # ========== DEBUG MEMORY MONITORING START ==========
            if iteration % 5 == 0:  # Monitor every 5 iterations
                print(f"\n{'='*60}")
                print(f"MEMORY DEBUG - Iteration {iteration}")
                print(f"{'='*60}")
                initial_mem = log_memory_usage("Iteration Start", detailed=True)
            # ========== DEBUG MEMORY MONITORING END ==========
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= config.gradient_accumulation_steps:
                    break # accumulate the gradients over some iterations

                # ========== DEBUG MEMORY MONITORING START ==========
                if iteration % 5 == 0:
                    print(f"\n--- Batch {batch_idx}/{config.gradient_accumulation_steps} ---")
                    log_memory_usage(f"Batch {batch_idx} Start")
                # ========== DEBUG MEMORY MONITORING END ==========

                device = 'cuda'
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', None)
                attention_mask = attention_mask.to(device)
                
                # ========== DEBUG MEMORY MONITORING START ==========
                if iteration % 5 == 0:
                    log_memory_usage(f"After Data Transfer")
                # ========== DEBUG MEMORY MONITORING END ==========
                
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
                
                # ========== DEBUG MEMORY MONITORING START ==========
                if iteration % 5 == 0:
                    log_memory_usage(f"After Forward Pass")
                # ========== DEBUG MEMORY MONITORING END ==========
                
                ce_loss = outputs['loss']
                kd_loss = torch.tensor(0.0, device=device)
                
                # Get teacher logits without caching
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask)
                        teacher_logits = teacher_outputs.logits.detach()
                
                # ========== DEBUG MEMORY MONITORING START ==========
                if iteration % 5 == 0:
                    log_memory_usage(f"After Teacher Forward")
                # ========== DEBUG MEMORY MONITORING END ==========
                    
                student_logits = outputs['logits']
                kd_loss = knowledge_distillation_loss(student_logits, teacher_logits)
                loss = 0.7 * ce_loss + 0.3 * kd_loss

                loss = loss / config.gradient_accumulation_steps
                
                # Backward pass to compute gradients
                scaler.scale(loss).backward(retain_graph=False)
                
                # ========== DEBUG MEMORY MONITORING START ==========
                if iteration % 5 == 0:
                    log_memory_usage(f"After Backward Pass")
                # ========== DEBUG MEMORY MONITORING END ==========
                
                # Immediately move gradients to CPU and accumulate, then clear GPU
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            # Use .data to avoid keeping autograd graph
                            grad_data = param.grad.data
                            # Accumulate on CPU
                            cpu_gradients[name].add_(grad_data.cpu())
                            # Clear GPU gradient completely
                            param.grad = None
                    
                    # Force clear CUDA cache after each batch to prevent memory growth
                    torch.cuda.empty_cache()
                
                # ========== DEBUG MEMORY MONITORING START ==========
                if iteration % 5 == 0:
                    log_memory_usage(f"After Gradient Transfer to CPU")
                # ========== DEBUG MEMORY MONITORING END ==========
                
                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_kd_loss += kd_loss.item()
                print(f"total_loss so far: {total_loss:.4f}")
            # End of batch accumulation
            
            # ========== DEBUG MEMORY MONITORING START ==========
            if iteration % 5 == 0:
                log_memory_usage(f"Before Loading Gradients to GPU")
            # ========== DEBUG MEMORY MONITORING END ==========
            
            # Load accumulated gradients from CPU back to GPU for optimizer step
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param.grad = cpu_gradients[name].to('cuda')
            
            # ========== DEBUG MEMORY MONITORING START ==========
            if iteration % 5 == 0:
                log_memory_usage(f"After Loading Gradients to GPU")
            # ========== DEBUG MEMORY MONITORING END ==========
            
            # Optimizer step with scaled gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
            # ========== DEBUG MEMORY MONITORING START ==========
            if iteration % 5 == 0:
                log_memory_usage(f"After Optimizer Step")
            # ========== DEBUG MEMORY MONITORING END ==========
            
            # Clear all gradients after optimizer step
            optimizer.zero_grad(set_to_none=True)  # Use set_to_none=True for better memory cleanup
            
            # ========== DEBUG MEMORY MONITORING START ==========
            if iteration % 5 == 0:
                log_memory_usage(f"After Clearing Gradients")
            # ========== DEBUG MEMORY MONITORING END ==========
            
            # Force clear any remaining GPU cache periodically
            if iteration % 10 == 0:
                torch.cuda.empty_cache()
                # ========== DEBUG MEMORY MONITORING START ==========
                if iteration % 5 == 0:
                    log_memory_usage(f"After CUDA Cache Clear")
                # ========== DEBUG MEMORY MONITORING END ==========
            
            # Additional aggressive memory cleanup every 50 iterations
            if iteration % 50 == 0 and iteration > 0:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all CUDA operations are complete
                gc.collect()  # Force Python garbage collection
                # ========== DEBUG MEMORY MONITORING START ==========
                print(f"\n>>> Aggressive cleanup at iteration {iteration}")
                log_memory_usage(f"After Aggressive Cleanup", detailed=True)
                # ========== DEBUG MEMORY MONITORING END ==========
            
            if iteration < config.warmup_steps:
                scheduler.step()
            
            # Record training loss
            avg_loss = total_loss / config.gradient_accumulation_steps
            training_stats['iteration_losses'].append(avg_loss)

            log_memory_usage(f"End of Iteration {iteration}")
            
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
                print("GPU cache cleared.")
                
                
        # After loop completion
        print("Training loop completed")
        log_memory_usage("Training Loop Completed")
    
    except Exception as e:
        print(f"\n!!! TRAINING ERROR CAUGHT !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        log_memory_usage("At Error")
        import traceback
        traceback.print_exc()
        raise e
    
    # Save training statistics
    log_memory_usage("Before Stats Save")
    try:
        import json
        json.dump(training_stats, open('training_stats.json', 'w'), indent=2)
        log_memory_usage("After Stats Save")
        print("Stats saved")
    except Exception as e:
        print(f"Stats error: {e}")
    
    print("Before return")
    log_memory_usage("Before Return")
    return model
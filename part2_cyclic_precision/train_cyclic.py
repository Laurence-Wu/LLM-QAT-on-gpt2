"""
Training module for Cyclic Precision Training (CPT)
Implements cyclic bit-width switching and static precision training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import gc
from typing import Dict, List, Union, Tuple
import math


class CyclicPrecisionScheduler:
    """
    Scheduler for cyclic precision training.
    Manages bit-width cycling patterns and transitions.
    """
    
    def __init__(self, config):
        """
        Initialize the cyclic precision scheduler.
        
        Args:
            config: CyclicPrecisionConfig object
        """
        self.config = config
        self.cycle_length = config.cycle_length
        self.bit_width_pattern = config.bit_width_pattern
        self.current_iteration = 0
        self.current_cycle = 0
        
        # Calculate steps per bit width in pattern
        self.steps_per_bit = self.cycle_length // len(self.bit_width_pattern)
        
    def get_current_bit_width(self, iteration: int) -> int:
        """
        Get the current bit width for the given iteration.
        
        Args:
            iteration: Current training iteration
        
        Returns:
            Bit width for the current iteration
        """
        # Calculate position in cycle
        position_in_cycle = iteration % self.cycle_length
        
        # Determine which bit width to use
        pattern_index = position_in_cycle // self.steps_per_bit
        pattern_index = min(pattern_index, len(self.bit_width_pattern) - 1)
        
        # Get bit width
        bit_width = self.bit_width_pattern[pattern_index]
        
        # Apply progressive cycling if enabled
        if self.config.progressive_cycles and self.current_cycle > 0:
            # Reduce bit width over cycles
            reduction = self.current_cycle * self.config.progression_rate
            bit_width = max(2, int(bit_width - reduction))
        
        return bit_width
    
    def get_layer_bit_widths(self, iteration: int, n_layers: int) -> List[int]:
        """
        Get bit widths for each layer (for layer-wise cycling).
        
        Args:
            iteration: Current training iteration
            n_layers: Number of layers in the model
        
        Returns:
            List of bit widths for each layer
        """
        if not self.config.layer_wise_cycling:
            # All layers use the same bit width
            bit_width = self.get_current_bit_width(iteration)
            return [bit_width] * n_layers
        
        # Layer-wise cycling with offset
        layer_bit_widths = []
        for layer_idx in range(n_layers):
            offset_iteration = iteration + (layer_idx * self.config.layer_cycle_offset)
            bit_width = self.get_current_bit_width(offset_iteration)
            layer_bit_widths.append(bit_width)
        
        return layer_bit_widths
    
    def get_learning_rate_scale(self, bit_width: int) -> float:
        """
        Get learning rate scale factor for current bit width.
        
        Args:
            bit_width: Current bit width
        
        Returns:
            Scale factor for learning rate
        """
        if not self.config.adjust_lr_with_bits:
            return 1.0
        
        return self.config.lr_scale_factors.get(bit_width, 1.0)
    
    def update(self, iteration: int):
        """
        Update scheduler state.
        
        Args:
            iteration: Current iteration
        """
        self.current_iteration = iteration
        self.current_cycle = iteration // self.cycle_length


def create_layer_config_from_bit_width(bit_width: Union[int, List[int]], 
                                       n_layers: int) -> List[Dict]:
    """
    Create layer configuration from bit width specification.
    
    Args:
        bit_width: Single bit width or list of bit widths
        n_layers: Number of layers
    
    Returns:
        List of layer configurations
    """
    if isinstance(bit_width, int):
        # Uniform bit width
        return [{'attn_bits': bit_width, 'mlp_bits': bit_width} 
                for _ in range(n_layers)]
    
    elif isinstance(bit_width, list):
        # Alternating bit widths
        config = []
        for i in range(n_layers):
            bits = bit_width[i % len(bit_width)]
            config.append({'attn_bits': bits, 'mlp_bits': bits})
        return config
    
    elif bit_width == 'progressive':
        # Progressive: higher precision at boundaries
        config = []
        for i in range(n_layers):
            if i < 2 or i >= n_layers - 2:
                bits = 8  # Higher precision at boundaries
            else:
                bits = 4  # Lower precision in middle
            config.append({'attn_bits': bits, 'mlp_bits': bits})
        return config
    
    else:
        raise ValueError(f"Unknown bit width configuration: {bit_width}")


def train_with_cpt(model, train_loader, val_loader, training_config, 
                   cyclic_config, n_layers: int = 12) -> Tuple[nn.Module, Dict]:
    """
    Train model with Cyclic Precision Training.
    
    Args:
        model: SwitchableQuantizedGPT2 model
        train_loader: Training data loader
        val_loader: Validation data loader
        training_config: Training configuration
        cyclic_config: Cyclic precision configuration
        n_layers: Number of layers in model
    
    Returns:
        Tuple of (trained model, training statistics)
    """
    print("\nStarting Cyclic Precision Training...")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=training_config.adam_betas,
        eps=training_config.adam_epsilon,
        weight_decay=training_config.weight_decay
    )
    
    # Setup scheduler
    cyclic_scheduler = CyclicPrecisionScheduler(cyclic_config)
    
    # Setup mixed precision
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() and training_config.use_amp else None
    
    # Training statistics
    stats = {
        'iteration_losses': [],
        'validation_losses': [],
        'bit_width_history': [],
        'cycle_metrics': [],
        'best_val_loss': float('inf'),
        'best_iteration': 0,
        'final_loss': 0
    }
    
    # Main training loop
    for iteration in tqdm(range(training_config.num_cpt_iterations), desc="CPT Training"):
        model.train()
        
        # Get current bit widths
        if cyclic_config.layer_wise_cycling:
            layer_bit_widths = cyclic_scheduler.get_layer_bit_widths(iteration, n_layers)
            layer_config = []
            for bits in layer_bit_widths:
                layer_config.append({'attn_bits': bits, 'mlp_bits': bits})
        else:
            current_bit_width = cyclic_scheduler.get_current_bit_width(iteration)
            layer_config = create_layer_config_from_bit_width(current_bit_width, n_layers)
            stats['bit_width_history'].append(current_bit_width)
        
        # Set model precision
        model.set_layer_precision(layer_config)
        
        # Adjust learning rate based on bit width
        if not cyclic_config.layer_wise_cycling:
            lr_scale = cyclic_scheduler.get_learning_rate_scale(current_bit_width)
            for param_group in optimizer.param_groups:
                param_group['lr'] = training_config.learning_rate * lr_scale
        
        # Training step
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= training_config.gradient_accumulation_steps:
                break
            
            device = next(model.parameters()).device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward pass
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
                    loss = outputs['loss'] / training_config.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
                loss = outputs['loss'] / training_config.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item()
        
        # Optimizer step
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
            optimizer.step()
        
        optimizer.zero_grad()
        
        # Update scheduler
        cyclic_scheduler.update(iteration)
        
        # Record loss
        avg_loss = total_loss
        stats['iteration_losses'].append(avg_loss)
        stats['final_loss'] = avg_loss
        
        # Logging
        if iteration % training_config.log_interval == 0:
            if not cyclic_config.layer_wise_cycling:
                print(f"\nIteration {iteration}/{training_config.num_cpt_iterations}")
                print(f"  Cycle: {cyclic_scheduler.current_cycle}")
                print(f"  Bit width: {current_bit_width}")
                print(f"  Loss: {avg_loss:.4f}")
                print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Track cycle metrics
        if cyclic_config.track_cycle_metrics:
            if iteration % cyclic_scheduler.cycle_length == 0 and iteration > 0:
                cycle_losses = stats['iteration_losses'][-cyclic_scheduler.cycle_length:]
                cycle_metric = {
                    'cycle': cyclic_scheduler.current_cycle - 1,
                    'avg_loss': np.mean(cycle_losses),
                    'std_loss': np.std(cycle_losses),
                    'min_loss': np.min(cycle_losses),
                    'max_loss': np.max(cycle_losses)
                }
                stats['cycle_metrics'].append(cycle_metric)
                
                if training_config.verbose:
                    print(f"\nCycle {cycle_metric['cycle']} Summary:")
                    print(f"  Avg Loss: {cycle_metric['avg_loss']:.4f}")
                    print(f"  Std Loss: {cycle_metric['std_loss']:.4f}")
        
        # Validation
        if iteration % training_config.eval_interval == 0 and iteration > 0:
            model.eval()
            val_loss = 0
            val_steps = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= 10:
                        break
                    
                    device = next(model.parameters()).device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch.get('attention_mask')
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    
                    outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
                    val_loss += outputs['loss'].item()
                    val_steps += 1
            
            avg_val_loss = val_loss / max(val_steps, 1)
            stats['validation_losses'].append(avg_val_loss)
            
            print(f"  Validation Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < stats['best_val_loss']:
                stats['best_val_loss'] = avg_val_loss
                stats['best_iteration'] = iteration
                print(f"  New best validation loss!")
        
        # Clear cache periodically
        if iteration % training_config.empty_cache_interval == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    return model, stats


def train_with_static_precision(model, train_loader, val_loader, training_config,
                               bit_config: Union[int, List[int], str],
                               n_layers: int = 12) -> Tuple[nn.Module, Dict]:
    """
    Train model with static precision configuration.
    
    Args:
        model: SwitchableQuantizedGPT2 model
        train_loader: Training data loader
        val_loader: Validation data loader
        training_config: Training configuration
        bit_config: Static bit configuration
        n_layers: Number of layers
    
    Returns:
        Tuple of (trained model, training statistics)
    """
    print(f"\nTraining with static precision: {bit_config}")
    
    # Create layer configuration
    layer_config = create_layer_config_from_bit_width(bit_config, n_layers)
    model.set_layer_precision(layer_config)
    
    # Setup optimizer with reduced learning rate for fine-tuning
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate * 0.5,  # Reduced LR for fine-tuning
        betas=training_config.adam_betas,
        eps=training_config.adam_epsilon,
        weight_decay=training_config.weight_decay
    )
    
    # Setup mixed precision
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() and training_config.use_amp else None
    
    # Training statistics
    stats = {
        'iteration_losses': [],
        'validation_losses': [],
        'best_val_loss': float('inf'),
        'best_iteration': 0,
        'final_loss': 0,
        'bit_config': str(bit_config)
    }
    
    # Training loop
    for iteration in tqdm(range(training_config.num_static_iterations), 
                         desc=f"Static {bit_config}"):
        model.train()
        
        # Training step
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= training_config.gradient_accumulation_steps:
                break
            
            device = next(model.parameters()).device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward pass
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
                    loss = outputs['loss'] / training_config.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
                loss = outputs['loss'] / training_config.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item()
        
        # Optimizer step
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
            optimizer.step()
        
        optimizer.zero_grad()
        
        # Record loss
        avg_loss = total_loss
        stats['iteration_losses'].append(avg_loss)
        stats['final_loss'] = avg_loss
        
        # Validation at the end
        if iteration == training_config.num_static_iterations - 1:
            model.eval()
            val_loss = 0
            val_steps = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= 10:
                        break
                    
                    device = next(model.parameters()).device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch.get('attention_mask')
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    
                    outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
                    val_loss += outputs['loss'].item()
                    val_steps += 1
            
            avg_val_loss = val_loss / max(val_steps, 1)
            stats['validation_losses'].append(avg_val_loss)
            stats['best_val_loss'] = avg_val_loss
            
            print(f"\nFinal validation loss: {avg_val_loss:.4f}")
    
    return model, stats
"""
Evaluation module for Cyclic Precision Training
Evaluates CPT models and compares with static configurations.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Any
import gc


def evaluate_cyclic_training(model, data_loader, cyclic_config, max_batches: int = 50) -> Dict:
    """
    Evaluate a model trained with cyclic precision.
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        cyclic_config: Cyclic precision configuration used in training
        max_batches: Maximum number of batches to evaluate
    
    Returns:
        Dictionary containing evaluation metrics
    """
    print("\nEvaluating cyclic precision trained model...")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Test with different bit widths from the cycle
    bit_widths_to_test = list(set(cyclic_config.bit_width_pattern))
    results_per_bit = {}
    
    for bit_width in bit_widths_to_test:
        print(f"  Testing with {bit_width}-bit precision...")
        
        # Set uniform bit width
        n_layers = len(model.h)
        layer_config = [{'attn_bits': bit_width, 'mlp_bits': bit_width} 
                       for _ in range(n_layers)]
        model.set_layer_precision(layer_config)
        
        # Evaluate
        total_loss = 0
        total_tokens = 0
        total_correct = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"{bit_width}-bit", leave=False)):
                if batch_idx >= max_batches:
                    break
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_loss += loss.item()
                
                # Calculate accuracy
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                if attention_mask is not None:
                    shift_mask = attention_mask[..., 1:].contiguous()
                    valid_tokens = shift_mask.sum().item()
                else:
                    valid_tokens = shift_labels.numel()
                
                predictions = torch.argmax(shift_logits, dim=-1)
                correct = (predictions == shift_labels)
                
                if attention_mask is not None:
                    correct = correct * shift_mask
                
                total_correct += correct.sum().item()
                total_tokens += valid_tokens
                batch_count += 1
        
        # Calculate metrics
        avg_loss = total_loss / max(batch_count, 1)
        perplexity = np.exp(avg_loss)
        accuracy = total_correct / max(total_tokens, 1)
        
        results_per_bit[bit_width] = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'accuracy': accuracy,
            'total_tokens': total_tokens
        }
        
        print(f"    Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}, Accuracy: {accuracy:.4f}")
    
    # Calculate average metrics across all bit widths
    avg_metrics = {
        'avg_loss': np.mean([r['loss'] for r in results_per_bit.values()]),
        'avg_perplexity': np.mean([r['perplexity'] for r in results_per_bit.values()]),
        'avg_accuracy': np.mean([r['accuracy'] for r in results_per_bit.values()]),
        'std_loss': np.std([r['loss'] for r in results_per_bit.values()]),
        'std_perplexity': np.std([r['perplexity'] for r in results_per_bit.values()]),
        'std_accuracy': np.std([r['accuracy'] for r in results_per_bit.values()])
    }
    
    # Analyze cycle stability
    stability_score = 1.0 / (1.0 + avg_metrics['std_loss'])  # Higher score = more stable
    
    return {
        'results_per_bit': results_per_bit,
        'average_metrics': avg_metrics,
        'stability_score': stability_score,
        'bit_widths_tested': bit_widths_to_test
    }


def compare_bit_configurations(cpt_model, static_models: Dict, data_loader,
                              n_layers: int = 12, max_batches: int = 50) -> Dict:
    """
    Compare CPT model with static bit-width configurations.
    
    Args:
        cpt_model: Model trained with cyclic precision
        static_models: Dictionary of models trained with static configurations
        data_loader: Data loader for evaluation
        n_layers: Number of layers in the models
        max_batches: Maximum batches to evaluate
    
    Returns:
        Dictionary containing comparison results
    """
    print("\nComparing bit-width configurations...")
    
    device = next(cpt_model.parameters()).device
    comparison_results = {}
    
    # Evaluate CPT model with its best configuration (8-bit as default)
    print("\nEvaluating CPT model...")
    cpt_model.eval()
    layer_config = [{'attn_bits': 8, 'mlp_bits': 8} for _ in range(n_layers)]
    cpt_model.set_layer_precision(layer_config)
    
    cpt_metrics = evaluate_single_model(cpt_model, data_loader, max_batches)
    comparison_results['cpt'] = cpt_metrics
    print(f"  CPT - Loss: {cpt_metrics['loss']:.4f}, Perplexity: {cpt_metrics['perplexity']:.2f}")
    
    # Evaluate each static configuration
    for config_name, model_data in static_models.items():
        print(f"\nEvaluating {config_name} model...")
        static_model = model_data['model']
        static_model.eval()
        
        static_metrics = evaluate_single_model(static_model, data_loader, max_batches)
        comparison_results[config_name] = static_metrics
        print(f"  {config_name} - Loss: {static_metrics['loss']:.4f}, "
              f"Perplexity: {static_metrics['perplexity']:.2f}")
    
    # Find best configuration
    best_config = None
    best_perplexity = float('inf')
    best_accuracy = 0
    
    for config_name, metrics in comparison_results.items():
        if metrics['perplexity'] < best_perplexity:
            best_perplexity = metrics['perplexity']
            best_config = config_name
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
    
    # Calculate efficiency metrics
    efficiency_results = {}
    for config_name, metrics in comparison_results.items():
        # Estimate effective bits based on configuration name
        if '2bit' in config_name:
            effective_bits = 2
        elif '4bit' in config_name:
            effective_bits = 4
        elif '8bit' in config_name:
            effective_bits = 8
        elif 'mixed_2_4' in config_name:
            effective_bits = 3  # Average of 2 and 4
        elif 'mixed_4_8' in config_name:
            effective_bits = 6  # Average of 4 and 8
        else:
            effective_bits = 8  # Default for CPT
        
        efficiency = metrics['accuracy'] / effective_bits
        efficiency_results[config_name] = {
            'efficiency': efficiency,
            'effective_bits': effective_bits,
            'accuracy': metrics['accuracy']
        }
    
    # Generate insights
    insights = []
    
    # Compare CPT vs best static
    if 'cpt' in comparison_results and best_config != 'cpt':
        cpt_perf = comparison_results['cpt']['perplexity']
        best_static_perf = comparison_results[best_config]['perplexity']
        
        if cpt_perf < best_static_perf:
            improvement = ((best_static_perf - cpt_perf) / best_static_perf) * 100
            insights.append(f"CPT outperforms best static config by {improvement:.1f}%")
        else:
            degradation = ((cpt_perf - best_static_perf) / best_static_perf) * 100
            insights.append(f"Static {best_config} outperforms CPT by {degradation:.1f}%")
    
    # Check if mixed precision helps
    if '4bit' in comparison_results and 'mixed_4_8' in comparison_results:
        uniform_perf = comparison_results['4bit']['perplexity']
        mixed_perf = comparison_results['mixed_4_8']['perplexity']
        
        if mixed_perf < uniform_perf:
            improvement = ((uniform_perf - mixed_perf) / uniform_perf) * 100
            insights.append(f"Mixed 4-8 bit improves over uniform 4-bit by {improvement:.1f}%")
    
    # Analyze low-bit performance
    if '2bit' in comparison_results:
        low_bit_acc = comparison_results['2bit']['accuracy']
        if low_bit_acc > 0.3:  # Threshold for "good" 2-bit performance
            insights.append(f"2-bit quantization achieves {low_bit_acc:.2%} accuracy, showing good compression potential")
    
    # Summary
    summary = {
        'best_configuration': best_config,
        'best_perplexity': best_perplexity,
        'best_accuracy': best_accuracy,
        'efficiency_rankings': sorted(efficiency_results.items(), 
                                     key=lambda x: x[1]['efficiency'], 
                                     reverse=True),
        'insights': insights
    }
    
    comparison_results['summary'] = summary
    
    return comparison_results


def evaluate_single_model(model, data_loader, max_batches: int = 50) -> Dict:
    """
    Evaluate a single model configuration.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        max_batches: Maximum batches to evaluate
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating", leave=False)):
            if batch_idx >= max_batches:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
            loss = outputs['loss']
            logits = outputs['logits']
            
            total_loss += loss.item()
            
            # Calculate accuracy
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            if attention_mask is not None:
                shift_mask = attention_mask[..., 1:].contiguous()
                valid_tokens = shift_mask.sum().item()
            else:
                valid_tokens = shift_labels.numel()
            
            predictions = torch.argmax(shift_logits, dim=-1)
            correct = (predictions == shift_labels)
            
            if attention_mask is not None:
                correct = correct * shift_mask
            
            total_correct += correct.sum().item()
            total_tokens += valid_tokens
            batch_count += 1
    
    # Calculate metrics
    avg_loss = total_loss / max(batch_count, 1)
    perplexity = np.exp(avg_loss)
    accuracy = total_correct / max(total_tokens, 1)
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy,
        'total_tokens': total_tokens,
        'batch_count': batch_count
    }


def analyze_cycle_progression(training_stats: Dict) -> Dict:
    """
    Analyze how model performance progresses through cycles.
    
    Args:
        training_stats: Training statistics from CPT
    
    Returns:
        Dictionary containing cycle analysis
    """
    if 'cycle_metrics' not in training_stats or not training_stats['cycle_metrics']:
        return {'error': 'No cycle metrics available'}
    
    cycle_metrics = training_stats['cycle_metrics']
    
    # Calculate cycle trends
    cycle_losses = [m['avg_loss'] for m in cycle_metrics]
    cycle_stds = [m['std_loss'] for m in cycle_metrics]
    
    # Check for convergence
    if len(cycle_losses) > 1:
        loss_trend = np.polyfit(range(len(cycle_losses)), cycle_losses, 1)[0]
        convergence_rate = -loss_trend  # Negative slope means improvement
    else:
        convergence_rate = 0
    
    # Check for stability
    avg_std = np.mean(cycle_stds)
    stability_improved = cycle_stds[-1] < cycle_stds[0] if len(cycle_stds) > 1 else False
    
    analysis = {
        'num_cycles': len(cycle_metrics),
        'initial_loss': cycle_losses[0] if cycle_losses else None,
        'final_loss': cycle_losses[-1] if cycle_losses else None,
        'improvement': cycle_losses[0] - cycle_losses[-1] if len(cycle_losses) > 1 else 0,
        'convergence_rate': convergence_rate,
        'average_stability': avg_std,
        'stability_improved': stability_improved,
        'best_cycle': min(enumerate(cycle_losses), key=lambda x: x[1])[0] if cycle_losses else None
    }
    
    return analysis
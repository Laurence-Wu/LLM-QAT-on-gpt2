"""
Visualization Module for Model Analysis
Creates plots and charts for analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from typing import Dict, Any


def create_analysis_plots(analysis_results: Dict[str, Any], output_dir: str):
    """
    Create comprehensive visualization plots for analysis results.
    
    Args:
        analysis_results: Dictionary containing all analysis results
        output_dir: Directory to save plots
    """
    print("Creating visualization plots...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Performance plots
    if 'performance' in analysis_results:
        create_performance_plots(analysis_results['performance'], plots_dir)
    
    # Bit-width analysis plots
    if 'bit_width' in analysis_results:
        create_bit_width_plots(analysis_results['bit_width'], plots_dir)
    
    # Robustness plots
    if 'robustness' in analysis_results:
        create_robustness_plots(analysis_results['robustness'], plots_dir)
    
    # Combined summary plot
    create_summary_plot(analysis_results, plots_dir)
    
    print(f"Plots saved to: {plots_dir}")


def create_performance_plots(performance_results: Dict, plots_dir: str):
    """
    Create performance-related plots.
    
    Args:
        performance_results: Performance analysis results
        plots_dir: Directory to save plots
    """
    # Speed metrics plot
    if 'speed_metrics' in performance_results:
        speed_metrics = performance_results['speed_metrics']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Latency distribution
        if 'mean_latency' in speed_metrics:
            latencies = [
                speed_metrics.get('mean_latency', 0),
                speed_metrics.get('p50_latency', 0),
                speed_metrics.get('p95_latency', 0),
                speed_metrics.get('p99_latency', 0)
            ]
            labels = ['Mean', 'P50', 'P95', 'P99']
            
            ax1.bar(labels, latencies, color='skyblue')
            ax1.set_title('Latency Distribution')
            ax1.set_ylabel('Latency (seconds)')
            
        # Throughput metrics
        if 'tokens_per_second' in speed_metrics:
            metrics = ['Tokens/sec', 'Batches/sec']
            values = [
                speed_metrics.get('tokens_per_second', 0),
                speed_metrics.get('batches_per_second', 0)
            ]
            
            ax2.bar(metrics, values, color='lightgreen')
            ax2.set_title('Throughput Metrics')
            ax2.set_ylabel('Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'performance_speed.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Memory usage plot
    if 'memory_metrics' in performance_results:
        memory_metrics = performance_results['memory_metrics']
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        memory_types = []
        memory_values = []
        
        if 'model_size_mb' in memory_metrics:
            memory_types.append('Model Size')
            memory_values.append(memory_metrics['model_size_mb'])
        
        if 'peak_memory_mb' in memory_metrics:
            memory_types.append('Peak Memory')
            memory_values.append(memory_metrics['peak_memory_mb'])
        
        if 'avg_memory_increase_mb' in memory_metrics:
            memory_types.append('Avg Increase')
            memory_values.append(memory_metrics['avg_memory_increase_mb'])
        
        if memory_types:
            ax.bar(memory_types, memory_values, color='coral')
            ax.set_title('Memory Usage Analysis')
            ax.set_ylabel('Memory (MB)')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'performance_memory.png'), dpi=300, bbox_inches='tight')
        plt.close()


def create_bit_width_plots(bit_width_results: Dict, plots_dir: str):
    """
    Create bit-width analysis plots.
    
    Args:
        bit_width_results: Bit-width analysis results
        plots_dir: Directory to save plots
    """
    # Configuration comparison plot
    if 'configuration_results' in bit_width_results:
        config_results = bit_width_results['configuration_results']
        
        configs = list(config_results.keys())
        accuracies = [config_results[c]['metrics']['accuracy'] for c in configs]
        effective_bits = [config_results[c]['effective_bits'] for c in configs]
        compression_ratios = [config_results[c]['compression_ratio'] for c in configs]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Accuracy comparison
        ax1.bar(range(len(configs)), accuracies, color='lightblue')
        ax1.set_title('Accuracy by Configuration')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(range(len(configs)))
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        
        # Effective bits
        ax2.bar(range(len(configs)), effective_bits, color='lightgreen')
        ax2.set_title('Effective Bits by Configuration')
        ax2.set_ylabel('Effective Bits')
        ax2.set_xticks(range(len(configs)))
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        
        # Compression ratio
        ax3.bar(range(len(configs)), compression_ratios, color='lightyellow')
        ax3.set_title('Compression Ratio by Configuration')
        ax3.set_ylabel('Compression Ratio')
        ax3.set_xticks(range(len(configs)))
        ax3.set_xticklabels(configs, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'bit_width_configurations.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Accuracy vs Effective Bits scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        scatter = ax.scatter(effective_bits, accuracies, s=100, alpha=0.7, 
                           c=compression_ratios, cmap='viridis')
        
        # Add labels for each point
        for i, config in enumerate(configs):
            ax.annotate(config, (effective_bits[i], accuracies[i]), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.8)
        
        ax.set_xlabel('Effective Bits')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Effective Bits\n(Color represents compression ratio)')
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Compression Ratio')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'accuracy_vs_bits.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Layer sensitivity plot
    if 'layer_sensitivity' in bit_width_results:
        layer_sens = bit_width_results['layer_sensitivity']['layer_sensitivity']
        
        # Extract layer sensitivity data
        layers = []
        sensitivities_4bit = []
        sensitivities_2bit = []
        
        for layer_name, layer_data in layer_sens.items():
            layers.append(layer_name)
            sensitivities_4bit.append(layer_data.get('4bit', {}).get('sensitivity', 0))
            if '2bit' in layer_data:
                sensitivities_2bit.append(layer_data['2bit']['sensitivity'])
            else:
                sensitivities_2bit.append(0)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        x = np.arange(len(layers))
        width = 0.35
        
        ax.bar(x - width/2, sensitivities_4bit, width, label='4-bit', alpha=0.8)
        ax.bar(x + width/2, sensitivities_2bit, width, label='2-bit', alpha=0.8)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Sensitivity (Loss Increase)')
        ax.set_title('Layer Sensitivity to Quantization')
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'layer_sensitivity.png'), dpi=300, bbox_inches='tight')
        plt.close()


def create_robustness_plots(robustness_results: Dict, plots_dir: str):
    """
    Create robustness analysis plots.
    
    Args:
        robustness_results: Robustness analysis results
        plots_dir: Directory to save plots
    """
    # Adversarial robustness plot
    if 'adversarial' in robustness_results:
        adv_results = robustness_results['adversarial']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Attack results
        attack_names = []
        robust_accuracies = []
        
        for key, value in adv_results.items():
            if key not in ['clean_accuracy', 'worst_case_accuracy', 'robustness_gap', 'robustness_ratio']:
                attack_names.append(key.replace('_', ' ').title())
                robust_accuracies.append(value)
        
        if attack_names:
            clean_acc = adv_results.get('clean_accuracy', 0)
            
            ax1.bar(attack_names, robust_accuracies, color='lightcoral', alpha=0.7)
            ax1.axhline(y=clean_acc, color='green', linestyle='--', label=f'Clean Accuracy ({clean_acc:.3f})')
            ax1.set_title('Adversarial Attack Results')
            ax1.set_ylabel('Robust Accuracy')
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend()
        
        # Robustness metrics
        metrics = ['Clean Accuracy', 'Worst Case', 'Robustness Gap']
        values = [
            adv_results.get('clean_accuracy', 0),
            adv_results.get('worst_case_accuracy', 0),
            adv_results.get('robustness_gap', 0)
        ]
        colors = ['green', 'orange', 'red']
        
        ax2.bar(metrics, values, color=colors, alpha=0.7)
        ax2.set_title('Robustness Summary Metrics')
        ax2.set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'adversarial_robustness.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Noise robustness plot
    if 'quantization_noise' in robustness_results:
        noise_results = robustness_results['quantization_noise']
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        noise_levels = []
        accuracies = []
        
        baseline_acc = noise_results.get('baseline', 0)
        
        for key, value in noise_results.items():
            if key.startswith('noise_'):
                noise_level = key.replace('noise_', '')
                noise_levels.append(float(noise_level))
                accuracies.append(value)
        
        if noise_levels:
            # Sort by noise level
            sorted_data = sorted(zip(noise_levels, accuracies))
            noise_levels, accuracies = zip(*sorted_data)
            
            ax.plot([0] + list(noise_levels), [baseline_acc] + list(accuracies), 
                   'o-', linewidth=2, markersize=8)
            ax.axhline(y=baseline_acc, color='green', linestyle='--', alpha=0.7, 
                      label=f'Baseline ({baseline_acc:.3f})')
            
            ax.set_xlabel('Noise Level')
            ax.set_ylabel('Accuracy')
            ax.set_title('Robustness to Quantization Noise')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'noise_robustness.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Input perturbation robustness
    if 'input_perturbations' in robustness_results:
        pert_results = robustness_results['input_perturbations']
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        pert_names = []
        pert_accuracies = []
        
        baseline_acc = pert_results.get('baseline', 0)
        
        for key, value in pert_results.items():
            if key not in ['baseline', 'worst_perturbation_accuracy', 'perturbation_robustness_gap']:
                pert_names.append(key.replace('_', ' ').title())
                pert_accuracies.append(value)
        
        if pert_names:
            ax.bar(pert_names, pert_accuracies, color='lightsalmon', alpha=0.7)
            ax.axhline(y=baseline_acc, color='green', linestyle='--', 
                      label=f'Baseline ({baseline_acc:.3f})')
            ax.set_title('Robustness to Input Perturbations')
            ax.set_ylabel('Accuracy')
            ax.tick_params(axis='x', rotation=45)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'perturbation_robustness.png'), dpi=300, bbox_inches='tight')
        plt.close()


def create_summary_plot(analysis_results: Dict[str, Any], plots_dir: str):
    """
    Create a comprehensive summary plot.
    
    Args:
        analysis_results: All analysis results
        plots_dir: Directory to save plots
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Performance summary (top row)
    if 'performance' in analysis_results:
        perf = analysis_results['performance']
        
        # Basic metrics
        ax1 = fig.add_subplot(gs[0, 0])
        if 'basic_metrics' in perf:
            basic = perf['basic_metrics']
            metrics = ['Loss', 'Perplexity', 'Accuracy', 'Top-5 Acc']
            values = [
                basic.get('loss', 0),
                basic.get('perplexity', 0) / 10,  # Scale down for visibility
                basic.get('accuracy', 0),
                basic.get('top5_accuracy', 0)
            ]
            ax1.bar(metrics, values, color='lightblue')
            ax1.set_title('Performance Metrics')
            ax1.tick_params(axis='x', rotation=45)
        
        # Speed metrics
        ax2 = fig.add_subplot(gs[0, 1])
        if 'speed_metrics' in perf:
            speed = perf['speed_metrics']
            ax2.bar(['Tokens/sec'], [speed.get('tokens_per_second', 0)], color='lightgreen')
            ax2.set_title('Speed (Tokens/sec)')
        
        # Memory usage
        ax3 = fig.add_subplot(gs[0, 2])
        if 'memory_metrics' in perf:
            memory = perf['memory_metrics']
            ax3.bar(['Model Size (MB)'], [memory.get('model_size_mb', 0)], color='coral')
            ax3.set_title('Memory Usage')
    
    # Bit-width analysis (middle row)
    if 'bit_width' in analysis_results:
        bw = analysis_results['bit_width']
        
        # Configuration comparison
        ax4 = fig.add_subplot(gs[1, :2])
        if 'configuration_results' in bw:
            config_results = bw['configuration_results']
            configs = list(config_results.keys())[:8]  # Limit to first 8
            accuracies = [config_results[c]['metrics']['accuracy'] for c in configs]
            effective_bits = [config_results[c]['effective_bits'] for c in configs]
            
            x = np.arange(len(configs))
            width = 0.35
            
            ax4_twin = ax4.twinx()
            bars1 = ax4.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.7, color='skyblue')
            bars2 = ax4_twin.bar(x + width/2, effective_bits, width, label='Effective Bits', alpha=0.7, color='lightcoral')
            
            ax4.set_xlabel('Configuration')
            ax4.set_ylabel('Accuracy', color='blue')
            ax4_twin.set_ylabel('Effective Bits', color='red')
            ax4.set_title('Bit-width Configuration Analysis')
            ax4.set_xticks(x)
            ax4.set_xticklabels([c[:10] for c in configs], rotation=45, ha='right')
            
            ax4.legend(loc='upper left')
            ax4_twin.legend(loc='upper right')
        
        # Optimal configuration info
        ax5 = fig.add_subplot(gs[1, 2])
        if 'optimal_configuration' in bw:
            optimal = bw['optimal_configuration']
            ax5.text(0.1, 0.7, f"Best: {optimal.get('name', 'N/A')[:15]}", fontsize=10, weight='bold')
            ax5.text(0.1, 0.5, f"Score: {optimal.get('score', 0):.3f}", fontsize=9)
            ax5.text(0.1, 0.3, f"Bits: {optimal.get('effective_bits', 0):.1f}", fontsize=9)
            ax5.text(0.1, 0.1, f"Compression: {optimal.get('compression_ratio', 0):.1f}x", fontsize=9)
            ax5.set_title('Optimal Configuration')
            ax5.set_xlim(0, 1)
            ax5.set_ylim(0, 1)
            ax5.axis('off')
    
    # Robustness analysis (bottom row)
    if 'robustness' in analysis_results:
        rob = analysis_results['robustness']
        
        # Adversarial robustness
        ax6 = fig.add_subplot(gs[2, 0])
        if 'adversarial' in rob:
            adv = rob['adversarial']
            clean_acc = adv.get('clean_accuracy', 0)
            robust_acc = adv.get('worst_case_accuracy', 0)
            
            ax6.bar(['Clean', 'Robust'], [clean_acc, robust_acc], 
                   color=['green', 'orange'], alpha=0.7)
            ax6.set_title('Adversarial Robustness')
            ax6.set_ylabel('Accuracy')
        
        # Noise robustness
        ax7 = fig.add_subplot(gs[2, 1])
        if 'quantization_noise' in rob:
            noise = rob['quantization_noise']
            baseline = noise.get('baseline', 0)
            worst_noise = noise.get('worst_noise_accuracy', 0)
            
            ax7.bar(['Baseline', 'With Noise'], [baseline, worst_noise], 
                   color=['blue', 'red'], alpha=0.7)
            ax7.set_title('Noise Robustness')
            ax7.set_ylabel('Accuracy')
        
        # Perturbation robustness
        ax8 = fig.add_subplot(gs[2, 2])
        if 'input_perturbations' in rob:
            pert = rob['input_perturbations']
            baseline = pert.get('baseline', 0)
            worst_pert = pert.get('worst_perturbation_accuracy', 0)
            
            ax8.bar(['Baseline', 'Perturbed'], [baseline, worst_pert], 
                   color=['purple', 'yellow'], alpha=0.7)
            ax8.set_title('Input Robustness')
            ax8.set_ylabel('Accuracy')
    
    plt.suptitle('Model Analysis Summary Dashboard', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(plots_dir, 'analysis_summary_dashboard.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
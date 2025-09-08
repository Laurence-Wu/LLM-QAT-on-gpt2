#!/usr/bin/env python3
"""
Part 3: Post-Training Analysis
Comprehensive analysis tool for trained GPT-2 models with quantization.
Analyzes model performance, bit-width efficiency, and generates detailed reports.
"""

import os
import sys
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from tqdm import tqdm

# Add shared folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

# Import shared components
from shared.models import SwitchableQuantizedGPT2
from shared.dataset import create_dataloaders

# Import analysis modules
from performance_analyzer import PerformanceAnalyzer
from bit_width_analyzer import BitWidthAnalyzer
from robustness_analyzer import RobustnessAnalyzer
from visualization import create_analysis_plots


class ModelAnalyzer:
    """
    Main class for comprehensive model analysis.
    Coordinates different analysis components and generates reports.
    """
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Initialize the model analyzer.
        
        Args:
            model_path: Path to the saved model checkpoint
            config_path: Optional path to configuration file
        """
        self.model_path = model_path
        self.config_path = config_path
        
        # Load model and configuration
        self.model, self.config = self.load_model()
        
        # Initialize analysis components
        self.performance_analyzer = PerformanceAnalyzer(self.model)
        self.bit_width_analyzer = BitWidthAnalyzer(self.model)
        self.robustness_analyzer = RobustnessAnalyzer(self.model)
        
        # Results storage
        self.analysis_results = {}
        
    def load_model(self) -> tuple:
        """
        Load the trained model and its configuration.
        
        Returns:
            Tuple of (model, config)
        """
        print(f"Loading model from: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Extract configuration
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
        elif self.config_path:
            with open(self.config_path, 'r') as f:
                model_config = json.load(f)
        else:
            # Default configuration
            model_config = {
                'n_layer': 6,
                'n_embd': 768,
                'n_head': 12,
                'vocab_size': 50257,
                'n_positions': 256,
                'bit_widths': [2, 4, 8]
            }
        
        # Create model
        from transformers import GPT2Config
        gpt2_config = GPT2Config(
            vocab_size=model_config.get('vocab_size', 50257),
            n_positions=model_config.get('n_positions', 256),
            n_embd=model_config.get('n_embd', 768),
            n_layer=model_config.get('n_layer', 6),
            n_head=model_config.get('n_head', 12),
            bit_widths=model_config.get('bit_widths', [2, 4, 8])
        )
        
        model = SwitchableQuantizedGPT2(gpt2_config)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Move to appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
        
        return model, model_config
    
    def analyze_performance(self, data_loader) -> Dict:
        """
        Analyze model performance metrics.
        
        Args:
            data_loader: DataLoader for evaluation
        
        Returns:
            Dictionary of performance metrics
        """
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Basic metrics
        print("\n1. Computing basic metrics...")
        basic_metrics = self.performance_analyzer.compute_basic_metrics(data_loader)
        
        # Speed benchmarks
        print("\n2. Running speed benchmarks...")
        speed_metrics = self.performance_analyzer.benchmark_inference_speed(data_loader)
        
        # Memory profiling
        print("\n3. Profiling memory usage...")
        memory_metrics = self.performance_analyzer.profile_memory_usage(data_loader)
        
        # Combine results
        performance_results = {
            'basic_metrics': basic_metrics,
            'speed_metrics': speed_metrics,
            'memory_metrics': memory_metrics
        }
        
        self.analysis_results['performance'] = performance_results
        return performance_results
    
    def analyze_bit_widths(self, data_loader) -> Dict:
        """
        Analyze bit-width configurations and their impact.
        
        Args:
            data_loader: DataLoader for evaluation
        
        Returns:
            Dictionary of bit-width analysis results
        """
        print("\n" + "="*60)
        print("BIT-WIDTH ANALYSIS")
        print("="*60)
        
        # Test different configurations
        print("\n1. Testing bit-width configurations...")
        config_results = self.bit_width_analyzer.test_configurations(data_loader)
        
        # Analyze layer sensitivity
        print("\n2. Analyzing layer sensitivity...")
        sensitivity_results = self.bit_width_analyzer.analyze_layer_sensitivity(data_loader)
        
        # Find optimal configuration
        print("\n3. Finding optimal configuration...")
        optimal_config = self.bit_width_analyzer.find_optimal_configuration(
            config_results, 
            sensitivity_results
        )
        
        # Combine results
        bit_width_results = {
            'configuration_results': config_results,
            'layer_sensitivity': sensitivity_results,
            'optimal_configuration': optimal_config
        }
        
        self.analysis_results['bit_width'] = bit_width_results
        return bit_width_results
    
    def analyze_robustness(self, data_loader) -> Dict:
        """
        Analyze model robustness and stability.
        
        Args:
            data_loader: DataLoader for evaluation
        
        Returns:
            Dictionary of robustness analysis results
        """
        print("\n" + "="*60)
        print("ROBUSTNESS ANALYSIS")
        print("="*60)
        
        # Test adversarial robustness
        print("\n1. Testing adversarial robustness...")
        adversarial_results = self.robustness_analyzer.test_adversarial_robustness(data_loader)
        
        # Test quantization noise robustness
        print("\n2. Testing quantization noise robustness...")
        noise_results = self.robustness_analyzer.test_quantization_noise(data_loader)
        
        # Test input perturbation robustness
        print("\n3. Testing input perturbation robustness...")
        perturbation_results = self.robustness_analyzer.test_input_perturbations(data_loader)
        
        # Combine results
        robustness_results = {
            'adversarial': adversarial_results,
            'quantization_noise': noise_results,
            'input_perturbations': perturbation_results
        }
        
        self.analysis_results['robustness'] = robustness_results
        return robustness_results
    
    def generate_comprehensive_report(self, output_dir: str = './analysis_results/') -> None:
        """
        Generate a comprehensive analysis report.
        
        Args:
            output_dir: Directory to save the report and visualizations
        """
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results as JSON
        results_path = os.path.join(output_dir, 'analysis_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_path}")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        create_analysis_plots(self.analysis_results, output_dir)
        
        # Generate summary report
        summary = self.generate_summary()
        summary_path = os.path.join(output_dir, 'summary_report.md')
        with open(summary_path, 'w') as f:
            f.write(summary)
        print(f"Summary report saved to: {summary_path}")
        
        # Generate detailed insights
        insights = self.generate_insights()
        insights_path = os.path.join(output_dir, 'insights.md')
        with open(insights_path, 'w') as f:
            f.write(insights)
        print(f"Insights saved to: {insights_path}")
        
    def generate_summary(self) -> str:
        """
        Generate a markdown summary report.
        
        Returns:
            Markdown formatted summary string
        """
        summary = "# Model Analysis Summary Report\n\n"
        summary += f"**Model Path**: {self.model_path}\n"
        summary += f"**Analysis Date**: {pd.Timestamp.now()}\n\n"
        
        # Model configuration
        summary += "## Model Configuration\n\n"
        summary += f"- **Layers**: {self.config.get('n_layer', 'N/A')}\n"
        summary += f"- **Embedding Dimension**: {self.config.get('n_embd', 'N/A')}\n"
        summary += f"- **Attention Heads**: {self.config.get('n_head', 'N/A')}\n"
        summary += f"- **Bit Widths**: {self.config.get('bit_widths', 'N/A')}\n\n"
        
        # Performance summary
        if 'performance' in self.analysis_results:
            summary += "## Performance Metrics\n\n"
            perf = self.analysis_results['performance']
            
            if 'basic_metrics' in perf:
                basic = perf['basic_metrics']
                summary += "### Basic Metrics\n"
                summary += f"- **Loss**: {basic.get('loss', 'N/A'):.4f}\n"
                summary += f"- **Perplexity**: {basic.get('perplexity', 'N/A'):.2f}\n"
                summary += f"- **Accuracy**: {basic.get('accuracy', 'N/A'):.4f}\n\n"
            
            if 'speed_metrics' in perf:
                speed = perf['speed_metrics']
                summary += "### Speed Metrics\n"
                summary += f"- **Tokens/Second**: {speed.get('tokens_per_second', 'N/A'):.2f}\n"
                summary += f"- **ms/Token**: {speed.get('ms_per_token', 'N/A'):.2f}\n\n"
            
            if 'memory_metrics' in perf:
                memory = perf['memory_metrics']
                summary += "### Memory Usage\n"
                summary += f"- **Model Size (MB)**: {memory.get('model_size_mb', 'N/A'):.2f}\n"
                summary += f"- **Peak Memory (MB)**: {memory.get('peak_memory_mb', 'N/A'):.2f}\n\n"
        
        # Bit-width analysis summary
        if 'bit_width' in self.analysis_results:
            summary += "## Bit-Width Analysis\n\n"
            bw = self.analysis_results['bit_width']
            
            if 'optimal_configuration' in bw:
                optimal = bw['optimal_configuration']
                summary += "### Optimal Configuration\n"
                summary += f"- **Configuration**: {optimal.get('name', 'N/A')}\n"
                summary += f"- **Effective Bits**: {optimal.get('effective_bits', 'N/A'):.2f}\n"
                summary += f"- **Performance Score**: {optimal.get('score', 'N/A'):.4f}\n\n"
        
        # Robustness summary
        if 'robustness' in self.analysis_results:
            summary += "## Robustness Analysis\n\n"
            rob = self.analysis_results['robustness']
            
            if 'adversarial' in rob:
                adv = rob['adversarial']
                summary += "### Adversarial Robustness\n"
                summary += f"- **Clean Accuracy**: {adv.get('clean_accuracy', 'N/A'):.4f}\n"
                summary += f"- **Robust Accuracy**: {adv.get('robust_accuracy', 'N/A'):.4f}\n"
                summary += f"- **Robustness Gap**: {adv.get('robustness_gap', 'N/A'):.4f}\n\n"
        
        return summary
    
    def generate_insights(self) -> str:
        """
        Generate detailed insights from the analysis.
        
        Returns:
            Markdown formatted insights string
        """
        insights = "# Model Analysis Insights\n\n"
        
        # Performance insights
        insights += "## Performance Insights\n\n"
        
        if 'performance' in self.analysis_results:
            perf = self.analysis_results['performance']
            
            # Check if model is efficient
            if 'speed_metrics' in perf:
                tps = perf['speed_metrics'].get('tokens_per_second', 0)
                if tps > 1000:
                    insights += f"✅ **High Speed**: Model processes {tps:.0f} tokens/second, suitable for real-time applications.\n\n"
                elif tps > 100:
                    insights += f"⚠️ **Moderate Speed**: Model processes {tps:.0f} tokens/second, may need optimization for real-time use.\n\n"
                else:
                    insights += f"❌ **Low Speed**: Model processes only {tps:.0f} tokens/second, optimization required.\n\n"
            
            # Memory efficiency
            if 'memory_metrics' in perf:
                size_mb = perf['memory_metrics'].get('model_size_mb', 0)
                if size_mb < 100:
                    insights += f"✅ **Memory Efficient**: Model size is {size_mb:.1f} MB, suitable for edge deployment.\n\n"
                elif size_mb < 500:
                    insights += f"⚠️ **Moderate Memory**: Model size is {size_mb:.1f} MB, suitable for server deployment.\n\n"
                else:
                    insights += f"❌ **High Memory**: Model size is {size_mb:.1f} MB, may require memory optimization.\n\n"
        
        # Bit-width insights
        insights += "## Quantization Insights\n\n"
        
        if 'bit_width' in self.analysis_results:
            bw = self.analysis_results['bit_width']
            
            if 'layer_sensitivity' in bw:
                sensitivity = bw['layer_sensitivity']
                # Identify most and least sensitive layers
                if sensitivity:
                    insights += "### Layer Sensitivity\n"
                    insights += "- Early layers typically show higher sensitivity to quantization\n"
                    insights += "- Consider using mixed precision with higher bits for sensitive layers\n\n"
            
            if 'optimal_configuration' in bw:
                optimal = bw['optimal_configuration']
                insights += f"### Recommended Configuration\n"
                insights += f"- Use **{optimal.get('name', 'N/A')}** configuration for best efficiency\n"
                insights += f"- This achieves {optimal.get('compression_ratio', 'N/A'):.1f}x compression\n\n"
        
        # Robustness insights
        insights += "## Robustness Insights\n\n"
        
        if 'robustness' in self.analysis_results:
            rob = self.analysis_results['robustness']
            
            if 'adversarial' in rob:
                adv = rob['adversarial']
                gap = adv.get('robustness_gap', 1.0)
                
                if gap < 0.1:
                    insights += "✅ **Strong Robustness**: Model shows excellent resistance to adversarial attacks.\n\n"
                elif gap < 0.3:
                    insights += "⚠️ **Moderate Robustness**: Model has some vulnerability to adversarial attacks.\n\n"
                else:
                    insights += "❌ **Weak Robustness**: Model is vulnerable to adversarial attacks, consider robustness training.\n\n"
        
        # Recommendations
        insights += "## Recommendations\n\n"
        insights += self.generate_recommendations()
        
        return insights
    
    def generate_recommendations(self) -> str:
        """
        Generate actionable recommendations based on analysis.
        
        Returns:
            Markdown formatted recommendations
        """
        recommendations = []
        
        # Performance-based recommendations
        if 'performance' in self.analysis_results:
            perf = self.analysis_results['performance']
            if 'basic_metrics' in perf:
                perplexity = perf['basic_metrics'].get('perplexity', float('inf'))
                if perplexity > 50:
                    recommendations.append(
                        "- **Improve Training**: High perplexity indicates the model may benefit from additional training or data."
                    )
        
        # Quantization recommendations
        if 'bit_width' in self.analysis_results:
            bw = self.analysis_results['bit_width']
            if 'optimal_configuration' in bw:
                optimal = bw['optimal_configuration']
                if optimal.get('effective_bits', 8) > 4:
                    recommendations.append(
                        "- **Explore Lower Bits**: Consider testing 2-bit or 3-bit quantization for further compression."
                    )
        
        # Robustness recommendations
        if 'robustness' in self.analysis_results:
            rob = self.analysis_results['robustness']
            if 'adversarial' in rob:
                gap = rob['adversarial'].get('robustness_gap', 0)
                if gap > 0.2:
                    recommendations.append(
                        "- **Enhance Robustness**: Consider adversarial training or defensive distillation."
                    )
        
        # Deployment recommendations
        recommendations.append(
            "- **Deployment Strategy**: Based on the analysis, this model is best suited for "
            + self.determine_deployment_scenario()
        )
        
        return "\n".join(recommendations) if recommendations else "No specific recommendations at this time.\n"
    
    def determine_deployment_scenario(self) -> str:
        """
        Determine the best deployment scenario based on analysis.
        
        Returns:
            Deployment scenario description
        """
        if 'performance' not in self.analysis_results:
            return "server-side deployment (insufficient data for edge deployment assessment)."
        
        perf = self.analysis_results['performance']
        
        # Check memory and speed
        memory_ok = False
        speed_ok = False
        
        if 'memory_metrics' in perf:
            size_mb = perf['memory_metrics'].get('model_size_mb', float('inf'))
            memory_ok = size_mb < 100
        
        if 'speed_metrics' in perf:
            tps = perf['speed_metrics'].get('tokens_per_second', 0)
            speed_ok = tps > 100
        
        if memory_ok and speed_ok:
            return "edge deployment on mobile or embedded devices."
        elif memory_ok:
            return "edge deployment with hardware acceleration."
        elif speed_ok:
            return "cloud deployment with good latency characteristics."
        else:
            return "batch processing or offline inference scenarios."


def main():
    """
    Main function for post-training analysis.
    """
    import argparse
    from transformers import GPT2TokenizerFast
    
    parser = argparse.ArgumentParser(description='Analyze trained GPT-2 models')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to model configuration file')
    parser.add_argument('--output_dir', type=str, default='./analysis_results/',
                       help='Directory to save analysis results')
    parser.add_argument('--data_split', type=str, default='validation[:100]',
                       help='Data split to use for analysis')
    
    args = parser.parse_args()
    
    print("="*60)
    print("POST-TRAINING MODEL ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = ModelAnalyzer(args.model_path, args.config_path)
    
    # Setup data loader
    print("\nPreparing data loader...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    _, val_loader = create_dataloaders(
        tokenizer,
        train_split='train[:1]',  # Dummy train split
        val_split=args.data_split,
        batch_size=2,
        max_length=256
    )
    
    # Run analyses
    print("\nRunning comprehensive analysis...")
    
    # Performance analysis
    performance_results = analyzer.analyze_performance(val_loader)
    
    # Bit-width analysis
    bit_width_results = analyzer.analyze_bit_widths(val_loader)
    
    # Robustness analysis
    robustness_results = analyzer.analyze_robustness(val_loader)
    
    # Generate report
    analyzer.generate_comprehensive_report(args.output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  - analysis_results.json: Raw analysis data")
    print("  - summary_report.md: High-level summary")
    print("  - insights.md: Detailed insights and recommendations")
    print("  - Various plots and visualizations")
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
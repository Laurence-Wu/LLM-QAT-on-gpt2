"""
Bit-Width Analysis Module
Analyzes optimal bit-width configurations and layer sensitivity.
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import itertools
import gc


class BitWidthAnalyzer:
    """
    Analyzes bit-width configurations and their impact on model performance.
    """
    
    def __init__(self, model):
        """
        Initialize the bit-width analyzer.
        
        Args:
            model: The trained model to analyze
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.n_layers = len(model.h)
        
        # Available bit widths from model config
        self.available_bits = getattr(model.config, 'bit_widths', [2, 4, 8])
        
    def test_configurations(self, data_loader, max_batches: int = 20) -> Dict:
        """
        Test various bit-width configurations.
        
        Args:
            data_loader: DataLoader for evaluation
            max_batches: Maximum batches to evaluate per configuration
        
        Returns:
            Dictionary of configuration results
        """
        print("Testing bit-width configurations...")
        
        # Define configurations to test
        configs = self._generate_test_configurations()
        results = {}
        
        for config_name, layer_config in configs.items():
            print(f"  Testing {config_name}...")
            
            # Set model configuration
            self.model.set_layer_precision(layer_config)
            
            # Evaluate
            metrics = self._evaluate_configuration(data_loader, max_batches)
            
            # Calculate effective bits
            effective_bits = self._calculate_effective_bits(layer_config)
            
            # Calculate compression ratio
            baseline_bits = 32  # Assume float32 baseline
            compression_ratio = baseline_bits / effective_bits
            
            results[config_name] = {
                'metrics': metrics,
                'effective_bits': effective_bits,
                'compression_ratio': compression_ratio,
                'layer_config': layer_config
            }
            
            print(f"    Loss: {metrics['loss']:.4f}, "
                  f"Accuracy: {metrics['accuracy']:.4f}, "
                  f"Effective bits: {effective_bits:.2f}")
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()
        
        return results
    
    def _generate_test_configurations(self) -> Dict[str, List[Dict]]:
        """
        Generate different bit-width configurations to test.
        
        Returns:
            Dictionary mapping configuration names to layer configurations
        """
        configs = {}
        
        # Uniform configurations
        for bits in self.available_bits:
            config_name = f"uniform_{bits}bit"
            layer_config = [{'attn_bits': bits, 'mlp_bits': bits} 
                           for _ in range(self.n_layers)]
            configs[config_name] = layer_config
        
        # Mixed precision configurations
        bit_pairs = list(itertools.combinations(self.available_bits, 2))
        for bits1, bits2 in bit_pairs:
            # Alternating pattern
            config_name = f"alternating_{bits1}_{bits2}bit"
            layer_config = []
            for i in range(self.n_layers):
                if i % 2 == 0:
                    layer_config.append({'attn_bits': bits1, 'mlp_bits': bits2})
                else:
                    layer_config.append({'attn_bits': bits2, 'mlp_bits': bits1})
            configs[config_name] = layer_config
        
        # Progressive configurations (higher precision at boundaries)
        for low_bits in [2, 4]:
            for high_bits in [8, 16] if 16 in self.available_bits else [8]:
                if low_bits < high_bits:
                    config_name = f"progressive_{low_bits}_{high_bits}bit"
                    layer_config = []
                    for i in range(self.n_layers):
                        if i < 2 or i >= self.n_layers - 2:
                            bits = high_bits
                        else:
                            bits = low_bits
                        layer_config.append({'attn_bits': bits, 'mlp_bits': bits})
                    configs[config_name] = layer_config
        
        # Attention-focused (higher precision for attention)
        config_name = "attention_focused"
        layer_config = [{'attn_bits': 8, 'mlp_bits': 4} 
                       for _ in range(self.n_layers)]
        configs[config_name] = layer_config
        
        # MLP-focused (higher precision for MLP)
        config_name = "mlp_focused"
        layer_config = [{'attn_bits': 4, 'mlp_bits': 8} 
                       for _ in range(self.n_layers)]
        configs[config_name] = layer_config
        
        return configs
    
    def analyze_layer_sensitivity(self, data_loader, max_batches: int = 10) -> Dict:
        """
        Analyze sensitivity of each layer to bit-width reduction.
        
        Args:
            data_loader: DataLoader for evaluation
            max_batches: Maximum batches per test
        
        Returns:
            Dictionary of layer sensitivity results
        """
        print("Analyzing layer sensitivity to quantization...")
        
        # Baseline: all layers at highest precision
        baseline_bits = max(self.available_bits)
        baseline_config = [{'attn_bits': baseline_bits, 'mlp_bits': baseline_bits} 
                          for _ in range(self.n_layers)]
        
        self.model.set_layer_precision(baseline_config)
        baseline_metrics = self._evaluate_configuration(data_loader, max_batches)
        baseline_loss = baseline_metrics['loss']
        
        print(f"Baseline loss ({baseline_bits}-bit): {baseline_loss:.4f}")
        
        layer_sensitivity = {}
        
        # Test reducing precision for each layer individually
        for layer_idx in range(self.n_layers):
            layer_results = {}
            
            for test_bits in self.available_bits:
                if test_bits >= baseline_bits:
                    continue  # Skip if not actually reducing precision
                
                # Create config with one layer at reduced precision
                test_config = baseline_config.copy()
                test_config[layer_idx] = {'attn_bits': test_bits, 'mlp_bits': test_bits}
                
                self.model.set_layer_precision(test_config)
                test_metrics = self._evaluate_configuration(data_loader, max_batches // 2)
                
                # Calculate sensitivity as loss increase
                sensitivity = test_metrics['loss'] - baseline_loss
                
                layer_results[f'{test_bits}bit'] = {
                    'loss': test_metrics['loss'],
                    'sensitivity': sensitivity,
                    'accuracy_drop': baseline_metrics['accuracy'] - test_metrics['accuracy']
                }
            
            layer_sensitivity[f'layer_{layer_idx}'] = layer_results
            
            print(f"  Layer {layer_idx}: sensitivity analyzed")
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()
        
        # Analyze attention vs MLP sensitivity
        component_sensitivity = self._analyze_component_sensitivity(
            data_loader, baseline_config, baseline_loss, max_batches // 2
        )
        
        return {
            'layer_sensitivity': layer_sensitivity,
            'component_sensitivity': component_sensitivity,
            'baseline_metrics': baseline_metrics
        }
    
    def _analyze_component_sensitivity(self, data_loader, baseline_config, 
                                     baseline_loss, max_batches) -> Dict:
        """
        Analyze sensitivity of attention vs MLP components.
        
        Args:
            data_loader: DataLoader for evaluation
            baseline_config: Baseline configuration
            baseline_loss: Baseline loss value
            max_batches: Maximum batches to evaluate
        
        Returns:
            Dictionary of component sensitivity results
        """
        component_results = {}
        
        # Test attention-only quantization
        for test_bits in [2, 4]:
            if test_bits in self.available_bits:
                attn_config = baseline_config.copy()
                for i in range(len(attn_config)):
                    attn_config[i] = {
                        'attn_bits': test_bits,
                        'mlp_bits': attn_config[i]['mlp_bits']
                    }
                
                self.model.set_layer_precision(attn_config)
                metrics = self._evaluate_configuration(data_loader, max_batches)
                
                component_results[f'attention_{test_bits}bit'] = {
                    'loss': metrics['loss'],
                    'sensitivity': metrics['loss'] - baseline_loss
                }
        
        # Test MLP-only quantization
        for test_bits in [2, 4]:
            if test_bits in self.available_bits:
                mlp_config = baseline_config.copy()
                for i in range(len(mlp_config)):
                    mlp_config[i] = {
                        'attn_bits': mlp_config[i]['attn_bits'],
                        'mlp_bits': test_bits
                    }
                
                self.model.set_layer_precision(mlp_config)
                metrics = self._evaluate_configuration(data_loader, max_batches)
                
                component_results[f'mlp_{test_bits}bit'] = {
                    'loss': metrics['loss'],
                    'sensitivity': metrics['loss'] - baseline_loss
                }
        
        return component_results
    
    def find_optimal_configuration(self, config_results: Dict, 
                                 sensitivity_results: Dict) -> Dict:
        """
        Find the optimal bit-width configuration based on performance and efficiency.
        
        Args:
            config_results: Results from configuration testing
            sensitivity_results: Results from sensitivity analysis
        
        Returns:
            Dictionary describing the optimal configuration
        """
        print("Finding optimal configuration...")
        
        # Calculate efficiency score for each configuration
        scores = {}
        
        for config_name, results in config_results.items():
            metrics = results['metrics']
            effective_bits = results['effective_bits']
            
            # Score based on accuracy per bit (higher is better)
            accuracy_efficiency = metrics['accuracy'] / effective_bits
            
            # Score based on perplexity efficiency (lower perplexity is better)
            perplexity_efficiency = 1.0 / (metrics['perplexity'] * effective_bits)
            
            # Combined score
            combined_score = 0.6 * accuracy_efficiency + 0.4 * perplexity_efficiency
            
            scores[config_name] = {
                'accuracy_efficiency': accuracy_efficiency,
                'perplexity_efficiency': perplexity_efficiency,
                'combined_score': combined_score,
                'metrics': metrics,
                'effective_bits': effective_bits
            }
        
        # Find configuration with highest combined score
        best_config = max(scores.keys(), key=lambda k: scores[k]['combined_score'])
        
        # Generate recommendations based on sensitivity analysis
        recommendations = self._generate_optimization_recommendations(sensitivity_results)
        
        return {
            'name': best_config,
            'score': scores[best_config]['combined_score'],
            'metrics': scores[best_config]['metrics'],
            'effective_bits': scores[best_config]['effective_bits'],
            'compression_ratio': config_results[best_config]['compression_ratio'],
            'layer_config': config_results[best_config]['layer_config'],
            'all_scores': scores,
            'recommendations': recommendations
        }
    
    def _generate_optimization_recommendations(self, sensitivity_results: Dict) -> List[str]:
        """
        Generate recommendations for further optimization.
        
        Args:
            sensitivity_results: Results from sensitivity analysis
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if 'layer_sensitivity' not in sensitivity_results:
            return recommendations
        
        layer_sensitivity = sensitivity_results['layer_sensitivity']
        
        # Find least sensitive layers
        least_sensitive_layers = []
        most_sensitive_layers = []
        
        for layer_name, layer_data in layer_sensitivity.items():
            if '4bit' in layer_data:
                sensitivity = layer_data['4bit']['sensitivity']
                if sensitivity < 0.1:  # Low sensitivity threshold
                    least_sensitive_layers.append((layer_name, sensitivity))
                elif sensitivity > 0.5:  # High sensitivity threshold
                    most_sensitive_layers.append((layer_name, sensitivity))
        
        # Sort by sensitivity
        least_sensitive_layers.sort(key=lambda x: x[1])
        most_sensitive_layers.sort(key=lambda x: x[1], reverse=True)
        
        # Generate recommendations
        if least_sensitive_layers:
            layers_str = ", ".join([layer[0] for layer in least_sensitive_layers[:3]])
            recommendations.append(
                f"Consider further reducing precision in: {layers_str} (low sensitivity)"
            )
        
        if most_sensitive_layers:
            layers_str = ", ".join([layer[0] for layer in most_sensitive_layers[:3]])
            recommendations.append(
                f"Maintain higher precision in: {layers_str} (high sensitivity)"
            )
        
        # Component recommendations
        if 'component_sensitivity' in sensitivity_results:
            comp_sens = sensitivity_results['component_sensitivity']
            
            # Compare attention vs MLP sensitivity
            attn_sens = comp_sens.get('attention_4bit', {}).get('sensitivity', float('inf'))
            mlp_sens = comp_sens.get('mlp_4bit', {}).get('sensitivity', float('inf'))
            
            if attn_sens < mlp_sens:
                recommendations.append(
                    "Attention layers are less sensitive to quantization than MLP layers"
                )
            else:
                recommendations.append(
                    "MLP layers are less sensitive to quantization than attention layers"
                )
        
        return recommendations
    
    def _evaluate_configuration(self, data_loader, max_batches: int) -> Dict:
        """
        Evaluate a specific bit-width configuration.
        
        Args:
            data_loader: DataLoader for evaluation
            max_batches: Maximum batches to evaluate
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0
        total_tokens = 0
        total_correct = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                outputs = self.model(input_ids, labels=input_ids, attention_mask=attention_mask)
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
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'accuracy': accuracy,
            'total_tokens': total_tokens
        }
    
    def _calculate_effective_bits(self, layer_config: List[Dict]) -> float:
        """
        Calculate the effective bit-width across all layers.
        
        Args:
            layer_config: List of layer configurations
        
        Returns:
            Effective bit-width as a weighted average
        """
        total_bits = 0
        total_components = 0
        
        for config in layer_config:
            total_bits += config['attn_bits'] + config['mlp_bits']
            total_components += 2  # Attention + MLP
        
        return total_bits / max(total_components, 1)
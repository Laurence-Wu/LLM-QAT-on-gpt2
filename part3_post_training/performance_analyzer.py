"""
Performance Analysis Module
Analyzes model performance metrics including speed, memory, and accuracy.
"""

import torch
import time
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
import gc
import psutil
import os


class PerformanceAnalyzer:
    """
    Analyzes various performance aspects of the trained model.
    """
    
    def __init__(self, model):
        """
        Initialize the performance analyzer.
        
        Args:
            model: The trained model to analyze
        """
        self.model = model
        self.device = next(model.parameters()).device
        
    def compute_basic_metrics(self, data_loader, max_batches: int = 50) -> Dict:
        """
        Compute basic performance metrics (loss, perplexity, accuracy).
        
        Args:
            data_loader: DataLoader for evaluation
            max_batches: Maximum number of batches to evaluate
        
        Returns:
            Dictionary of basic metrics
        """
        self.model.eval()
        
        total_loss = 0
        total_tokens = 0
        total_correct = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Computing metrics", total=max_batches)):
                if batch_idx >= max_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Forward pass
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
        
        # Calculate final metrics
        avg_loss = total_loss / max(batch_count, 1)
        perplexity = np.exp(avg_loss)
        accuracy = total_correct / max(total_tokens, 1)
        
        # Calculate top-k accuracy
        top5_accuracy = self.compute_topk_accuracy(data_loader, k=5, max_batches=20)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'accuracy': accuracy,
            'top5_accuracy': top5_accuracy,
            'total_tokens_evaluated': total_tokens,
            'batches_evaluated': batch_count
        }
    
    def compute_topk_accuracy(self, data_loader, k: int = 5, max_batches: int = 20) -> float:
        """
        Compute top-k accuracy.
        
        Args:
            data_loader: DataLoader for evaluation
            k: Top-k value
            max_batches: Maximum batches to evaluate
        
        Returns:
            Top-k accuracy
        """
        self.model.eval()
        
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                outputs = self.model(input_ids, labels=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                # Get top-k predictions
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                _, topk_preds = torch.topk(shift_logits, k, dim=-1)
                
                # Check if true label is in top-k
                correct = (topk_preds == shift_labels.unsqueeze(-1)).any(dim=-1)
                
                if attention_mask is not None:
                    shift_mask = attention_mask[..., 1:].contiguous()
                    correct = correct * shift_mask
                    valid_tokens = shift_mask.sum().item()
                else:
                    valid_tokens = shift_labels.numel()
                
                total_correct += correct.sum().item()
                total_tokens += valid_tokens
        
        return total_correct / max(total_tokens, 1)
    
    def benchmark_inference_speed(self, data_loader, num_warmup: int = 5, 
                                 num_measure: int = 20) -> Dict:
        """
        Benchmark inference speed of the model.
        
        Args:
            data_loader: DataLoader for benchmarking
            num_warmup: Number of warmup iterations
            num_measure: Number of iterations to measure
        
        Returns:
            Dictionary of speed metrics
        """
        self.model.eval()
        data_iter = iter(data_loader)
        
        print("Running warmup iterations...")
        # Warmup
        for _ in range(num_warmup):
            try:
                batch = next(data_iter)
                input_ids = batch['input_ids'].to(self.device)
                with torch.no_grad():
                    _ = self.model(input_ids)
            except StopIteration:
                data_iter = iter(data_loader)
        
        print("Measuring inference speed...")
        # Measurement
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        latencies = []
        total_tokens = 0
        
        for _ in range(num_measure):
            try:
                batch = next(data_iter)
                input_ids = batch['input_ids'].to(self.device)
                num_tokens = input_ids.numel()
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    _ = self.model(input_ids)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.perf_counter()
                
                latency = end_time - start_time
                latencies.append(latency)
                total_tokens += num_tokens
                
            except StopIteration:
                data_iter = iter(data_loader)
        
        # Calculate statistics
        latencies = np.array(latencies)
        total_time = np.sum(latencies)
        
        return {
            'mean_latency': np.mean(latencies),
            'std_latency': np.std(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'p50_latency': np.percentile(latencies, 50),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'tokens_per_second': total_tokens / total_time,
            'ms_per_token': (total_time / total_tokens) * 1000,
            'batches_per_second': num_measure / total_time
        }
    
    def profile_memory_usage(self, data_loader, num_iterations: int = 10) -> Dict:
        """
        Profile memory usage of the model.
        
        Args:
            data_loader: DataLoader for profiling
            num_iterations: Number of iterations to profile
        
        Returns:
            Dictionary of memory metrics
        """
        # Get model size
        model_size = self.calculate_model_size()
        
        # Memory profiling during inference
        memory_usage = []
        peak_memory = 0
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()
            
            initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        else:
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024**2  # MB
        
        self.model.eval()
        data_iter = iter(data_loader)
        
        for _ in range(num_iterations):
            try:
                batch = next(data_iter)
                input_ids = batch['input_ids'].to(self.device)
                
                with torch.no_grad():
                    _ = self.model(input_ids)
                
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() / 1024**2
                    peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / 1024**2)
                else:
                    process = psutil.Process(os.getpid())
                    current_memory = process.memory_info().rss / 1024**2
                    peak_memory = max(peak_memory, current_memory)
                
                memory_usage.append(current_memory - initial_memory)
                
            except StopIteration:
                data_iter = iter(data_loader)
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'model_size_mb': model_size,
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'avg_memory_increase_mb': np.mean(memory_usage) if memory_usage else 0,
            'max_memory_increase_mb': np.max(memory_usage) if memory_usage else 0,
            'device': str(self.device),
            'gpu_available': torch.cuda.is_available()
        }
    
    def calculate_model_size(self) -> float:
        """
        Calculate the size of the model in MB.
        
        Returns:
            Model size in MB
        """
        param_size = 0
        buffer_size = 0
        
        # Calculate parameter size
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        # Calculate buffer size
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        # Total size in MB
        total_size = (param_size + buffer_size) / 1024**2
        
        return total_size
    
    def profile_layer_execution_time(self, data_loader, num_iterations: int = 5) -> Dict:
        """
        Profile execution time for each layer.
        
        Args:
            data_loader: DataLoader for profiling
            num_iterations: Number of iterations
        
        Returns:
            Dictionary of layer execution times
        """
        self.model.eval()
        layer_times = {}
        
        # Hook to measure layer execution time
        def create_hook(layer_name):
            def hook(module, input, output):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.perf_counter()
                if layer_name not in layer_times:
                    layer_times[layer_name] = []
                layer_times[layer_name].append(end_time - hook.start_time)
            
            def pre_hook(module, input):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                hook.start_time = time.perf_counter()
            
            return pre_hook, hook
        
        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                pre_hook, hook = create_hook(name)
                hooks.append(module.register_forward_pre_hook(pre_hook))
                hooks.append(module.register_forward_hook(hook))
        
        # Run profiling
        data_iter = iter(data_loader)
        for _ in range(num_iterations):
            try:
                batch = next(data_iter)
                input_ids = batch['input_ids'].to(self.device)
                
                with torch.no_grad():
                    _ = self.model(input_ids)
                    
            except StopIteration:
                data_iter = iter(data_loader)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Calculate statistics
        layer_stats = {}
        for layer_name, times in layer_times.items():
            layer_stats[layer_name] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'total_time': np.sum(times)
            }
        
        return layer_stats
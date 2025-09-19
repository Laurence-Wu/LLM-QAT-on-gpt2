#!/usr/bin/env python3
"""
Complete SP Training Monitor with Real Data
Monitors the entire training procedure including data loading, training loop, and evaluation
"""

import sys
import os
import torch
import torch.nn as nn
import gc
import time
import json
import psutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2Config, GPT2TokenizerFast, GPT2Model
from shared.models_sp import SPModel, SPLMHeadModel
from part1_switchable_precision.config_sp import ModelConfig, TrainingConfig
from part1_switchable_precision.train_sp import train_sp, get_next_bitwidth, evaluate
from shared.dataset import create_dataloaders


class ComprehensiveTrainingMonitor:
    """Complete training monitor with detailed tracking."""

    def __init__(self, output_dir="training_monitor_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.log = {
            "timestamp_start": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "model_architecture": {},
            "data_pipeline": {},
            "training_progress": [],
            "shape_tracking": {},
            "gradient_flow": {},
            "memory_timeline": [],
            "precision_switches": [],
            "evaluation_results": [],
            "issues_detected": [],
            "performance_metrics": {}
        }

    def _get_system_info(self):
        """Get comprehensive system information."""
        info = {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cpu_info": {
                "count": psutil.cpu_count(),
                "freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else "N/A"
            },
            "memory_gb": psutil.virtual_memory().total / (1024**3)
        }

        if torch.cuda.is_available():
            info["gpu_info"] = {
                "count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "capability": torch.cuda.get_device_capability(0),
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "cuda_version": torch.version.cuda
            }

        return info

    def monitor_model_architecture(self, model, config):
        """Analyze and log model architecture."""
        print("\n📊 Analyzing Model Architecture...")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        # Count by component
        component_params = {}
        lora_params_by_bit = {}

        for name, param in model.named_parameters():
            # Categorize parameters
            if 'wte' in name:
                component = 'token_embeddings'
            elif 'wpe' in name:
                component = 'position_embeddings'
            elif 'ln' in name:
                component = 'layer_norms'
            elif 'lora' in name:
                component = 'lora_adapters'
                # Track LoRA params by bit width
                for bits in config.bit_widths:
                    if f'{bits}bit' in name:
                        key = f'lora_{bits}bit'
                        lora_params_by_bit[key] = lora_params_by_bit.get(key, 0) + param.numel()
            elif 'attn' in name:
                component = 'attention'
            elif 'mlp' in name:
                component = 'mlp'
            elif 'lm_head' in name:
                component = 'lm_head'
            else:
                component = 'other'

            component_params[component] = component_params.get(component, 0) + param.numel()

        self.log["model_architecture"] = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "parameter_breakdown": component_params,
            "lora_parameters_by_bit": lora_params_by_bit,
            "model_size_mb": total_params * 4 / (1024**2),
            "config": {
                "n_layer": config.n_layer,
                "n_embd": config.n_embd,
                "n_head": config.n_head,
                "vocab_size": config.vocab_size,
                "n_positions": config.n_positions,
                "bit_widths": config.bit_widths,
                "lora_rank_per_bit": config.lora_rank_per_bit,
                "lora_alpha_per_bit": config.lora_alpha_per_bit
            }
        }

        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / (1024**2):.1f} MB")
        print(f"  LoRA parameters by bit-width:")
        for bit_key, count in lora_params_by_bit.items():
            print(f"    {bit_key}: {count:,}")

    def monitor_data_pipeline(self, train_loader, val_loader, tokenizer):
        """Monitor data loading pipeline."""
        print("\n📊 Analyzing Data Pipeline...")

        # Sample batch analysis
        sample_batch = next(iter(train_loader))

        batch_info = {
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "batch_keys": list(sample_batch.keys()),
            "batch_shapes": {},
            "tokenizer_info": {
                "vocab_size": tokenizer.vocab_size,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id
            }
        }

        # Analyze batch structure
        for key, tensor in sample_batch.items():
            if isinstance(tensor, torch.Tensor):
                batch_info["batch_shapes"][key] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "min": int(tensor.min().item()),
                    "max": int(tensor.max().item())
                }

        self.log["data_pipeline"] = batch_info

        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Batch shape: {batch_info['batch_shapes']['input_ids']['shape']}")

    def monitor_training_step(self, step: int, model, loss: float,
                            current_bits: int, optimizer, input_ids: torch.Tensor):
        """Monitor a single training step."""

        step_info = {
            "step": step,
            "loss": float(loss),
            "current_bits": current_bits,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "timestamp": datetime.now().isoformat()
        }

        # Track shapes
        if step % 10 == 0:  # Sample every 10 steps
            self._track_shapes(model, input_ids, step)

        # Track gradients
        grad_info = None
        if step % 10 == 0:
            grad_info = self._analyze_gradients(model)
            step_info["gradient_norm"] = grad_info["total_norm"]
            step_info["gradient_stats"] = grad_info

        # Track memory
        memory_info = self._get_memory_info()
        step_info["memory_mb"] = memory_info["gpu_allocated_mb"] if "gpu_allocated_mb" in memory_info else 0

        self.log["training_progress"].append(step_info)

        # Check for issues
        if np.isnan(loss) or np.isinf(loss):
            self._log_issue(f"Invalid loss at step {step}: {loss}", "critical")

        # Only check gradient norm if we have grad_info
        if grad_info and grad_info.get("total_norm", 0) > 100:
            self._log_issue(f"Large gradient norm at step {step}: {grad_info['total_norm']:.2f}", "warning")

        return step_info

    def _track_shapes(self, model, input_ids, step):
        """Track tensor shapes through the model."""
        shapes = {}

        # Input shape
        shapes["input"] = list(input_ids.shape)

        # Track key layer outputs using hooks
        def get_shape_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    shapes[name] = list(output.shape)
                elif isinstance(output, tuple) and len(output) > 0:
                    shapes[name] = list(output[0].shape)
            return hook

        # Register temporary hooks
        hooks = []
        hooks.append(model.transformer.wte.register_forward_hook(get_shape_hook("embeddings")))
        hooks.append(model.transformer.ln_f.register_forward_hook(get_shape_hook("final_ln")))
        hooks.append(model.lm_head.register_forward_hook(get_shape_hook("lm_head")))

        # Forward pass to collect shapes
        with torch.no_grad():
            _ = model(input_ids)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Store shapes
        if step not in self.log["shape_tracking"]:
            self.log["shape_tracking"][step] = shapes

    def _analyze_gradients(self, model):
        """Analyze gradient statistics."""
        grad_info = {
            "total_norm": 0.0,
            "num_params_with_grad": 0,
            "num_params_without_grad": 0,
            "max_grad": 0.0,
            "min_grad": float('inf'),
            "has_nan": False,
            "has_inf": False
        }

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_info["total_norm"] += grad_norm ** 2
                grad_info["num_params_with_grad"] += 1

                grad_info["max_grad"] = max(grad_info["max_grad"], grad_norm)
                grad_info["min_grad"] = min(grad_info["min_grad"], grad_norm)

                if torch.isnan(param.grad).any():
                    grad_info["has_nan"] = True
                if torch.isinf(param.grad).any():
                    grad_info["has_inf"] = True
            else:
                grad_info["num_params_without_grad"] += 1

        grad_info["total_norm"] = (grad_info["total_norm"] ** 0.5)

        return grad_info

    def _get_memory_info(self):
        """Get current memory usage."""
        info = {
            "cpu_percent": psutil.virtual_memory().percent,
            "cpu_available_gb": psutil.virtual_memory().available / (1024**3)
        }

        if torch.cuda.is_available():
            info.update({
                "gpu_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                "gpu_reserved_mb": torch.cuda.memory_reserved() / (1024**2),
                "gpu_max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2)
            })

        return info

    def monitor_precision_switch(self, step: int, old_bits: int, new_bits: int):
        """Log precision switching event."""
        switch_info = {
            "step": step,
            "old_bits": old_bits,
            "new_bits": new_bits,
            "timestamp": datetime.now().isoformat()
        }
        self.log["precision_switches"].append(switch_info)

    def monitor_evaluation(self, step: int, val_loss: float, current_bits: int):
        """Monitor evaluation results."""
        eval_info = {
            "step": step,
            "val_loss": float(val_loss),
            "current_bits": current_bits,
            "timestamp": datetime.now().isoformat()
        }
        self.log["evaluation_results"].append(eval_info)

    def _log_issue(self, issue: str, severity: str = "info"):
        """Log detected issues."""
        issue_info = {
            "issue": issue,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }
        self.log["issues_detected"].append(issue_info)
        print(f"  ⚠️ {severity.upper()}: {issue}")

    def finalize(self):
        """Finalize monitoring and compute summary statistics."""
        self.log["timestamp_end"] = datetime.now().isoformat()

        # Compute performance metrics
        if self.log["training_progress"]:
            losses = [s["loss"] for s in self.log["training_progress"]]
            self.log["performance_metrics"] = {
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "min_loss": min(losses),
                "max_loss": max(losses),
                "loss_reduction": losses[0] - losses[-1],
                "total_steps": len(losses)
            }

        # Save all logs
        self.save_logs()

    def save_logs(self):
        """Save monitoring logs to files."""
        # Main log file
        log_file = os.path.join(self.output_dir, "training_monitor.json")
        with open(log_file, 'w') as f:
            json.dump(self.log, f, indent=2, default=str)
        print(f"\n💾 Monitoring logs saved to: {log_file}")

        # Create summary report
        self._create_summary_report()

    def _create_summary_report(self):
        """Create a human-readable summary report."""
        report_file = os.path.join(self.output_dir, "training_summary.txt")

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SP TRAINING SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")

            # System Info
            f.write("SYSTEM INFORMATION:\n")
            f.write("-"*40 + "\n")
            if "gpu_info" in self.log["system_info"]:
                f.write(f"GPU: {self.log['system_info']['gpu_info']['device_name']}\n")
                f.write(f"GPU Memory: {self.log['system_info']['gpu_info']['memory_gb']:.1f} GB\n")
            f.write(f"PyTorch: {self.log['system_info']['pytorch_version']}\n\n")

            # Model Architecture
            f.write("MODEL ARCHITECTURE:\n")
            f.write("-"*40 + "\n")
            arch = self.log["model_architecture"]
            f.write(f"Total Parameters: {arch['total_parameters']:,}\n")
            f.write(f"Trainable Parameters: {arch['trainable_parameters']:,}\n")
            f.write(f"Model Size: {arch['model_size_mb']:.1f} MB\n\n")

            # Training Performance
            if "performance_metrics" in self.log:
                f.write("TRAINING PERFORMANCE:\n")
                f.write("-"*40 + "\n")
                metrics = self.log["performance_metrics"]
                f.write(f"Initial Loss: {metrics['initial_loss']:.4f}\n")
                f.write(f"Final Loss: {metrics['final_loss']:.4f}\n")
                f.write(f"Loss Reduction: {metrics['loss_reduction']:.4f}\n")
                f.write(f"Total Steps: {metrics['total_steps']}\n\n")

            # Precision Switches
            f.write("PRECISION SWITCHING:\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Switches: {len(self.log['precision_switches'])}\n")
            if self.log['precision_switches']:
                switches = self.log['precision_switches'][:5]
                for s in switches:
                    f.write(f"  Step {s['step']}: {s['old_bits']}-bit → {s['new_bits']}-bit\n")
                if len(self.log['precision_switches']) > 5:
                    f.write("  ...\n")
            f.write("\n")

            # Issues Detected
            f.write("ISSUES DETECTED:\n")
            f.write("-"*40 + "\n")
            if self.log["issues_detected"]:
                by_severity = {}
                for issue in self.log["issues_detected"]:
                    severity = issue["severity"]
                    by_severity[severity] = by_severity.get(severity, 0) + 1
                for severity, count in by_severity.items():
                    f.write(f"  {severity.upper()}: {count}\n")
            else:
                f.write("  No issues detected\n")

        print(f"📄 Summary report saved to: {report_file}")


def run_monitored_training():
    """Run SP training with comprehensive monitoring."""

    print("\n" + "="*80)
    print("STARTING MONITORED SP TRAINING")
    print("="*80)

    # Initialize monitor
    monitor = ComprehensiveTrainingMonitor()

    try:
        # ========== Configuration ==========
        print("\n1. Setting up configuration...")
        model_config = ModelConfig()
        model_config.n_layer = 2  # Small for demonstration
        model_config.n_embd = 256
        model_config.n_head = 4
        model_config.vocab_size = 50257  # GPT-2 vocab

        training_config = TrainingConfig()
        training_config.train_split = 'train[:100]'  # Small dataset for demo
        training_config.val_split = 'validation[:20]'
        training_config.batch_size = 2
        training_config.max_seq_length = 128
        training_config.num_iterations = 30  # Short run for demo
        training_config.eval_interval = 10
        training_config.gradient_accumulation_steps = 1

        # ========== Model Creation ==========
        print("\n2. Creating model...")
        gpt2_config = GPT2Config(
            vocab_size=model_config.vocab_size,
            n_positions=model_config.n_positions,
            n_embd=model_config.n_embd,
            n_layer=model_config.n_layer,
            n_head=model_config.n_head
        )
        gpt2_config.bit_widths = model_config.bit_widths
        gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
        gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit

        model = SPLMHeadModel(gpt2_config)

        # Monitor model architecture
        monitor.monitor_model_architecture(model, model_config)

        # ========== Data Loading ==========
        print("\n3. Loading data...")
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        train_loader, val_loader = create_dataloaders(
            tokenizer,
            train_split=training_config.train_split,
            val_split=training_config.val_split,
            batch_size=training_config.batch_size,
            max_length=training_config.max_seq_length,
            doc_stride=training_config.doc_stride
        )

        # Monitor data pipeline
        monitor.monitor_data_pipeline(train_loader, val_loader, tokenizer)

        # ========== Setup Training ==========
        print("\n4. Setting up training...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {device}")
        model = model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )

        # ========== Training Loop ==========
        print("\n5. Starting training loop...")
        model.train()
        train_iter = iter(train_loader)
        current_bits = model_config.bit_widths[0]

        for step in range(training_config.num_iterations):
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch['input_ids'].to(device)
            labels = input_ids.clone()

            # Check for precision switch
            new_bits = get_next_bitwidth(step, model_config)
            if new_bits != current_bits:
                monitor.monitor_precision_switch(step, current_bits, new_bits)
                current_bits = new_bits
                model.set_precision(current_bits)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss']

            # Backward pass
            loss.backward()

            # Monitor this step
            step_info = monitor.monitor_training_step(
                step, model, loss.item(), current_bits, optimizer, input_ids
            )

            # Optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
            optimizer.step()

            # Progress update
            if step % 5 == 0:
                print(f"  Step {step}/{training_config.num_iterations}: "
                      f"Loss={loss.item():.4f}, Bits={current_bits}, "
                      f"LR={optimizer.param_groups[0]['lr']:.6f}")

            # Evaluation
            if step % training_config.eval_interval == 0 and step > 0:
                model.eval()
                val_loss = evaluate(model, val_loader, device, training_config.use_amp)
                monitor.monitor_evaluation(step, val_loss, current_bits)
                print(f"  📊 Validation loss: {val_loss:.4f}")
                model.train()

        # ========== Finalize ==========
        print("\n6. Finalizing...")
        monitor.finalize()

        print("\n" + "="*80)
        print("✅ MONITORED TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)

        return monitor

    except Exception as e:
        print(f"\n❌ Error during monitored training: {e}")
        monitor._log_issue(str(e), "critical")
        monitor.finalize()
        import traceback
        traceback.print_exc()
        return monitor


if __name__ == "__main__":
    # Run the monitored training
    monitor = run_monitored_training()

    print("\n📊 Training Monitor Results:")
    print(f"  Output directory: {monitor.output_dir}")
    print(f"  Total issues detected: {len(monitor.log['issues_detected'])}")
    print(f"  Precision switches: {len(monitor.log['precision_switches'])}")

    if monitor.log["performance_metrics"]:
        metrics = monitor.log["performance_metrics"]
        print(f"  Loss reduction: {metrics['loss_reduction']:.4f}")
        print(f"  Final loss: {metrics['final_loss']:.4f}")
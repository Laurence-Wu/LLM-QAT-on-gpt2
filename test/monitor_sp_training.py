#!/usr/bin/env python3
"""
Comprehensive SP Training Monitor
Watches the entire training procedure and tracks shapes, memory, and potential issues
"""

import sys
import os
import torch
import torch.nn as nn
import gc
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback
import psutil
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2Config, GPT2TokenizerFast
from shared.models_sp import SPModel, SPLMHeadModel
from part1_switchable_precision.config_sp import ModelConfig, TrainingConfig
from part1_switchable_precision.train_sp import train_sp, get_next_bitwidth
from shared.dataset import create_dataloaders

class TrainingMonitor:
    """Monitor for tracking training progress and diagnosing issues."""

    def __init__(self, log_file="sp_training_monitor.json"):
        self.log_file = log_file
        self.logs = {
            "start_time": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "model_info": {},
            "training_steps": [],
            "shape_tracking": {},
            "memory_tracking": [],
            "errors": [],
            "warnings": [],
            "precision_switches": []
        }

    def _get_system_info(self):
        """Get system information."""
        info = {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cpu_count": psutil.cpu_count(),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3)
        }

        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            })

        return info

    def log_model_info(self, model, config):
        """Log model architecture information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logs["model_info"] = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024**2),  # Assuming float32
            "config": {
                "n_layer": config.n_layer,
                "n_embd": config.n_embd,
                "n_head": config.n_head,
                "vocab_size": config.vocab_size,
                "bit_widths": config.bit_widths,
                "lora_rank_per_bit": config.lora_rank_per_bit,
                "lora_alpha_per_bit": config.lora_alpha_per_bit
            }
        }

        # Count LoRA parameters per bit-width
        lora_params = {}
        for bits in config.bit_widths:
            lora_params[f"{bits}bit"] = 0

        for name, module in model.named_modules():
            if hasattr(module, 'lora_adapters'):
                for bit_key, adapter in module.lora_adapters.items():
                    bits = int(bit_key.replace('bit', ''))
                    # Count A and B matrices
                    params = adapter['A'].numel() + adapter['B'].numel()
                    lora_params[bit_key] += params

        self.logs["model_info"]["lora_parameters_per_bit"] = lora_params

    def log_shape(self, name: str, tensor: torch.Tensor, step: Optional[int] = None):
        """Log tensor shape information."""
        shape_info = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "min": float(tensor.min().item()) if tensor.numel() > 0 else None,
            "max": float(tensor.max().item()) if tensor.numel() > 0 else None,
            "mean": float(tensor.mean().item()) if tensor.numel() > 0 else None,
            "std": float(tensor.std().item()) if tensor.numel() > 0 else None,
            "has_nan": bool(torch.isnan(tensor).any().item()),
            "has_inf": bool(torch.isinf(tensor).any().item())
        }

        if step is not None:
            shape_info["step"] = step

        if name not in self.logs["shape_tracking"]:
            self.logs["shape_tracking"][name] = []
        self.logs["shape_tracking"][name].append(shape_info)

        # Check for issues
        if shape_info["has_nan"]:
            self.add_warning(f"NaN detected in {name} at step {step}")
        if shape_info["has_inf"]:
            self.add_warning(f"Inf detected in {name} at step {step}")

    def log_memory(self, step: int, tag: str = ""):
        """Log memory usage."""
        memory_info = {
            "step": step,
            "tag": tag,
            "timestamp": datetime.now().isoformat()
        }

        # CPU memory
        cpu_memory = psutil.virtual_memory()
        memory_info["cpu"] = {
            "used_gb": cpu_memory.used / (1024**3),
            "available_gb": cpu_memory.available / (1024**3),
            "percent": cpu_memory.percent
        }

        # GPU memory if available
        if torch.cuda.is_available():
            memory_info["gpu"] = {
                "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
                "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2)
            }

        self.logs["memory_tracking"].append(memory_info)

    def log_training_step(self, step: int, loss: float, current_bits: int,
                         learning_rate: float, grad_norm: Optional[float] = None):
        """Log training step information."""
        step_info = {
            "step": step,
            "loss": float(loss),
            "current_bits": current_bits,
            "learning_rate": learning_rate,
            "grad_norm": grad_norm,
            "timestamp": datetime.now().isoformat()
        }

        self.logs["training_steps"].append(step_info)

    def log_precision_switch(self, step: int, old_bits: int, new_bits: int):
        """Log precision switching events."""
        switch_info = {
            "step": step,
            "old_bits": old_bits,
            "new_bits": new_bits,
            "timestamp": datetime.now().isoformat()
        }

        self.logs["precision_switches"].append(switch_info)

    def add_error(self, error: str, traceback_str: Optional[str] = None):
        """Add error to log."""
        error_info = {
            "error": error,
            "traceback": traceback_str,
            "timestamp": datetime.now().isoformat()
        }
        self.logs["errors"].append(error_info)

    def add_warning(self, warning: str):
        """Add warning to log."""
        warning_info = {
            "warning": warning,
            "timestamp": datetime.now().isoformat()
        }
        self.logs["warnings"].append(warning_info)

    def save_logs(self):
        """Save logs to file."""
        self.logs["end_time"] = datetime.now().isoformat()
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2, default=str)

    def print_summary(self):
        """Print summary of monitoring results."""
        print("\n" + "="*80)
        print("TRAINING MONITOR SUMMARY")
        print("="*80)

        print(f"\nTotal training steps: {len(self.logs['training_steps'])}")
        print(f"Precision switches: {len(self.logs['precision_switches'])}")
        print(f"Errors encountered: {len(self.logs['errors'])}")
        print(f"Warnings: {len(self.logs['warnings'])}")

        if self.logs['training_steps']:
            losses = [step['loss'] for step in self.logs['training_steps']]
            print(f"\nLoss statistics:")
            print(f"  Initial loss: {losses[0]:.4f}")
            print(f"  Final loss: {losses[-1]:.4f}")
            print(f"  Min loss: {min(losses):.4f}")
            print(f"  Max loss: {max(losses):.4f}")

        if self.logs['memory_tracking']:
            gpu_mem = [m['gpu']['allocated_mb'] for m in self.logs['memory_tracking']
                      if 'gpu' in m]
            if gpu_mem:
                print(f"\nGPU memory usage:")
                print(f"  Peak: {max(gpu_mem):.1f} MB")
                print(f"  Average: {sum(gpu_mem)/len(gpu_mem):.1f} MB")

        print("\n" + "="*80)


def monitor_sp_training_step_by_step():
    """Monitor SP training with detailed step-by-step tracking."""

    print("\n" + "="*80)
    print("SP TRAINING STEP-BY-STEP MONITOR")
    print("="*80)

    monitor = TrainingMonitor("sp_training_detailed_monitor.json")

    try:
        # ========== 1. Configuration Setup ==========
        print("\n1. Setting up configuration...")
        model_config = ModelConfig()
        model_config.n_layer = 2  # Small for testing
        model_config.n_embd = 256
        model_config.n_head = 4
        model_config.vocab_size = 1000
        model_config.bit_widths = [4, 8, 16]
        model_config.lora_rank_per_bit = {4: 4, 8: 8, 16: 16}
        model_config.lora_alpha_per_bit = {4: 8, 8: 16, 16: 32}

        training_config = TrainingConfig()
        training_config.batch_size = 2
        training_config.max_seq_length = 64
        training_config.num_iterations = 20  # Small for testing
        training_config.gradient_accumulation_steps = 1
        training_config.learning_rate = 1e-4
        training_config.use_amp = False  # Disable for clearer debugging

        print("   ✓ Configuration created")

        # ========== 2. Model Creation ==========
        print("\n2. Creating SP model...")
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
        monitor.log_model_info(model, model_config)

        # Check model structure
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Model layers: {model_config.n_layer}")
        print(f"   Embedding dimension: {model_config.n_embd}")
        print(f"   Vocabulary size: {model_config.vocab_size}")

        # ========== 3. Check Model Components ==========
        print("\n3. Checking model components...")

        # Check embeddings
        print("   Checking embeddings:")
        print(f"     Token embeddings shape: {model.transformer.wte.weight.shape}")
        print(f"     Position embeddings shape: {model.transformer.wpe.weight.shape}")
        monitor.log_shape("token_embeddings", model.transformer.wte.weight)
        monitor.log_shape("position_embeddings", model.transformer.wpe.weight)

        # Check transformer blocks
        print("   Checking transformer blocks:")
        for i, block in enumerate(model.transformer.h):
            print(f"     Block {i}:")

            # Check attention
            attn = block.attn
            print(f"       Attention c_attn shape: {attn.c_attn.linear.weight.shape}")
            print(f"       Attention c_proj shape: {attn.c_proj.linear.weight.shape}")

            # Check LoRA adapters
            for bit_key in attn.c_attn.lora_adapters:
                lora_a = attn.c_attn.lora_adapters[bit_key]['A']
                lora_b = attn.c_attn.lora_adapters[bit_key]['B']
                print(f"       LoRA {bit_key} - A: {lora_a.shape}, B: {lora_b.shape}")
                monitor.log_shape(f"block_{i}_attn_lora_{bit_key}_A", lora_a)
                monitor.log_shape(f"block_{i}_attn_lora_{bit_key}_B", lora_b)

            # Check MLP
            mlp = block.mlp
            print(f"       MLP c_fc shape: {mlp.c_fc.linear.weight.shape}")
            print(f"       MLP c_proj shape: {mlp.c_proj.linear.weight.shape}")

        # Check LM head
        print(f"   LM head shape: {model.lm_head.weight.shape}")
        monitor.log_shape("lm_head", model.lm_head.weight)

        # ========== 4. Device Setup ==========
        print("\n4. Setting up device...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Using device: {device}")

        if device.type == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
            monitor.log_memory(0, "initial")

        model = model.to(device)
        monitor.log_memory(0, "model_loaded")

        # ========== 5. Test Forward Pass ==========
        print("\n5. Testing forward pass with different bit-widths...")

        # Create dummy input
        batch_size = 2
        seq_length = 32
        input_ids = torch.randint(0, model_config.vocab_size,
                                 (batch_size, seq_length)).to(device)
        labels = input_ids.clone()

        print(f"   Input shape: {input_ids.shape}")
        monitor.log_shape("input_ids", input_ids, step=0)

        for bits in model_config.bit_widths:
            print(f"\n   Testing {bits}-bit precision:")

            # Set precision
            old_bits = model.get_current_precision()
            model.set_precision(bits)
            monitor.log_precision_switch(0, old_bits, bits)

            # Forward pass
            with torch.no_grad():
                output = model(input_ids, labels=labels)

            print(f"     Loss: {output['loss'].item():.4f}")
            print(f"     Logits shape: {output['logits'].shape}")

            monitor.log_shape(f"logits_{bits}bit", output['logits'], step=0)
            monitor.log_shape(f"loss_{bits}bit", output['loss'], step=0)

            # Check for issues
            if torch.isnan(output['loss']):
                monitor.add_error(f"NaN loss detected at {bits}-bit precision")
            if torch.isinf(output['loss']):
                monitor.add_error(f"Inf loss detected at {bits}-bit precision")

            monitor.log_memory(0, f"forward_{bits}bit")

        # ========== 6. Test Backward Pass ==========
        print("\n6. Testing backward pass...")

        optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

        for i, bits in enumerate(model_config.bit_widths):
            print(f"\n   Testing backward with {bits}-bit:")

            model.set_precision(bits)
            optimizer.zero_grad()

            # Forward
            output = model(input_ids, labels=labels)
            loss = output['loss']

            # Backward
            loss.backward()

            # Check gradients
            total_grad_norm = 0
            num_params_with_grad = 0
            num_params_without_grad = 0

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm ** 2
                    num_params_with_grad += 1

                    # Log gradient info for key parameters
                    if any(key in name for key in ['lora', 'c_attn', 'c_fc', 'wte']):
                        monitor.log_shape(f"grad_{name}_{bits}bit", param.grad, step=i)

                    # Check for gradient issues
                    if torch.isnan(param.grad).any():
                        monitor.add_warning(f"NaN gradient in {name} at {bits}-bit")
                    if torch.isinf(param.grad).any():
                        monitor.add_warning(f"Inf gradient in {name} at {bits}-bit")
                else:
                    num_params_without_grad += 1

            total_grad_norm = (total_grad_norm ** 0.5)

            print(f"     Loss: {loss.item():.4f}")
            print(f"     Total gradient norm: {total_grad_norm:.4f}")
            print(f"     Parameters with gradients: {num_params_with_grad}")
            print(f"     Parameters without gradients: {num_params_without_grad}")

            monitor.log_training_step(i, loss.item(), bits,
                                     training_config.learning_rate, total_grad_norm)
            monitor.log_memory(i, f"backward_{bits}bit")

        # ========== 7. Test Training Loop ==========
        print("\n7. Testing training loop with precision switching...")

        model.train()
        current_bits = model_config.bit_widths[0]

        for iteration in range(10):  # Short test
            # Get next bit width
            old_bits = current_bits
            current_bits = get_next_bitwidth(iteration, model_config)

            if old_bits != current_bits:
                print(f"\n   Step {iteration}: Switching from {old_bits}-bit to {current_bits}-bit")
                monitor.log_precision_switch(iteration, old_bits, current_bits)

            model.set_precision(current_bits)

            # Training step
            optimizer.zero_grad()

            # Generate new batch
            input_ids = torch.randint(0, model_config.vocab_size,
                                    (batch_size, seq_length)).to(device)
            labels = input_ids.clone()

            # Forward
            output = model(input_ids, labels=labels)
            loss = output['loss']

            # Backward
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                training_config.max_grad_norm
            )

            # Optimizer step
            optimizer.step()

            print(f"   Step {iteration}: {current_bits}-bit, "
                  f"Loss: {loss.item():.4f}, Grad norm: {grad_norm:.4f}")

            monitor.log_training_step(iteration, loss.item(), current_bits,
                                     training_config.learning_rate, grad_norm)
            monitor.log_memory(iteration, f"train_step_{iteration}")

            # Periodic checks
            if iteration % 5 == 0:
                # Check weight statistics
                for name, param in model.named_parameters():
                    if 'weight' in name and 'lora' not in name:
                        monitor.log_shape(f"weight_{name}", param, step=iteration)

        # ========== 8. Final Checks ==========
        print("\n8. Final checks...")

        # Check final model state
        print("   Checking final model state:")
        model.eval()

        with torch.no_grad():
            # Test at each precision
            for bits in model_config.bit_widths:
                model.set_precision(bits)
                output = model(input_ids, labels=labels)
                print(f"     {bits}-bit final loss: {output['loss'].item():.4f}")

        # Memory cleanup
        if device.type == "cuda":
            print(f"\n   Final GPU memory: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")
            torch.cuda.empty_cache()
            gc.collect()
            print(f"   After cleanup: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")

        # Save monitoring results
        monitor.save_logs()
        monitor.print_summary()

        print(f"\nMonitoring logs saved to: {monitor.log_file}")

        return True

    except Exception as e:
        print(f"\n❌ Error during monitoring: {e}")
        monitor.add_error(str(e), traceback.format_exc())
        monitor.save_logs()
        monitor.print_summary()
        traceback.print_exc()
        return False


def analyze_training_issues(log_file="sp_training_detailed_monitor.json"):
    """Analyze training logs for potential issues."""

    print("\n" + "="*80)
    print("TRAINING ISSUE ANALYSIS")
    print("="*80)

    try:
        with open(log_file, 'r') as f:
            logs = json.load(f)
    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
        print("Please run monitor_sp_training_step_by_step() first")
        return

    issues_found = []

    # 1. Check for NaN/Inf
    print("\n1. Checking for NaN/Inf values...")
    nan_count = 0
    inf_count = 0

    for name, shapes in logs.get("shape_tracking", {}).items():
        for shape_info in shapes:
            if shape_info.get("has_nan"):
                nan_count += 1
                issues_found.append(f"NaN in {name}")
            if shape_info.get("has_inf"):
                inf_count += 1
                issues_found.append(f"Inf in {name}")

    if nan_count > 0:
        print(f"   ⚠ Found {nan_count} NaN occurrences")
    else:
        print("   ✓ No NaN values detected")

    if inf_count > 0:
        print(f"   ⚠ Found {inf_count} Inf occurrences")
    else:
        print("   ✓ No Inf values detected")

    # 2. Check loss progression
    print("\n2. Checking loss progression...")
    training_steps = logs.get("training_steps", [])

    if training_steps:
        losses = [step["loss"] for step in training_steps]

        # Check for loss explosion
        if max(losses) > min(losses) * 100:
            print("   ⚠ Large loss variation detected (possible instability)")
            issues_found.append("Large loss variation")
        else:
            print("   ✓ Loss variation is reasonable")

        # Check for loss plateau
        if len(losses) > 5:
            recent_losses = losses[-5:]
            if max(recent_losses) - min(recent_losses) < 0.001:
                print("   ⚠ Loss appears to have plateaued")
                issues_found.append("Loss plateau")
            else:
                print("   ✓ Loss is still changing")

    # 3. Check gradient norms
    print("\n3. Checking gradient norms...")
    grad_norms = [step["grad_norm"] for step in training_steps if step.get("grad_norm")]

    if grad_norms:
        max_grad = max(grad_norms)
        avg_grad = sum(grad_norms) / len(grad_norms)

        print(f"   Max gradient norm: {max_grad:.4f}")
        print(f"   Average gradient norm: {avg_grad:.4f}")

        if max_grad > 100:
            print("   ⚠ Very large gradients detected")
            issues_found.append("Large gradients")
        elif max_grad < 0.0001:
            print("   ⚠ Very small gradients detected (vanishing gradients)")
            issues_found.append("Vanishing gradients")
        else:
            print("   ✓ Gradient norms are in reasonable range")

    # 4. Check memory usage
    print("\n4. Checking memory usage...")
    memory_tracking = logs.get("memory_tracking", [])

    if memory_tracking:
        gpu_memories = [m["gpu"]["allocated_mb"] for m in memory_tracking if "gpu" in m]
        if gpu_memories:
            max_memory = max(gpu_memories)
            memory_growth = gpu_memories[-1] - gpu_memories[0] if len(gpu_memories) > 1 else 0

            print(f"   Peak GPU memory: {max_memory:.1f} MB")
            print(f"   Memory growth: {memory_growth:.1f} MB")

            if memory_growth > 100:
                print("   ⚠ Significant memory growth detected (possible memory leak)")
                issues_found.append("Memory growth")
            else:
                print("   ✓ Memory usage is stable")

    # 5. Check precision switching
    print("\n5. Checking precision switching...")
    switches = logs.get("precision_switches", [])

    if switches:
        print(f"   Total switches: {len(switches)}")
        switch_pattern = [(s["old_bits"], s["new_bits"]) for s in switches]
        print(f"   Switch pattern: {switch_pattern[:5]}...")
        print("   ✓ Precision switching is working")
    else:
        print("   ⚠ No precision switches recorded")

    # 6. Summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)

    if issues_found:
        print("\n⚠ Issues found:")
        for issue in set(issues_found):
            print(f"  - {issue}")
    else:
        print("\n✅ No major issues detected!")

    # Recommendations
    if issues_found:
        print("\nRecommendations:")
        if "NaN in" in str(issues_found) or "Inf in" in str(issues_found):
            print("  - Check learning rate (try reducing it)")
            print("  - Check weight initialization")
            print("  - Enable gradient clipping")
        if "Large gradients" in issues_found:
            print("  - Reduce learning rate")
            print("  - Increase gradient clipping threshold")
        if "Vanishing gradients" in issues_found:
            print("  - Increase learning rate")
            print("  - Check model architecture")
            print("  - Consider using different activation functions")
        if "Memory growth" in issues_found:
            print("  - Check for retained computation graphs")
            print("  - Ensure proper tensor cleanup")
            print("  - Use torch.cuda.empty_cache() periodically")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Monitor SP Training')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze existing logs instead of running training')
    parser.add_argument('--log-file', type=str, default='sp_training_detailed_monitor.json',
                       help='Log file to analyze')

    args = parser.parse_args()

    if args.analyze:
        analyze_training_issues(args.log_file)
    else:
        success = monitor_sp_training_step_by_step()
        if success:
            print("\n✅ Monitoring completed successfully!")
            print("\nRun with --analyze flag to analyze the results:")
            print("  python monitor_sp_training.py --analyze")
        else:
            print("\n❌ Monitoring encountered errors!")
            print("Check the log file for details")
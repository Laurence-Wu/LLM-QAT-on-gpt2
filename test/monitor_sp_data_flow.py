#!/usr/bin/env python3
"""
SP Training Data Flow Monitor
Tracks data flow through the model during training, monitoring shapes at each layer
"""

import sys
import os
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2Config
from shared.models_sp import SPModel, SPLMHeadModel, SPBlock, SPAttention, SPMLP
from part1_switchable_precision.config_sp import ModelConfig


class DataFlowMonitor:
    """Monitor data flow through SP model layers."""

    def __init__(self):
        self.flow_log = {
            "timestamp": datetime.now().isoformat(),
            "layer_inputs": {},
            "layer_outputs": {},
            "attention_patterns": {},
            "activation_stats": {},
            "precision_effects": {}
        }
        self.hooks = []

    def register_hooks(self, model):
        """Register forward hooks on all layers."""

        def create_hook(layer_name):
            def hook(module, input, output):
                self._log_layer_flow(layer_name, module, input, output)
            return hook

        # Register hooks on key layers
        # Embeddings
        self.hooks.append(
            model.transformer.wte.register_forward_hook(create_hook("token_embedding"))
        )
        self.hooks.append(
            model.transformer.wpe.register_forward_hook(create_hook("position_embedding"))
        )

        # Transformer blocks
        for i, block in enumerate(model.transformer.h):
            # Block level
            self.hooks.append(
                block.register_forward_hook(create_hook(f"block_{i}"))
            )

            # Attention
            self.hooks.append(
                block.attn.register_forward_hook(create_hook(f"block_{i}_attention"))
            )

            # Attention components
            self.hooks.append(
                block.attn.c_attn.register_forward_hook(create_hook(f"block_{i}_attn_c_attn"))
            )
            self.hooks.append(
                block.attn.c_proj.register_forward_hook(create_hook(f"block_{i}_attn_c_proj"))
            )

            # MLP
            self.hooks.append(
                block.mlp.register_forward_hook(create_hook(f"block_{i}_mlp"))
            )

            # MLP components
            self.hooks.append(
                block.mlp.c_fc.register_forward_hook(create_hook(f"block_{i}_mlp_c_fc"))
            )
            self.hooks.append(
                block.mlp.c_proj.register_forward_hook(create_hook(f"block_{i}_mlp_c_proj"))
            )

            # Layer norms
            self.hooks.append(
                block.ln_1.register_forward_hook(create_hook(f"block_{i}_ln_1"))
            )
            self.hooks.append(
                block.ln_2.register_forward_hook(create_hook(f"block_{i}_ln_2"))
            )

        # Final layer norm
        self.hooks.append(
            model.transformer.ln_f.register_forward_hook(create_hook("final_layer_norm"))
        )

        # LM head
        self.hooks.append(
            model.lm_head.register_forward_hook(create_hook("lm_head"))
        )

    def _log_layer_flow(self, layer_name: str, module: nn.Module,
                       input: Tuple[torch.Tensor], output: torch.Tensor):
        """Log data flow through a layer."""

        # Handle different input types
        if isinstance(input, tuple):
            input_tensor = input[0] if len(input) > 0 else None
        else:
            input_tensor = input

        # Handle different output types
        if isinstance(output, tuple):
            output_tensor = output[0] if len(output) > 0 else None
        elif isinstance(output, dict):
            output_tensor = output.get('hidden_states', output.get('logits'))
        else:
            output_tensor = output

        # Log input shape and stats
        if input_tensor is not None and isinstance(input_tensor, torch.Tensor):
            if layer_name not in self.flow_log["layer_inputs"]:
                self.flow_log["layer_inputs"][layer_name] = []

            input_stats = self._get_tensor_stats(input_tensor)
            self.flow_log["layer_inputs"][layer_name].append(input_stats)

        # Log output shape and stats
        if output_tensor is not None and isinstance(output_tensor, torch.Tensor):
            if layer_name not in self.flow_log["layer_outputs"]:
                self.flow_log["layer_outputs"][layer_name] = []

            output_stats = self._get_tensor_stats(output_tensor)
            self.flow_log["layer_outputs"][layer_name].append(output_stats)

        # Log activation patterns for attention layers
        if "attention" in layer_name and hasattr(module, 'last_attn_weights'):
            if layer_name not in self.flow_log["attention_patterns"]:
                self.flow_log["attention_patterns"][layer_name] = []

            attn_stats = self._get_attention_stats(module.last_attn_weights)
            self.flow_log["attention_patterns"][layer_name].append(attn_stats)

    def _get_tensor_stats(self, tensor: torch.Tensor) -> Dict:
        """Get statistics for a tensor."""
        with torch.no_grad():
            stats = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "min": float(tensor.min().item()) if tensor.numel() > 0 else None,
                "max": float(tensor.max().item()) if tensor.numel() > 0 else None,
                "mean": float(tensor.mean().item()) if tensor.numel() > 0 else None,
                "std": float(tensor.std().item()) if tensor.numel() > 0 else None,
                "abs_mean": float(tensor.abs().mean().item()) if tensor.numel() > 0 else None,
                "sparsity": float((tensor == 0).sum().item() / tensor.numel()) if tensor.numel() > 0 else None,
                "has_nan": bool(torch.isnan(tensor).any().item()),
                "has_inf": bool(torch.isinf(tensor).any().item()),
                "grad_ready": tensor.requires_grad
            }

            # Add quantiles for better distribution understanding
            if tensor.numel() > 0:
                flat_tensor = tensor.flatten()
                stats["quantiles"] = {
                    "q01": float(torch.quantile(flat_tensor, 0.01).item()),
                    "q25": float(torch.quantile(flat_tensor, 0.25).item()),
                    "q50": float(torch.quantile(flat_tensor, 0.50).item()),
                    "q75": float(torch.quantile(flat_tensor, 0.75).item()),
                    "q99": float(torch.quantile(flat_tensor, 0.99).item()),
                }

        return stats

    def _get_attention_stats(self, attn_weights: torch.Tensor) -> Dict:
        """Get statistics for attention weights."""
        with torch.no_grad():
            stats = {
                "shape": list(attn_weights.shape),
                "max_attention": float(attn_weights.max().item()),
                "min_attention": float(attn_weights.min().item()),
                "mean_attention": float(attn_weights.mean().item()),
                "attention_entropy": float(-(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1).mean().item())
            }

            # Find most attended positions
            if len(attn_weights.shape) >= 2:
                max_attn_pos = attn_weights.mean(dim=0).argmax(dim=-1)
                stats["most_attended_positions"] = max_attn_pos.tolist() if max_attn_pos.numel() <= 10 else "too_many"

        return stats

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def analyze_flow(self):
        """Analyze the logged data flow."""
        analysis = {
            "shape_consistency": self._check_shape_consistency(),
            "activation_health": self._check_activation_health(),
            "layer_statistics": self._compute_layer_statistics(),
            "bottlenecks": self._identify_bottlenecks()
        }
        return analysis

    def _check_shape_consistency(self) -> Dict:
        """Check if shapes are consistent across forward passes."""
        consistency = {}

        for layer_name, outputs in self.flow_log["layer_outputs"].items():
            if outputs:
                shapes = [o["shape"] for o in outputs]
                unique_shapes = list(set(map(tuple, shapes)))

                consistency[layer_name] = {
                    "consistent": len(unique_shapes) == 1,
                    "unique_shapes": [list(s) for s in unique_shapes]
                }

        return consistency

    def _check_activation_health(self) -> Dict:
        """Check for issues in activations."""
        health = {}

        for layer_name, outputs in self.flow_log["layer_outputs"].items():
            if outputs:
                issues = []

                # Check for NaN/Inf
                if any(o["has_nan"] for o in outputs):
                    issues.append("contains_nan")
                if any(o["has_inf"] for o in outputs):
                    issues.append("contains_inf")

                # Check for dead neurons (high sparsity)
                avg_sparsity = sum(o["sparsity"] for o in outputs) / len(outputs)
                if avg_sparsity > 0.9:
                    issues.append("high_sparsity")

                # Check for saturation
                if outputs[0]["mean"] is not None:
                    means = [o["mean"] for o in outputs]
                    if all(abs(m) < 1e-6 for m in means):
                        issues.append("near_zero_activations")

                health[layer_name] = {
                    "healthy": len(issues) == 0,
                    "issues": issues,
                    "avg_sparsity": avg_sparsity
                }

        return health

    def _compute_layer_statistics(self) -> Dict:
        """Compute statistics for each layer."""
        statistics = {}

        for layer_name, outputs in self.flow_log["layer_outputs"].items():
            if outputs and outputs[0]["mean"] is not None:
                statistics[layer_name] = {
                    "mean_magnitude": sum(abs(o["mean"]) for o in outputs) / len(outputs),
                    "mean_std": sum(o["std"] for o in outputs) / len(outputs),
                    "activation_range": sum(o["max"] - o["min"] for o in outputs) / len(outputs)
                }

        return statistics

    def _identify_bottlenecks(self) -> List[str]:
        """Identify potential bottlenecks in the network."""
        bottlenecks = []

        stats = self._compute_layer_statistics()

        for layer_name, layer_stats in stats.items():
            # Check for vanishing activations
            if layer_stats["mean_magnitude"] < 1e-4:
                bottlenecks.append(f"{layer_name}: vanishing_activations")

            # Check for exploding activations
            if layer_stats["mean_magnitude"] > 1e4:
                bottlenecks.append(f"{layer_name}: exploding_activations")

            # Check for low variance
            if layer_stats["mean_std"] < 1e-5:
                bottlenecks.append(f"{layer_name}: low_variance")

        return bottlenecks

    def save_log(self, filename="sp_data_flow.json"):
        """Save the flow log to file."""
        # Convert any remaining tensors to lists
        def convert_tensors(obj):
            if torch.is_tensor(obj):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(v) for v in obj]
            return obj

        serializable_log = convert_tensors(self.flow_log)

        with open(filename, 'w') as f:
            json.dump(serializable_log, f, indent=2, default=str)

    def print_summary(self):
        """Print a summary of the data flow."""
        print("\n" + "="*80)
        print("DATA FLOW SUMMARY")
        print("="*80)

        # Shape information
        print("\n📐 Layer Output Shapes:")
        for layer_name, outputs in self.flow_log["layer_outputs"].items():
            if outputs:
                shape = outputs[0]["shape"]
                print(f"  {layer_name}: {shape}")

        # Health check
        health = self._check_activation_health()
        unhealthy = [name for name, h in health.items() if not h["healthy"]]

        if unhealthy:
            print("\n⚠ Unhealthy Layers:")
            for layer in unhealthy:
                issues = health[layer]["issues"]
                print(f"  {layer}: {', '.join(issues)}")
        else:
            print("\n✅ All layers healthy")

        # Statistics
        stats = self._compute_layer_statistics()
        if stats:
            print("\n📊 Layer Statistics (averaged):")
            for layer_name, layer_stats in list(stats.items())[:5]:  # Show first 5
                print(f"  {layer_name}:")
                print(f"    Mean magnitude: {layer_stats['mean_magnitude']:.6f}")
                print(f"    Mean std: {layer_stats['mean_std']:.6f}")

        # Bottlenecks
        bottlenecks = self._identify_bottlenecks()
        if bottlenecks:
            print("\n🚧 Potential Bottlenecks:")
            for bottleneck in bottlenecks:
                print(f"  {bottleneck}")

        print("\n" + "="*80)


def monitor_training_data_flow():
    """Monitor data flow through SP model during training."""

    print("\n" + "="*80)
    print("MONITORING SP TRAINING DATA FLOW")
    print("="*80)

    # Setup configuration
    model_config = ModelConfig()
    model_config.n_layer = 2
    model_config.n_embd = 256
    model_config.n_head = 4
    model_config.vocab_size = 1000

    # Create model
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create monitor and register hooks
    monitor = DataFlowMonitor()
    monitor.register_hooks(model)

    print("✓ Hooks registered on all layers")

    # Test with different precisions
    batch_size = 2
    seq_length = 32
    input_ids = torch.randint(0, model_config.vocab_size,
                             (batch_size, seq_length)).to(device)
    labels = input_ids.clone()

    for bits in model_config.bit_widths:
        print(f"\n📊 Testing {bits}-bit precision...")
        model.set_precision(bits)

        # Forward pass
        with torch.no_grad():
            output = model(input_ids, labels=labels)
            loss = output['loss']

        print(f"  Loss: {loss.item():.4f}")

        # Store precision-specific effects
        monitor.flow_log["precision_effects"][f"{bits}bit"] = {
            "loss": float(loss.item()),
            "timestamp": datetime.now().isoformat()
        }

    # Analyze and save
    analysis = monitor.analyze_flow()

    print("\n🔍 Analysis Results:")
    print(f"  Shape consistency: {sum(1 for v in analysis['shape_consistency'].values() if v['consistent'])} / {len(analysis['shape_consistency'])} layers")
    print(f"  Healthy layers: {sum(1 for v in analysis['activation_health'].values() if v['healthy'])} / {len(analysis['activation_health'])}")
    print(f"  Bottlenecks found: {len(analysis['bottlenecks'])}")

    # Clean up hooks
    monitor.remove_hooks()

    # Save and summarize
    monitor.save_log("sp_data_flow_monitor.json")
    monitor.print_summary()

    print(f"\n💾 Data flow log saved to: sp_data_flow_monitor.json")

    return monitor


def visualize_layer_flow(log_file="sp_data_flow_monitor.json"):
    """Visualize the data flow from the log file."""

    print("\n" + "="*80)
    print("LAYER FLOW VISUALIZATION")
    print("="*80)

    try:
        with open(log_file, 'r') as f:
            flow_log = json.load(f)
    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
        return

    # Create flow diagram
    print("\n🔄 Data Flow Diagram:")
    print("\n[Input]")

    layers_in_order = [
        "token_embedding",
        "position_embedding",
        "block_0",
        "block_0_ln_1",
        "block_0_attention",
        "block_0_ln_2",
        "block_0_mlp",
        "block_1",
        "block_1_ln_1",
        "block_1_attention",
        "block_1_ln_2",
        "block_1_mlp",
        "final_layer_norm",
        "lm_head"
    ]

    for layer in layers_in_order:
        if layer in flow_log["layer_outputs"]:
            outputs = flow_log["layer_outputs"][layer]
            if outputs:
                shape = outputs[0]["shape"]
                mean = outputs[0]["mean"]
                print(f"   ↓")
                print(f"[{layer}]")
                print(f"  Shape: {shape}")
                if mean is not None:
                    print(f"  Mean: {mean:.6f}")

    print("   ↓")
    print("[Output]")

    # Show precision effects
    if "precision_effects" in flow_log:
        print("\n🎯 Precision Effects:")
        for precision, effects in flow_log["precision_effects"].items():
            print(f"  {precision}: Loss = {effects['loss']:.4f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Monitor SP Training Data Flow')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize existing data flow log')
    parser.add_argument('--log-file', type=str, default='sp_data_flow_monitor.json',
                       help='Log file for visualization')

    args = parser.parse_args()

    if args.visualize:
        visualize_layer_flow(args.log_file)
    else:
        monitor = monitor_training_data_flow()
        print("\n✅ Data flow monitoring completed!")
        print("\nRun with --visualize flag to see the flow diagram:")
        print("  python monitor_sp_data_flow.py --visualize")
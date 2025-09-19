#!/usr/bin/env python3
"""
Visualize Backpropagation Graph for Switchable Precision Training
Shows the gradient flow through quantized layers, LoRA adapters, and distillation components
"""

import sys
import os
import torch
import torch.nn as nn
from graphviz import Digraph
from collections import defaultdict
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2Config
from shared.models_sp import SPLMHeadModel
from part1_switchable_precision.config_sp import ModelConfig, TrainingConfig
from part1_switchable_precision.distillation import DistillationConfig, SelfDistillationTrainer


def make_dot(var, params=None, show_attrs=False, show_saved=False):
    """
    Create a graphviz Digraph object to visualize the computation graph.

    Args:
        var: Output tensor to trace back from
        params: Dict of parameter names (optional)
        show_attrs: Show tensor attributes
        show_saved: Show saved tensors for backward
    """
    if params is not None:
        params = {id(v): k for k, v in params.items()}
    else:
        params = {}

    node_attr = dict(style='filled', shape='box', align='left', fontsize='10',
                    ranksep='0.1', height='0.2', fontname='monospace')

    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12", rankdir='TB'))
    seen = set()

    def size_to_str(size):
        return '×'.join(['%d' % v for v in size])

    def get_var_name(var, name=None):
        if not name:
            name = params.get(id(var), '')
        return '%s\n %s' % (name, size_to_str(var.size()))

    def add_nodes(var, name=None, _parent=None):
        if var not in seen:
            node_id = str(id(var))

            # Determine node properties
            if torch.is_tensor(var):
                # Color code by properties
                if var.requires_grad:
                    fillcolor = 'lightblue'
                elif var.is_leaf:
                    fillcolor = 'lightgreen'
                else:
                    fillcolor = 'white'

                node_name = get_var_name(var, name)

                # Add additional info
                info = []
                if show_attrs:
                    if var.grad_fn:
                        info.append(f"grad_fn: {type(var.grad_fn).__name__}")
                    info.append(f"dtype: {var.dtype}")
                    info.append(f"device: {var.device}")
                    if var.grad is not None:
                        info.append(f"grad: {size_to_str(var.grad.size())}")

                if info:
                    node_name += '\n' + '\n'.join(info)

                dot.node(node_id, node_name, fillcolor=fillcolor)
            else:
                # For grad_fn nodes
                fillcolor = 'orange'
                if hasattr(var, '__name__'):
                    node_name = var.__name__
                else:
                    node_name = str(type(var).__name__)

                dot.node(node_id, node_name, fillcolor=fillcolor)

            seen.add(var)

            # Add edges
            if hasattr(var, 'grad_fn'):
                add_nodes(var.grad_fn, _parent=var)

            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        if not (torch.is_tensor(u[0]) and u[0].is_leaf):
                            add_nodes(u[0], _parent=var)
                            edge_id = str(id(u[0]))
                            if edge_id in [str(id(n)) for n in seen]:
                                dot.edge(edge_id, node_id)

            # Handle saved tensors
            if show_saved and hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    add_nodes(t, f'saved_{size_to_str(t.size())}', _parent=var)

    # Start from the variable
    add_nodes(var.grad_fn if hasattr(var, 'grad_fn') else var)
    return dot


def trace_gradient_flow(model, loss, print_details=True):
    """
    Trace and print the gradient flow through the model.

    Args:
        model: The model being trained
        loss: The loss tensor with grad_fn
        print_details: Whether to print detailed gradient information
    """
    print("\n" + "="*80)
    print("GRADIENT FLOW ANALYSIS")
    print("="*80)

    # Build parameter name mapping
    param_map = {}
    for name, param in model.named_parameters():
        param_map[id(param)] = name

    # Analyze gradient flow
    grad_fn = loss.grad_fn

    print("\n1. LOSS BACKWARD GRAPH:")
    print(f"   Loss value: {loss.item():.4f}")
    print(f"   Root grad_fn: {grad_fn}")

    # Recursively trace grad_fn
    visited = set()
    grad_fn_tree = []

    def trace_grad_fn(fn, depth=0, max_depth=10):
        if fn is None or depth > max_depth or id(fn) in visited:
            return

        visited.add(id(fn))
        indent = "  " * depth
        fn_name = type(fn).__name__

        # Track the gradient function
        grad_fn_tree.append((depth, fn_name))

        if print_details and depth < 5:  # Limit printing depth
            print(f"{indent}└─ {fn_name}")

        # Check for specific operations
        if 'Quantization' in fn_name:
            print(f"{indent}   ⚡ QUANTIZATION DETECTED (uses STE)")
        elif 'Round' in fn_name:
            print(f"{indent}   ⚡ ROUND OPERATION (non-differentiable, needs STE)")
        elif 'LoRA' in fn_name or 'Lora' in fn_name:
            print(f"{indent}   🔧 LoRA ADAPTER")

        if hasattr(fn, 'next_functions'):
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    trace_grad_fn(next_fn, depth + 1, max_depth)

    trace_grad_fn(grad_fn)

    # Analyze gradient flow patterns
    print("\n2. GRADIENT FLOW PATTERNS:")

    # Count operation types
    op_counts = defaultdict(int)
    for _, op_name in grad_fn_tree:
        op_counts[op_name] += 1

    print("\n   Operation frequency:")
    for op_name, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"     {op_name}: {count}")

    # Check for STE usage
    ste_ops = [op for _, op in grad_fn_tree if 'Backward' in op and 'Quantization' in op]
    if ste_ops:
        print(f"\n   ✓ STE (Straight-Through Estimator) detected: {len(ste_ops)} quantization operations")

    return grad_fn_tree


def analyze_sp_backprop():
    """
    Main function to analyze backpropagation in switchable precision training.
    """
    print("\n" + "="*80)
    print("SWITCHABLE PRECISION BACKPROPAGATION GRAPH ANALYSIS")
    print("="*80)

    # Create configuration
    model_config = ModelConfig()
    training_config = TrainingConfig()

    # Small model for visualization
    model_config.n_layer = 2
    model_config.n_embd = 128
    model_config.n_head = 4

    # Create model
    gpt2_config = GPT2Config(
        vocab_size=1000,
        n_positions=256,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head
    )

    gpt2_config.bit_widths = model_config.bit_widths
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SPLMHeadModel(gpt2_config).to(device)

    # Create distillation trainer
    distill_config = DistillationConfig(
        use_distillation=training_config.use_distillation,
        alpha_output=training_config.distillation_alpha_output,
        alpha_feature=training_config.distillation_alpha_feature,
        temperature=training_config.distillation_temperature
    )

    distillation_trainer = SelfDistillationTrainer(model, distill_config, device)

    # Create dummy input
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    print("\n3. ANALYZING DIFFERENT PRECISION MODES:")

    for bits in [16, 8, 4]:
        print(f"\n" + "-"*60)
        print(f"PRECISION: {bits}-bit")
        print("-"*60)

        # Set precision
        model.set_precision(bits)

        # Forward pass
        outputs = model(input_ids, output_hidden_states=True, return_dict=True)

        # Compute loss (with or without distillation)
        if training_config.use_distillation:
            loss, components = distillation_trainer.compute_distillation_loss(
                outputs, labels, input_ids
            )

            print(f"\nLoss components:")
            for key, value in components.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
        else:
            # Standard loss
            loss = outputs['loss'] if 'loss' in outputs else \
                   nn.functional.cross_entropy(
                       outputs['logits'][..., :-1, :].contiguous().view(-1, outputs['logits'].size(-1)),
                       labels[..., 1:].contiguous().view(-1)
                   )

        # Trace gradient flow
        print(f"\nGradient flow for {bits}-bit precision:")
        grad_fn_tree = trace_gradient_flow(model, loss, print_details=(bits == 8))

        # Backward pass to check gradient flow
        loss.backward(retain_graph=True)

        # Check gradient statistics
        print(f"\n4. GRADIENT STATISTICS ({bits}-bit):")

        grad_stats = {
            'lora': {'count': 0, 'mean': [], 'max': []},
            'quantizer': {'count': 0, 'mean': [], 'max': []},
            'base': {'count': 0, 'mean': [], 'max': []}
        }

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()

                if 'lora' in name.lower():
                    grad_stats['lora']['count'] += 1
                    grad_stats['lora']['mean'].append(grad_norm)
                    grad_stats['lora']['max'].append(param.grad.abs().max().item())
                elif 'quant' in name.lower():
                    grad_stats['quantizer']['count'] += 1
                    grad_stats['quantizer']['mean'].append(grad_norm)
                    grad_stats['quantizer']['max'].append(param.grad.abs().max().item())
                else:
                    grad_stats['base']['count'] += 1
                    grad_stats['base']['mean'].append(grad_norm)
                    grad_stats['base']['max'].append(param.grad.abs().max().item())

        for component, stats in grad_stats.items():
            if stats['count'] > 0:
                print(f"\n  {component.upper()} parameters:")
                print(f"    Count: {stats['count']}")
                print(f"    Mean gradient norm: {np.mean(stats['mean']):.6f}")
                print(f"    Max gradient value: {np.max(stats['max']):.6f}")

        # Zero gradients for next iteration
        model.zero_grad()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    print("\nKEY FINDINGS:")
    print("• STE enables gradient flow through quantization operations")
    print("• LoRA adapters receive gradients at all precision levels")
    print("• Distillation adds additional gradient paths for low-precision modes")
    print("• Gradient magnitudes vary with precision (typically lower for lower bits)")


def visualize_computation_graph():
    """
    Create a visual representation of the computation graph.
    """
    print("\n" + "="*80)
    print("CREATING COMPUTATION GRAPH VISUALIZATION")
    print("="*80)

    # Create small model for cleaner visualization
    model_config = ModelConfig()
    model_config.n_layer = 1  # Single layer for clarity
    model_config.n_embd = 64
    model_config.n_head = 2

    gpt2_config = GPT2Config(
        vocab_size=100,
        n_positions=128,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head
    )

    gpt2_config.bit_widths = [8]  # Single bit-width for simplicity
    gpt2_config.lora_rank_per_bit = {8: 4}
    gpt2_config.lora_alpha_per_bit = {8: 8}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SPLMHeadModel(gpt2_config).to(device)
    model.set_precision(8)

    # Small input
    input_ids = torch.randint(0, 100, (1, 16), device=device)
    labels = input_ids.clone()

    # Forward pass
    outputs = model(input_ids, labels=labels, return_dict=True)
    loss = outputs['loss']

    # Create visualization
    params = dict(model.named_parameters())
    dot = make_dot(loss, params, show_attrs=True, show_saved=False)

    # Save to file
    output_file = 'sp_backprop_graph'
    try:
        dot.render(output_file, format='png', cleanup=True)
        print(f"\n✓ Computation graph saved to {output_file}.png")
    except Exception as e:
        print(f"\n⚠️ Could not save graph image: {e}")
        print("  (Install graphviz to generate the image)")

    # Save as text representation
    with open(f"{output_file}.dot", 'w') as f:
        f.write(dot.source)
    print(f"✓ Graph source saved to {output_file}.dot")

    return dot


def print_model_graph_structure():
    """
    Print the model structure focusing on gradient flow paths.
    """
    print("\n" + "="*80)
    print("MODEL STRUCTURE FOR GRADIENT FLOW")
    print("="*80)

    model_config = ModelConfig()
    training_config = TrainingConfig()

    # Create model
    gpt2_config = GPT2Config(
        vocab_size=1000,
        n_positions=256,
        n_embd=128,
        n_layer=2,
        n_head=4
    )

    gpt2_config.bit_widths = model_config.bit_widths
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SPLMHeadModel(gpt2_config).to(device)

    print("\nMODEL HIERARCHY:")

    def print_module_tree(module, name='model', depth=0, max_depth=3):
        if depth > max_depth:
            return

        indent = "  " * depth

        # Identify special modules
        markers = []
        if 'SPLinearWithLoRA' in module.__class__.__name__:
            markers.append("🔄 SWITCHABLE")
        if 'LoRA' in module.__class__.__name__:
            markers.append("🔧 LoRA")
        if 'Quantize' in module.__class__.__name__:
            markers.append("📊 QUANTIZE")

        marker_str = " ".join(markers) if markers else ""

        print(f"{indent}├─ {name}: {module.__class__.__name__} {marker_str}")

        # Print parameters if they exist
        params = list(module.parameters(recurse=False))
        if params:
            for param in params:
                if param.requires_grad:
                    print(f"{indent}│  └─ param: {tuple(param.shape)} [trainable]")
                else:
                    print(f"{indent}│  └─ param: {tuple(param.shape)} [frozen]")

        # Recurse through children
        for child_name, child_module in module.named_children():
            print_module_tree(child_module, child_name, depth + 1, max_depth)

    print_module_tree(model)

    print("\n" + "="*80)
    print("GRADIENT FLOW PATHS:")
    print("="*80)

    print("\n1. STANDARD FORWARD (16-bit, teacher mode):")
    print("   Input → Embeddings → Transformer Blocks → LM Head → Cross-Entropy Loss")

    print("\n2. DISTILLED FORWARD (8-bit or 4-bit, student mode):")
    print("   Input → Embeddings → Quantized Weights → LoRA Adapters →")
    print("   → Transformer Blocks → LM Head → KL Loss + Feature Loss")

    print("\n3. BACKWARD (via STE):")
    print("   Loss → LM Head → Transformer Blocks →")
    print("   → LoRA Adapters (gradients) → Quantizers (STE) → Base Weights (if not frozen)")

    print("\n4. KEY GRADIENT PATHS:")
    print("   • LoRA A & B matrices: Direct gradients")
    print("   • Quantizer scales: Learnable parameters with gradients")
    print("   • Base weights: Gradients if unfrozen, otherwise no gradient")
    print("   • Quantization operation: STE passes gradients through")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SWITCHABLE PRECISION BACKPROPAGATION ANALYSIS")
    print("="*80)

    # Run analyses
    print("\nSelect analysis to run:")
    print("1. Full backpropagation analysis")
    print("2. Model structure for gradient flow")
    print("3. Computation graph visualization")
    print("4. All analyses")

    choice = input("\nEnter choice (1-4): ").strip() or "4"

    if choice == "1" or choice == "4":
        analyze_sp_backprop()

    if choice == "2" or choice == "4":
        print_model_graph_structure()

    if choice == "3" or choice == "4":
        visualize_computation_graph()

    print("\n✅ Analysis complete!")
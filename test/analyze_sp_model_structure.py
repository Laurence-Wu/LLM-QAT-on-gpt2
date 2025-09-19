#!/usr/bin/env python3
"""
Analyze SP Model Structure with Distillation Support
Comprehensive analysis of the Switchable Precision model architecture including distillation components
"""

import sys
import os
import torch
import torch.nn as nn
from transformers import GPT2Config

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models_sp import SPLMHeadModel, SPModel, SPBlock, SPAttention, SPMLP
from part1_switchable_precision.config_sp import ModelConfig, TrainingConfig
from part1_switchable_precision.distillation import DistillationConfig, SelfDistillationTrainer


def analyze_model_structure():
    """Analyze the SP model structure in detail."""
    print("=" * 80)
    print("SP MODEL STRUCTURE ANALYSIS")
    print("=" * 80)

    # Create model configuration
    model_config = ModelConfig()

    print(f"\n1. MODEL CONFIGURATION:")
    print(f"   Vocabulary size: {model_config.vocab_size:,}")
    print(f"   Embedding dimension: {model_config.n_embd}")
    print(f"   Number of layers: {model_config.n_layer}")
    print(f"   Number of heads: {model_config.n_head}")
    print(f"   Max positions: {model_config.n_positions}")
    print(f"   Supported bit widths: {model_config.bit_widths}")
    print(f"   LoRA rank per bit: {model_config.lora_rank_per_bit}")
    print(f"   LoRA alpha per bit: {model_config.lora_alpha_per_bit}")

    # Display distillation configuration
    training_config = TrainingConfig()
    print(f"\n   DISTILLATION CONFIGURATION:")
    print(f"   Distillation enabled: {training_config.use_distillation}")
    print(f"   KL loss weight (α₁): {training_config.distillation_alpha_output}")
    print(f"   Feature loss weight (α₂): {training_config.distillation_alpha_feature}")
    print(f"   Temperature: {training_config.distillation_temperature}")
    print(f"   Teacher update freq: {training_config.teacher_update_freq}")
    print(f"   Warmup steps: {training_config.distillation_warmup}")
    print(f"   Cache size: {training_config.distillation_cache_size}")

    # Create GPT-2 config
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        layer_norm_epsilon=model_config.layer_norm_epsilon,
        embd_pdrop=model_config.embd_pdrop
    )

    # Add SP-specific configs
    gpt2_config.bit_widths = model_config.bit_widths
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit

    # Create model
    print(f"\n2. CREATING SP MODEL...")
    model = SPLMHeadModel(gpt2_config)

    print(f"\n3. MODEL HIERARCHY:")
    print(f"   SPLMHeadModel")
    print(f"   ├── transformer (SPModel)")
    print(f"   │   ├── wte (Token Embeddings): {model.transformer.wte}")
    print(f"   │   ├── wpe (Position Embeddings): {model.transformer.wpe}")
    print(f"   │   ├── drop (Dropout): {model.transformer.drop}")
    print(f"   │   ├── h (Transformer Blocks): {len(model.transformer.h)} blocks")
    print(f"   │   └── ln_f (Final LayerNorm): {model.transformer.ln_f}")
    print(f"   └── lm_head (Language Model Head): {model.lm_head}")

    # Weight tying check
    is_tied = model.lm_head.weight is model.transformer.wte.weight
    print(f"\n4. WEIGHT TYING:")
    print(f"   LM head tied to embeddings: {'✓ Yes' if is_tied else '✗ No'}")

    return model, model_config


def analyze_transformer_block(model):
    """Analyze a single transformer block structure."""
    print(f"\n5. TRANSFORMER BLOCK STRUCTURE (Block 0):")
    block = model.transformer.h[0]

    print(f"   SPBlock")
    print(f"   ├── ln_1 (LayerNorm): {block.ln_1}")
    print(f"   ├── attn (SPAttention)")
    print(f"   │   ├── n_head: {block.attn.n_head}")
    print(f"   │   ├── head_dim: {block.attn.head_dim}")
    print(f"   │   ├── c_attn (SPLinearWithLoRA): {block.attn.c_attn}")
    print(f"   │   ├── c_proj (SPLinearWithLoRA): {block.attn.c_proj}")
    print(f"   │   ├── kv_quantizer: {block.attn.kv_quantizer}")
    print(f"   │   └── bias: {block.attn.bias.shape}")
    print(f"   ├── ln_2 (LayerNorm): {block.ln_2}")
    print(f"   └── mlp (SPMLP)")
    print(f"       ├── c_fc (SPLinearWithLoRA): {block.mlp.c_fc}")
    print(f"       ├── act (GELU): {block.mlp.act}")
    print(f"       └── c_proj (SPLinearWithLoRA): {block.mlp.c_proj}")


def analyze_switchable_linear(model):
    """Analyze the SPLinearWithLoRA structure."""
    print(f"\n6. SP LINEAR STRUCTURE (c_attn from block 0):")
    linear = model.transformer.h[0].attn.c_attn

    print(f"   SPLinearWithLoRA")
    print(f"   ├── in_features: {linear.in_features}")
    print(f"   ├── out_features: {linear.out_features}")
    print(f"   ├── bit_widths: {linear.bit_widths}")
    print(f"   ├── current_bits: {linear.current_bits}")
    print(f"   ├── linear (Base FP32): {linear.linear}")
    print(f"   ├── quantizers_weight: {len(linear.quantizers_weight)} quantizers")

    for bits_key in linear.quantizers_weight.keys():
        print(f"   │   └── {bits_key}: {linear.quantizers_weight[bits_key]}")

    print(f"   ├── quantizers_input: {len(linear.quantizers_input)} quantizers")
    for bits_key in linear.quantizers_input.keys():
        print(f"   │   └── {bits_key}: {linear.quantizers_input[bits_key]}")

    print(f"   └── lora_adapters: {len(linear.lora_adapters)} LoRA adapters")
    for bits_key, lora in linear.lora_adapters.items():
        print(f"       └── {bits_key}: {lora}")
        print(f"           ├── lora_A: {lora.lora_A.shape}")
        print(f"           ├── lora_B: {lora.lora_B.shape}")
        print(f"           ├── rank: {lora.rank}")
        print(f"           ├── alpha: {getattr(lora, 'alpha', 'N/A')}")
        print(f"           └── scaling: {getattr(lora, 'scaling', 'N/A')}")


def count_parameters(model):
    """Count and categorize model parameters."""
    print(f"\n7. PARAMETER ANALYSIS:")

    # Base model parameters (without LoRA)
    base_params = 0
    lora_params = 0
    quantizer_params = 0

    for name, param in model.named_parameters():
        param_count = param.numel()

        if 'lora_' in name:
            lora_params += param_count
        elif 'quantizer' in name:
            quantizer_params += param_count
        else:
            base_params += param_count

    total_params = base_params + lora_params + quantizer_params

    print(f"   Base model parameters: {base_params:,} ({100*base_params/total_params:.1f}%)")
    print(f"   LoRA parameters: {lora_params:,} ({100*lora_params/total_params:.1f}%)")
    print(f"   Quantizer parameters: {quantizer_params:,} ({100*quantizer_params/total_params:.1f}%)")
    print(f"   Total parameters: {total_params:,}")

    # LoRA parameters by bit width
    print(f"\n   LoRA parameters by bit width:")
    for bits in [4, 8, 16]:
        bit_params = sum(p.numel() for name, p in model.named_parameters()
                        if f'{bits}bit' in name and 'lora_' in name)
        print(f"     {bits}-bit LoRA: {bit_params:,} parameters")

    return total_params, base_params, lora_params, quantizer_params


def analyze_precision_switching(model):
    """Analyze precision switching capabilities."""
    print(f"\n8. PRECISION SWITCHING ANALYSIS:")

    # Test precision switching
    for bits in [16, 8, 4]:
        print(f"\n   Setting precision to {bits}-bit:")
        model.set_precision(bits)
        current_precision = model.get_current_precision()
        print(f"     Current precision: {current_precision}")

        # Check first block's settings
        block = model.transformer.h[0]
        attn_bits = block.attn.c_attn.current_bits
        mlp_bits = block.mlp.c_fc.current_bits
        print(f"     Attention layer bits: {attn_bits}")
        print(f"     MLP layer bits: {mlp_bits}")


def test_forward_pass(model, model_config):
    """Test forward pass with different precisions."""
    print(f"\n9. FORWARD PASS TEST:")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Create test input
    batch_size = 2
    seq_length = 16
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_length)).to(device)

    with torch.no_grad():
        for bits in [16, 8, 4]:
            model.set_precision(bits)
            outputs = model(input_ids, labels=input_ids)

            logits = outputs['logits']
            loss = outputs['loss']

            print(f"   {bits}-bit precision:")
            print(f"     Input shape: {input_ids.shape}")
            print(f"     Logits shape: {logits.shape}")
            print(f"     Loss: {loss.item():.4f}")
            print(f"     Memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")


def analyze_memory_usage(model):
    """Analyze memory usage of different components."""
    print(f"\n10. MEMORY USAGE ANALYSIS:")

    def get_param_memory(params):
        return sum(p.numel() * p.element_size() for p in params) / 1024**2

    # Base parameters memory
    base_params = [p for name, p in model.named_parameters()
                  if 'lora_' not in name and 'quantizer' not in name]
    base_memory = get_param_memory(base_params)

    # LoRA parameters memory
    lora_params = [p for name, p in model.named_parameters() if 'lora_' in name]
    lora_memory = get_param_memory(lora_params)

    # Quantizer parameters memory
    quantizer_params = [p for name, p in model.named_parameters() if 'quantizer' in name]
    quantizer_memory = get_param_memory(quantizer_params)

    total_memory = base_memory + lora_memory + quantizer_memory

    print(f"   Base model memory: {base_memory:.1f} MB ({100*base_memory/total_memory:.1f}%)")
    print(f"   LoRA memory: {lora_memory:.1f} MB ({100*lora_memory/total_memory:.1f}%)")
    print(f"   Quantizer memory: {quantizer_memory:.1f} MB ({100*quantizer_memory/total_memory:.1f}%)")
    print(f"   Total model memory: {total_memory:.1f} MB")


def analyze_distillation_setup():
    """Analyze the distillation setup for the model."""
    print(f"\n" + "=" * 80)
    print("DISTILLATION SETUP ANALYSIS")
    print("=" * 80)

    # Create configs
    model_config = ModelConfig()
    training_config = TrainingConfig()

    # Create simplified model for testing
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

    # Setup distillation
    distill_config = DistillationConfig(
        use_distillation=training_config.use_distillation,
        alpha_output=training_config.distillation_alpha_output,
        alpha_feature=training_config.distillation_alpha_feature,
        temperature=training_config.distillation_temperature
    )

    trainer = SelfDistillationTrainer(model, distill_config, device)

    print(f"\n1. DISTILLATION CONFIGURATION:")
    print(f"   Teacher precision: {trainer.full_precision_bits}-bit")
    print(f"   Student precisions: {[b for b in model_config.bit_widths if b != trainer.full_precision_bits]}")
    print(f"   Loss formula: L = {distill_config.alpha_output}*KL + {distill_config.alpha_feature}*MSE")

    # Test distillation at different precisions
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    print(f"\n2. LOSS COMPONENTS BY PRECISION:")
    for bits in model_config.bit_widths:
        model.set_precision(bits)
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True, return_dict=True)
        loss, components = trainer.compute_distillation_loss(outputs, labels, input_ids)

        print(f"\n   {bits}-bit:")
        if bits == trainer.full_precision_bits:
            print(f"     Mode: TEACHER (standard cross-entropy)")
        else:
            print(f"     Mode: STUDENT (distillation)")
        for k, v in components.items():
            if isinstance(v, float):
                print(f"     {k}: {v:.6f}")

    # Analyze cache behavior
    print(f"\n3. TEACHER CACHE ANALYSIS:")
    print(f"   Cache statistics: {trainer.get_stats()}")

    return model, trainer


def main():
    """Main analysis function with distillation support."""
    # Analyze model structure
    model, model_config = analyze_model_structure()

    # Analyze transformer block
    analyze_transformer_block(model)

    # Analyze switchable linear layer
    analyze_switchable_linear(model)

    # Count parameters
    count_parameters(model)

    # Analyze precision switching
    analyze_precision_switching(model)

    # Test forward pass
    test_forward_pass(model, model_config)

    # Analyze memory usage
    analyze_memory_usage(model)

    # NEW: Analyze distillation setup
    analyze_distillation_setup()

    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print(f"\nKEY FINDINGS:")
    print(f"• The SP model maintains separate LoRA adapters for each bit-width")
    print(f"• Quantizers are applied to both weights and activations")
    print(f"• Base GPT-2 weights can be frozen while training only LoRA adapters")
    print(f"• Precision switching is done at the layer level")
    print(f"• Memory overhead is primarily from LoRA adapters (3x for 3 bit-widths)")
    print(f"• NEW: Self-distillation uses full-precision as teacher for low-precision students")
    print(f"• NEW: Distillation combines KL divergence and feature matching losses")


if __name__ == "__main__":
    main()
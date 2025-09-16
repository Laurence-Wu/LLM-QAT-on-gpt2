#!/usr/bin/env python3
"""
Main script to run LLM-QAT paper evaluation suite
"""

import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import SwitchableQATGPT2
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

from part3_evaluation.llm_qat_metrics import LLMQATEvaluation
from part3_evaluation.bit_configurations import BitConfigurations
from part3_evaluation.generate_tables import ResultTableGenerator
from part3_evaluation.baseline_comparison import BaselineComparison


def load_pretrained_weights_into_qat(qat_model, model_name='gpt2'):
    """Load pre-trained GPT-2 weights into QAT model"""
    print(f"Loading pre-trained weights from {model_name}...")

    # Load pre-trained GPT-2
    pretrained_model = GPT2LMHeadModel.from_pretrained(model_name)
    pretrained_state = pretrained_model.state_dict()

    # Copy embeddings
    qat_model.wte.weight.data = pretrained_state['transformer.wte.weight']

    # Handle position embeddings size mismatch
    pretrained_wpe = pretrained_state['transformer.wpe.weight']
    if pretrained_wpe.shape[0] != qat_model.wpe.weight.shape[0]:
        min_pos = min(pretrained_wpe.shape[0], qat_model.wpe.weight.shape[0])
        qat_model.wpe.weight.data[:min_pos] = pretrained_wpe[:min_pos]
        print(f"Adjusted position embeddings from {pretrained_wpe.shape[0]} to {qat_model.wpe.weight.shape[0]}")
    else:
        qat_model.wpe.weight.data = pretrained_wpe

    # Copy transformer blocks
    for i in range(len(qat_model.h)):
        # Layer norms
        qat_model.h[i].ln_1.weight.data = pretrained_state[f'transformer.h.{i}.ln_1.weight']
        qat_model.h[i].ln_1.bias.data = pretrained_state[f'transformer.h.{i}.ln_1.bias']
        qat_model.h[i].ln_2.weight.data = pretrained_state[f'transformer.h.{i}.ln_2.weight']
        qat_model.h[i].ln_2.bias.data = pretrained_state[f'transformer.h.{i}.ln_2.bias']

        # Attention weights (transpose from conv1d to linear)
        qat_model.h[i].attn.c_attn.linear.weight.data = pretrained_state[f'transformer.h.{i}.attn.c_attn.weight'].t()
        qat_model.h[i].attn.c_attn.linear.bias.data = pretrained_state[f'transformer.h.{i}.attn.c_attn.bias']
        qat_model.h[i].attn.c_proj.linear.weight.data = pretrained_state[f'transformer.h.{i}.attn.c_proj.weight'].t()
        qat_model.h[i].attn.c_proj.linear.bias.data = pretrained_state[f'transformer.h.{i}.attn.c_proj.bias']

        # MLP weights (transpose from conv1d to linear)
        qat_model.h[i].mlp.c_fc.linear.weight.data = pretrained_state[f'transformer.h.{i}.mlp.c_fc.weight'].t()
        qat_model.h[i].mlp.c_fc.linear.bias.data = pretrained_state[f'transformer.h.{i}.mlp.c_fc.bias']
        qat_model.h[i].mlp.c_proj.linear.weight.data = pretrained_state[f'transformer.h.{i}.mlp.c_proj.weight'].t()
        qat_model.h[i].mlp.c_proj.linear.bias.data = pretrained_state[f'transformer.h.{i}.mlp.c_proj.bias']

        # Handle attention bias if exists
        if f'transformer.h.{i}.attn.bias' in pretrained_state:
            pretrained_bias = pretrained_state[f'transformer.h.{i}.attn.bias']
            model_bias_shape = qat_model.h[i].attn.bias.shape
            if pretrained_bias.shape != model_bias_shape:
                min_size = min(pretrained_bias.shape[0], model_bias_shape[0])
                qat_model.h[i].attn.bias.data[:min_size, :min_size] = pretrained_bias[:min_size, :min_size]
            else:
                qat_model.h[i].attn.bias.data = pretrained_bias

    # Final layer norm
    qat_model.ln_f.weight.data = pretrained_state['transformer.ln_f.weight']
    qat_model.ln_f.bias.data = pretrained_state['transformer.ln_f.bias']

    # LM head shares weight with embeddings
    qat_model.lm_head.weight = qat_model.wte.weight

    # Initialize LoRA weights to small/zero values
    with torch.no_grad():
        for module in qat_model.modules():
            try:
                lora_adapters = module.lora_adapters
                for lora in lora_adapters.values():
                    try:
                        nn.init.zeros_(lora.lora_B)
                    except AttributeError:
                        pass  # lora_B doesn't exist
                    try:
                        nn.init.normal_(lora.lora_A, std=0.01)
                    except AttributeError:
                        pass  # lora_A doesn't exist
            except AttributeError:
                pass  # module doesn't have lora_adapters

    print("Pre-trained weights loaded successfully!")
    return qat_model


def load_switchable_model(model_path: str = None, config_path: str = None, use_pretrained: bool = True):
    """Load switchable precision model with proper configuration"""

    # Force CUDA availability check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This evaluation requires CUDA.")

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Default bit widths - will be overridden if loading from checkpoint
    default_bit_widths = [4, 8, 16]

    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")

        # Determine JSON config path
        json_path = config_path  # Use provided config path if available

        if not json_path and model_path.endswith('.pth'):
            # Try to auto-detect matching JSON file
            timestamp = model_path.split('_')[-1].replace('.pth', '')
            possible_json = f"qat_training_stats_{timestamp}.json"
            if os.path.exists(possible_json):
                json_path = possible_json
                print(f"Auto-detected matching JSON config: {json_path}")

        if json_path and os.path.exists(json_path):
            print(f"Using config from: {json_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cuda')

        # Try to get config from JSON file first (more complete), then fall back to checkpoint
        model_config = None
        training_config = None

        if json_path:
            try:
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                    model_config = json_data.get('model_config', {})
                    training_config = json_data.get('training_config', {})
                    print(f"Loaded configuration from JSON file")
            except Exception as e:
                print(f"Warning: Could not load JSON config: {e}")

        # Fall back to checkpoint config if JSON not available
        if not model_config and isinstance(checkpoint, dict):
            model_config = checkpoint.get('model_config', {})
            training_config = checkpoint.get('training_config', {})

        if model_config:
            # Extract bit widths - handle both old and new format
            bit_widths = model_config.get('bit_widths', default_bit_widths)
            quantization_bits = model_config.get('quantization_bits', 8)

            # If bit_widths not specified but quantization_bits is, use standard set
            if bit_widths == default_bit_widths and quantization_bits:
                print(f"Model trained with quantization_bits={quantization_bits}, using standard bit widths")
                bit_widths = [4, 8, 16]  # Standard switchable widths

            # Use training config's max_seq_length as n_positions if available
            actual_n_positions = 256  # Default
            if training_config and 'max_seq_length' in training_config:
                actual_n_positions = training_config['max_seq_length']
                print(f"Using max_seq_length from training config: {actual_n_positions}")
            elif 'model_state_dict' in checkpoint and 'wpe.weight' in checkpoint['model_state_dict']:
                # Get actual n_positions from the weight shape
                wpe_shape = checkpoint['model_state_dict']['wpe.weight'].shape
                actual_n_positions = wpe_shape[0]
                print(f"Detected actual n_positions from weights: {actual_n_positions}")

            # Use exact configuration from model_config
            config = GPT2Config(
                vocab_size=model_config.get('vocab_size', 50257),
                n_positions=actual_n_positions,
                n_embd=model_config.get('n_embd', 768),
                n_layer=model_config.get('n_layer', 6),
                n_head=model_config.get('n_head', 12),
                layer_norm_epsilon=model_config.get('layer_norm_epsilon', 1e-5),
                embd_pdrop=model_config.get('embd_pdrop', 0.1),
                lora_rank=model_config.get('lora_rank', 16),
                lora_alpha=model_config.get('lora_alpha', 32),
                lora_dropout=model_config.get('lora_dropout', 0.1)
            )

            print(f"\nLoaded Model Configuration:")
            print(f"  - n_layer: {config.n_layer}")
            print(f"  - n_embd: {config.n_embd}")
            print(f"  - n_head: {config.n_head}")
            print(f"  - n_positions: {config.n_positions}")
            print(f"  - vocab_size: {config.vocab_size}")
            print(f"  - quantization_bits (training): {quantization_bits}")
            print(f"  - bit_widths (switchable): {bit_widths}")

            if training_config:
                print(f"\nTraining Configuration:")
                print(f"  - batch_size: {training_config.get('batch_size', 'N/A')}")
                print(f"  - max_seq_length: {training_config.get('max_seq_length', 'N/A')}")
                print(f"  - learning_rate: {training_config.get('learning_rate', 'N/A')}")
                print(f"  - num_iterations: {training_config.get('num_iterations', 'N/A')}")
        else:
            # Use default configuration
            bit_widths = default_bit_widths
            config = GPT2Config(
                vocab_size=50257,
                n_positions=256,
                n_embd=768,
                n_layer=6,
                n_head=12,
                layer_norm_epsilon=1e-5,
                embd_pdrop=0.1,
                lora_rank=16,
                lora_alpha=32,
                lora_dropout=0.1
            )

        print(f"Creating model with bit-widths: {bit_widths}")
        # Create model WITHOUT random initialization
        model = SwitchableQATGPT2(config, bit_widths=bit_widths, initialize_weights=False)

        # Load pre-trained weights first if requested
        if use_pretrained:
            model = load_pretrained_weights_into_qat(model, 'gpt2')

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Load state dict with size mismatch handling for attention bias
            state_dict = checkpoint['model_state_dict']
            model_state = model.state_dict()

            # Handle size mismatches for attention bias
            for key in list(state_dict.keys()):
                if '.bias' in key and key in model_state:
                    saved_bias = state_dict[key]
                    model_bias = model_state[key]

                    if saved_bias.shape != model_bias.shape:
                        # Resize the attention bias to match model's n_positions
                        print(f"Resizing {key} from {saved_bias.shape} to {model_bias.shape}")
                        # For attention bias (2D causal mask)
                        if len(saved_bias.shape) == 2 and len(model_bias.shape) == 2:
                            min_size = min(saved_bias.shape[0], model_bias.shape[0])
                            new_bias = torch.zeros_like(model_bias)
                            new_bias[:min_size, :min_size] = saved_bias[:min_size, :min_size]
                            # Fill the rest with the appropriate causal mask pattern if needed
                            if min_size < model_bias.shape[0] and 'attn' in key:
                                # Create proper causal mask for remaining positions
                                for i in range(min_size, model_bias.shape[0]):
                                    new_bias[i, :i+1] = 1
                        else:
                            # For 1D biases, just copy what we can
                            min_size = min(saved_bias.shape[0], model_bias.shape[0])
                            new_bias = torch.zeros_like(model_bias)
                            if len(saved_bias.shape) == 1:
                                new_bias[:min_size] = saved_bias[:min_size]
                            else:
                                new_bias = saved_bias  # Keep original if shape is unexpected
                        state_dict[key] = new_bias

            # Load the modified state dict
            model.load_state_dict(state_dict, strict=False)
            print("Model checkpoint loaded successfully")
        elif not isinstance(checkpoint, dict):
            model = checkpoint
    else:
        print(f"Creating new model with pre-trained weights")
        config = GPT2Config(
            vocab_size=50257,
            n_positions=256,
            n_embd=768,
            n_layer=6,
            n_head=12,
            layer_norm_epsilon=1e-5,
            embd_pdrop=0.1,
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.1
        )
        # Create model WITHOUT random initialization
        model = SwitchableQATGPT2(config, bit_widths=default_bit_widths, initialize_weights=False)

        # Load pre-trained GPT-2 weights
        if use_pretrained:
            model = load_pretrained_weights_into_qat(model, 'gpt2')
            print("Using GPT-2 pre-trained weights")
        else:
            # Apply random initialization for backward compatibility
            model.apply(model._init_weights)
            print("Using random initialization (not recommended for evaluation)")

    # Force model to CUDA
    device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    print(f"Model moved to {device}")
    print(f"Model device check: {next(model.parameters()).device}")
    return model


def load_tokenizer():
    """Load GPT-2 tokenizer"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description='LLM-QAT Paper Evaluation Suite')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to training config JSON file (optional, will auto-detect if not provided)')
    parser.add_argument('--output_dir', type=str, default='part3_evaluation/results',
                       help='Directory to save results')
    parser.add_argument('--configs', nargs='+',
                       default=['INT4', 'INT8', 'FP16'],
                       help='Configurations to evaluate (e.g., INT4 INT8 FP16)')
    parser.add_argument('--skip_few_shot', action='store_true',
                       help='Skip few-shot evaluation (faster)')
    parser.add_argument('--skip_zero_shot', action='store_true',
                       help='Skip zero-shot evaluation')
    parser.add_argument('--skip_perplexity', action='store_true',
                       help='Skip perplexity evaluation')
    parser.add_argument('--compare_baselines', action='store_true',
                       help='Compare with baseline methods')
    parser.add_argument('--use_pretrained', action='store_true', default=True,
                       help='Use pre-trained GPT-2 weights (strongly recommended)')
    parser.add_argument('--use_random_init', action='store_true',
                       help='Use random initialization instead of pre-trained (for testing only)')
    parser.add_argument('--force_cuda', action='store_true', default=True,
                       help='Force CUDA usage (default: True)')
    args = parser.parse_args()

    # Force CUDA check
    if args.force_cuda and not torch.cuda.is_available():
        print("ERROR: CUDA is required but not available!")
        print("Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support installed.")
        sys.exit(1)

    # Determine whether to use pre-trained weights
    use_pretrained = not args.use_random_init

    model = load_switchable_model(args.model_path, config_path=args.config_path, use_pretrained=use_pretrained)
    tokenizer = load_tokenizer()

    evaluator = LLMQATEvaluation(model, tokenizer)

    # Verify model has required attributes
    if not hasattr(model, 'bit_widths'):
        raise AttributeError("Model does not support switchable precision. Please ensure the model was trained with SwitchableQATGPT2.")

    # Model supports switchable precision
    supported_bit_widths = model.bit_widths
    print(f"Model supports bit-widths: {supported_bit_widths}")

    # Map bit-widths to configuration names
    bit_to_config = {
        2: 'INT2',
        4: 'INT4',
        8: 'INT8',
        16: 'FP16'
    }

    # Override args.configs with supported configurations
    if not args.configs or args.configs == ['INT4', 'INT8', 'FP16']:
        # Use default or auto-detect
        args.configs = [bit_to_config.get(b, f'INT{b}') for b in supported_bit_widths if b in bit_to_config]
        print(f"Auto-detected configurations to evaluate: {args.configs}")

    # Validate that requested configs are supported
    for config_name in args.configs:
        if config_name in BitConfigurations.STANDARD_CONFIGS:
            config = BitConfigurations.STANDARD_CONFIGS[config_name]
            weight_bits = config['W']
            if weight_bits not in supported_bit_widths:
                raise ValueError(f"Configuration {config_name} requires {weight_bits}-bit precision, "
                               f"but model only supports {supported_bit_widths}. "
                               f"Please train the model with the required bit-width.")

    print("="*70)
    print("Running LLM-QAT Paper Evaluation Suite")
    print("="*70)
    print(f"Model: GPT-2 ({evaluator.model_params:.1f}M parameters)")
    print(f"Configurations to evaluate: {args.configs}")
    print(f"Output directory: {args.output_dir}")
    print("="*70)

    results = {}

    for config_name in args.configs:
        print(f"\n{'='*60}")
        print(f"Evaluating configuration: {config_name}")
        print('='*60)

        if config_name not in BitConfigurations.STANDARD_CONFIGS:
            print(f"Warning: Configuration {config_name} not found in standard configs")
            continue

        config = BitConfigurations.STANDARD_CONFIGS[config_name]

        BitConfigurations.apply_config_to_model(model, config)

        results[config_name] = {
            'config_name': config['name'],
            'bits': f"{config['W']}-{config['A']}-{config['KV']}",
            'model_size_gb': evaluator.calculate_model_size(config),
            'description': config.get('description', '')
        }

        print(f"Configuration: {config['name']} ({config['description']})")
        print(f"Model size: {results[config_name]['model_size_gb']} GB")
        print(f"Applying bit configuration W={config['W']}, A={config['A']}, KV={config['KV']}")

        if not args.skip_perplexity:
            print("\n2. Perplexity evaluation...")
            perplexity_results = evaluator.evaluate_perplexity(config)
            results[config_name]['perplexity'] = perplexity_results
            print(f"   WikiText2: {perplexity_results['WikiText2']:.1f}")
            print(f"   C4: {perplexity_results['C4']:.1f}")

        if not args.skip_zero_shot:
            print("\n1. Zero-shot common sense evaluation...")
            zero_shot_results = evaluator.evaluate_zero_shot_common_sense(config)
            results[config_name]['zero_shot'] = zero_shot_results
            print(f"   Average score: {zero_shot_results['Average']:.1f}%")

            # Print only the tasks that were actually evaluated
            for task, score in zero_shot_results.items():
                if task != 'Average':
                    print(f"   {task}: {score:.1f}%")



        if not args.skip_few_shot:
            print("\n3. Few-shot evaluation...")
            few_shot_results = evaluator.evaluate_few_shot(config)
            results[config_name]['few_shot'] = few_shot_results

            if 'MMLU' in few_shot_results and isinstance(few_shot_results['MMLU'], dict):
                mmlu = few_shot_results['MMLU']
                print(f"   MMLU:")
                for category in ['Humanities', 'STEM', 'Social Sciences', 'Other']:
                    if category in mmlu:
                        score = mmlu[category]
                        if isinstance(score, (int, float)) and not np.isnan(score):
                            print(f"     {category}: {score:.1f}%")
                        else:
                            print(f"     {category}: 0.0%")
                avg_score = mmlu.get('Average', 0)
                if isinstance(avg_score, (int, float)) and not np.isnan(avg_score):
                    print(f"     Average: {avg_score:.1f}%")
                else:
                    print(f"     Average: 0.0%")

            if 'TriviaQA' in few_shot_results:
                print(f"   TriviaQA: {few_shot_results['TriviaQA']:.1f}%")

    print("\n" + "="*70)
    print("Generating result tables...")
    print("="*70)

    table_gen = ResultTableGenerator(results)

    if not args.skip_zero_shot:
        table_gen.generate_table_1_zero_shot()

    if not args.skip_perplexity:
        table_gen.generate_table_2_perplexity()

    if not args.skip_few_shot:
        table_gen.generate_table_7_few_shot()

    table_gen.export_to_markdown()
    table_gen.export_to_latex()

    if args.compare_baselines:
        print("\n" + "="*70)
        print("Comparing with baseline methods...")
        print("="*70)

        comparison = BaselineComparison(results)
        comparison.compare_with_baselines()
        comparison.plot_accuracy_vs_bits()
        comparison.calculate_degradation_from_fp16()

    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    with open(output_path / 'llm_qat_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Evaluation complete!")
    print(f"Results saved to {output_path}")
    print(f"{'='*70}")

    print("\nSummary of Results:")
    for config_name, result in results.items():
        print(f"\n{config_name} ({result['bits']}):")
        if 'zero_shot' in result and result['zero_shot']:
            print(f"  Zero-shot avg: {result['zero_shot'].get('Average', 0):.1f}%")
        if 'perplexity' in result and result['perplexity']:
            print(f"  WikiText2 PPL: {result['perplexity'].get('WikiText2', float('inf')):.1f}")
            print(f"  C4 PPL: {result['perplexity'].get('C4', float('inf')):.1f}")


if __name__ == "__main__":
    main()
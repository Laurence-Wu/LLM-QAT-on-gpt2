#!/usr/bin/env python3
"""
Main script to run LLM-QAT paper evaluation suite with standard evaluation methods
"""

import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np
import time
from tqdm import tqdm
import math
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part1_switchable_precision.models_sp import SPModel, SPLMHeadModel
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

from part3_evaluation.llm_qat_metrics import LLMQATEvaluation
from part3_evaluation.bit_configurations import BitConfigurations
from part3_evaluation.generate_tables import ResultTableGenerator
from part3_evaluation.baseline_comparison import BaselineComparison
from part3_evaluation.zero_shot_tasks import ZeroShotEvaluator
from part3_evaluation.few_shot_eval import FewShotEvaluator
from part3_evaluation.perplexity_eval import PerplexityEvaluator


def load_pretrained_weights_into_qat(sp_model, model_name='gpt2'):
    """Load pre-trained GPT-2 weights into SP model"""
    print(f"Loading pre-trained weights from {model_name}...")

    # Load pre-trained GPT-2
    pretrained_model = GPT2LMHeadModel.from_pretrained(model_name)
    pretrained_state = pretrained_model.state_dict()

    # Access the transformer part of SPLMHeadModel
    transformer = sp_model.transformer

    # Copy embeddings
    transformer.wte.weight.data = pretrained_state['transformer.wte.weight']

    # Handle position embeddings size mismatch
    pretrained_wpe = pretrained_state['transformer.wpe.weight']
    if pretrained_wpe.shape[0] != transformer.wpe.weight.shape[0]:
        min_pos = min(pretrained_wpe.shape[0], transformer.wpe.weight.shape[0])
        transformer.wpe.weight.data[:min_pos] = pretrained_wpe[:min_pos]
        print(f"Adjusted position embeddings from {pretrained_wpe.shape[0]} to {transformer.wpe.weight.shape[0]}")
    else:
        transformer.wpe.weight.data = pretrained_wpe

    # Copy transformer blocks
    for i in range(len(transformer.h)):
        # For SwitchableLayerNorm, copy weights to each precision's LayerNorm
        for ln_key in transformer.h[i].ln_1.ln_layers:
            transformer.h[i].ln_1.ln_layers[ln_key].weight.data = pretrained_state[f'transformer.h.{i}.ln_1.weight']
            transformer.h[i].ln_1.ln_layers[ln_key].bias.data = pretrained_state[f'transformer.h.{i}.ln_1.bias']

        for ln_key in transformer.h[i].ln_2.ln_layers:
            transformer.h[i].ln_2.ln_layers[ln_key].weight.data = pretrained_state[f'transformer.h.{i}.ln_2.weight']
            transformer.h[i].ln_2.ln_layers[ln_key].bias.data = pretrained_state[f'transformer.h.{i}.ln_2.bias']

        # Attention weights (transpose from conv1d to linear)
        transformer.h[i].attn.c_attn.linear.weight.data = pretrained_state[f'transformer.h.{i}.attn.c_attn.weight'].t()
        transformer.h[i].attn.c_attn.linear.bias.data = pretrained_state[f'transformer.h.{i}.attn.c_attn.bias']
        transformer.h[i].attn.c_proj.linear.weight.data = pretrained_state[f'transformer.h.{i}.attn.c_proj.weight'].t()
        transformer.h[i].attn.c_proj.linear.bias.data = pretrained_state[f'transformer.h.{i}.attn.c_proj.bias']

        # MLP weights (transpose from conv1d to linear)
        transformer.h[i].mlp.c_fc.linear.weight.data = pretrained_state[f'transformer.h.{i}.mlp.c_fc.weight'].t()
        transformer.h[i].mlp.c_fc.linear.bias.data = pretrained_state[f'transformer.h.{i}.mlp.c_fc.bias']
        transformer.h[i].mlp.c_proj.linear.weight.data = pretrained_state[f'transformer.h.{i}.mlp.c_proj.weight'].t()
        transformer.h[i].mlp.c_proj.linear.bias.data = pretrained_state[f'transformer.h.{i}.mlp.c_proj.bias']

        # Handle attention bias if exists
        if f'transformer.h.{i}.attn.bias' in pretrained_state:
            pretrained_bias = pretrained_state[f'transformer.h.{i}.attn.bias']
            model_bias_shape = transformer.h[i].attn.bias.shape
            if pretrained_bias.shape != model_bias_shape:
                min_size = min(pretrained_bias.shape[0], model_bias_shape[0])
                transformer.h[i].attn.bias.data[:min_size, :min_size] = pretrained_bias[:min_size, :min_size]
            else:
                transformer.h[i].attn.bias.data = pretrained_bias

    # Final layer norm - also uses SwitchableLayerNorm
    for ln_key in transformer.ln_f.ln_layers:
        transformer.ln_f.ln_layers[ln_key].weight.data = pretrained_state['transformer.ln_f.weight']
        transformer.ln_f.ln_layers[ln_key].bias.data = pretrained_state['transformer.ln_f.bias']

    # LM head shares weight with embeddings
    sp_model.lm_head.weight = transformer.wte.weight

    # Initialize LoRA weights to small/zero values
    with torch.no_grad():
        for module in sp_model.modules():
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
    return sp_model


class CalibrationManager:
    """Manages quantizer calibration with warm-up phase"""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.calibrated_bits = set()

    def warm_up_calibration(self, data_loader, bit_widths: List[int], num_batches: int = 10):
        """Warm-up phase for calibrating all bit-widths"""
        print("\n" + "="*60)
        print("WARM-UP CALIBRATION PHASE")
        print("="*60)

        for bits in bit_widths:
            if bits < 16:  # No calibration needed for FP16
                print(f"\nCalibrating {bits}-bit precision...")
                self.calibrate_precision(bits, data_loader, num_batches)
                self.calibrated_bits.add(bits)

        print("\n✅ Warm-up calibration complete for all precisions")
        print("="*60)

    def calibrate_precision(self, bits: int, data_loader, num_batches: int):
        """Calibrate quantizers for specific bit-width"""
        # Set model to calibration bit-width
        self.model.set_precision(bits)
        self.model.train()  # Calibration requires training mode

        # Start calibration for all quantizers
        bits_key = f'{bits}bit'
        for name, module in self.model.named_modules():
            try:
                # Try to access quantizers_weight and quantizers_input
                quantizers_weight = module.quantizers_weight
                quantizers_input = module.quantizers_input

                if bits_key in quantizers_weight:
                    quantizers_weight[bits_key].start_calibration()
                if bits_key in quantizers_input:
                    quantizers_input[bits_key].start_calibration()
            except AttributeError:
                # Module doesn't have quantizers, continue
                continue

        # Collect statistics
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break

                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                else:
                    input_ids = batch.to(self.device)

                _ = self.model(input_ids)

        # Finish calibration
        for name, module in self.model.named_modules():
            try:
                # Try to access quantizers_weight and quantizers_input
                quantizers_weight = module.quantizers_weight
                quantizers_input = module.quantizers_input

                if bits_key in quantizers_weight:
                    quantizers_weight[bits_key].finish_calibration()
                if bits_key in quantizers_input:
                    quantizers_input[bits_key].finish_calibration()
            except AttributeError:
                # Module doesn't have quantizers, continue
                continue

        self.model.eval()  # Return to eval mode
        torch.cuda.empty_cache()


def fix_state_dict_shapes(state_dict):
    """Fix shape mismatches in loaded state dict.

    Handles conversion between scalar tensors and size [1] tensors.
    """
    fixed_dict = {}
    for key, value in state_dict.items():
        # Check if this is a quantizer parameter
        if any(param in key for param in ['scale', 'zero_point', 'running_min', 'running_max']):
            if torch.is_tensor(value) and value.dim() == 0:
                # Convert scalar to size [1] tensor
                fixed_dict[key] = value.unsqueeze(0)
            else:
                fixed_dict[key] = value
        else:
            fixed_dict[key] = value
    return fixed_dict


def validate_model_config(config):
    """Validate that all required model configuration parameters are present."""
    required_params = [
        'vocab_size', 'n_positions', 'n_embd', 'n_layer', 'n_head',
        'layer_norm_epsilon', 'embd_pdrop', 'bit_widths',
        'lora_rank_per_bit', 'lora_alpha_per_bit',
        'activation_bits_per_bit', 'quantizer_per_bit'
    ]

    missing = [key for key in required_params if key not in config]
    if missing:
        raise ValueError(f"Missing required configuration parameters: {missing}\n"
                        f"Please ensure your checkpoint was saved with complete configuration.")

    print(f"✅ Configuration validation passed: {len(required_params)} required parameters found")


def load_switchable_model(model_path: str = None, config_path: str = None, use_pretrained: bool = True):
    """Load switchable precision model with proper configuration"""

    # Force CUDA availability check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This evaluation requires CUDA.")

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")

        # Determine JSON config path
        json_path = config_path  # Use provided config path if available

        if not json_path and model_path.endswith('.pth'):
            # Try to auto-detect matching JSON file
            dir_path = os.path.dirname(model_path) if os.path.dirname(model_path) else '.'
            base_name = os.path.basename(model_path)
            timestamp = base_name.split('_')[-1].replace('.pth', '')

            # Try multiple patterns
            possible_jsons = [
                os.path.join(dir_path, f"sp_gpt2_config_{timestamp}.json"),  # New format
                os.path.join(dir_path, f"qat_training_stats_{timestamp}.json"),  # Old format
                f"sp_gpt2_config_{timestamp}.json",
                f"qat_training_stats_{timestamp}.json"
            ]

            for possible_json in possible_jsons:
                if os.path.exists(possible_json):
                    json_path = possible_json
                    print(f"Auto-detected matching JSON config: {json_path}")
                    break

        # Load checkpoint first
        checkpoint = torch.load(model_path, map_location='cuda')

        # Try to get config from JSON file first (most complete), then fall back to checkpoint
        model_config = None
        training_config = None

        # Priority 1: User-specified config path
        if config_path and os.path.exists(config_path):
            print(f"Using user-specified config from: {config_path}")
            try:
                with open(config_path, 'r') as f:
                    json_data = json.load(f)
                    model_config = json_data.get('model_config', {})
                    training_config = json_data.get('training_config', {})
                    print(f"✓ Loaded configuration from specified JSON file")
            except Exception as e:
                print(f"Warning: Could not load specified JSON config: {e}")

        # Priority 2: Auto-detected JSON config
        elif json_path and os.path.exists(json_path):
            print(f"Using auto-detected config from: {json_path}")
            try:
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                    model_config = json_data.get('model_config', {})
                    training_config = json_data.get('training_config', {})
                    print(f"✓ Loaded configuration from auto-detected JSON file")
            except Exception as e:
                print(f"Warning: Could not load auto-detected JSON config: {e}")

        # Priority 3: Config embedded in checkpoint
        if not model_config and isinstance(checkpoint, dict):
            if 'model_config' in checkpoint:
                model_config = checkpoint.get('model_config', {})
                training_config = checkpoint.get('training_config', {})
                print("Using configuration embedded in checkpoint")

        if not model_config:
            raise ValueError("No model configuration found! Please provide a valid checkpoint with model_config or specify --config_path.\n"
                           "The checkpoint must be saved with complete configuration using the updated training script.")

        # STRICT CONFIG USAGE - Use exactly what's in the config, no defaults
        print("\n" + "="*50)
        print("USING STRICT CONFIGURATION FROM CHECKPOINT/JSON")
        print("="*50)

        # Validate configuration has all required parameters
        validate_model_config(model_config)

        # Extract required values from model_config (NO DEFAULTS)
        n_layer = model_config['n_layer']
        n_embd = model_config['n_embd']
        n_head = model_config['n_head']
        quantization_bits = model_config.get('quantization_bits')  # Optional field

        # Validate required fields
        if n_layer is None or n_embd is None or n_head is None:
            raise ValueError(f"Missing required model config fields. Got: n_layer={n_layer}, n_embd={n_embd}, n_head={n_head}")

        # Bit widths MUST be specified in config
        bit_widths = model_config['bit_widths']  # Will raise KeyError if missing
        print(f"Using bit widths from config: {bit_widths}")

        # Get n_positions from actual weights in checkpoint (most reliable)
        actual_n_positions = None
        if 'model_state_dict' in checkpoint:
            # Check transformer.wpe.weight first (SP model format)
            if 'transformer.wpe.weight' in checkpoint['model_state_dict']:
                wpe_shape = checkpoint['model_state_dict']['transformer.wpe.weight'].shape
                actual_n_positions = wpe_shape[0]
                print(f"Detected n_positions from transformer.wpe.weight shape: {actual_n_positions}")
            elif 'wpe.weight' in checkpoint['model_state_dict']:
                wpe_shape = checkpoint['model_state_dict']['wpe.weight'].shape
                actual_n_positions = wpe_shape[0]
                print(f"Detected n_positions from wpe.weight shape: {actual_n_positions}")

        # Fallback to config if needed
        if actual_n_positions is None:
            if training_config and 'max_seq_length' in training_config:
                actual_n_positions = training_config['max_seq_length']
                print(f"Using max_seq_length from training config: {actual_n_positions}")
            else:
                # Default to GPT-2 standard
                actual_n_positions = 1024
                print("WARNING: Could not determine n_positions, using GPT-2 standard 1024")

        # Build config with values from the loaded configuration (NO DEFAULTS)
        config = GPT2Config(
            vocab_size=model_config['vocab_size'],  # Required
            n_positions=actual_n_positions,  # Detected from weights
            n_embd=n_embd,  # Required
            n_layer=n_layer,  # Required
            n_head=n_head,  # Required
            layer_norm_epsilon=model_config['layer_norm_epsilon'],  # Required
            embd_pdrop=model_config['embd_pdrop'],  # Required
            lora_rank=model_config.get('lora_rank', None),  # Optional for SP models
            lora_alpha=model_config.get('lora_alpha', None),  # Optional for SP models
            lora_dropout=model_config.get('lora_dropout', 0.0)  # Optional, default to 0 if not set
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
            print(f"  - batch_size: {training_config.get('batch_size')}")
            print(f"  - max_seq_length: {training_config.get('max_seq_length')}")
            print(f"  - learning_rate: {training_config.get('learning_rate')}")
            print(f"  - num_iterations: {training_config.get('num_iterations')}")

        print(f"Creating model with bit-widths: {bit_widths}")
        # Add SP-specific configurations to config
        config.bit_widths = bit_widths

        # Get SP-specific configurations from model_config (NO DEFAULTS)
        config.lora_rank_per_bit = model_config['lora_rank_per_bit']  # Required for SP
        config.lora_alpha_per_bit = model_config['lora_alpha_per_bit']  # Required for SP
        config.activation_bits_per_bit = model_config['activation_bits_per_bit']  # Required for SP
        config.quantizer_per_bit = model_config['quantizer_per_bit']  # Required for SP

        # Convert string keys to int if necessary (JSON serialization converts int keys to strings)
        if isinstance(config.lora_rank_per_bit, dict):
            config.lora_rank_per_bit = {int(k) if isinstance(k, str) else k: v
                                       for k, v in config.lora_rank_per_bit.items()}
        if isinstance(config.lora_alpha_per_bit, dict):
            config.lora_alpha_per_bit = {int(k) if isinstance(k, str) else k: v
                                        for k, v in config.lora_alpha_per_bit.items()}
        if isinstance(config.activation_bits_per_bit, dict):
            config.activation_bits_per_bit = {int(k) if isinstance(k, str) else k: v
                                             for k, v in config.activation_bits_per_bit.items()}
        if isinstance(config.quantizer_per_bit, dict) and config.quantizer_per_bit is not None:
            config.quantizer_per_bit = {int(k) if isinstance(k, str) else k: v
                                       for k, v in config.quantizer_per_bit.items()}

        # Print what we're using to help debug
        print(f"LoRA rank per bit: {config.lora_rank_per_bit}")
        print(f"LoRA alpha per bit: {config.lora_alpha_per_bit}")
        print(f"Activation bits per bit: {config.activation_bits_per_bit}")

        # Create SPLMHeadModel instead of SwitchableQATGPT2
        model = SPLMHeadModel(config)

        # Don't load pretrained weights - we'll load from checkpoint directly
        # This avoids resizing issues

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Load state dict with size mismatch handling for attention bias
            state_dict = checkpoint['model_state_dict']

            # Fix quantizer shape mismatches only
            state_dict = fix_state_dict_shapes(state_dict)

            # Don't resize matrices - model should be created with correct dimensions

            # Load the modified state dict with strict=False to handle minor mismatches
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
                # Only show first 10 missing keys to avoid clutter
                for key in missing_keys[:10]:
                    print(f"  - {key}")
                if len(missing_keys) > 10:
                    print(f"  ... and {len(missing_keys) - 10} more")

            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
                # Only show first 10 unexpected keys
                for key in unexpected_keys[:10]:
                    print(f"  - {key}")
                if len(unexpected_keys) > 10:
                    print(f"  ... and {len(unexpected_keys) - 10} more")

            print("Model checkpoint loaded successfully (with shape fixes and warnings handled)")
        elif not isinstance(checkpoint, dict):
            model = checkpoint
    else:
        raise ValueError("No model path provided! Please specify --model_path with a trained checkpoint file.")

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


# Removed EvaluationMetrics class - using specialized evaluators instead


def load_evaluation_config(config_path):
    """Load evaluation configuration from JSON file. NO DEFAULTS ALLOWED."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Evaluation config required but not found: {config_path}\n"
                              f"Please ensure evaluation_config.json exists at: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Validate required sections
    required_sections = ['device', 'calibration', 'zero_shot', 'few_shot', 'perplexity', 'output', 'model']
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    return config


def main():
    parser = argparse.ArgumentParser(description='LLM-QAT Paper Evaluation Suite with Standard Methods')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--config_path', type=str,
                       help='Path to training config JSON file (optional, will auto-detect if not provided)')
    parser.add_argument('--eval_config', type=str,
                       default='part3_evaluation/evaluation_config.json',
                       help='Path to evaluation configuration JSON file')
    args = parser.parse_args()

    # Load evaluation configuration (NO DEFAULTS)
    eval_config = load_evaluation_config(args.eval_config)
    print(f"Loaded evaluation config from: {args.eval_config}")

    # Check CUDA requirement from config
    if eval_config['model']['require_cuda']:
        if not torch.cuda.is_available():
            print("ERROR: CUDA is required (per evaluation config) but not available!")
            print("Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support installed.")
            sys.exit(1)

    # Load model from checkpoint
    model = load_switchable_model(args.model_path, config_path=args.config_path, use_pretrained=False)
    tokenizer = load_tokenizer()

    # Initialize all evaluation components with config
    device = eval_config['device']
    evaluator = LLMQATEvaluation(model, tokenizer)
    zero_shot_evaluator = ZeroShotEvaluator(model, tokenizer, device=device, config=eval_config['zero_shot'])
    few_shot_evaluator = FewShotEvaluator(model, tokenizer, device=device, config=eval_config['few_shot'])
    perplexity_evaluator = PerplexityEvaluator(model, tokenizer, device=device, config=eval_config['perplexity'])

    # Prepare calibration data from config
    calib_cfg = eval_config['calibration']
    print(f"\nPreparing calibration data from {calib_cfg['dataset']}...")
    calibration_dataset = load_dataset(
        calib_cfg['dataset'],
        calib_cfg['dataset_config'],
        split=calib_cfg['split']
    )
    calibration_texts = [sample['text'] for sample in calibration_dataset
                        if len(sample['text'].strip()) > 0][:calib_cfg['num_samples']]

    # Tokenize calibration data with config parameters
    calibration_data = []
    for text in calibration_texts:
        tokens = tokenizer(
            text,
            return_tensors='pt',
            max_length=calib_cfg['max_length'],
            truncation=calib_cfg['truncation'],
            padding=calib_cfg['padding']
        )
        # Squeeze to remove batch dimension since DataLoader will add it back
        calibration_data.append(tokens['input_ids'].squeeze(0))

    calibration_loader = torch.utils.data.DataLoader(
        calibration_data,
        batch_size=calib_cfg['batch_size'],
        shuffle=False
    )

    # Initialize calibration manager and perform warm-up
    calib_manager = CalibrationManager(model, device=device)
    try:
        # Try to access bit_widths attribute
        model_bit_widths = model.transformer.bit_widths  # For SPLMHeadModel, bit_widths is in transformer
        print("\nPerforming warm-up calibration for all bit-widths...")
        warm_up_batches = calib_cfg['warm_up_batches']  # From config, no default
        calib_manager.warm_up_calibration(calibration_loader, model_bit_widths, num_batches=warm_up_batches)
    except AttributeError as e:
        raise AttributeError(f"Model does not support switchable precision: {e}\nPlease ensure the model was trained with SPLMHeadModel.")

    # Get current model's bit configuration
    try:
        current_bits = model.transformer.current_bits
    except AttributeError:
        # Use config default if attribute doesn't exist
        current_bits = eval_config['model']['default_precision']
        print(f"Warning: Could not get current_bits from model, using config default: {current_bits}")
    print(f"Current model precision: {current_bits}-bit")

    # Map current bit-width to configuration name from config
    bit_to_config = {int(k): v for k, v in eval_config['model']['bit_to_config_mapping'].items()}

    config_name = bit_to_config.get(current_bits, f'INT{current_bits}')
    print(f"Configuration to evaluate: {config_name}")

    # Get evaluation settings from config
    output_dir = eval_config['output']['directory']

    print("="*70)
    print("Running SP Model Evaluation")
    print("="*70)
    print(f"Model: GPT-2 ({evaluator.model_params:.1f}M parameters)")
    print(f"Current precision: {current_bits}-bit ({config_name})")
    print(f"Output directory: {output_dir}")
    print(f"Max zero-shot samples: {eval_config['zero_shot']['max_samples']}")
    print(f"Max few-shot samples: {eval_config['few_shot']['max_samples']}")
    print(f"Max perplexity samples: {eval_config['perplexity']['max_samples']}")
    print("="*70)

    results = {}

    # Run comprehensive evaluation for the current model configuration
    print(f"\n{'='*60}")
    print(f"Evaluating current model configuration: {config_name}")
    print('='*60)

    if config_name not in BitConfigurations.STANDARD_CONFIGS:
        # Create config based on current bits if not in standard configs
        config = {
            'W': current_bits,
            'A': current_bits,
            'KV': current_bits,
            'name': f'{current_bits}-{current_bits}-{current_bits}',
            'description': f'{current_bits}-bit quantization'
        }
    else:
        config = BitConfigurations.STANDARD_CONFIGS[config_name]

    # No need to apply config - model is already at current precision

        results[config_name] = {
            'config_name': config['name'],
            'bits': f"{config['W']}-{config['A']}-{config['KV']}",
            'model_size_gb': evaluator.calculate_model_size(config),
            'description': config.get('description', '')
        }

        print(f"Configuration: {config['name']} ({config['description']})")
        print(f"Model size: {results[config_name]['model_size_gb']} GB")
        print(f"Applying bit configuration W={config['W']}, A={config['A']}, KV={config['KV']}")

        # 1. Zero-shot evaluation (6 benchmarks)
        print("\n1. Zero-shot common sense evaluation...")
        try:
            zero_shot_results = zero_shot_evaluator.evaluate_all_tasks(config)
            results[config_name]['zero_shot'] = zero_shot_results
            print(f"   BoolQ: {zero_shot_results.get('BoolQ', 0):.1f}%")
            print(f"   HellaSwag: {zero_shot_results.get('HellaSwag', 0):.1f}%")
            print(f"   WinoGrande: {zero_shot_results.get('WinoGrande', 0):.1f}%")
            print(f"   ARC-e: {zero_shot_results.get('ARC-e', 0):.1f}%")
            print(f"   ARC-c: {zero_shot_results.get('ARC-c', 0):.1f}%")
            print(f"   OBQA: {zero_shot_results.get('OBQA', 0):.1f}%")
            print(f"   Average: {zero_shot_results.get('Average', 0):.1f}%")
        except Exception as e:
            print(f"   Warning: Zero-shot evaluation failed: {e}")
            results[config_name]['zero_shot'] = {'Average': 0.0}

        # 2. Perplexity evaluation with sliding window
        print("\n2. Perplexity evaluation (sliding window)...")
        try:
            perplexity_results = perplexity_evaluator.evaluate_all_datasets(config)
            results[config_name]['perplexity'] = perplexity_results
            print(f"   WikiText2: {perplexity_results['WikiText2']:.1f}")
            print(f"   C4: {perplexity_results['C4']:.1f}")
        except Exception as e:
            print(f"   Warning: Perplexity evaluation failed: {e}")
            results[config_name]['perplexity'] = {'WikiText2': float('inf'), 'C4': float('inf')}

        # 3. Few-shot evaluation (5-shot)
        print("\n3. Few-shot evaluation (5-shot)...")
        try:
            mmlu_scores = few_shot_evaluator.evaluate_mmlu(config, num_shots=5)
            triviaqa_score = few_shot_evaluator.evaluate_triviaqa(config, num_shots=5)

            results[config_name]['few_shot'] = {
                'MMLU': mmlu_scores,
                'TriviaQA': triviaqa_score
            }
            print(f"   MMLU by category:")
            print(f"     - Humanities: {mmlu_scores['Humanities']:.1f}%")
            print(f"     - STEM: {mmlu_scores['STEM']:.1f}%")
            print(f"     - Social Sciences: {mmlu_scores['Social Sciences']:.1f}%")
            print(f"     - Other: {mmlu_scores['Other']:.1f}%")
            print(f"     - Average: {mmlu_scores['Average']:.1f}%")
            print(f"   TriviaQA: {triviaqa_score:.1f}%")
        except Exception as e:
            print(f"   Warning: Few-shot evaluation failed: {e}")
            results[config_name]['few_shot'] = {
                'MMLU': {'Average': 0.0},
                'TriviaQA': 0.0
            }

        # Calculate compression ratio
        compression = 32 / config['W']  # Simple compression based on weight bits
        results[config_name]['compression_ratio'] = compression
        print(f"\n   Compression ratio: {compression:.2f}x")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    results_filename = eval_config['output']['results_filename']
    with open(output_path / results_filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Evaluation complete!")
    print(f"Results saved to {output_path}")
    print(f"{'='*70}")

    print("\nSummary of Results:")
    print("="*70)
    if config_name in results:
        result = results[config_name]
        print(f"\n{config_name} ({result['bits']}):")
        print(f"  Model size: {result['model_size_gb']} GB")
        print(f"  Compression: {result['compression_ratio']:.1f}x")

        if 'zero_shot' in result and result['zero_shot']:
            print(f"  Zero-shot avg: {result['zero_shot'].get('Average', 0):.1f}%")

        if 'perplexity' in result and result['perplexity']:
            if 'WikiText2' in result['perplexity']:
                print(f"  WikiText2 PPL: {result['perplexity']['WikiText2']:.1f}")
            if 'C4' in result['perplexity']:
                print(f"  C4 PPL: {result['perplexity']['C4']:.1f}")

        if 'few_shot' in result and result['few_shot']:
            if 'MMLU' in result['few_shot']:
                print(f"  MMLU avg: {result['few_shot']['MMLU'].get('Average', 0):.1f}%")
            if 'TriviaQA' in result['few_shot']:
                print(f"  TriviaQA: {result['few_shot']['TriviaQA']:.1f}%")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
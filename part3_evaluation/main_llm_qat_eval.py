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

from shared.models_sp import SPModel, SPLMHeadModel
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

from part3_evaluation.llm_qat_metrics import LLMQATEvaluation
from part3_evaluation.bit_configurations import BitConfigurations
from part3_evaluation.generate_tables import ResultTableGenerator
from part3_evaluation.baseline_comparison import BaselineComparison


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

    def __init__(self, model, device='cuda'):
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


def load_switchable_model(model_path: str = None, config_path: str = None, use_pretrained: bool = True):
    """Load switchable precision model with proper configuration"""

    # Force CUDA availability check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This evaluation requires CUDA.")

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Default bit widths - will be overridden if loading from checkpoint
    default_bit_widths = [6, 8, 16]

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
            raise ValueError("No model configuration found! Please provide a valid checkpoint with model_config or specify --config_path")

        # STRICT CONFIG USAGE - Use exactly what's in the config, no defaults
        print("\n" + "="*50)
        print("USING STRICT CONFIGURATION FROM CHECKPOINT/JSON")
        print("="*50)

        # Extract required values from model_config
        n_layer = model_config.get('n_layer')
        n_embd = model_config.get('n_embd')
        n_head = model_config.get('n_head')
        quantization_bits = model_config.get('quantization_bits')

        # Validate required fields
        if n_layer is None or n_embd is None or n_head is None:
            raise ValueError(f"Missing required model config fields. Got: n_layer={n_layer}, n_embd={n_embd}, n_head={n_head}")

        # Determine bit widths from config or use switchable defaults only if not specified
        bit_widths = model_config.get('bit_widths')
        if bit_widths is None:
            # Only use defaults if bit_widths not explicitly set
            if quantization_bits:
                print(f"Model trained with quantization_bits={quantization_bits}, using switchable bit widths [6, 8, 16]")
                bit_widths = [6, 8, 16]
            else:
                raise ValueError("No bit_widths or quantization_bits specified in model config")

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

        # Build config with ONLY values from the loaded configuration
        config = GPT2Config(
            vocab_size=model_config.get('vocab_size', 50257),  # GPT-2 standard
            n_positions=actual_n_positions,
            n_embd=n_embd,  # From config, no default
            n_layer=n_layer,  # From config, no default
            n_head=n_head,  # From config, no default
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

        print(f"Creating model with bit-widths: {bit_widths}")
        # Add SP-specific configurations to config
        config.bit_widths = bit_widths

        # Get LoRA configurations from model_config - use CORRECT defaults from config_sp.py
        config.lora_rank_per_bit = model_config.get('lora_rank_per_bit', {6: 32, 8: 16, 16: 16, 32: 0})  # Fixed: 16-bit should be rank=16
        config.lora_alpha_per_bit = model_config.get('lora_alpha_per_bit', {6: 64, 8: 32, 16: 32, 32: 0})  # Fixed: 16-bit alpha=32
        config.activation_bits_per_bit = model_config.get('activation_bits_per_bit', {6: 6, 8: 8, 16: 16, 32: 32})
        config.quantizer_per_bit = model_config.get('quantizer_per_bit', None)

        # Print what we're using to help debug
        print(f"LoRA rank per bit: {config.lora_rank_per_bit}")
        print(f"LoRA alpha per bit: {config.lora_alpha_per_bit}")

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


class EvaluationMetrics:
    """Standard evaluation metrics for switchable precision models"""

    @staticmethod
    def calculate_perplexity(model, data_loader, max_samples: int = 100):
        """Calculate perplexity on a dataset"""
        model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, desc="Calculating perplexity", total=max_samples)):
                if i >= max_samples:
                    break

                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].cuda()
                    attention_mask = batch.get('attention_mask', None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.cuda()
                else:
                    input_ids = batch.cuda()
                    attention_mask = None

                outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss

                # Count actual tokens (excluding padding)
                if attention_mask is not None:
                    num_tokens = attention_mask.sum().item()
                else:
                    num_tokens = input_ids.numel()

                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        return perplexity

    @staticmethod
    def evaluate_accuracy(model, data_loader, task_type: str = 'classification', max_samples: int = 100):
        """Evaluate accuracy on downstream tasks"""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, desc=f"Evaluating {task_type}", total=max_samples)):
                if i >= max_samples:
                    break

                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].cuda()
                    labels = batch.get('labels', batch['input_ids']).cuda()
                else:
                    input_ids = batch.cuda()
                    labels = input_ids

                outputs = model(input_ids)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits

                if task_type == 'classification':
                    # For classification tasks
                    predictions = torch.argmax(logits[:, -1, :], dim=-1)
                    correct += (predictions == labels[:, -1]).sum().item()
                    total += labels.shape[0]
                else:
                    # For language modeling tasks
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    predictions = torch.argmax(shift_logits, dim=-1)
                    correct += (predictions == shift_labels).sum().item()
                    total += shift_labels.numel()

        accuracy = (correct / total * 100) if total > 0 else 0
        return accuracy

    @staticmethod
    def measure_inference_speed(model, input_shape=(1, 128), num_iterations: int = 100, warmup: int = 10):
        """Measure inference speed in tokens/second"""
        model.eval()
        device = next(model.parameters()).device

        # Create dummy input
        dummy_input = torch.randint(0, 50257, input_shape).to(device)

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(dummy_input)

        # Measure
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(dummy_input)

        torch.cuda.synchronize()
        end_time = time.time()

        elapsed_time = end_time - start_time
        tokens_processed = num_iterations * input_shape[0] * input_shape[1]
        tokens_per_second = tokens_processed / elapsed_time

        return tokens_per_second

    @staticmethod
    def calculate_compression_ratio(model, baseline_bits: int = 32):
        """Calculate model compression ratio"""
        try:
            # Try to get current_bits from transformer
            current_bits = model.transformer.current_bits
        except AttributeError:
            # Default to 16 if not found
            current_bits = 16
        compression_ratio = baseline_bits / current_bits
        return compression_ratio

    @staticmethod
    def evaluate_robustness(model, data_loader, noise_levels: List[float] = [0.01, 0.05, 0.1]):
        """Evaluate model robustness to input noise"""
        model.eval()
        robustness_scores = {}

        for noise_level in noise_levels:
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in data_loader:
                    if isinstance(batch, dict):
                        input_ids = batch['input_ids'].cuda()
                    else:
                        input_ids = batch.cuda()

                    # Add noise to embeddings
                    # For SPLMHeadModel, wte is in transformer
                    embeddings = model.transformer.wte(input_ids)
                    noise = torch.randn_like(embeddings) * noise_level
                    noisy_embeddings = embeddings + noise

                    # Forward pass with noisy embeddings
                    # Note: This requires model to accept embeddings directly
                    # For standard implementation, we'll skip actual noise injection
                    outputs = model(input_ids)

                    # Calculate accuracy (simplified)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
                    predictions = torch.argmax(logits[:, :-1, :], dim=-1)
                    targets = input_ids[:, 1:]
                    correct += (predictions == targets).sum().item()
                    total += targets.numel()

                    if total > 1000:  # Limit evaluation
                        break

            accuracy = (correct / total * 100) if total > 0 else 0
            robustness_scores[f'noise_{noise_level}'] = accuracy

        return robustness_scores


def main():
    parser = argparse.ArgumentParser(description='LLM-QAT Paper Evaluation Suite with Standard Methods')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to training config JSON file (optional, will auto-detect if not provided)')
    parser.add_argument('--output_dir', type=str, default='part3_evaluation/results',
                       help='Directory to save results')
    args = parser.parse_args()

    # Force CUDA check - always required
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required but not available!")
        print("Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support installed.")
        sys.exit(1)

    # Load model from checkpoint
    model = load_switchable_model(args.model_path, config_path=args.config_path, use_pretrained=False)
    tokenizer = load_tokenizer()

    # Initialize evaluation components
    evaluator = LLMQATEvaluation(model, tokenizer)
    metrics = EvaluationMetrics()

    # Prepare calibration data
    print("\nPreparing calibration data...")
    calibration_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
    calibration_texts = [sample['text'] for sample in calibration_dataset if len(sample['text'].strip()) > 0][:100]

    # Tokenize calibration data
    calibration_data = []
    for text in calibration_texts:
        tokens = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
        # Squeeze to remove batch dimension since DataLoader will add it back
        calibration_data.append(tokens['input_ids'].squeeze(0))

    calibration_loader = torch.utils.data.DataLoader(
        calibration_data, batch_size=4, shuffle=False
    )

    # Initialize calibration manager and perform warm-up
    calib_manager = CalibrationManager(model)
    try:
        # Try to access bit_widths attribute
        model_bit_widths = model.transformer.bit_widths  # For SPLMHeadModel, bit_widths is in transformer
        print("\nPerforming warm-up calibration for all bit-widths...")
        warm_up_samples = 10  # Default warm-up samples
        calib_manager.warm_up_calibration(calibration_loader, model_bit_widths, num_batches=warm_up_samples)
    except AttributeError as e:
        raise AttributeError(f"Model does not support switchable precision: {e}\nPlease ensure the model was trained with SPLMHeadModel.")

    # Model supports switchable precision
    supported_bit_widths = model.transformer.bit_widths  # Access through transformer
    print(f"Model supports bit-widths: {supported_bit_widths}")

    # Map bit-widths to configuration names
    bit_to_config = {
        6: 'INT6',
        8: 'INT8',
        16: 'FP16',
        32: 'FP32'
    }

    # Auto-detect configurations based on model's supported bit widths
    configs_to_eval = [bit_to_config.get(b, f'INT{b}') for b in supported_bit_widths if b in bit_to_config]
    print(f"Configurations to evaluate: {configs_to_eval}")

    # Hardcoded evaluation settings
    max_eval_samples = 100
    output_dir = args.output_dir

    print("="*70)
    print("Running SP Model Evaluation")
    print("="*70)
    print(f"Model: GPT-2 ({evaluator.model_params:.1f}M parameters)")
    print(f"Configurations: {configs_to_eval}")
    print(f"Output directory: {output_dir}")
    print(f"Max evaluation samples: {max_eval_samples}")
    print("="*70)

    results = {}

    # Run simplified evaluation for each supported bit width
    for config_name in configs_to_eval:
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

        # Run basic perplexity evaluation
        print("\nPerplexity evaluation...")

        # Load evaluation datasets
        wiki_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        wiki_texts = [sample['text'] for sample in wiki_dataset if len(sample['text'].strip()) > 0][:max_eval_samples]

        # Tokenize data
        wiki_data = []
        for text in wiki_texts:
            tokens = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
            wiki_data.append(tokens)

        wiki_loader = torch.utils.data.DataLoader(wiki_data, batch_size=4, shuffle=False)

        # Calculate perplexity
        wiki_ppl = metrics.calculate_perplexity(model, wiki_loader, max_samples=max_eval_samples)

        results[config_name]['perplexity'] = {'WikiText2': wiki_ppl}
        print(f"   WikiText2 Perplexity: {wiki_ppl:.1f}")

        # Simple inference speed test
        print("\nInference Speed evaluation...")
        tokens_per_sec = metrics.measure_inference_speed(
            model, input_shape=(1, 128), num_iterations=20
        )
        results[config_name]['inference_speed'] = tokens_per_sec
        print(f"   Speed: {tokens_per_sec:.1f} tokens/second")

        # Calculate compression ratio
        compression = metrics.calculate_compression_ratio(model)
        results[config_name]['compression_ratio'] = compression
        print(f"   Compression ratio: {compression:.2f}x")

    # Save results
    output_path = Path(output_dir)
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
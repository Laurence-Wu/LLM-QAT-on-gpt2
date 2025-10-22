import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List, Tuple
import json
import os
import sys
import argparse
from datetime import datetime
from transformers import GPT2Config, GPT2TokenizerFast

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
part5_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(part5_dir)
sys.path.insert(0, project_root)

from part5_squad.squad_metrics import evaluate_squad
from part5_squad.models_squad import SPQuestionAnsweringModel
from part5_squad.dataset_squad import SQuADDataset


def extract_answer(start_logits, end_logits, input_ids, tokenizer,
                   max_answer_length=30, n_best_size=20, question_length=None):
    """
    Extract answer span from logits using beam search

    Searches over top-N start and top-N end positions to find
    the best valid span (start <= end, length <= max_answer_length)

    Args:
        start_logits: [seq_length] - Start position logits
        end_logits: [seq_length] - End position logits
        input_ids: [seq_length] - Input token IDs
        tokenizer: Tokenizer for decoding
        max_answer_length: Maximum answer length in tokens
        n_best_size: Number of top positions to consider
        question_length: Length of question (to exclude from answer)

    Returns:
        Dict with 'text', 'start', 'end', 'score'
    """
    seq_length = start_logits.shape[0]

    # Get top N start and end positions
    start_top_log_probs, start_top_indices = torch.topk(start_logits, min(n_best_size, seq_length))
    end_top_log_probs, end_top_indices = torch.topk(end_logits, min(n_best_size, seq_length))

    # Find best valid span
    best_score = float('-inf')
    best_start = 0
    best_end = 0

    for start_idx in start_top_indices:
        for end_idx in end_top_indices:
            start_pos = start_idx.item()
            end_pos = end_idx.item()

            # Validate span
            if end_pos < start_pos:
                continue
            if end_pos - start_pos + 1 > max_answer_length:
                continue
            # Exclude question part if specified
            if question_length and start_pos < question_length:
                continue

            # Score is sum of log probabilities
            score = (start_logits[start_pos] + end_logits[end_pos]).item()

            if score > best_score:
                best_score = score
                best_start = start_pos
                best_end = end_pos

    # Decode answer
    answer_tokens = input_ids[best_start:best_end+1]
    answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return {
        'text': answer_text,
        'start': best_start,
        'end': best_end,
        'score': best_score
    }


def load_squad_model_from_checkpoint(checkpoint_path, device):
    """
    Load SPQuestionAnsweringModel from checkpoint

    Follows part5_squad/eval_squad.py pattern for model creation.
    Loads calibrated quantizers from checkpoint state_dict.

    Args:
        checkpoint_path: Path to .pth checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, bit_width)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract configs from checkpoint
    model_config = checkpoint.get('model_config')
    training_config = checkpoint.get('training_config')
    bit_width = checkpoint.get('bit_width')

    if model_config is None:
        raise ValueError("Checkpoint missing model_config")
    if bit_width is None:
        raise ValueError("Checkpoint missing bit_width")

    print(f"Checkpoint bit-width: {bit_width}")

    # Create GPT2Config
    gpt2_config = GPT2Config(
        vocab_size=model_config['vocab_size'],
        n_positions=model_config['n_positions'],
        n_embd=model_config['n_embd'],
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        activation_function='gelu_new',
        layer_norm_epsilon=model_config.get('layer_norm_epsilon', 1e-5),
        embd_pdrop=model_config.get('embd_pdrop', 0.1)
    )

    # Add switchable precision config
    gpt2_config.quantization_bits = model_config.get('quantization_bits', 8)
    gpt2_config.lora_rank = model_config.get('lora_rank', 16)
    gpt2_config.lora_alpha = model_config.get('lora_alpha', 32)
    gpt2_config.lora_rank_per_bit = model_config['lora_rank_per_bit']
    gpt2_config.lora_alpha_per_bit = model_config['lora_alpha_per_bit']
    gpt2_config.activation_bits_per_bit = model_config['activation_bits_per_bit']
    gpt2_config.quantizer_per_bit = model_config['quantizer_per_bit']
    gpt2_config.bit_widths = model_config['bit_widths']

    # Convert string keys to int if needed
    for attr_name in ['lora_rank_per_bit', 'lora_alpha_per_bit', 'activation_bits_per_bit', 'quantizer_per_bit']:
        attr_val = getattr(gpt2_config, attr_name)
        if isinstance(attr_val, dict):
            setattr(gpt2_config, attr_name, {int(k) if isinstance(k, str) else k: v for k, v in attr_val.items()})

    # Create QA model
    print("Initializing SPQuestionAnsweringModel...")
    model = SPQuestionAnsweringModel(gpt2_config)

    # Load state_dict (includes calibrated quantizers from training)
    print("Loading model weights and calibrated quantizers from checkpoint...")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move to device
    model = model.to(device)

    # Set precision
    model.set_precision(bit_width)

    print(f"Model loaded successfully at {bit_width}-bit precision")
    print("Calibrated quantizers loaded from checkpoint (no re-calibration needed)")

    return model, bit_width


def evaluate_squad_model_at_dtype(model, dataset, tokenizer, device, dtype_name,
                                   max_answer_length=30, n_best_size=20, max_examples=None):
    """
    Evaluate QA model on SQuAD dataset at specified dtype

    Args:
        model: SPQuestionAnsweringModel (already cast to target dtype)
        dataset: SQuADDataset
        tokenizer: Tokenizer
        device: Device
        dtype_name: Name of dtype for logging (fp32, fp16, bf16)
        max_answer_length: Maximum answer length
        n_best_size: Number of top positions for beam search
        max_examples: Maximum number of examples to evaluate (for debugging)

    Returns:
        Dict with 'exact_match', 'f1', 'total' scores
    """
    model.eval()

    print(f"Evaluating at {dtype_name.upper()} precision...")

    predictions = []
    num_examples = 0

    with torch.no_grad():
        for example in tqdm(dataset, desc=f"Evaluating {dtype_name.upper()}"):
            if max_examples and num_examples >= max_examples:
                break

            # Move inputs to device and ensure they match model dtype
            input_ids = example['input_ids'].unsqueeze(0).to(device)
            attention_mask = example['attention_mask'].unsqueeze(0).to(device)

            # Cast inputs to model's dtype if needed
            # Note: input_ids are long, so only attention_mask needs dtype casting if it's float

            # Forward pass
            try:
                outputs = model(input_ids, attention_mask=attention_mask)

                # Cast logits back to FP32 for stable computation
                start_logits = outputs['start_logits'][0].float()
                end_logits = outputs['end_logits'][0].float()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: OOM at example {num_examples}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            # Extract answer
            answer = extract_answer(
                start_logits,
                end_logits,
                input_ids[0],
                tokenizer,
                max_answer_length=max_answer_length,
                n_best_size=n_best_size
            )

            predictions.append({
                'id': example['example_id'],
                'prediction_text': answer['text']
            })

            num_examples += 1

    # Compute metrics
    results = evaluate_squad(predictions, dataset.dataset)

    print(f"\n{dtype_name.upper()} Results:")
    print(f"  Exact Match: {results['exact_match']:.2f}%")
    print(f"  F1 Score: {results['f1']:.2f}%")
    print(f"  Total Examples: {results['total']}")

    return results


def evaluate_fp32(checkpoint_path, dataset, tokenizer, device, config):
    """
    Evaluate model at FP32 precision (baseline)

    Args:
        checkpoint_path: Path to checkpoint
        dataset: SQuADDataset
        tokenizer: Tokenizer
        device: Device
        config: Evaluation config dict

    Returns:
        Dict with evaluation results
    """
    print("\n" + "="*70)
    print("Loading model for FP32 evaluation")
    print("="*70)

    # Load model at original precision
    model, bit_width = load_squad_model_from_checkpoint(checkpoint_path, device)
    model.eval()

    # Evaluate
    dataset_name = 'squad_v1' if hasattr(dataset, 'version') and dataset.version == 'v1' else 'squad_v2'
    results = evaluate_squad_model_at_dtype(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        device=device,
        dtype_name='fp32',
        max_answer_length=config[dataset_name]['max_answer_length'],
        n_best_size=config[dataset_name]['n_best_size'],
        max_examples=config[dataset_name]['max_examples']
    )

    # Clean up
    del model
    torch.cuda.empty_cache()

    return results


def evaluate_fp16(checkpoint_path, dataset, tokenizer, device, config):
    """
    Evaluate model at FP16 precision

    Args:
        checkpoint_path: Path to checkpoint
        dataset: SQuADDataset
        tokenizer: Tokenizer
        device: Device
        config: Evaluation config dict

    Returns:
        Dict with evaluation results
    """
    print("\n" + "="*70)
    print("Loading model for FP16 evaluation")
    print("="*70)

    # Check if FP16 is supported
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, FP16 may not be supported on CPU")

    # Load model at original precision
    model, bit_width = load_squad_model_from_checkpoint(checkpoint_path, device)

    # Switch to FP16 via set_precision flag
    print("Setting model to FP16...")
    model.set_precision(16.0)
    model.eval()

    # Evaluate
    dataset_name = 'squad_v1' if hasattr(dataset, 'version') and dataset.version == 'v1' else 'squad_v2'
    results = evaluate_squad_model_at_dtype(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        device=device,
        dtype_name='fp16',
        max_answer_length=config[dataset_name]['max_answer_length'],
        n_best_size=config[dataset_name]['n_best_size'],
        max_examples=config[dataset_name]['max_examples']
    )

    # Clean up
    del model
    torch.cuda.empty_cache()

    return results


def evaluate_bf16(checkpoint_path, dataset, tokenizer, device, config):
    """
    Evaluate model at BF16 precision

    Args:
        checkpoint_path: Path to checkpoint
        dataset: SQuADDataset
        tokenizer: Tokenizer
        device: Device
        config: Evaluation config dict

    Returns:
        Dict with evaluation results
    """
    print("\n" + "="*70)
    print("Loading model for BF16 evaluation")
    print("="*70)

    # Check if BF16 is supported
    if device.type == 'cuda':
        if not torch.cuda.is_bf16_supported():
            print("WARNING: BF16 not supported on this GPU, skipping BF16 evaluation")
            return None
    else:
        print("WARNING: BF16 evaluation on CPU may be slow")

    # Load model at original precision
    model, bit_width = load_squad_model_from_checkpoint(checkpoint_path, device)

    # Switch to BF16 via set_precision flag
    print("Setting model to BF16...")
    model.set_precision(16.5)
    model.eval()

    # Evaluate
    dataset_name = 'squad_v1' if hasattr(dataset, 'version') and dataset.version == 'v1' else 'squad_v2'
    results = evaluate_squad_model_at_dtype(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        device=device,
        dtype_name='bf16',
        max_answer_length=config[dataset_name]['max_answer_length'],
        n_best_size=config[dataset_name]['n_best_size'],
        max_examples=config[dataset_name]['max_examples']
    )

    # Clean up
    del model
    torch.cuda.empty_cache()

    return results


def load_evaluation_config(config_path='evaluation_config.json'):
    """
    Load evaluation configuration from JSON file

    Args:
        config_path: Path to evaluation config

    Returns:
        Config dictionary
    """
    # Get config path relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_config_path = os.path.join(current_dir, config_path)

    if not os.path.exists(full_config_path):
        raise FileNotFoundError(f"Evaluation config not found: {full_config_path}")

    with open(full_config_path, 'r') as f:
        config = json.load(f)

    return config


def main():
    """
    Main evaluation function for SQuAD QA with FP16/BF16 comparison

    Evaluates model at FP32, FP16, and BF16 precisions on both SQuAD v1.1 and v2.0
    Automatically saves results to JSON with timestamp
    """
    parser = argparse.ArgumentParser(description='Evaluate SQuAD QA Model at FP32/FP16/BF16')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('--skip-fp16', action='store_true',
                       help='Skip FP16 evaluation')
    parser.add_argument('--skip-bf16', action='store_true',
                       help='Skip BF16 evaluation')
    parser.add_argument('--squad-v1-only', action='store_true',
                       help='Evaluate only on SQuAD v1.1')
    parser.add_argument('--squad-v2-only', action='store_true',
                       help='Evaluate only on SQuAD v2.0')
    args = parser.parse_args()

    # Load evaluation config
    print("Loading evaluation configuration...")
    config = load_evaluation_config()

    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, using CPU")
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare results dictionary
    all_results = {
        'model_path': args.checkpoint,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'device': str(device),
        'squad_v1': {},
        'squad_v2': {}
    }

    # Evaluate on SQuAD v1.1
    if not args.squad_v2_only:
        print("\n" + "="*70)
        print("EVALUATING ON SQUAD V1.1")
        print("="*70)

        squad_v1_config = config['squad_v1']
        squad_v1_dataset = SQuADDataset(
            tokenizer=tokenizer,
            split=squad_v1_config['split'],
            max_length=384,
            version='v1'
        )

        # FP32 evaluation
        v1_fp32 = evaluate_fp32(args.checkpoint, squad_v1_dataset, tokenizer, device, config)
        all_results['squad_v1']['fp32'] = v1_fp32

        # FP16 evaluation
        if not args.skip_fp16:
            v1_fp16 = evaluate_fp16(args.checkpoint, squad_v1_dataset, tokenizer, device, config)
            all_results['squad_v1']['fp16'] = v1_fp16

        # BF16 evaluation
        if not args.skip_bf16:
            v1_bf16 = evaluate_bf16(args.checkpoint, squad_v1_dataset, tokenizer, device, config)
            if v1_bf16 is not None:
                all_results['squad_v1']['bf16'] = v1_bf16

    # Evaluate on SQuAD v2.0
    if not args.squad_v1_only:
        print("\n" + "="*70)
        print("EVALUATING ON SQUAD V2.0")
        print("="*70)

        squad_v2_config = config['squad_v2']
        squad_v2_dataset = SQuADDataset(
            tokenizer=tokenizer,
            split=squad_v2_config['split'],
            max_length=384,
            version='v2'
        )

        # FP32 evaluation
        v2_fp32 = evaluate_fp32(args.checkpoint, squad_v2_dataset, tokenizer, device, config)
        all_results['squad_v2']['fp32'] = v2_fp32

        # FP16 evaluation
        if not args.skip_fp16:
            v2_fp16 = evaluate_fp16(args.checkpoint, squad_v2_dataset, tokenizer, device, config)
            all_results['squad_v2']['fp16'] = v2_fp16

        # BF16 evaluation
        if not args.skip_bf16:
            v2_bf16 = evaluate_bf16(args.checkpoint, squad_v2_dataset, tokenizer, device, config)
            if v2_bf16 is not None:
                all_results['squad_v2']['bf16'] = v2_bf16

    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Model: {args.checkpoint}")
    print(f"Device: {device}")
    print()

    if not args.squad_v2_only and all_results['squad_v1']:
        print("SQuAD v1.1:")
        print(f"{'Precision':<12} {'Exact Match':<15} {'F1 Score':<15}")
        print("-"*70)
        for dtype in ['fp32', 'fp16', 'bf16']:
            if dtype in all_results['squad_v1']:
                res = all_results['squad_v1'][dtype]
                print(f"{dtype.upper():<12} {res['exact_match']:>6.2f}%{'':<8} {res['f1']:>6.2f}%")
        print()

    if not args.squad_v1_only and all_results['squad_v2']:
        print("SQuAD v2.0:")
        print(f"{'Precision':<12} {'Exact Match':<15} {'F1 Score':<15}")
        print("-"*70)
        for dtype in ['fp32', 'fp16', 'bf16']:
            if dtype in all_results['squad_v2']:
                res = all_results['squad_v2'][dtype]
                print(f"{dtype.upper():<12} {res['exact_match']:>6.2f}%{'':<8} {res['f1']:>6.2f}%")

    print("="*70)

    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"fp_comparison_results_{timestamp}.json"
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), results_filename)

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == '__main__':
    main()

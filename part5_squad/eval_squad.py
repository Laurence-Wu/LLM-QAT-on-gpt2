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


def extract_answer_batch(start_logits, end_logits, input_ids, tokenizer,
                         max_answer_length=30, n_best_size=20):
    """
    Extract answers for a batch

    Args:
        start_logits: [batch_size, seq_length]
        end_logits: [batch_size, seq_length]
        input_ids: [batch_size, seq_length]
        tokenizer: Tokenizer
        max_answer_length: Maximum answer length
        n_best_size: Number of top positions to consider

    Returns:
        List of answer dicts
    """
    batch_size = start_logits.shape[0]
    answers = []

    for i in range(batch_size):
        answer = extract_answer(
            start_logits[i],
            end_logits[i],
            input_ids[i],
            tokenizer,
            max_answer_length=max_answer_length,
            n_best_size=n_best_size
        )
        answers.append(answer)

    return answers


def evaluate_squad_model(model, dataset, tokenizer, device, bit_width,
                        max_answer_length=30, n_best_size=20, max_examples=None):
    """
    Evaluate QA model on SQuAD dataset

    Args:
        model: SPQuestionAnsweringModel
        dataset: SQuADDataset
        tokenizer: Tokenizer
        device: Device
        bit_width: Precision to evaluate at
        max_answer_length: Maximum answer length
        n_best_size: Number of top positions for beam search
        max_examples: Maximum number of examples to evaluate (for debugging)

    Returns:
        Dict with 'exact_match', 'f1', 'total' scores
    """
    model.eval()
    model.set_precision(bit_width)

    print(f"Evaluating at {bit_width}-bit precision...")

    predictions = []
    num_examples = 0

    with torch.no_grad():
        for example in tqdm(dataset, desc=f"Evaluating {bit_width}-bit"):
            if max_examples and num_examples >= max_examples:
                break

            input_ids = example['input_ids'].unsqueeze(0).to(device)
            attention_mask = example['attention_mask'].unsqueeze(0).to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)

            # Extract answer
            answer = extract_answer(
                outputs['start_logits'][0],
                outputs['end_logits'][0],
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

    print(f"\n{bit_width}-bit Results:")
    print(f"  Exact Match: {results['exact_match']:.2f}%")
    print(f"  F1 Score: {results['f1']:.2f}%")
    print(f"  Total Examples: {results['total']}")

    return results


def evaluate_all_precisions(model, dataset, tokenizer, device, bit_widths,
                           max_answer_length=30, n_best_size=20, max_examples=None):
    """
    Evaluate model at all precision levels

    Args:
        model: SPQuestionAnsweringModel
        dataset: SQuADDataset
        tokenizer: Tokenizer
        device: Device
        bit_widths: List of bit-widths to evaluate
        max_answer_length: Maximum answer length
        n_best_size: Number of top positions for beam search
        max_examples: Maximum number of examples (for debugging)

    Returns:
        Dict mapping bit_width -> results
    """
    all_results = {}

    for bits in bit_widths:
        results = evaluate_squad_model(
            model, dataset, tokenizer, device, bits,
            max_answer_length=max_answer_length,
            n_best_size=n_best_size,
            max_examples=max_examples
        )
        all_results[bits] = results

    # Print summary
    print("\n" + "="*70)
    print("Evaluation Summary")
    print("="*70)
    print(f"{'Precision':<15} {'Exact Match':<15} {'F1 Score':<15}")
    print("-"*70)
    for bits in sorted(bit_widths):
        results = all_results[bits]
        print(f"{bits}-bit{'':<10} {results['exact_match']:>6.2f}%{'':<8} {results['f1']:>6.2f}%")
    print("="*70)

    return all_results


def save_predictions(predictions, output_file):
    """
    Save predictions to JSON file

    Args:
        predictions: List of prediction dicts
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {output_file}")


def save_results(results, output_file):
    """
    Save evaluation results to JSON file

    Args:
        results: Results dict
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


def load_squad_model_from_checkpoint(checkpoint_path, device):
    """
    Load SPQuestionAnsweringModel from checkpoint

    Follows main_squad.py initialize_model() pattern for model creation.
    Loads calibrated quantizers from checkpoint state_dict.

    Args:
        checkpoint_path: Path to .pth checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, bit_width)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract configs from checkpoint
    model_config = checkpoint.get('model_config')
    training_config = checkpoint.get('training_config')
    bit_width = checkpoint.get('bit_width')

    if model_config is None:
        raise ValueError("Checkpoint missing model_config")
    if bit_width is None:
        raise ValueError("Checkpoint missing bit_width")

    print(f"Checkpoint bit-width: {bit_width}")

    # Create GPT2Config following main_squad.py initialize_model() and part3_eval_sp pattern
    # model_config is a dictionary, not an object (saved by deploy.py lines 50-56)
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

    # Add switchable precision config (following part3_eval_sp pattern)
    gpt2_config.quantization_bits = model_config.get('quantization_bits', 8)
    gpt2_config.lora_rank = model_config.get('lora_rank', 16)
    gpt2_config.lora_alpha = model_config.get('lora_alpha', 32)
    gpt2_config.lora_rank_per_bit = model_config['lora_rank_per_bit']
    gpt2_config.lora_alpha_per_bit = model_config['lora_alpha_per_bit']
    gpt2_config.activation_bits_per_bit = model_config['activation_bits_per_bit']
    gpt2_config.quantizer_per_bit = model_config['quantizer_per_bit']
    gpt2_config.bit_widths = model_config['bit_widths']

    # Convert string keys to int if needed (part3_eval_sp lines 61-64)
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


def load_evaluation_config_squad(config_path='evaluation_config_squad.json'):
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
    Main evaluation function for SQuAD QA

    Evaluates model on both SQuAD v1.1 and v2.0
    Automatically saves results to JSON with timestamp
    """
    parser = argparse.ArgumentParser(description='Evaluate SQuAD QA Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint (.pth)')
    args = parser.parse_args()

    # Load evaluation config
    print("Loading evaluation configuration...")
    config = load_evaluation_config_squad()

    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    print(f"Using device: {device}")

    # Load model from checkpoint (calibration params loaded from .pth)
    model, bit_width = load_squad_model_from_checkpoint(args.model_path, device)
    model.eval()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare results dictionary
    all_results = {
        'model_path': args.model_path,
        'bit_width': bit_width,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'squad_v1': {},
        'squad_v2': {}
    }

    # Evaluate on SQuAD v1.1
    print("\n" + "="*70)
    print("Evaluating on SQuAD v1.1")
    print("="*70)

    squad_v1_config = config['squad_v1']
    squad_v1_dataset = SQuADDataset(
        tokenizer=tokenizer,
        split=squad_v1_config['split'],
        max_length=384,
        version='v1'
    )

    v1_results = evaluate_squad_model(
        model=model,
        dataset=squad_v1_dataset,
        tokenizer=tokenizer,
        device=device,
        bit_width=bit_width,
        max_answer_length=squad_v1_config['max_answer_length'],
        n_best_size=squad_v1_config['n_best_size'],
        max_examples=squad_v1_config['max_examples']
    )
    all_results['squad_v1'] = v1_results

    # Evaluate on SQuAD v2.0
    print("\n" + "="*70)
    print("Evaluating on SQuAD v2.0")
    print("="*70)

    squad_v2_config = config['squad_v2']
    squad_v2_dataset = SQuADDataset(
        tokenizer=tokenizer,
        split=squad_v2_config['split'],
        max_length=384,
        version='v2'
    )

    v2_results = evaluate_squad_model(
        model=model,
        dataset=squad_v2_dataset,
        tokenizer=tokenizer,
        device=device,
        bit_width=bit_width,
        max_answer_length=squad_v2_config['max_answer_length'],
        n_best_size=squad_v2_config['n_best_size'],
        max_examples=squad_v2_config['max_examples']
    )
    all_results['squad_v2'] = v2_results

    # Print summary
    print("\n" + "="*70)
    print("Evaluation Summary")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Precision: {bit_width}-bit")
    print()
    print(f"{'Dataset':<20} {'Exact Match':<15} {'F1 Score':<15}")
    print("-"*70)
    print(f"{'SQuAD v1.1':<20} {v1_results['exact_match']:>6.2f}%{'':<8} {v1_results['f1']:>6.2f}%")
    print(f"{'SQuAD v2.0':<20} {v2_results['exact_match']:>6.2f}%{'':<8} {v2_results['f1']:>6.2f}%")
    print("="*70)

    # Always save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"squad_eval_results_{bit_width}bit_{timestamp}.json"

    with open(results_filename, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_filename}")

    return all_results


if __name__ == '__main__':
    main()

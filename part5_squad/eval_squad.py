import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List, Tuple
import json
import os
import sys

from part5_squad.squad_metrics import evaluate_squad


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

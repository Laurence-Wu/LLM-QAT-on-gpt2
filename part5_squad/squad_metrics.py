"""
SQuAD Evaluation Metrics: Exact Match and F1 Score

Official SQuAD evaluation metrics following the original implementation.
"""

import string
import re
from collections import Counter
from typing import List, Dict


def normalize_answer(s: str) -> str:
    """
    Normalize answer text for comparison

    Normalization steps:
    1. Lowercase
    2. Remove punctuation
    3. Remove articles (a, an, the)
    4. Remove extra whitespace

    Args:
        s: Answer text

    Returns:
        Normalized answer text
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute Exact Match score

    Returns 1.0 if the normalized prediction exactly matches
    any normalized ground truth, otherwise 0.0.

    Args:
        prediction: Predicted answer text
        ground_truths: List of ground truth answer texts

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_prediction == normalized_ground_truth:
            return 1.0

    return 0.0


def f1_score(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute F1 score (token-level)

    F1 score is computed as:
    1. Tokenize prediction and ground truth into words
    2. Compute precision = (# common tokens) / (# prediction tokens)
    3. Compute recall = (# common tokens) / (# ground truth tokens)
    4. F1 = 2 * (precision * recall) / (precision + recall)

    Returns the maximum F1 over all ground truths.

    Args:
        prediction: Predicted answer text
        ground_truths: List of ground truth answer texts

    Returns:
        F1 score (between 0.0 and 1.0)
    """
    normalized_prediction = normalize_answer(prediction)
    prediction_tokens = normalized_prediction.split()

    # Handle empty prediction
    if not prediction_tokens:
        return 0.0

    max_f1 = 0.0

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        ground_truth_tokens = normalized_ground_truth.split()

        # Handle empty ground truth
        if not ground_truth_tokens:
            continue

        # Compute token overlap using Counter
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            f1 = 0.0
        else:
            precision = num_common / len(prediction_tokens)
            recall = num_common / len(ground_truth_tokens)
            f1 = 2 * precision * recall / (precision + recall)

        max_f1 = max(max_f1, f1)

    return max_f1


def evaluate_squad(predictions: List[Dict[str, str]], dataset) -> Dict[str, float]:
    """
    Evaluate predictions on SQuAD dataset

    Args:
        predictions: List of dicts with keys 'id' and 'prediction_text'
                    [{'id': 'example_id', 'prediction_text': 'answer'}, ...]
        dataset: SQuAD dataset (HuggingFace dataset or list of examples)
                 Each example should have 'id' and 'answers' keys

    Returns:
        Dict with keys:
            - 'exact_match': EM score (0-100)
            - 'f1': F1 score (0-100)
            - 'total': Number of examples evaluated
    """
    # Build dataset lookup by ID
    if hasattr(dataset, '__iter__'):
        dataset_lookup = {example['id']: example for example in dataset}
    else:
        # Handle HuggingFace Dataset
        dataset_lookup = {}
        for example in dataset:
            dataset_lookup[example['id']] = example

    em_total = 0.0
    f1_total = 0.0
    count = 0

    for pred in predictions:
        example_id = pred['id']
        pred_text = pred['prediction_text']

        if example_id not in dataset_lookup:
            print(f"Warning: Example ID {example_id} not found in dataset")
            continue

        example = dataset_lookup[example_id]
        ground_truths = example['answers']['text']

        # Compute metrics
        em = exact_match_score(pred_text, ground_truths)
        f1 = f1_score(pred_text, ground_truths)

        em_total += em
        f1_total += f1
        count += 1

    if count == 0:
        return {
            'exact_match': 0.0,
            'f1': 0.0,
            'total': 0
        }

    return {
        'exact_match': 100.0 * em_total / count,
        'f1': 100.0 * f1_total / count,
        'total': count
    }


def compute_metrics_single(prediction: str, ground_truths: List[str]) -> Dict[str, float]:
    """
    Compute EM and F1 for a single prediction

    Args:
        prediction: Predicted answer text
        ground_truths: List of ground truth answer texts

    Returns:
        Dict with 'exact_match' and 'f1' scores
    """
    return {
        'exact_match': exact_match_score(prediction, ground_truths),
        'f1': f1_score(prediction, ground_truths)
    }

"""
Test SQuAD evaluation metrics (EM and F1)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from squad_metrics import normalize_answer, exact_match_score, f1_score, evaluate_squad


def test_normalize_answer():
    """Test answer normalization"""
    print("Testing answer normalization...")

    # Test lowercase
    assert normalize_answer("The Cat") == "cat", "Should lowercase"

    # Test punctuation removal
    assert normalize_answer("cat!") == "cat", "Should remove punctuation"

    # Test article removal
    assert normalize_answer("the cat") == "cat", "Should remove articles"
    assert normalize_answer("a dog") == "dog", "Should remove 'a'"
    assert normalize_answer("an apple") == "apple", "Should remove 'an'"

    # Test whitespace normalization
    assert normalize_answer("the  cat") == "cat", "Should normalize whitespace"

    # Test combined
    assert normalize_answer("The Cat!") == "cat", "Should normalize fully"

    print("✓ Answer normalization works")


def test_exact_match():
    """Test Exact Match metric"""
    print("Testing Exact Match...")

    # Perfect match
    assert exact_match_score("the cat", ["the cat"]) == 1.0, \
        "Should match identical strings"

    # Match after normalization
    assert exact_match_score("The Cat!", ["the cat"]) == 1.0, \
        "Should match after normalization"

    # No match
    assert exact_match_score("dog", ["cat"]) == 0.0, \
        "Should not match different strings"

    # Multiple ground truths
    assert exact_match_score("cat", ["dog", "cat", "bird"]) == 1.0, \
        "Should match one of multiple ground truths"

    # Partial match (should fail)
    assert exact_match_score("the cat sat", ["the cat"]) == 0.0, \
        "Should not match partial strings"

    print("✓ Exact Match works")


def test_f1_score():
    """Test F1 Score metric"""
    print("Testing F1 Score...")

    # Perfect overlap
    assert f1_score("the cat", ["the cat"]) == 1.0, \
        "Perfect overlap should give F1=1.0"

    # Partial overlap
    f1 = f1_score("the big cat", ["the cat"])
    assert 0 < f1 < 1, "Partial overlap should give 0 < F1 < 1"
    # Expected: After normalization "big cat" vs "cat"
    # precision = 1/2, recall = 1/1, F1 = 2 * (1/2 * 1) / (1/2 + 1) ≈ 0.667
    assert abs(f1 - 0.667) < 0.01, f"Expected F1≈0.667, got {f1}"

    # No overlap
    assert f1_score("dog", ["cat"]) == 0.0, \
        "No overlap should give F1=0.0"

    # Multiple ground truths (max F1)
    f1 = f1_score("cat", ["dog", "cat", "bird"])
    assert f1 == 1.0, "Should return max F1 over ground truths"

    # Empty prediction
    assert f1_score("", ["cat"]) == 0.0, \
        "Empty prediction should give F1=0.0"

    print("✓ F1 Score works")


def test_evaluate_squad():
    """Test full SQuAD evaluation"""
    print("Testing SQuAD evaluation...")

    # Create mock predictions
    predictions = [
        {'id': 'q1', 'prediction_text': 'Paris'},
        {'id': 'q2', 'prediction_text': 'the Eiffel Tower'},
        {'id': 'q3', 'prediction_text': 'France'}
    ]

    # Create mock dataset
    class MockDataset:
        def __iter__(self):
            return iter([
                {'id': 'q1', 'answers': {'text': ['Paris', 'paris']}},
                {'id': 'q2', 'answers': {'text': ['Eiffel Tower']}},
                {'id': 'q3', 'answers': {'text': ['France']}}
            ])

    dataset = MockDataset()

    # Evaluate
    results = evaluate_squad(predictions, dataset)

    # Check results
    assert 'exact_match' in results, "Results should have exact_match"
    assert 'f1' in results, "Results should have f1"
    assert 'total' in results, "Results should have total"

    assert results['total'] == 3, "Should evaluate 3 examples"
    assert results['exact_match'] == 100.0, "All predictions are exact matches"
    # F1 for q2 is not perfect due to "the" article
    assert results['f1'] > 80.0, "F1 should be high"

    print("✓ SQuAD evaluation works")


if __name__ == '__main__':
    test_normalize_answer()
    test_exact_match()
    test_f1_score()
    test_evaluate_squad()
    print("\n✅ All metrics tests passed!")

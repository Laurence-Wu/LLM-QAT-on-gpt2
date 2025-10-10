"""
Run all tests for part5_squad

Usage:
    python tests/run_all_tests.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*70)
print("Running SQuAD QA Implementation Tests")
print("="*70)

# Run all test modules
test_modules = [
    'test_model',
    'test_dataset',
    'test_metrics',
    'test_loss',
    'test_distillation',
    'test_evaluation',
    'test_training_step'
]

failed_tests = []

for module_name in test_modules:
    print(f"\n{'='*70}")
    print(f"Running {module_name}")
    print(f"{'='*70}")

    try:
        module = __import__(module_name)
        # Run the module's tests
        if hasattr(module, '__name__'):
            exec(open(f"{module_name}.py").read())
    except Exception as e:
        print(f"❌ {module_name} failed: {e}")
        failed_tests.append((module_name, str(e)))

print(f"\n{'='*70}")
print("Test Summary")
print(f"{'='*70}")

if not failed_tests:
    print("✅ All tests passed!")
else:
    print(f"❌ {len(failed_tests)} test module(s) failed:")
    for module_name, error in failed_tests:
        print(f"  - {module_name}: {error}")

print(f"{'='*70}")

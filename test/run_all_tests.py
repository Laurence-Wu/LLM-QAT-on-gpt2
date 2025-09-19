#!/usr/bin/env python3
"""
Master Test Runner
Runs all test suites for the LLM Quantization project
"""

import sys
import os
import subprocess
import time
from datetime import datetime

# Test suites to run
TEST_SUITES = [
    {
        'name': 'Data Pipeline Tests',
        'file': 'test_data_pipeline.py',
        'description': 'Tests dataset loading, tokenization, and dataloader creation'
    },
    {
        'name': 'SP Model Tests',
        'file': 'test_sp_model.py',
        'description': 'Tests Switchable Precision model components and workflow'
    },
    {
        'name': 'CPT Model Tests',
        'file': 'test_cpt_model.py',
        'description': 'Tests Cyclic Precision Training model components and workflow'
    },
    {
        'name': 'Integration Tests',
        'file': 'test_integration.py',
        'description': 'Tests full workflows and model comparisons'
    }
]

def print_header():
    """Print test runner header."""
    print("\n" + "="*80)
    print(" " * 20 + "LLM QUANTIZATION TEST RUNNER")
    print(" " * 25 + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*80)
    print("\nThis will run all test suites for the project:")
    for suite in TEST_SUITES:
        print(f"  • {suite['name']}: {suite['description']}")
    print("\n" + "="*80)

def run_test_suite(suite):
    """Run a single test suite."""
    print(f"\n{'='*80}")
    print(f" Running: {suite['name']}")
    print(f" File: {suite['file']}")
    print(f" Description: {suite['description']}")
    print("="*80)

    test_path = os.path.join(os.path.dirname(__file__), suite['file'])

    if not os.path.exists(test_path):
        print(f"✗ Test file not found: {test_path}")
        return False

    start_time = time.time()

    try:
        # Run the test file
        result = subprocess.run(
            [sys.executable, test_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per test suite
        )

        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✅ {suite['name']} PASSED (took {elapsed_time:.2f}s)")
            return True
        else:
            print(f"\n❌ {suite['name']} FAILED (took {elapsed_time:.2f}s)")
            return False

    except subprocess.TimeoutExpired:
        print(f"\n⏱️ {suite['name']} TIMEOUT (exceeded 5 minutes)")
        return False
    except Exception as e:
        print(f"\n❌ {suite['name']} ERROR: {e}")
        return False

def run_quick_tests():
    """Run a quick subset of tests for rapid validation."""
    print("\n" + "="*80)
    print(" QUICK TEST MODE")
    print("="*80)
    print("\nRunning essential tests only...")

    # Only run SP and CPT basic tests
    quick_suites = [TEST_SUITES[1], TEST_SUITES[2]]  # SP and CPT tests

    passed = 0
    failed = 0

    for suite in quick_suites:
        if run_test_suite(suite):
            passed += 1
        else:
            failed += 1

    return passed, failed

def run_all_tests():
    """Run all test suites."""
    passed = 0
    failed = 0
    skipped = 0

    start_time = time.time()

    for suite in TEST_SUITES:
        try:
            if run_test_suite(suite):
                passed += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            print("\n\n⚠️ Test run interrupted by user")
            skipped = len(TEST_SUITES) - (passed + failed)
            break
        except Exception as e:
            print(f"\n❌ Unexpected error running {suite['name']}: {e}")
            failed += 1

    total_time = time.time() - start_time

    # Print summary
    print("\n" + "="*80)
    print(" TEST RUN SUMMARY")
    print("="*80)
    print(f"\n  Total time: {total_time:.2f} seconds")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    if skipped > 0:
        print(f"  Skipped: {skipped}")

    if failed == 0 and skipped == 0:
        print("\n" + "🎉 "*10)
        print("  ALL TESTS PASSED SUCCESSFULLY!")
        print("🎉 "*10)
    elif failed > 0:
        print(f"\n⚠️ {failed} test suite(s) failed. Please review the errors above.")
    else:
        print(f"\n⚠️ Test run incomplete. {skipped} test(s) were skipped.")

    print("\n" + "="*80)

    return failed == 0

def main():
    """Main test runner function."""
    import argparse

    parser = argparse.ArgumentParser(description='Run all tests for LLM Quantization project')
    parser.add_argument('--quick', action='store_true',
                       help='Run only essential tests for quick validation')
    parser.add_argument('--suite', type=str,
                       help='Run only a specific test suite (sp, cpt, data, integration)')
    args = parser.parse_args()

    print_header()

    if args.quick:
        passed, failed = run_quick_tests()
        success = failed == 0
    elif args.suite:
        # Run specific suite
        suite_map = {
            'data': TEST_SUITES[0],
            'sp': TEST_SUITES[1],
            'cpt': TEST_SUITES[2],
            'integration': TEST_SUITES[3]
        }

        if args.suite in suite_map:
            success = run_test_suite(suite_map[args.suite])
        else:
            print(f"Unknown suite: {args.suite}")
            print(f"Available suites: {', '.join(suite_map.keys())}")
            success = False
    else:
        success = run_all_tests()

    # Return appropriate exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
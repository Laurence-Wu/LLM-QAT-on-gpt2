#!/usr/bin/env python3
"""
Test Runner for SP Model Debug Suite
This file sits outside the test folder and provides a clean interface to run tests.
"""

import sys
import os
import argparse

# Add test directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test'))

from debug_sp_model import run_comprehensive_test, print_test_summary


def main():
    parser = argparse.ArgumentParser(
        description='SP Model Test Runner - Comprehensive testing for multi-precision models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Suites:
  all        - Run all test suites (default)
  basic      - Basic functionality tests (32-bit equivalence, quantization, LoRA)
  precision  - Precision mismatch detection and consistency tests
  batchnorm  - Batch normalization behavior and statistics tests
  training   - Training dynamics, distillation, and QAT tests

Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --suite basic      # Run only basic tests
  python run_tests.py --quick            # Run tests in quick mode
  python run_tests.py --suite precision  # Run precision mismatch tests
        """
    )

    parser.add_argument(
        '--suite',
        type=str,
        default='all',
        choices=['all', 'basic', 'precision', 'batchnorm', 'training'],
        help='Test suite to run (default: all)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run tests in quick mode with reduced samples'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output during tests'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("SP MODEL TEST RUNNER")
    print("="*80)
    print(f"\n🔧 Configuration:")
    print(f"   Test Suite: {args.suite}")
    print(f"   Quick Mode: {'Enabled' if args.quick else 'Disabled'}")
    print(f"   Verbose: {'Enabled' if args.verbose else 'Disabled'}")

    try:
        # Run the test suite
        print("\n🚀 Starting tests...\n")
        results = run_comprehensive_test(
            test_suite=args.suite,
            quick_mode=args.quick
        )

        # Print summary
        print_test_summary(results)

        # Determine exit code based on results
        exit_code = 0

        # Check for failures in results
        for test_name, test_result in results.items():
            if isinstance(test_result, dict):
                if 'status' in test_result and 'FAILED' in str(test_result['status']):
                    exit_code = 1
                    break

        if exit_code == 0:
            print("\n✅ All tests completed successfully!")
        else:
            print("\n⚠️ Some tests had issues. Please review the output above.")

        return exit_code

    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        print(f"\n❌ Test runner failed with error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
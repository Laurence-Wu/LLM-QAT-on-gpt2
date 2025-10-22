"""
Test runner for additionalBP test suite

Runs all tests for BF16/FP16 precision conversion and evaluation pipeline.

Usage:
    python run_additionalBP_tests.py
    python run_additionalBP_tests.py --verbose
    python run_additionalBP_tests.py --test <test_name>
"""

import sys
import os
import subprocess
import argparse
from datetime import datetime

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_section(text):
    """Print formatted section header"""
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{'-'*70}{Colors.ENDC}")


def run_test_file(test_file, verbose=False):
    """
    Run a single test file and return result

    Args:
        test_file: Path to test file
        verbose: Whether to show detailed output

    Returns:
        (success: bool, duration: float)
    """
    test_name = os.path.basename(test_file)
    print_section(f"Running {test_name}")

    start_time = datetime.now()

    try:
        if verbose:
            # Show all output
            result = subprocess.run(
                [sys.executable, test_file],
                cwd=os.path.dirname(test_file)
            )
            success = result.returncode == 0
        else:
            # Capture output, only show on failure
            result = subprocess.run(
                [sys.executable, test_file],
                cwd=os.path.dirname(test_file),
                capture_output=True,
                text=True
            )
            success = result.returncode == 0

            if success:
                # Extract summary line
                lines = result.stdout.strip().split('\n')
                for line in lines[-10:]:  # Show last 10 lines
                    if 'passed' in line.lower() or 'summary' in line.lower() or '✓' in line or '✅' in line:
                        print(line)
            else:
                # Show error output
                print(result.stdout)
                if result.stderr:
                    print(f"{Colors.FAIL}STDERR:{Colors.ENDC}")
                    print(result.stderr)

        duration = (datetime.now() - start_time).total_seconds()

        if success:
            print(f"{Colors.OKGREEN}✅ {test_name} passed ({duration:.1f}s){Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}❌ {test_name} failed ({duration:.1f}s){Colors.ENDC}")

        return success, duration

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        print(f"{Colors.FAIL}❌ {test_name} crashed: {e} ({duration:.1f}s){Colors.ENDC}")
        return False, duration


def main():
    """Run all test files"""
    parser = argparse.ArgumentParser(description='Run additionalBP test suite')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed output for all tests')
    parser.add_argument('--test', '-t', type=str,
                       help='Run specific test file (e.g., test_precision_conversion)')
    args = parser.parse_args()

    # Get current directory (additionalBP)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define test files in order
    all_test_files = [
        'test_precision_conversion.py',
        'test_eval_pipeline.py',
        'test_precision_stress.py',
    ]

    # Filter if specific test requested
    if args.test:
        # Allow partial matches
        test_files = [f for f in all_test_files if args.test in f]
        if not test_files:
            print(f"{Colors.FAIL}Error: No test file matching '{args.test}'{Colors.ENDC}")
            print(f"Available tests: {', '.join(all_test_files)}")
            return 1
    else:
        test_files = all_test_files

    # Print header
    print_header("additionalBP Test Suite")
    print(f"{Colors.OKCYAN}Running {len(test_files)} test file(s){Colors.ENDC}")
    print(f"{Colors.OKCYAN}Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")

    # Run tests
    results = []
    total_duration = 0.0

    for test_file in test_files:
        test_path = os.path.join(current_dir, test_file)

        if not os.path.exists(test_path):
            print(f"{Colors.WARNING}⚠️  Test file not found: {test_file}{Colors.ENDC}")
            results.append((test_file, False, 0.0))
            continue

        success, duration = run_test_file(test_path, args.verbose)
        results.append((test_file, success, duration))
        total_duration += duration

    # Print summary
    print_header("Test Summary")

    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed

    print(f"{Colors.BOLD}Results:{Colors.ENDC}")
    for test_file, success, duration in results:
        status = f"{Colors.OKGREEN}✅ PASS{Colors.ENDC}" if success else f"{Colors.FAIL}❌ FAIL{Colors.ENDC}"
        print(f"  {status}  {test_file:<35} ({duration:.1f}s)")

    print()
    print(f"{Colors.BOLD}Total: {passed}/{len(results)} passed, {failed}/{len(results)} failed{Colors.ENDC}")
    print(f"{Colors.BOLD}Total duration: {total_duration:.1f}s{Colors.ENDC}")
    print(f"{Colors.OKCYAN}End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")

    # Final result
    if failed == 0:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 All tests passed!{Colors.ENDC}\n")
        return 0
    else:
        print(f"\n{Colors.FAIL}{Colors.BOLD}💥 {failed} test file(s) failed{Colors.ENDC}\n")
        return 1


if __name__ == '__main__':
    exit(main())

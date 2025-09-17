#!/usr/bin/env python3
"""
Verify all import statements in test folder work correctly.
"""

import sys
import os
import importlib.util
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_imports(file_path):
    """Check if all imports in a Python file work."""
    print(f"\nChecking: {file_path.name}")
    print("-" * 40)

    errors = []

    # Read the file and extract import statements
    with open(file_path, 'r') as f:
        lines = f.readlines()

    imports = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('import ') or line.startswith('from '):
            # Skip relative imports within test files
            if 'from .' in line or 'import .' in line:
                continue
            imports.append((i+1, line))

    # Test each import
    for line_num, import_stmt in imports:
        try:
            # Handle different import types
            if import_stmt.startswith('from '):
                # Extract module name
                parts = import_stmt.split()
                if len(parts) >= 2:
                    module_name = parts[1]
                    # Skip if it's a relative import or local module
                    if module_name.startswith('.'):
                        continue

                    # Try to import the module
                    if module_name in ['shared', 'part3_evaluation', 'part2_cyclic_precision']:
                        # These are local project modules
                        try:
                            exec(import_stmt)
                            print(f"  ✓ Line {line_num}: {import_stmt[:50]}...")
                        except Exception as e:
                            errors.append(f"  ✗ Line {line_num}: {import_stmt[:50]}... - {str(e)[:50]}")
                    else:
                        # External module
                        try:
                            __import__(module_name.split('.')[0])
                            print(f"  ✓ Line {line_num}: {import_stmt[:50]}...")
                        except ImportError as e:
                            errors.append(f"  ✗ Line {line_num}: {import_stmt[:50]}... - Missing: {module_name}")

            elif import_stmt.startswith('import '):
                # Simple import
                module_name = import_stmt.split()[1].split('.')[0]
                try:
                    __import__(module_name)
                    print(f"  ✓ Line {line_num}: {import_stmt[:50]}...")
                except ImportError as e:
                    errors.append(f"  ✗ Line {line_num}: {import_stmt[:50]}... - Missing: {module_name}")

        except Exception as e:
            errors.append(f"  ✗ Line {line_num}: {import_stmt[:50]}... - Error: {str(e)[:50]}")

    return errors

def main():
    """Check all Python files in the test directory."""
    print("="*60)
    print("VERIFYING IMPORTS IN TEST FOLDER")
    print("="*60)

    test_dir = Path(__file__).parent
    python_files = list(test_dir.glob("*.py"))

    # Exclude this verification script
    python_files = [f for f in python_files if f.name != "verify_imports.py"]

    print(f"Found {len(python_files)} Python files to check")

    all_errors = {}

    for file_path in sorted(python_files):
        errors = check_imports(file_path)
        if errors:
            all_errors[file_path.name] = errors

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if not all_errors:
        print("✅ All imports are working correctly!")
        print(f"Checked {len(python_files)} files with no import errors.")
    else:
        print(f"⚠️ Found import issues in {len(all_errors)} files:")
        for filename, errors in all_errors.items():
            print(f"\n{filename}:")
            for error in errors:
                print(error)

    # Check for commonly needed modules
    print("\n" + "="*60)
    print("CHECKING PROJECT STRUCTURE")
    print("="*60)

    required_dirs = [
        "shared",
        "part3_evaluation",
        "part2_cyclic_precision"
    ]

    parent_dir = test_dir.parent

    for dir_name in required_dirs:
        dir_path = parent_dir / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/ exists")
            # Check for __init__.py or key files
            init_file = dir_path / "__init__.py"
            if init_file.exists():
                print(f"  ✓ {dir_name}/__init__.py exists")
            else:
                # Check for key Python files
                py_files = list(dir_path.glob("*.py"))[:3]  # Show first 3
                if py_files:
                    print(f"  ⓘ Found {len(list(dir_path.glob('*.py')))} .py files")
        else:
            print(f"✗ {dir_name}/ NOT FOUND")

    # Test critical imports
    print("\n" + "="*60)
    print("TESTING CRITICAL IMPORTS")
    print("="*60)

    critical_imports = [
        ("shared.models", "SwitchableQATGPT2"),
        ("shared.models", "QATGPT2"),
        ("shared.quantization", "LearnableFakeQuantize"),
        ("part3_evaluation.main_llm_qat_eval", "load_switchable_model"),
        ("part3_evaluation.zero_shot_tasks", "ZeroShotEvaluator"),
        ("part3_evaluation.few_shot_eval", "FewShotEvaluator"),
    ]

    for module_name, class_name in critical_imports:
        try:
            module = __import__(module_name, fromlist=[class_name])
            if hasattr(module, class_name):
                print(f"✓ Can import {class_name} from {module_name}")
            else:
                print(f"✗ {class_name} not found in {module_name}")
        except ImportError as e:
            print(f"✗ Cannot import {module_name}: {e}")
        except Exception as e:
            print(f"✗ Error importing {module_name}: {e}")

    return len(all_errors) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Checkpoint validation script for SP models.
Checks if a checkpoint file is valid and can be loaded for evaluation.
"""

import sys
import os
import torch
import json
import argparse
from pathlib import Path
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_file_basics(filepath: str) -> Dict[str, Any]:
    """Check basic file properties."""
    results = {
        'exists': False,
        'readable': False,
        'size_bytes': 0,
        'size_mb': 0.0,
        'path': filepath
    }

    if not os.path.exists(filepath):
        print(f"❌ File does not exist: {filepath}")
        return results

    results['exists'] = True

    try:
        # Check if readable
        with open(filepath, 'rb') as f:
            f.read(1)
        results['readable'] = True
    except Exception as e:
        print(f"❌ File is not readable: {e}")
        return results

    # Get file size
    size_bytes = os.path.getsize(filepath)
    results['size_bytes'] = size_bytes
    results['size_mb'] = size_bytes / (1024 * 1024)

    print(f"✅ File exists and is readable")
    print(f"   Path: {filepath}")
    print(f"   Size: {results['size_mb']:.2f} MB ({size_bytes:,} bytes)")

    if size_bytes == 0:
        print(f"❌ File is empty (0 bytes)")
        results['error'] = 'Empty file'
    elif size_bytes < 1000:
        print(f"⚠️ File is suspiciously small ({size_bytes} bytes)")
        results['warning'] = 'File too small'

    return results


def check_checkpoint_loading(filepath: str) -> Dict[str, Any]:
    """Try to load the checkpoint and check its contents."""
    results = {
        'loadable': False,
        'type': None,
        'keys': [],
        'has_model_state': False,
        'has_model_config': False,
        'has_training_config': False,
        'bit_width_info': None,
        'model_type': None
    }

    print("\n" + "="*60)
    print("Attempting to load checkpoint...")

    try:
        # Try different loading methods
        checkpoint = None
        error_messages = []

        # Method 1: Standard torch.load
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            print("✅ Loaded with standard torch.load")
            results['loadable'] = True
        except EOFError:
            error_messages.append("EOFError: File appears to be truncated or corrupted")
        except Exception as e:
            error_messages.append(f"Standard load failed: {e}")

        # Method 2: Try with weights_only=True (safer)
        if checkpoint is None:
            try:
                checkpoint = torch.load(filepath, map_location='cpu', weights_only=True)
                print("✅ Loaded with weights_only=True")
                results['loadable'] = True
            except Exception as e:
                error_messages.append(f"Weights-only load failed: {e}")

        # Method 3: Try with pickle directly
        if checkpoint is None:
            try:
                import pickle
                with open(filepath, 'rb') as f:
                    checkpoint = pickle.load(f)
                print("✅ Loaded with pickle directly")
                results['loadable'] = True
            except Exception as e:
                error_messages.append(f"Pickle load failed: {e}")

        # If all methods failed
        if checkpoint is None:
            print("❌ Failed to load checkpoint with any method:")
            for msg in error_messages:
                print(f"   - {msg}")
            results['error'] = error_messages
            return results

    except Exception as e:
        print(f"❌ Unexpected error loading checkpoint: {e}")
        results['error'] = str(e)
        return results

    # Analyze checkpoint contents
    print("\n" + "-"*60)
    print("Checkpoint contents:")

    # Check type
    if isinstance(checkpoint, dict):
        results['type'] = 'dict'
        results['keys'] = list(checkpoint.keys())

        print(f"Type: Dictionary with {len(results['keys'])} keys")
        print(f"Keys: {results['keys'][:10]}")  # Show first 10 keys

        # Check for expected keys
        if 'model_state_dict' in checkpoint:
            results['has_model_state'] = True
            print("✅ Found 'model_state_dict'")

            # Count parameters
            state_dict = checkpoint['model_state_dict']
            num_params = len(state_dict)
            total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
            print(f"   - {num_params} parameter tensors")
            print(f"   - {total_params:,} total parameters")

            # Check for SP model signatures
            sp_signatures = [
                'transformer.h.0.attn.c_attn.lora_adapters',
                'transformer.h.0.ln_1.ln_layers',
                'transformer.bit_widths'
            ]

            for sig in sp_signatures:
                if any(sig in key for key in state_dict.keys()):
                    print(f"   ✅ Found SP model signature: {sig}")
                    results['model_type'] = 'SPModel'
                    break

        if 'model_config' in checkpoint:
            results['has_model_config'] = True
            config = checkpoint['model_config']
            print("✅ Found 'model_config'")

            # Check for bit width information
            if 'bit_widths' in config:
                results['bit_width_info'] = config['bit_widths']
                print(f"   - Bit widths: {config['bit_widths']}")
            if 'quantization_bits' in config:
                print(f"   - Quantization bits: {config['quantization_bits']}")
            if 'n_layer' in config:
                print(f"   - Layers: {config['n_layer']}")
            if 'n_embd' in config:
                print(f"   - Embedding dim: {config['n_embd']}")

        if 'training_config' in checkpoint:
            results['has_training_config'] = True
            print("✅ Found 'training_config'")

        # Check for INT8 checkpoint format
        if 'int8_state_dict' in checkpoint:
            print("✅ Found 'int8_state_dict' (INT8 checkpoint)")
            results['model_type'] = 'INT8'

            if 'model_info' in checkpoint:
                info = checkpoint['model_info']
                print(f"   - Target bits: {info.get('target_bits', 'N/A')}")
                print(f"   - Compression ratio: {info.get('compression_ratio', 'N/A'):.2f}x")
                print(f"   - INT8 size: {info.get('int8_size_mb', 'N/A'):.2f} MB")

        # Check for other common keys
        if 'bit_width' in checkpoint:
            print(f"   - Checkpoint bit width: {checkpoint['bit_width']}")
            results['bit_width_info'] = checkpoint['bit_width']

        if 'timestamp' in checkpoint:
            print(f"   - Timestamp: {checkpoint['timestamp']}")

    elif isinstance(checkpoint, torch.nn.Module):
        results['type'] = 'model'
        print("Type: Direct model object")
        results['model_type'] = 'DirectModel'

    else:
        results['type'] = 'unknown'
        print(f"Type: Unknown ({type(checkpoint)})")

    return results


def suggest_fixes(file_results: Dict, load_results: Dict):
    """Suggest fixes based on the issues found."""
    print("\n" + "="*60)
    print("Diagnosis and Suggestions:")
    print("="*60)

    if not file_results['exists']:
        print("\n❌ ISSUE: File not found")
        print("SUGGESTIONS:")
        print("  1. Check if the file path is correct")
        print("  2. Ensure you're running from the correct directory")
        print("  3. Try using absolute path instead of relative path")
        return

    if file_results.get('size_bytes', 0) == 0:
        print("\n❌ ISSUE: File is empty (0 bytes)")
        print("SUGGESTIONS:")
        print("  1. The file save was interrupted - retrain the model")
        print("  2. Check disk space when saving checkpoints")
        print("  3. Verify the training script completed successfully")
        return

    if not load_results['loadable']:
        print("\n❌ ISSUE: Checkpoint file is corrupted or incomplete")
        print("LIKELY CAUSES:")
        print("  - Training was interrupted during checkpoint save")
        print("  - File transfer was incomplete")
        print("  - Disk was full during save")
        print("  - PyTorch version mismatch")
        print("\nSUGGESTIONS:")
        print("  1. Retrain the model or use a backup checkpoint")
        print("  2. Check if there are other checkpoint files with similar timestamps")
        print("  3. Verify PyTorch versions match between training and evaluation")
        return

    if load_results['type'] == 'dict':
        if not load_results['has_model_state']:
            print("\n⚠️ ISSUE: Missing 'model_state_dict' key")
            print("SUGGESTIONS:")
            print("  1. This might be a raw state_dict - wrap it properly")
            print("  2. Check if the checkpoint format matches expected structure")

        if not load_results['has_model_config']:
            print("\n⚠️ ISSUE: Missing 'model_config' key")
            print("SUGGESTIONS:")
            print("  1. Provide --config_path argument with matching JSON config")
            print("  2. Ensure checkpoint was saved with config information")

        if not load_results['bit_width_info']:
            print("\n⚠️ ISSUE: No bit width information found")
            print("SUGGESTIONS:")
            print("  1. This might be a standard model, not an SP model")
            print("  2. Ensure the model was trained with switchable precision")

    if load_results['model_type'] == 'INT8':
        print("\n✅ This is an INT8 checkpoint")
        print("  - Use specialized INT8 loading functions")
        print("  - This represents a specific precision, not switchable")

    elif load_results['model_type'] == 'SPModel':
        print("\n✅ This appears to be a Switchable Precision model")
        print("  - Can be used with Part 3 evaluation")
        print(f"  - Supports bit widths: {load_results.get('bit_width_info', 'Unknown')}")

    print("\n" + "="*60)


def check_companion_files(filepath: str):
    """Check for companion configuration files."""
    print("\n" + "="*60)
    print("Checking for companion files...")
    print("="*60)

    base_path = Path(filepath)
    directory = base_path.parent
    stem = base_path.stem

    # Try to find matching JSON config
    json_patterns = [
        f"*{stem[-15:]}*.json",  # Match timestamp if present
        f"{stem}*.json",
        "qat_training_stats*.json"
    ]

    json_files = []
    for pattern in json_patterns:
        json_files.extend(directory.glob(pattern))

    if json_files:
        print(f"✅ Found {len(json_files)} potential config file(s):")
        for jf in json_files[:5]:  # Show max 5
            print(f"   - {jf.name}")

            # Try to load and show basic info
            try:
                with open(jf, 'r') as f:
                    config = json.load(f)
                    if 'model_config' in config:
                        mc = config['model_config']
                        print(f"     Model config: layers={mc.get('n_layer')}, "
                              f"embd={mc.get('n_embd')}, "
                              f"bits={mc.get('bit_widths')}")
            except Exception:
                pass
    else:
        print("⚠️ No companion JSON config files found")
        print("   Consider providing --config_path for better results")

    # Check for other checkpoints
    other_checkpoints = list(directory.glob("*.pth"))
    other_checkpoints = [cp for cp in other_checkpoints if cp != base_path]

    if other_checkpoints:
        print(f"\n✅ Found {len(other_checkpoints)} other checkpoint(s) in directory:")
        for cp in other_checkpoints[:5]:  # Show max 5
            size_mb = cp.stat().st_size / (1024 * 1024)
            print(f"   - {cp.name} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description='Validate checkpoint files for SP model evaluation'
    )
    parser.add_argument('checkpoint_path', type=str,
                       help='Path to checkpoint file to validate')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed information')
    parser.add_argument('--check-companion', action='store_true',
                       help='Check for companion config files')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("CHECKPOINT VALIDATION TOOL")
    print("="*80)

    # Check file basics
    file_results = check_file_basics(args.checkpoint_path)

    if file_results['exists'] and file_results['readable']:
        # Try to load checkpoint
        load_results = check_checkpoint_loading(args.checkpoint_path)
    else:
        load_results = {'loadable': False}

    # Check for companion files if requested
    if args.check_companion and file_results['exists']:
        check_companion_files(args.checkpoint_path)

    # Provide diagnosis and suggestions
    suggest_fixes(file_results, load_results)

    # Final status
    print("\nFINAL STATUS:")
    if file_results['exists'] and load_results['loadable']:
        print("✅ Checkpoint is valid and can be used for evaluation")

        # Provide example command
        print("\nExample evaluation command:")
        print(f"python part3_evaluation/main_llm_qat_eval.py \\")
        print(f"    --model_path {args.checkpoint_path} \\")
        print(f"    --output_dir results \\")
        print(f"    --max_eval_samples 100")

        return 0  # Success
    else:
        print("❌ Checkpoint cannot be used for evaluation")
        print("   Please address the issues above")
        return 1  # Error


if __name__ == "__main__":
    sys.exit(main())
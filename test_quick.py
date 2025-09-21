#!/usr/bin/env python3
"""
Quick test to verify the recursion error is fixed
"""

import sys
import os
import torch

# Add test directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test'))

from fix_model_initialization import create_properly_initialized_model
from transformers import GPT2Tokenizer

def test_basic():
    """Quick test to check if model loads without recursion error"""
    print("Testing model loading...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print("Loading SP model...")
    model, config = create_properly_initialized_model(use_pretrained=True, num_layers=2)
    model = model.to(device)

    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Test precision switching
    print("\nTesting precision switching...")
    for precision in [32, 16, 8, 4]:
        print(f"  Setting precision to {precision}-bit...")
        model.set_precision(precision)

    # Test forward pass
    print("\nTesting forward pass...")
    text = "Hello world"
    tokens = tokenizer(text, return_tensors='pt')['input_ids'].to(device)

    model.eval()
    with torch.no_grad():
        output = model(tokens)

    print(f"  Output shape: {output['last_hidden_state'].shape}")

    # Check for quantizers
    print("\nChecking for quantizers...")
    has_quantizers = False
    for name, module in model.named_modules():
        quantizers_weight = getattr(module, 'quantizers_weight', None)
        if quantizers_weight is not None:
            has_quantizers = True
            print(f"  Found quantizers in: {name}")
            break

    if has_quantizers:
        print("✅ Quantizers found")
    else:
        print("⚠️ No quantizers found")

    print("\n✅ Quick test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_basic()
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
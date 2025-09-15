#!/usr/bin/env python3
"""
Test script to verify evaluation can run with or without optional dependencies
"""

import sys
import torch

print("Testing dependencies...")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test optional dependencies
optional_deps = {
    'tabulate': False,
    'pandas': False,
    'matplotlib': False,
    'sklearn': False
}

for dep, _ in optional_deps.items():
    try:
        __import__(dep)
        optional_deps[dep] = True
        print(f"✓ {dep} is installed")
    except ImportError:
        print(f"✗ {dep} is NOT installed (optional)")

print("\n" + "="*50)
print("Testing model loading...")

from shared.models import SwitchableQATGPT2
from transformers import GPT2Config, GPT2Tokenizer

# Create config
config = GPT2Config(
    vocab_size=50257,
    n_positions=256,
    n_embd=768,
    n_layer=6,
    n_head=12,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.1
)

# Create model
model = SwitchableQATGPT2(config, bit_widths=[2, 4, 8, 16])
print(f"✓ Model created successfully")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Test generate method
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

test_input = "Hello world"
input_ids = tokenizer.encode(test_input, return_tensors='pt')

try:
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=10)
    print(f"✓ Generate method works")
    print(f"  Input: {test_input}")
    print(f"  Output: {tokenizer.decode(output[0][:20])}")
except Exception as e:
    print(f"✗ Generate method failed: {e}")

print("\n" + "="*50)
print("Recommendations:")

if not optional_deps['tabulate']:
    print("• Install tabulate for better table formatting:")
    print("  python -m pip install tabulate")

if not optional_deps['pandas']:
    print("• Install pandas for data manipulation:")
    print("  python -m pip install pandas")

if not optional_deps['matplotlib']:
    print("• Install matplotlib for visualization:")
    print("  python -m pip install matplotlib")

if not all(optional_deps.values()):
    print("\nTo install all optional dependencies:")
    print("  python -m pip install tabulate pandas matplotlib scikit-learn")
else:
    print("All optional dependencies are installed!")

print("\nThe evaluation script will work without these optional dependencies,")
print("but some features may be limited.")
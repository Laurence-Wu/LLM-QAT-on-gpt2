#!/usr/bin/env python3
"""
Quick fixes to test if the model works better with different settings.
This script tries various workarounds to identify what fixes the issue.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import SwitchableQATGPT2
from part1_switchable_precision.main_qat import load_pretrained_weights
from transformers import GPT2Tokenizer, GPT2Config


def test_with_fixes():
    """Test model with various fixes applied."""
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, default=None)
    args = parser.parse_args()
    
    print("="*70)
    print("TESTING MODEL WITH VARIOUS FIXES")
    print("="*70)
    
    # Load model
    if os.path.exists(args.model_path):
        model = torch.load(args.model_path)
    else:
        config = GPT2Config()
        config.n_layer = 12
        config.lora_rank = 8
        config.lora_alpha = 16
        config.lora_dropout = 0.0
        model = SwitchableQATGPT2(config, bit_widths=[4, 8, 16], initialize_weights=False)
        load_pretrained_weights(model)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test prompt for all fixes
    test_prompts = [
        "The capital of France is",
        "Question: Is water wet? Answer (Yes/No):",
        "Two plus two equals"
    ]
    
    print("\n[FIX 1] Testing with FP16 (no quantization)")
    print("-"*40)
    if hasattr(model, 'set_precision'):
        model.set_precision(16)
        print("Set to 16-bit precision (should bypass quantization)")
    else:
        print("Model doesn't support set_precision")
    
    model.eval()
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_new_tokens=10, 
                                    temperature=0.7, do_sample=True,
                                    pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  {prompt[:30]}... -> {generated}")
    
    print("\n[FIX 2] Testing with LoRA disabled")
    print("-"*40)
    # Disable all LoRA modules
    lora_count = 0
    for module in model.modules():
        if hasattr(module, 'lora'):
            if hasattr(module.lora, 'scaling'):
                module.lora.scaling = 0
                lora_count += 1
    print(f"Disabled {lora_count} LoRA modules")
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_new_tokens=10,
                                    temperature=0.7, do_sample=True,
                                    pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  {prompt[:30]}... -> {generated}")
    
    print("\n[FIX 3] Testing with one-shot calibration per prompt")
    print("-"*40)
    # Set back to 8-bit and ensure calibration
    if hasattr(model, 'set_precision'):
        model.set_precision(8)
    
    # Force recalibration
    for module in model.modules():
        if hasattr(module, 'quantize_input'):
            module.quantize_input.calibrated = False
        if hasattr(module, 'quantize_weight'):
            module.quantize_weight.calibrated = False
    
    for prompt in test_prompts:
        # Reset calibration for each prompt
        for module in model.modules():
            if hasattr(module, 'quantize_input'):
                module.quantize_input.calibrated = False
        
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_new_tokens=10,
                                    temperature=0.7, do_sample=True,
                                    pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  {prompt[:30]}... -> {generated}")
    
    print("\n[FIX 4] Testing with modified generation parameters")
    print("-"*40)
    # Try different generation strategies
    strategies = [
        {'temperature': 1.0, 'do_sample': True, 'top_p': 0.9, 'top_k': 50},
        {'temperature': 0.0, 'do_sample': False},  # Greedy
        {'num_beams': 4, 'do_sample': False},  # Beam search
    ]
    
    for i, strategy in enumerate(strategies):
        print(f"\nStrategy {i+1}: {strategy}")
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'], 
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id,
                **strategy
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Generated: {generated}")
    
    print("\n[FIX 5] Testing BoolQ with simplified prompts")
    print("-"*40)
    # Test BoolQ-style with shorter prompts
    boolq_tests = [
        {
            'passage': "The sky appears blue during the day.",
            'question': "Is the sky blue?",
            'answer': True
        },
        {
            'passage': "Water freezes at 0 degrees Celsius.",
            'question': "Does water freeze at 100 degrees?",
            'answer': False
        }
    ]
    
    for test in boolq_tests:
        # Short prompt that won't get truncated
        prompt = f"Question: {test['question']} Answer (True/False):"
        inputs = tokenizer(prompt, return_tensors='pt', max_length=50, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_new_tokens=5,
                                    temperature=0.1, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        predicted = 'true' in generated.lower()
        correct = predicted == test['answer']
        
        print(f"  Q: {test['question']}")
        print(f"  Generated: {generated}")
        print(f"  Correct: {correct}")
    
    print("\n[FIX 6] Testing with manual scale setting")
    print("-"*40)
    # Manually set reasonable scales for quantizers
    for module in model.modules():
        if hasattr(module, 'quantize_input'):
            # Set reasonable scale for activations
            module.quantize_input.scale.data.fill_(0.02)
            module.quantize_input.running_min.data.fill_(-3.0)
            module.quantize_input.running_max.data.fill_(3.0)
            module.quantize_input.calibrated = True
        
        if hasattr(module, 'quantize_weight'):
            # Set reasonable scale for weights
            if hasattr(module, 'linear') and module.linear.weight is not None:
                weight_max = module.linear.weight.abs().max().item()
                module.quantize_weight.scale.data.fill_(weight_max / 127)
                module.quantize_weight.calibrated = True
    
    print("Set manual scales for all quantizers")
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_new_tokens=10,
                                    temperature=0.7, do_sample=True,
                                    pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  {prompt[:30]}... -> {generated}")


if __name__ == "__main__":
    test_with_fixes()

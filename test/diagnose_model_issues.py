#!/usr/bin/env python3
"""
Comprehensive diagnostic script to identify why the model performs poorly.
Run this to systematically test each potential issue.
"""

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import SwitchableQATGPT2
from part1_switchable_precision.main_qat import load_pretrained_weights


class ModelDiagnostics:
    """Comprehensive diagnostics for QAT model issues."""
    
    def __init__(self, model_path, config_path=None):
        """Initialize diagnostics with model."""
        print("="*70)
        print("INITIALIZING MODEL DIAGNOSTICS")
        print("="*70)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        print("\nLoading model...")
        # Create model from checkpoint or create new one
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)

            # Load config from checkpoint or use default
            from transformers import GPT2Config

            # Check for both 'config' and 'model_config' keys
            config_dict = None
            if 'model_config' in checkpoint:
                config_dict = checkpoint['model_config']
                print("Found 'model_config' in checkpoint")
            elif 'config' in checkpoint:
                config_dict = checkpoint['config']
                print("Found 'config' in checkpoint")

            if config_dict is not None and isinstance(config_dict, dict):
                # Convert dict config to GPT2Config
                config = GPT2Config()

                # Required standard GPT2 attributes - no defaults, must exist
                required_attrs = ['n_layer', 'n_embd', 'n_head', 'n_positions', 'vocab_size',
                                 'lora_rank', 'lora_alpha', 'lora_dropout', 'layer_norm_epsilon', 'embd_pdrop']
                for attr in required_attrs:
                    if attr not in config_dict:
                        print(f"Error: Required attribute '{attr}' not found in config")
                        raise KeyError(f"Config missing required attribute: {attr}")
                    setattr(config, attr, config_dict[attr])

                # Required switchable precision configs - no defaults, must exist
                switchable_attrs = ['lora_rank_per_bit', 'lora_alpha_per_bit',
                                   'activation_bits_per_bit', 'kv_cache_bits_per_bit', 'kv_cache_bits']
                for attr in switchable_attrs:
                    if attr not in config_dict:
                        print(f"Error: Required switchable precision attribute '{attr}' not found in config")
                        raise KeyError(f"Config missing required switchable precision attribute: {attr}")
                    setattr(config, attr, config_dict[attr])

                # Copy any additional attributes from config_dict
                for key, value in config_dict.items():
                    if not hasattr(config, key):
                        setattr(config, key, value)

            elif config_dict is not None:
                # Config is already a GPT2Config object
                config = config_dict
                # Validate all required attributes exist - no defaults
                required_attrs = ['lora_rank_per_bit', 'lora_alpha_per_bit',
                                 'activation_bits_per_bit', 'kv_cache_bits_per_bit', 'kv_cache_bits']
                for attr in required_attrs:
                    if not hasattr(config, attr):
                        print(f"Error: Config object missing required attribute: {attr}")
                        raise AttributeError(f"Config must have attribute: {attr}")
            else:
                print("Error: No 'config' or 'model_config' found in checkpoint")
                raise KeyError("Checkpoint must contain 'config' or 'model_config' to load the model")

            # Create model and load state dict
            if 'bit_widths' in checkpoint:
                bit_widths = checkpoint['bit_widths']
            else:
                bit_widths = [4, 8, 16]

            self.model = SwitchableQATGPT2(config, bit_widths=bit_widths, initialize_weights=False)

            # Load model state with strict=False to handle mismatched keys
            if 'model_state_dict' in checkpoint:
                missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'model' in checkpoint:
                missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model'], strict=False)
            else:
                # Checkpoint might be just the state dict
                missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint, strict=False)

            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
                print(f"First few missing keys: {missing_keys[:3]}...")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
                print(f"First few unexpected keys: {unexpected_keys[:3]}...")

            self.model = self.model.to(self.device)
            print("Model loaded from checkpoint - NOT loading pretrained weights")
        else:
            print("Error: No checkpoint file provided or checkpoint could not be loaded")
            print(f"Attempted to load: {model_path}")
            raise FileNotFoundError(f"Checkpoint file not found or invalid: {model_path}")

        from transformers import GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        
        # Store results
        self.results = {}
        
    def run_all_tests(self):
        """Run all diagnostic tests."""
        print("\n" + "="*70)
        print("RUNNING DIAGNOSTIC TESTS")
        print("="*70)
        
        # Test 1: Basic generation capability
        print("\n[TEST 1] Basic Generation Capability")
        print("-"*40)
        self.test_basic_generation()
        
        # Test 2: Quantization stability
        print("\n[TEST 2] Quantization Stability During Generation")
        print("-"*40)
        self.test_quantization_stability()
        
        # Test 3: Compare bit precisions
        print("\n[TEST 3] Performance Across Bit Widths")
        print("-"*40)
        self.test_bit_precision_comparison()
        
        # Test 4: LoRA interference
        print("\n[TEST 4] LoRA Adapter Interference")
        print("-"*40)
        self.test_lora_interference()
        
        # Test 5: Pretrained weights verification
        print("\n[TEST 5] Pretrained Weights Verification")
        print("-"*40)
        self.test_pretrained_weights()
        
        # Test 6: Prompt truncation
        print("\n[TEST 6] Prompt Truncation Analysis")
        print("-"*40)
        self.test_prompt_truncation()
        
        # Test 7: Direct logits vs generation
        print("\n[TEST 7] Direct Logits vs Generation")
        print("-"*40)
        self.test_direct_logits()
        
        # Test 8: Zero-shot on simple tasks
        print("\n[TEST 8] Simple Zero-Shot Tasks")
        print("-"*40)
        self.test_simple_zero_shot()
        
        # Save results
        self.save_results()
        
    def test_basic_generation(self):
        """Test if model can generate coherent text."""
        prompts = [
            "The capital of France is",
            "Two plus two equals",
            "The sun rises in the",
            "Water freezes at",
            "The largest planet is"
        ]
        
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                # Try generation
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=10,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append({
                'prompt': prompt,
                'generated': generated,
                'makes_sense': self._check_if_makes_sense(prompt, generated)
            })
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated}")
            print(f"Makes sense: {results[-1]['makes_sense']}")
            print()
        
        success_rate = sum(r['makes_sense'] for r in results) / len(results)
        self.results['basic_generation'] = {
            'success_rate': success_rate,
            'examples': results
        }
        print(f"Success rate: {success_rate*100:.1f}%")
        
    def test_quantization_stability(self):
        """Test if quantization scales remain stable during generation."""
        prompt = "The weather today is"
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # Get first layer's quantizers
        first_layer = self.model.h[0].attn.c_attn

        # Handle different quantizer structures
        initial_state = {}
        if hasattr(first_layer, 'quantizers_input'):  # SwitchableQATLinearWithLoRA
            bits_key = f'{first_layer.current_bits}bit'
            if bits_key in first_layer.quantizers_input:
                initial_state['input_scale'] = first_layer.quantizers_input[bits_key].scale.item()
                initial_state['input_calibrated'] = first_layer.quantizers_input[bits_key].calibrated
            if bits_key in first_layer.quantizers_weight:
                initial_state['weight_scale'] = first_layer.quantizers_weight[bits_key].scale.item()
                initial_state['weight_calibrated'] = first_layer.quantizers_weight[bits_key].calibrated
        elif hasattr(first_layer, 'quantize_input'):  # QATLinearWithLoRA
            initial_state['input_scale'] = first_layer.quantize_input.scale.item()
            initial_state['input_calibrated'] = first_layer.quantize_input.calibrated
            initial_state['weight_scale'] = first_layer.quantize_weight.scale.item()
            initial_state['weight_calibrated'] = first_layer.quantize_weight.calibrated
        else:
            print("Warning: Could not find quantizers in first layer")
            self.results['quantization_stability'] = {'error': 'No quantizers found'}
            return
        
        print("Initial quantizer state:")
        for key, val in initial_state.items():
            print(f"  {key}: {val}")
        
        # Generate multiple tokens
        scales_during_generation = []
        
        with torch.no_grad():
            current_ids = inputs['input_ids'].clone()
            
            for step in range(10):
                outputs = self.model(current_ids)
                next_token_logits = outputs['logits'][0, -1, :]
                next_token = torch.argmax(next_token_logits)
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
                # Record scale
                scale_data = {'step': step}

                if hasattr(first_layer, 'quantizers_input'):  # SwitchableQATLinearWithLoRA
                    bits_key = f'{first_layer.current_bits}bit'
                    if bits_key in first_layer.quantizers_input:
                        scale_data['input_scale'] = first_layer.quantizers_input[bits_key].scale.item()
                    if bits_key in first_layer.quantizers_weight:
                        scale_data['weight_scale'] = first_layer.quantizers_weight[bits_key].scale.item()
                elif hasattr(first_layer, 'quantize_input'):  # QATLinearWithLoRA
                    scale_data['input_scale'] = first_layer.quantize_input.scale.item()
                    scale_data['weight_scale'] = first_layer.quantize_weight.scale.item()

                scales_during_generation.append(scale_data)
        
        # Check if scales changed
        scale_changes = []
        for i, scales in enumerate(scales_during_generation):
            input_change = abs(scales.get('input_scale', 0) - initial_state.get('input_scale', 0))
            weight_change = abs(scales.get('weight_scale', 0) - initial_state.get('weight_scale', 0))
            scale_changes.append(input_change + weight_change)
            
            if input_change > 0.001 or weight_change > 0.001:
                print(f"Step {i}: Scale changed! Input: {scales['input_scale']:.6f}, Weight: {scales['weight_scale']:.6f}")
        
        max_change = max(scale_changes)
        stable = max_change < 0.001
        
        self.results['quantization_stability'] = {
            'stable': stable,
            'max_scale_change': max_change,
            'initial_state': initial_state,
            'final_state': scales_during_generation[-1] if scales_during_generation else None
        }
        
        print(f"\nQuantization stable: {stable}")
        print(f"Max scale change: {max_change:.6f}")
        
    def test_bit_precision_comparison(self):
        """Compare model performance at different bit widths."""
        prompt = "The capital of France is"
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        bit_widths = self.model.bit_widths if hasattr(self.model, 'bit_widths') else [8]
        results = {}
        
        for bits in bit_widths:
            print(f"\nTesting {bits}-bit precision:")
            
            # Set precision
            self.model.set_precision(bits)
            
            with torch.no_grad():
                # Generate
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Get perplexity on a sample
                test_text = "The quick brown fox jumps over the lazy dog."
                test_inputs = self.tokenizer(test_text, return_tensors='pt').to(self.device)
                model_outputs = self.model(test_inputs['input_ids'])
                
                # Calculate simple perplexity
                logits = model_outputs['logits']
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = test_inputs['input_ids'][..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                perplexity = torch.exp(loss).item()
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results[bits] = {
                'generated': generated,
                'perplexity': perplexity,
                'correct': 'paris' in generated.lower()
            }
            
            print(f"  Generated: {generated}")
            print(f"  Perplexity: {perplexity:.2f}")
            print(f"  Correct: {results[bits]['correct']}")
        
        self.results['bit_precision_comparison'] = results
        
    def test_lora_interference(self):
        """Test if LoRA adapters are interfering with pretrained weights."""
        prompt = "The capital of France is"
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # Test with LoRA enabled (default)
        with torch.no_grad():
            outputs_with_lora = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_with_lora = self.tokenizer.decode(outputs_with_lora[0], skip_special_tokens=True)
        
        # Temporarily disable LoRA
        lora_modules = []
        for name, module in self.model.named_modules():
            # Handle different LoRA structures
            if hasattr(module, 'lora_adapters'):  # SwitchableQATLinearWithLoRA
                for bits_key, adapter in module.lora_adapters.items():
                    if hasattr(adapter, 'scaling'):
                        lora_modules.append((f"{name}.{bits_key}", adapter, adapter.scaling))
                        adapter.scaling = 0  # Disable LoRA
            elif hasattr(module, 'lora'):  # QATLinearWithLoRA
                if hasattr(module.lora, 'scaling'):
                    lora_modules.append((name, module.lora, module.lora.scaling))
                    module.lora.scaling = 0  # Disable LoRA
        
        print(f"Disabled {len(lora_modules)} LoRA modules")
        
        # Test without LoRA
        with torch.no_grad():
            outputs_without_lora = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_without_lora = self.tokenizer.decode(outputs_without_lora[0], skip_special_tokens=True)
        
        # Restore LoRA
        for name, lora_module, original_scaling in lora_modules:
            lora_module.scaling = original_scaling
        
        self.results['lora_interference'] = {
            'with_lora': generated_with_lora,
            'without_lora': generated_without_lora,
            'lora_makes_worse': ('paris' not in generated_with_lora.lower() and 
                                 'paris' in generated_without_lora.lower())
        }
        
        print(f"With LoRA: {generated_with_lora}")
        print(f"Without LoRA: {generated_without_lora}")
        print(f"LoRA interfering: {self.results['lora_interference']['lora_makes_worse']}")
        
    def test_pretrained_weights(self):
        """Verify if pretrained weights are actually loaded."""
        try:
            # Load reference GPT-2
            reference = GPT2Model.from_pretrained('gpt2')
            
            # Compare embeddings
            sample_indices = [100, 500, 1000, 5000]
            
            your_embeds = self.model.wte.weight[sample_indices, :10].detach().cpu()
            ref_embeds = reference.wte.weight[sample_indices, :10].detach().cpu()
            
            # Calculate similarity
            similarity = F.cosine_similarity(
                your_embeds.flatten().unsqueeze(0),
                ref_embeds.flatten().unsqueeze(0)
            ).item()
            
            close_enough = torch.allclose(your_embeds, ref_embeds, atol=0.1)
            
            self.results['pretrained_weights'] = {
                'similarity': similarity,
                'close_enough': close_enough,
                'sample_your': your_embeds[0, :5].tolist(),
                'sample_ref': ref_embeds[0, :5].tolist()
            }
            
            print(f"Embedding similarity: {similarity:.4f}")
            print(f"Weights match reference: {close_enough}")
            print(f"Your model sample: {your_embeds[0, :5]}")
            print(f"Reference sample: {ref_embeds[0, :5]}")
            
        except Exception as e:
            print(f"Could not load reference model: {e}")
            self.results['pretrained_weights'] = {'error': str(e)}
            
    def test_prompt_truncation(self):
        """Test if prompts are being truncated in evaluation."""
        # Create a long multiple choice prompt
        context = "The Earth's atmosphere is composed of several layers. " * 20  # Make it long
        question = "What is the primary gas in Earth's atmosphere?"
        choices = [
            "A: Oxygen (O2)",
            "B: Carbon Dioxide (CO2)",
            "C: Nitrogen (N2)",
            "D: Argon (Ar)"
        ]
        
        full_prompt = f"Context: {context}\n\nQuestion: {question}\n"
        for choice in choices:
            full_prompt += f"{choice}\n"
        full_prompt += "Answer:"
        
        # Check token length
        tokens = self.tokenizer.encode(full_prompt)
        truncated_tokens = self.tokenizer.encode(full_prompt, truncation=True, max_length=250)
        
        # Decode to see what gets cut off
        truncated_text = self.tokenizer.decode(truncated_tokens)
        
        # Check if answer choices are preserved
        choices_preserved = all(choice[0] in truncated_text for choice in choices)
        
        self.results['prompt_truncation'] = {
            'full_length': len(tokens),
            'truncated_length': len(truncated_tokens),
            'was_truncated': len(tokens) > 250,
            'choices_preserved': choices_preserved,
            'tokens_lost': len(tokens) - len(truncated_tokens)
        }
        
        print(f"Full prompt length: {len(tokens)} tokens")
        print(f"Truncated length: {len(truncated_tokens)} tokens")
        print(f"Was truncated: {len(tokens) > 250}")
        print(f"Answer choices preserved: {choices_preserved}")
        
        if not choices_preserved:
            print("\nWARNING: Answer choices are being cut off!")
            print("This explains 0% accuracy on multiple choice tasks!")
            
    def test_direct_logits(self):
        """Compare direct logit prediction vs generation."""
        prompt = "Two plus two equals"
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            # Method 1: Full generation
            generated_ids = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=5,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Method 2: Single forward pass
            outputs = self.model(inputs['input_ids'])
            next_token_logits = outputs['logits'][0, -1, :]
            
            # Get top 5 predictions
            top5_tokens = torch.topk(next_token_logits, 5)
            top5_words = [self.tokenizer.decode([idx]) for idx in top5_tokens.indices]
            
            # Get the actual next token
            next_token = next_token_logits.argmax()
            next_word = self.tokenizer.decode([next_token])
            
        self.results['direct_logits'] = {
            'generated': generated_text,
            'next_token_prediction': next_word,
            'top5_predictions': top5_words,
            'correct': 'four' in generated_text.lower() or '4' in generated_text
        }
        
        print(f"Generated: {generated_text}")
        print(f"Direct next token: {next_word}")
        print(f"Top 5 predictions: {top5_words}")
        print(f"Correct: {self.results['direct_logits']['correct']}")
        
    def test_simple_zero_shot(self):
        """Test on very simple zero-shot tasks."""
        simple_tasks = [
            {
                'prompt': "Question: Is the sky blue? Answer (Yes/No):",
                'correct_answers': ['yes'],
            },
            {
                'prompt': "Complete: Two plus two equals",
                'correct_answers': ['four', '4'],
            },
            {
                'prompt': "The capital of France is",
                'correct_answers': ['paris'],
            },
            {
                'prompt': "Question: What color is grass? Answer:",
                'correct_answers': ['green'],
            }
        ]
        
        results = []
        for task in simple_tasks:
            inputs = self.tokenizer(task['prompt'], return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=5,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            correct = any(ans in generated.lower() for ans in task['correct_answers'])
            
            results.append({
                'prompt': task['prompt'][:50] + "...",
                'generated': generated,
                'correct': correct
            })
            
            print(f"Prompt: {task['prompt'][:50]}...")
            print(f"Generated: {generated}")
            print(f"Correct: {correct}")
            print()
        
        accuracy = sum(r['correct'] for r in results) / len(results)
        self.results['simple_zero_shot'] = {
            'accuracy': accuracy,
            'tasks': results
        }
        
        print(f"Simple task accuracy: {accuracy*100:.1f}%")
        
    def _check_if_makes_sense(self, prompt, generated):
        """Simple heuristic to check if generation makes sense."""
        expected = {
            "The capital of France is": ["paris"],
            "Two plus two equals": ["four", "4"],
            "The sun rises in the": ["east", "morning"],
            "Water freezes at": ["0", "zero", "32"],
            "The largest planet is": ["jupiter"]
        }
        
        if prompt in expected:
            generated_lower = generated.lower()
            return any(exp in generated_lower for exp in expected[prompt])
        return False
    
    def save_results(self):
        """Save diagnostic results to file."""
        output_file = 'model_diagnostic_results.json'
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n{'='*70}")
        print("DIAGNOSTIC SUMMARY")
        print('='*70)
        
        # Print summary
        if 'basic_generation' in self.results:
            print(f"Basic generation success: {self.results['basic_generation']['success_rate']*100:.1f}%")
        
        if 'quantization_stability' in self.results:
            print(f"Quantization stable: {self.results['quantization_stability']['stable']}")
        
        if 'lora_interference' in self.results:
            print(f"LoRA interfering: {self.results['lora_interference']['lora_makes_worse']}")
        
        if 'prompt_truncation' in self.results:
            print(f"Prompts truncated: {self.results['prompt_truncation']['was_truncated']}")
            print(f"Choices preserved: {self.results['prompt_truncation']['choices_preserved']}")
        
        if 'simple_zero_shot' in self.results:
            print(f"Simple task accuracy: {self.results['simple_zero_shot']['accuracy']*100:.1f}%")
        
        print(f"\nFull results saved to: {output_file}")


def main():
    """Run diagnostics."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose QAT model issues')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to config JSON')
    
    args = parser.parse_args()
    
    diagnostics = ModelDiagnostics(args.model_path, args.config_path)
    diagnostics.run_all_tests()


if __name__ == "__main__":
    main()

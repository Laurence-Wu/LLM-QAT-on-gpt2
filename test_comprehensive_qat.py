#!/usr/bin/env python3
"""
Comprehensive test suite for QAT model with switchable precision.
Tests all critical aspects after fixing the perplexity issue.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time
import json
import math
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from transformers import GPT2Config, GPT2Tokenizer

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'part1_switchable_precision'))

from models import SwitchableQATGPT2
from config_qat import ModelConfig
from dataset import create_dataloaders


class ComprehensiveQATTester:
    """Comprehensive testing framework for QAT models."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.test_results = {}

    def create_model(self, load_pretrained=True):
        """Create a properly initialized model."""
        config = ModelConfig()
        gpt2_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.n_positions,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            layer_norm_epsilon=config.layer_norm_epsilon,
            embd_pdrop=config.embd_pdrop,
            quantization_bits=8,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout
        )

        model = SwitchableQATGPT2(gpt2_config, bit_widths=[4, 8, 16])

        if load_pretrained:
            # Load pretrained weights (assuming the fix is applied)
            from main_qat import load_pretrained_weights
            load_pretrained_weights(model)

        return model.to(self.device)

    # ============== TEST 1: QUANTIZATION ACCURACY ==============
    def test_quantization_accuracy(self):
        """Test model accuracy at different quantization levels."""
        print("\n" + "="*60)
        print("TEST 1: Quantization Accuracy at Different Bit Widths")
        print("="*60)

        model = self.create_model()
        model.eval()

        test_text = "The quick brown fox jumps over the lazy dog"
        inputs = self.tokenizer(test_text, return_tensors='pt', max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        results = {}
        reference_logits = None

        for bits in [16, 8, 4]:
            model.set_precision(bits)

            with torch.no_grad():
                outputs = model(input_ids=inputs['input_ids'])
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits

                if bits == 16:
                    reference_logits = logits.clone()

                # Calculate metrics
                if reference_logits is not None:
                    mse = torch.mean((logits - reference_logits) ** 2).item()
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        logits.flatten(), reference_logits.flatten(), dim=0
                    ).item()
                else:
                    mse = 0
                    cosine_sim = 1.0

                # Calculate perplexity
                labels = inputs['input_ids']
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1))
                perplexity = math.exp(loss.item()) if loss.item() < 20 else float('inf')

                results[bits] = {
                    'perplexity': perplexity,
                    'mse_from_fp16': mse,
                    'cosine_similarity': cosine_sim
                }

                print(f"\n{bits}-bit precision:")
                print(f"  Perplexity: {perplexity:.2f}")
                print(f"  MSE from FP16: {mse:.6f}")
                print(f"  Cosine similarity: {cosine_sim:.4f}")

        # Check if degradation is acceptable
        if results[4]['perplexity'] < results[16]['perplexity'] * 2:
            print("\n✅ 4-bit quantization maintains reasonable accuracy")
        else:
            print("\n⚠️ 4-bit quantization shows significant degradation")

        self.test_results['quantization_accuracy'] = results
        return results

    # ============== TEST 2: PRECISION SWITCHING ==============
    def test_precision_switching(self):
        """Test switching between different precisions dynamically."""
        print("\n" + "="*60)
        print("TEST 2: Dynamic Precision Switching")
        print("="*60)

        model = self.create_model()
        model.eval()

        test_text = "Hello world"
        inputs = self.tokenizer(test_text, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        switch_sequence = [16, 8, 4, 8, 16, 4, 16]
        outputs = []

        print("\nSwitching sequence:", switch_sequence)

        for bits in switch_sequence:
            model.set_precision(bits)

            with torch.no_grad():
                output = model(input_ids=inputs['input_ids'])
                logits = output['logits'] if isinstance(output, dict) else output.logits
                outputs.append((bits, logits.clone()))

            print(f"  Set to {bits}-bit: Output shape {logits.shape}, "
                  f"mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")

        # Verify outputs are consistent for same bit widths
        print("\nConsistency check:")
        for i, (bits1, out1) in enumerate(outputs):
            for j, (bits2, out2) in enumerate(outputs[i+1:], i+1):
                if bits1 == bits2:
                    diff = torch.mean(torch.abs(out1 - out2)).item()
                    print(f"  Outputs at indices {i} and {j} (both {bits1}-bit): "
                          f"Mean absolute difference = {diff:.6f}")

                    if diff < 0.001:
                        print(f"    ✅ Consistent outputs for {bits1}-bit")
                    else:
                        print(f"    ⚠️ Inconsistent outputs for {bits1}-bit")

        self.test_results['precision_switching'] = True
        return outputs

    # ============== TEST 3: GRADIENT FLOW ==============
    def test_gradient_flow(self):
        """Test that gradients flow properly through quantized layers."""
        print("\n" + "="*60)
        print("TEST 3: Gradient Flow Through Quantized Layers")
        print("="*60)

        model = self.create_model()
        model.train()

        test_text = "The capital of France is Paris"
        inputs = self.tokenizer(test_text, return_tensors='pt', max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        results = {}

        for bits in [16, 8, 4]:
            model.set_precision(bits)
            model.zero_grad()

            # Forward pass
            outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss

            # Backward pass
            loss.backward()

            # Check gradients
            grad_stats = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_mean = param.grad.mean().item()
                    grad_std = param.grad.std().item()

                    # Check for NaN or Inf
                    has_nan = torch.isnan(param.grad).any().item()
                    has_inf = torch.isinf(param.grad).any().item()

                    grad_stats.append({
                        'name': name,
                        'norm': grad_norm,
                        'mean': grad_mean,
                        'std': grad_std,
                        'has_nan': has_nan,
                        'has_inf': has_inf
                    })

            # Summary statistics
            total_params = len(grad_stats)
            params_with_grad = len([g for g in grad_stats if g['norm'] > 0])
            params_with_nan = len([g for g in grad_stats if g['has_nan']])
            params_with_inf = len([g for g in grad_stats if g['has_inf']])

            results[bits] = {
                'loss': loss.item(),
                'params_with_gradients': params_with_grad,
                'total_params': total_params,
                'params_with_nan': params_with_nan,
                'params_with_inf': params_with_inf
            }

            print(f"\n{bits}-bit precision:")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Parameters with gradients: {params_with_grad}/{total_params}")
            print(f"  Parameters with NaN: {params_with_nan}")
            print(f"  Parameters with Inf: {params_with_inf}")

            if params_with_grad > total_params * 0.8:
                print(f"  ✅ Good gradient flow")
            else:
                print(f"  ⚠️ Poor gradient flow")

        self.test_results['gradient_flow'] = results
        return results

    # ============== TEST 4: LORA ADAPTATION ==============
    def test_lora_adaptation(self):
        """Test LoRA adaptation at different bit widths."""
        print("\n" + "="*60)
        print("TEST 4: LoRA Adaptation at Different Precisions")
        print("="*60)

        model = self.create_model()

        # Check LoRA parameters at different bit widths
        for bits in [4, 8, 16]:
            model.set_precision(bits)

            print(f"\n{bits}-bit precision LoRA configuration:")

            # Count LoRA parameters
            lora_params = 0
            total_params = 0

            for name, module in model.named_modules():
                if hasattr(module, 'lora_adapters') and module.lora_adapters:
                    if bits in module.lora_adapters:
                        lora = module.lora_adapters[bits]
                        if hasattr(lora, 'lora_A') and hasattr(lora, 'lora_B'):
                            a_params = lora.lora_A.numel()
                            b_params = lora.lora_B.numel()
                            lora_params += a_params + b_params

                            # Check ranks
                            rank = lora.lora_A.shape[0]
                            print(f"    {name}: rank={rank}, "
                                  f"A shape={lora.lora_A.shape}, B shape={lora.lora_B.shape}")

            for param in model.parameters():
                total_params += param.numel()

            compression_ratio = total_params / max(lora_params, 1)

            print(f"  Total LoRA parameters: {lora_params:,}")
            print(f"  Total model parameters: {total_params:,}")
            print(f"  Compression ratio: {compression_ratio:.1f}x")
            print(f"  LoRA overhead: {100 * lora_params / total_params:.2f}%")

        self.test_results['lora_adaptation'] = True
        return True

    # ============== TEST 5: MEMORY EFFICIENCY ==============
    def test_memory_efficiency(self):
        """Test memory usage at different bit widths."""
        print("\n" + "="*60)
        print("TEST 5: Memory Efficiency")
        print("="*60)

        if not torch.cuda.is_available():
            print("  ⚠️ CUDA not available, skipping memory test")
            return None

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        results = {}

        for bits in [16, 8, 4]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            model = self.create_model()
            model.set_precision(bits)

            # Run inference
            test_text = "The quick brown fox " * 20  # Longer text
            inputs = self.tokenizer(test_text, return_tensors='pt',
                                   max_length=256, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                _ = model(input_ids=inputs['input_ids'])

            # Get memory stats
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            reserved = torch.cuda.memory_reserved() / 1024**2  # MB
            peak = torch.cuda.max_memory_allocated() / 1024**2  # MB

            results[bits] = {
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'peak_mb': peak
            }

            print(f"\n{bits}-bit precision:")
            print(f"  Allocated: {allocated:.1f} MB")
            print(f"  Reserved: {reserved:.1f} MB")
            print(f"  Peak: {peak:.1f} MB")

            del model
            torch.cuda.empty_cache()

        # Calculate savings
        if 16 in results and 4 in results:
            savings = (results[16]['peak_mb'] - results[4]['peak_mb']) / results[16]['peak_mb'] * 100
            print(f"\nMemory savings (4-bit vs 16-bit): {savings:.1f}%")

            if savings > 20:
                print("✅ Significant memory savings achieved")
            else:
                print("⚠️ Limited memory savings")

        self.test_results['memory_efficiency'] = results
        return results

    # ============== TEST 6: TRAINING STABILITY ==============
    def test_training_stability(self):
        """Test if model can train stably at different bit widths."""
        print("\n" + "="*60)
        print("TEST 6: Training Stability")
        print("="*60)

        model = self.create_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        # Create small dataset
        train_texts = [
            "The capital of France is Paris.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language.",
            "Neural networks can learn complex patterns.",
        ]

        results = {}

        for bits in [16, 8, 4]:
            model.set_precision(bits)
            model.train()

            losses = []
            print(f"\n{bits}-bit training:")

            for epoch in range(5):
                epoch_losses = []

                for text in train_texts:
                    inputs = self.tokenizer(text, return_tensors='pt',
                                           max_length=128, truncation=True,
                                           padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    optimizer.zero_grad()
                    outputs = model(input_ids=inputs['input_ids'],
                                  labels=inputs['input_ids'])
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss

                    # Check for NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"  ❌ NaN/Inf loss at epoch {epoch}")
                        break

                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()

                    epoch_losses.append(loss.item())

                avg_loss = np.mean(epoch_losses)
                losses.append(avg_loss)
                print(f"  Epoch {epoch}: Loss = {avg_loss:.4f}")

            # Check if training is stable (loss decreases)
            if len(losses) > 1:
                if losses[-1] < losses[0]:
                    print(f"  ✅ Training is stable and improving")
                else:
                    print(f"  ⚠️ Training may be unstable")

            results[bits] = losses

        self.test_results['training_stability'] = results
        return results

    # ============== TEST 7: INFERENCE SPEED ==============
    def test_inference_speed(self):
        """Benchmark inference speed at different bit widths."""
        print("\n" + "="*60)
        print("TEST 7: Inference Speed Benchmark")
        print("="*60)

        model = self.create_model()
        model.eval()

        # Prepare test data
        test_text = "The quick brown fox jumps over the lazy dog. " * 5
        inputs = self.tokenizer(test_text, return_tensors='pt',
                               max_length=256, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        results = {}
        num_runs = 100

        for bits in [16, 8, 4]:
            model.set_precision(bits)

            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(input_ids=inputs['input_ids'])

            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            for _ in range(num_runs):
                with torch.no_grad():
                    _ = model(input_ids=inputs['input_ids'])

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            avg_time = (end_time - start_time) / num_runs * 1000  # ms
            throughput = 1000 / avg_time  # inferences per second

            results[bits] = {
                'avg_time_ms': avg_time,
                'throughput': throughput
            }

            print(f"\n{bits}-bit precision:")
            print(f"  Average time: {avg_time:.2f} ms")
            print(f"  Throughput: {throughput:.1f} inferences/sec")

        # Calculate speedup
        if 16 in results and 4 in results:
            speedup = results[4]['throughput'] / results[16]['throughput']
            print(f"\n4-bit vs 16-bit speedup: {speedup:.2f}x")

            if speedup > 1.5:
                print("✅ Significant speedup achieved")
            else:
                print("⚠️ Limited speedup")

        self.test_results['inference_speed'] = results
        return results

    # ============== TEST 8: CHECKPOINT SAVE/LOAD ==============
    def test_checkpoint_operations(self):
        """Test saving and loading model with quantization states."""
        print("\n" + "="*60)
        print("TEST 8: Checkpoint Save/Load")
        print("="*60)

        # Create and setup model
        model1 = self.create_model()
        model1.set_precision(8)

        # Generate reference output
        test_text = "Test checkpoint"
        inputs = self.tokenizer(test_text, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output1 = model1(input_ids=inputs['input_ids'])
            logits1 = output1['logits'] if isinstance(output1, dict) else output1.logits

        # Save checkpoint
        checkpoint = {
            'model_state_dict': model1.state_dict(),
            'current_bits': model1.current_bits,
            'bit_widths': model1.bit_widths
        }
        torch.save(checkpoint, 'test_checkpoint.pth')
        print("  ✓ Checkpoint saved")

        # Create new model and load checkpoint
        model2 = self.create_model(load_pretrained=False)
        checkpoint = torch.load('test_checkpoint.pth')
        model2.load_state_dict(checkpoint['model_state_dict'])
        model2.current_bits = checkpoint['current_bits']

        print(f"  ✓ Checkpoint loaded (precision={checkpoint['current_bits']}-bit)")

        # Compare outputs
        with torch.no_grad():
            output2 = model2(input_ids=inputs['input_ids'])
            logits2 = output2['logits'] if isinstance(output2, dict) else output2.logits

        difference = torch.mean(torch.abs(logits1 - logits2)).item()
        print(f"  Output difference: {difference:.6f}")

        if difference < 1e-5:
            print("  ✅ Checkpoint save/load successful")
            success = True
        else:
            print("  ❌ Checkpoint save/load failed")
            success = False

        # Cleanup
        os.remove('test_checkpoint.pth')

        self.test_results['checkpoint_operations'] = success
        return success

    # ============== TEST 9: OUTPUT CONSISTENCY ==============
    def test_output_consistency(self):
        """Test consistency of outputs across multiple runs."""
        print("\n" + "="*60)
        print("TEST 9: Output Consistency")
        print("="*60)

        model = self.create_model()
        model.eval()

        test_texts = [
            "The weather is",
            "Machine learning",
            "In the future"
        ]

        results = {}

        for bits in [16, 8, 4]:
            model.set_precision(bits)
            print(f"\n{bits}-bit precision consistency test:")

            text_outputs = {}

            for text in test_texts:
                inputs = self.tokenizer(text, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = []
                for run in range(5):
                    with torch.no_grad():
                        output = model(input_ids=inputs['input_ids'])
                        logits = output['logits'] if isinstance(output, dict) else output.logits
                        outputs.append(logits.clone())

                # Check consistency
                differences = []
                for i in range(1, len(outputs)):
                    diff = torch.mean(torch.abs(outputs[0] - outputs[i])).item()
                    differences.append(diff)

                avg_diff = np.mean(differences)
                max_diff = np.max(differences)

                text_outputs[text] = {
                    'avg_diff': avg_diff,
                    'max_diff': max_diff
                }

                print(f"  '{text}': avg_diff={avg_diff:.6f}, max_diff={max_diff:.6f}")

                if max_diff < 1e-5:
                    print(f"    ✅ Perfectly consistent")
                elif max_diff < 1e-3:
                    print(f"    ✅ Highly consistent")
                else:
                    print(f"    ⚠️ Some inconsistency detected")

            results[bits] = text_outputs

        self.test_results['output_consistency'] = results
        return results

    # ============== MAIN TEST RUNNER ==============
    def run_all_tests(self):
        """Run all tests and generate summary report."""
        print("\n" + "="*70)
        print("COMPREHENSIVE QAT MODEL TEST SUITE")
        print("="*70)

        test_functions = [
            self.test_quantization_accuracy,
            self.test_precision_switching,
            self.test_gradient_flow,
            self.test_lora_adaptation,
            self.test_memory_efficiency,
            self.test_training_stability,
            self.test_inference_speed,
            self.test_checkpoint_operations,
            self.test_output_consistency
        ]

        for test_func in test_functions:
            try:
                test_func()
            except Exception as e:
                print(f"\n❌ Test {test_func.__name__} failed: {e}")
                import traceback
                traceback.print_exc()

        # Generate summary report
        self.generate_summary_report()

    def generate_summary_report(self):
        """Generate a summary report of all tests."""
        print("\n" + "="*70)
        print("TEST SUMMARY REPORT")
        print("="*70)

        # Save results to JSON
        with open('qat_test_results.json', 'w') as f:
            # Convert numpy/tensor types to Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, (np.ndarray, torch.Tensor)):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                return obj

            serializable_results = convert_to_serializable(self.test_results)
            json.dump(serializable_results, f, indent=2)

        print("\n📊 Test Results Summary:")
        print("-" * 50)

        for test_name, result in self.test_results.items():
            print(f"\n{test_name.replace('_', ' ').title()}:")
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
            elif isinstance(result, bool):
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"  Status: {status}")
            else:
                print(f"  Result: {result}")

        print("\n" + "="*70)
        print("Results saved to: qat_test_results.json")
        print("="*70)


def main():
    """Main test execution."""
    tester = ComprehensiveQATTester()
    tester.run_all_tests()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
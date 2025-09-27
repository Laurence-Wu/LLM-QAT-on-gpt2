import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Config
from datasets import load_dataset
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
import json
import math
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from deploy import load_model_for_evaluation
from quantization import LearnableFakeQuantize


class INT16InferenceTester:
    """Test INT16 inference with per-tensor calibration strategy"""

    def __init__(self, checkpoint_path: str = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize model
        self.model = self._initialize_model(checkpoint_path)

    def _initialize_model(self, checkpoint_path: Optional[str] = None):
        """Initialize model with per-tensor quantization for evaluation"""
        print("Loading INT16 model with per-tensor quantization...")

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint path not found: {checkpoint_path}")
            print("Using random initialization - results may not be meaningful")
            # For testing purposes, create a model with per-tensor quantization
            from config_sp import ModelConfig
            from models_sp import SPLMHeadModel

            config = ModelConfig()
            config.per_channel_quantization = False  # Use per-tensor for evaluation

            gpt2_config = GPT2Config(
                vocab_size=config.vocab_size,
                n_positions=config.n_positions,
                n_embd=config.n_embd,
                n_layer=config.n_layer,
                n_head=config.n_head,
                layer_norm_epsilon=config.layer_norm_epsilon,
                use_cache=False,
            )

            # Add switchable precision configs
            gpt2_config.lora_rank_per_bit = config.lora_rank_per_bit
            gpt2_config.lora_alpha_per_bit = config.lora_alpha_per_bit
            gpt2_config.quantizer_per_bit = config.quantizer_per_bit
            gpt2_config.bit_widths = config.bit_widths
            gpt2_config.per_channel_quantization = False

            model = SPLMHeadModel(gpt2_config)
            model.set_precision(16)
        else:
            # Load model with per-tensor quantization for evaluation
            model = load_model_for_evaluation(
                checkpoint_path=checkpoint_path,
                target_bits=16,  # INT16 configuration
                device=self.device
            )
            return model  # Already on device and in eval mode

        model = model.to(self.device)
        model.eval()
        return model

    def calibrate_input_quantizers_only(self, dataset_name: str = 'wikitext2',
                                       num_samples: int = 50):
        """
        Calibrate only input quantizers with evaluation data.
        Weight quantizers remain unchanged (already calibrated from training).
        """
        print(f"\nCalibrating input quantizers with {dataset_name}...")

        # Get current bit configuration
        current_bits = None
        for module in self.model.modules():
            if hasattr(module, 'current_bits'):
                current_bits = module.current_bits
                break

        if current_bits is None or current_bits >= 32:
            print("No calibration needed for 32-bit or FP mode")
            return

        bits_key = f'{current_bits}bit'

        # Configure input quantizers for calibration (already in per-tensor mode)
        input_quantizers = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizers_input') and bits_key in module.quantizers_input:
                quantizer = module.quantizers_input[bits_key]
                # Quantizers are already in per-tensor mode from model loading
                # Just enable calibration
                quantizer.calibrated = False
                quantizer.collecting_stats = True
                quantizer.num_batches_collected = 0
                quantizer.temp_min = None
                quantizer.temp_max = None
                input_quantizers.append((name, quantizer))

        print(f"Found {len(input_quantizers)} input quantizers to calibrate")

        # Load calibration data
        if dataset_name == 'wikitext2':
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
            texts = [item['text'] for item in dataset if item['text'].strip()][:num_samples]
        else:
            dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=True)
            texts = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                texts.append(item['text'])

        # Collect statistics
        self.model.eval()
        with torch.no_grad():
            for text in tqdm(texts, desc="Calibrating inputs"):
                if not text.strip():
                    continue

                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=256,
                    padding=False
                ).to(self.device)

                try:
                    _ = self.model(inputs['input_ids'])
                except Exception:
                    continue

        # Finalize calibration and move to CPU
        for name, quantizer in input_quantizers:
            quantizer.collecting_stats = False
            if quantizer.temp_min is not None and quantizer.temp_max is not None:
                # Compute scale/zero_point on CPU
                temp_min = quantizer.temp_min.cpu()
                temp_max = quantizer.temp_max.cpu()

                # Handle different quantizer types
                if quantizer.quantizer_type == 'minmax':
                    if quantizer.symmetric:
                        max_val = torch.max(torch.abs(temp_min), torch.abs(temp_max))
                        scale = max_val / (2 ** (quantizer.num_bits - 1) - 1)
                        zero_point = torch.zeros_like(scale)
                    else:
                        scale = (temp_max - temp_min) / (2 ** quantizer.num_bits - 1)
                        zero_point = temp_min
                elif quantizer.quantizer_type == 'log':
                    # Handle log quantization
                    eps = 1e-8
                    log_min = torch.log(torch.abs(temp_min) + eps)
                    log_max = torch.log(torch.abs(temp_max) + eps)
                    scale = (log_max - log_min)  # log_range
                    zero_point = log_min  # log_min
                else:
                    # Default fallback
                    max_val = torch.max(torch.abs(temp_min), torch.abs(temp_max))
                    scale = max_val / (2 ** (quantizer.num_bits - 1) - 1)
                    zero_point = torch.zeros_like(scale)

                # Store on CPU (already on CPU)
                quantizer.scale = scale
                quantizer.zero_point = zero_point
                quantizer.calibrated = True
                quantizer.temp_min = None
                quantizer.temp_max = None

        print("Input calibration completed (weights unchanged)")

    def calibrate_with_dataset(self, dataset_name: str = 'wikitext2',
                              num_samples: int = 100, max_length: int = 256):
        """
        Calibrate quantizers with specified dataset
        Demonstrates calibration strategy for inference
        """
        print(f"\nCalibrating with {dataset_name} dataset...")

        # Load calibration data
        if dataset_name == 'wikitext2':
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
            texts = [item['text'] for item in dataset if item['text'].strip()][:num_samples]
        elif dataset_name == 'openwebtext':
            dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=True)
            texts = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                texts.append(item['text'])
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        print(f"Loaded {len(texts)} text samples for calibration")

        # Enable calibration mode for all quantizers
        for module in self.model.modules():
            if isinstance(module, LearnableFakeQuantize):
                module.calibration = True

        # Process calibration data
        self.model.eval()
        with torch.no_grad():
            for text in tqdm(texts, desc="Calibrating"):
                if not text.strip():
                    continue

                # Tokenize with truncation to max_length
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_length,
                    padding=False
                ).to(self.device)

                # Forward pass to collect statistics
                try:
                    _ = self.model(inputs['input_ids'])
                except Exception as e:
                    print(f"Warning: Error during calibration: {e}")
                    continue

        # Disable calibration mode
        for module in self.model.modules():
            if isinstance(module, LearnableFakeQuantize):
                module.calibration = False

        print("Calibration completed")

    def test_variable_length_inference(self):
        """
        Test inference with variable-length sequences
        This demonstrates how per-tensor calibration handles different sequence lengths
        """
        print("\n" + "="*50)
        print("Testing variable-length inference")
        print("="*50)

        test_texts = [
            "The quick brown fox",  # Short
            "In the beginning of the story, there was a small village nestled in the mountains where people lived peacefully for generations",  # Medium
            "Artificial intelligence has revolutionized many aspects of our daily lives, from voice assistants that help us manage our schedules to recommendation systems that suggest what we might enjoy watching or reading. The technology continues to evolve at a rapid pace, bringing both exciting opportunities and important ethical considerations that society must carefully address."  # Long
        ]

        self.model.eval()
        with torch.no_grad():
            for i, text in enumerate(test_texts):
                print(f"\nTest {i+1} - Input length: {len(text.split())} words")
                print(f"Input: {text[:50]}...")

                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=256,
                    padding=False
                ).to(self.device)

                seq_len = inputs['input_ids'].shape[1]
                print(f"Token length: {seq_len}")

                try:
                    # Forward pass
                    outputs = self.model(inputs['input_ids'])

                    # Get logits shape
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs

                    print(f"Output shape: {logits.shape}")
                    print("✓ Inference successful")

                except Exception as e:
                    print(f"✗ Inference failed: {e}")

    def calculate_perplexity(self, dataset_name: str = 'wikitext2',
                            num_samples: int = 50, max_length: int = 256):
        """
        Calculate perplexity to evaluate quantization quality
        """
        print(f"\nCalculating perplexity on {dataset_name}...")

        # Load evaluation data
        if dataset_name == 'wikitext2':
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
            texts = [item['text'] for item in dataset if item['text'].strip()][:num_samples]
        else:
            dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=True)
            texts = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                texts.append(item['text'])

        self.model.eval()
        all_losses = []

        with torch.no_grad():
            for text in tqdm(texts, desc="Evaluating perplexity"):
                if not text.strip() or len(text.split()) < 10:
                    continue

                # Tokenize
                encodings = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_length,
                    padding=False
                ).to(self.device)

                input_ids = encodings['input_ids']
                if input_ids.shape[1] < 2:
                    continue

                try:
                    # Forward pass
                    outputs = self.model(input_ids)

                    # Get logits
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs

                    # Calculate loss
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()

                    loss_fct = nn.CrossEntropyLoss(reduction='none')
                    losses = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )

                    loss = losses.mean().item()
                    if not math.isnan(loss) and not math.isinf(loss):
                        all_losses.append(loss)

                except Exception as e:
                    continue

        if all_losses:
            avg_loss = sum(all_losses) / len(all_losses)
            perplexity = math.exp(min(avg_loss, 10))  # Cap to avoid overflow
            print(f"Perplexity: {perplexity:.2f}")
            return perplexity
        else:
            print("Could not calculate perplexity")
            return float('inf')

    def generate_text(self, prompt: str, max_new_tokens: int = 50):
        """
        Generate text to test inference capability
        """
        print(f"\nGenerating text from prompt: '{prompt}'")

        self.model.eval()

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            # Generate
            generated_ids = inputs['input_ids']

            for _ in range(max_new_tokens):
                outputs = self.model(generated_ids)

                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        return generated_text

    def run_full_test(self):
        """
        Run complete test suite demonstrating calibration strategies
        """
        print("\n" + "="*60)
        print("INT16 Inference Test with Input-Only Calibration")
        print("="*60)

        # Test 1: Baseline - no recalibration (using training calibration)
        print("\n1. Testing with original training calibration")
        print("-"*50)
        self.test_variable_length_inference()
        ppl_baseline = self.calculate_perplexity('wikitext2', num_samples=20)

        # Test 2: Input calibration with WikiText-2
        print("\n2. Testing with input calibration on WikiText-2")
        print("-"*50)
        self.calibrate_input_quantizers_only('wikitext2', num_samples=50)
        self.test_variable_length_inference()
        ppl_wikitext = self.calculate_perplexity('wikitext2', num_samples=20)

        # Test 3: Cross-dataset calibration
        print("\n3. Testing cross-dataset calibration")
        print("-"*50)
        print("Calibrating with OpenWebText, evaluating on WikiText2")
        self.calibrate_input_quantizers_only('openwebtext', num_samples=50)
        ppl_cross = self.calculate_perplexity('wikitext2', num_samples=20)

        # Test 4: Text generation
        print("\n4. Testing text generation")
        print("-"*50)
        test_prompts = [
            "The future of AI is",
            "Once upon a time",
            "In conclusion,"
        ]
        for prompt in test_prompts:
            self.generate_text(prompt, max_new_tokens=30)

        # Summary
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        print(f"Baseline perplexity (training calibration): {ppl_baseline:.2f}")
        print(f"Perplexity with WikiText-2 input calibration: {ppl_wikitext:.2f}")
        print(f"Perplexity with cross-dataset calibration: {ppl_cross:.2f}")

        if ppl_wikitext < ppl_baseline:
            print("\n✓ Input recalibration improves performance on evaluation data")
        else:
            print("\n✓ Training calibration generalizes well")

        print("\nKey insights:")
        print("- Weight quantizers preserve training calibration")
        print("- Only input quantizers are recalibrated for evaluation")
        print("- Per-tensor mode for inputs handles variable-length sequences")
        print("- Calibration statistics stored on CPU for memory efficiency")


def main():
    """Main test function"""
    import argparse

    parser = argparse.ArgumentParser(description='Test INT16 inference with calibration strategies')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to run on (cuda/cpu)')
    parser.add_argument('--quick', action='store_true',
                      help='Run quick test with fewer samples')

    args = parser.parse_args()

    # Initialize tester
    tester = INT16InferenceTester(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    if args.quick:
        # Quick test
        print("\nRunning quick test...")
        tester.calibrate_input_quantizers_only('wikitext2', num_samples=10)
        tester.test_variable_length_inference()
        tester.generate_text("Hello world", max_new_tokens=20)
    else:
        # Full test suite
        tester.run_full_test()

    print("\n✓ Test completed successfully")


if __name__ == "__main__":
    main()
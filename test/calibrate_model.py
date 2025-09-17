"""
Properly calibrate quantizers for QAT models.
"""

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Config as HFConfig, GPT2LMHeadModel
from tqdm import tqdm
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import QATGPT2
from shared.quantization import LearnableFakeQuantize
from part1_switchable_precision.main_qat import load_pretrained_weights


def calibrate_quantizers(model, tokenizer, device='cuda', num_samples=100):
    """
    Calibrate all quantizers in the model using sample data.

    Args:
        model: QAT model to calibrate
        tokenizer: Tokenizer for generating calibration data
        device: Device to run on
        num_samples: Number of calibration samples
    """
    print(f"Calibrating quantizers with {num_samples} samples...")

    # Sample calibration texts
    calibration_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we solve problems.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models have millions of parameters.",
        "Transformers revolutionized the field of NLP.",
        "Attention mechanisms help models focus on relevant information.",
        "Quantization reduces model size and improves inference speed.",
        "Neural networks are inspired by biological neurons.",
        "Training large models requires significant computational resources.",
        "Fine-tuning adapts pretrained models to specific tasks."
    ]

    # Extend by repeating if needed
    while len(calibration_texts) < num_samples:
        calibration_texts.extend(calibration_texts[:min(10, num_samples - len(calibration_texts))])

    # Put model in training mode for calibration
    model.train()

    # Run calibration
    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(calibration_texts))), desc="Calibrating"):
            text = calibration_texts[i % len(calibration_texts)]
            inputs = tokenizer(text, return_tensors='pt', max_length=128,
                             truncation=True, padding=True).to(device)

            # Forward pass to collect statistics
            _ = model(input_ids=inputs['input_ids'])

    # Put model back in eval mode
    model.eval()

    # Check calibration status
    calibrated_count = 0
    uncalibrated_count = 0

    for name, module in model.named_modules():
        if isinstance(module, LearnableFakeQuantize):
            if module.calibrated:
                calibrated_count += 1
            else:
                uncalibrated_count += 1
                print(f"  Warning: {name} is still uncalibrated")

    print(f"Calibration complete:")
    print(f"  Calibrated: {calibrated_count}")
    print(f"  Uncalibrated: {uncalibrated_count}")

    return model


def test_8bit_with_calibration():
    """Test 8-bit QAT model with proper calibration."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create config
    config = HFConfig()
    config.n_positions = 256
    config.n_layer = 12
    config.lora_rank = 8
    config.lora_alpha = 16
    config.lora_dropout = 0.0
    config.kv_cache_bits = 8

    # Create and load model
    print("Creating 8-bit QAT model...")
    model = QATGPT2(config, quantization_bits=8, initialize_weights=False)
    load_pretrained_weights(model)

    # Zero LoRA weights to isolate quantization effects
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'lora_b' in name.lower():
                param.data.zero_()

    model = model.to(device)

    # Create tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Test BEFORE calibration
    print("\n" + "="*60)
    print("BEFORE CALIBRATION")
    print("="*60)

    model.eval()
    test_text = "The capital of France is"
    inputs = tokenizer(test_text, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
        loss_before = outputs['loss'].item()
        perplexity_before = math.exp(loss_before) if loss_before < 20 else float('inf')
        print(f"Loss: {loss_before:.4f}")
        print(f"Perplexity: {perplexity_before:.1f}")

    # CALIBRATE the model
    print("\n" + "="*60)
    print("CALIBRATING MODEL")
    print("="*60)

    model = calibrate_quantizers(model, tokenizer, device, num_samples=50)

    # Test AFTER calibration
    print("\n" + "="*60)
    print("AFTER CALIBRATION")
    print("="*60)

    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
        loss_after = outputs['loss'].item()
        perplexity_after = math.exp(loss_after) if loss_after < 20 else float('inf')
        print(f"Loss: {loss_after:.4f}")
        print(f"Perplexity: {perplexity_after:.1f}")

    print(f"\nImprovement after calibration:")
    print(f"  Loss reduction: {loss_before - loss_after:.4f}")
    if perplexity_before != float('inf') and perplexity_after != float('inf'):
        print(f"  Perplexity improvement: {perplexity_before/perplexity_after:.2f}x")

    # Compare with reference
    print("\n" + "="*60)
    print("COMPARISON WITH REFERENCE GPT-2")
    print("="*60)

    ref_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()

    test_texts = [
        "The capital of France is",
        "Machine learning is",
        "Once upon a time",
    ]

    for text in test_texts:
        inputs = tokenizer(text, return_tensors='pt').to(device)

        with torch.no_grad():
            # Calibrated 8-bit QAT
            qat_out = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
            qat_loss = qat_out['loss'].item()
            qat_ppl = math.exp(qat_loss) if qat_loss < 20 else float('inf')

            # Reference
            ref_out = ref_model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
            ref_loss = ref_out.loss.item()
            ref_ppl = math.exp(ref_loss) if ref_loss < 20 else float('inf')

            print(f"\n'{text[:25]}...':")
            print(f"  8-bit QAT (calibrated): Loss={qat_loss:.3f}, PPL={qat_ppl:.1f}")
            print(f"  Reference GPT-2:        Loss={ref_loss:.3f}, PPL={ref_ppl:.1f}")
            if ref_ppl > 0:
                print(f"  Degradation:            {(qat_ppl/ref_ppl):.2f}x")


if __name__ == "__main__":
    test_8bit_with_calibration()
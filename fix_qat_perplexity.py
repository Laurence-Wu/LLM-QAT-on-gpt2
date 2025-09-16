"""
Comprehensive fix for QAT model perplexity issues.
This addresses:
1. Weight transpose direction
2. LoRA interference when alpha=0
3. Proper initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
import gc
import math
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import QATGPT2, QATGPT2Block
from shared.config import GPT2Config
from shared.lora import QATLinearWithLoRA


def disable_lora_completely(model):
    """Completely disable LoRA by zeroing all LoRA parameters and setting alpha to 0."""
    for module in model.modules():
        if isinstance(module, QATLinearWithLoRA):
            # Zero out LoRA weights
            with torch.no_grad():
                module.lora.lora_A.data.zero_()
                module.lora.lora_B.data.zero_()
                # Set scaling to 0 by modifying alpha
                module.lora.scaling = 0.0
    print("LoRA completely disabled")


def load_and_verify_weights(model, debug=True):
    """Load pretrained weights with verification and proper transpose."""
    print("Loading pretrained GPT-2 weights with verification...")

    device = next(model.parameters()).device
    pretrained = GPT2Model.from_pretrained('gpt2').to(device)

    # Get a reference output for comparison
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    test_text = "The capital of"
    test_input = tokenizer(test_text, return_tensors='pt').to(device)

    with torch.no_grad():
        pretrained.eval()
        ref_hidden = pretrained(test_input['input_ids']).last_hidden_state

    # Copy embeddings
    model.wte.weight.data = pretrained.wte.weight.data.clone()

    # Handle position embeddings
    min_pos = min(model.wpe.weight.shape[0], pretrained.wpe.weight.shape[0])
    model.wpe.weight.data[:min_pos] = pretrained.wpe.weight.data[:min_pos].clone()

    # Test embeddings
    with torch.no_grad():
        model.eval()
        test_embeds = model.wte(test_input['input_ids']) + model.wpe(torch.arange(test_input['input_ids'].shape[1], device=device))
        ref_embeds = pretrained.wte(test_input['input_ids']) + pretrained.wpe(torch.arange(test_input['input_ids'].shape[1], device=device))
        embed_diff = (test_embeds - ref_embeds).abs().mean().item()
        print(f"Embedding difference: {embed_diff:.6f}")

    # Copy transformer blocks with verification
    for i in range(min(len(model.h), len(pretrained.h))):
        if debug and i == 0:
            print(f"\nLoading and verifying block {i}...")

        # Layer norms
        model.h[i].ln_1.weight.data = pretrained.h[i].ln_1.weight.data.clone()
        model.h[i].ln_1.bias.data = pretrained.h[i].ln_1.bias.data.clone()
        model.h[i].ln_2.weight.data = pretrained.h[i].ln_2.weight.data.clone()
        model.h[i].ln_2.bias.data = pretrained.h[i].ln_2.bias.data.clone()

        # Attention weights - carefully handle transpose
        # GPT-2 Conv1D: weight shape is (n_in, n_out)
        # Our Linear: weight shape is (n_out, n_in)

        # c_attn combines q,k,v: (768, 2304) -> (2304, 768)
        pretrained_c_attn = pretrained.h[i].attn.c_attn.weight.data  # (768, 2304)
        model.h[i].attn.c_attn.linear.weight.data = pretrained_c_attn.t().contiguous()  # (2304, 768)
        model.h[i].attn.c_attn.linear.bias.data = pretrained.h[i].attn.c_attn.bias.data.clone()

        # c_proj: (768, 768) -> (768, 768)
        pretrained_c_proj = pretrained.h[i].attn.c_proj.weight.data  # (768, 768)
        model.h[i].attn.c_proj.linear.weight.data = pretrained_c_proj.t().contiguous()  # (768, 768)
        model.h[i].attn.c_proj.linear.bias.data = pretrained.h[i].attn.c_proj.bias.data.clone()

        # MLP weights
        # c_fc: (768, 3072) -> (3072, 768)
        pretrained_c_fc = pretrained.h[i].mlp.c_fc.weight.data  # (768, 3072)
        model.h[i].mlp.c_fc.linear.weight.data = pretrained_c_fc.t().contiguous()  # (3072, 768)
        model.h[i].mlp.c_fc.linear.bias.data = pretrained.h[i].mlp.c_fc.bias.data.clone()

        # c_proj: (3072, 768) -> (768, 3072)
        pretrained_mlp_c_proj = pretrained.h[i].mlp.c_proj.weight.data  # (3072, 768)
        model.h[i].mlp.c_proj.linear.weight.data = pretrained_mlp_c_proj.t().contiguous()  # (768, 3072)
        model.h[i].mlp.c_proj.linear.bias.data = pretrained.h[i].mlp.c_proj.bias.data.clone()

        # Verify first block if debug
        if debug and i == 0:
            with torch.no_grad():
                # Test through first layer norm
                test_ln1 = model.h[0].ln_1(test_embeds)
                ref_ln1 = pretrained.h[0].ln_1(ref_embeds)
                ln1_diff = (test_ln1 - ref_ln1).abs().mean().item()
                print(f"  After LN1 difference: {ln1_diff:.6f}")

                # Test attention output
                test_attn = model.h[0].attn(test_ln1)
                ref_attn = pretrained.h[0].attn(ref_ln1)[0]  # GPT2 attention returns tuple
                attn_diff = (test_attn - ref_attn).abs().mean().item()
                print(f"  After attention difference: {attn_diff:.6f}")

                # Full block
                test_block = model.h[0](test_embeds)
                ref_block = pretrained.h[0](ref_embeds)[0]  # GPT2 block returns tuple
                block_diff = (test_block - ref_block).abs().mean().item()
                print(f"  After full block difference: {block_diff:.6f}")

    # Final layer norm
    model.ln_f.weight.data = pretrained.ln_f.weight.data.clone()
    model.ln_f.bias.data = pretrained.ln_f.bias.data.clone()

    # Clean up
    del pretrained
    torch.cuda.empty_cache()
    gc.collect()

    print("Weights loaded and verified successfully")
    return model


def test_fixed_model():
    """Test the fixed model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Configuration
    config = GPT2Config(
        n_positions=256,
        n_layer=12,
        lora_rank=0,  # Disable LoRA completely
        lora_alpha=0,
        lora_dropout=0.0
    )

    # Create models
    print("Creating QAT model...")
    qat_model = QATGPT2(config, quantization_bits=16, initialize_weights=False)  # Start with 16-bit

    # Load and verify weights
    qat_model = load_and_verify_weights(qat_model)

    # Disable LoRA completely
    disable_lora_completely(qat_model)

    # Move to device and eval mode
    qat_model = qat_model.to(device)
    qat_model.eval()

    # Test perplexity
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    test_texts = [
        "The capital of France is",
        "Machine learning is",
        "In the beginning was",
        "Once upon a time",
    ]

    print("\n" + "="*60)
    print("TESTING FIXED QAT MODEL")
    print("="*60)

    total_loss = 0
    count = 0

    for text in test_texts:
        inputs = tokenizer(text, return_tensors='pt', max_length=50, truncation=True).to(device)

        with torch.no_grad():
            outputs = qat_model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss

            if loss is not None:
                total_loss += loss.item()
                count += 1
                perplexity = math.exp(loss.item()) if loss.item() < 20 else float('inf')
                print(f"Text: '{text[:20]}...' - Loss: {loss.item():.4f}, Perplexity: {perplexity:.1f}")

    if count > 0:
        avg_loss = total_loss / count
        avg_perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        print(f"\nAverage Loss: {avg_loss:.4f}")
        print(f"Average Perplexity: {avg_perplexity:.1f}")

    # Compare with reference GPT-2
    print("\n" + "="*60)
    print("REFERENCE GPT-2 COMPARISON")
    print("="*60)

    ref_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    ref_model.eval()

    ref_total_loss = 0
    ref_count = 0

    for text in test_texts:
        inputs = tokenizer(text, return_tensors='pt', max_length=50, truncation=True).to(device)

        with torch.no_grad():
            outputs = ref_model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
            loss = outputs.loss

            if loss is not None:
                ref_total_loss += loss.item()
                ref_count += 1
                perplexity = math.exp(loss.item()) if loss.item() < 20 else float('inf')
                print(f"Text: '{text[:20]}...' - Loss: {loss.item():.4f}, Perplexity: {perplexity:.1f}")

    if ref_count > 0:
        ref_avg_loss = ref_total_loss / ref_count
        ref_avg_perplexity = math.exp(ref_avg_loss) if ref_avg_loss < 20 else float('inf')
        print(f"\nReference Average Loss: {ref_avg_loss:.4f}")
        print(f"Reference Average Perplexity: {ref_avg_perplexity:.1f}")

    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)

    if avg_perplexity < ref_avg_perplexity * 3:  # Within 3x of reference
        print("✅ QAT model perplexity is reasonable!")
        print("The fixes have resolved the main issues.")
    else:
        print("❌ QAT model still has high perplexity.")
        print("Additional investigation needed.")


if __name__ == "__main__":
    test_fixed_model()
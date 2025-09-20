#!/usr/bin/env python3
"""
Test script to verify proper model initialization and weight loading
Prints detailed parameter information to verify all weights are loaded correctly
"""

import sys
import os
import torch
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from part1_switchable_precision.config_sp import ModelConfig
from shared.models_sp import SPLMHeadModel
from transformers import GPT2Config


def count_parameters_by_type(model, model_name="Model"):
    """Count and categorize parameters in the model."""
    param_info = defaultdict(lambda: {'count': 0, 'params': 0, 'names': []})

    for name, param in model.named_parameters():
        param_count = param.numel()

        # Categorize by type
        if 'wte' in name:
            category = 'Token Embeddings'
        elif 'wpe' in name:
            category = 'Position Embeddings'
        elif 'ln_f' in name:
            category = 'Final LayerNorm'
        elif 'lm_head' in name:
            category = 'LM Head'
        elif 'ln_1' in name:
            category = 'LayerNorm 1'
        elif 'ln_2' in name:
            category = 'LayerNorm 2'
        elif 'c_attn' in name:
            category = 'Attention QKV'
        elif 'c_proj' in name and 'attn' in name:
            category = 'Attention Projection'
        elif 'c_fc' in name:
            category = 'MLP FC'
        elif 'c_proj' in name and 'mlp' in name:
            category = 'MLP Projection'
        elif 'lora' in name.lower():
            category = 'LoRA Adapters'
        elif 'quantizer' in name.lower():
            category = 'Quantizers'
        else:
            category = 'Other'

        param_info[category]['count'] += 1
        param_info[category]['params'] += param_count
        param_info[category]['names'].append(name)

    print(f"\n{'='*60}")
    print(f"Parameter Analysis for {model_name}")
    print(f"{'='*60}")

    total_params = 0
    for category in sorted(param_info.keys()):
        info = param_info[category]
        print(f"\n{category}:")
        print(f"  Count: {info['count']}")
        print(f"  Parameters: {info['params']:,}")
        if info['count'] <= 5:  # Show names if few parameters
            for name in info['names']:
                print(f"    - {name}")
        total_params += info['params']

    print(f"\n{'='*60}")
    print(f"TOTAL PARAMETERS: {total_params:,}")
    print(f"{'='*60}")

    return param_info


def verify_weight_matching(sp_model, gpt2_model):
    """Verify that SP model weights match GPT-2 weights."""
    print("\n" + "="*60)
    print("Weight Matching Verification")
    print("="*60)

    mismatches = []
    matches = []

    # Check embeddings
    sp_wte = sp_model.transformer.wte.weight
    gpt2_wte = gpt2_model.transformer.wte.weight
    diff = (sp_wte - gpt2_wte).abs().max().item()
    if diff < 1e-5:
        matches.append(f"✅ Token embeddings match (diff={diff:.2e})")
    else:
        mismatches.append(f"❌ Token embeddings mismatch (diff={diff:.2e})")

    sp_wpe = sp_model.transformer.wpe.weight
    gpt2_wpe = gpt2_model.transformer.wpe.weight
    diff = (sp_wpe - gpt2_wpe).abs().max().item()
    if diff < 1e-5:
        matches.append(f"✅ Position embeddings match (diff={diff:.2e})")
    else:
        mismatches.append(f"❌ Position embeddings mismatch (diff={diff:.2e})")

    # Check layer norms
    for i in range(min(len(sp_model.transformer.h), len(gpt2_model.transformer.h))):
        # ln_1
        sp_ln1_w = sp_model.transformer.h[i].ln_1.weight
        gpt2_ln1_w = gpt2_model.transformer.h[i].ln_1.weight
        diff = (sp_ln1_w - gpt2_ln1_w).abs().max().item()
        if diff < 1e-5:
            if i == 0:  # Only print for first layer
                matches.append(f"✅ Layer {i} ln_1 weight matches (diff={diff:.2e})")
        else:
            mismatches.append(f"❌ Layer {i} ln_1 weight mismatch (diff={diff:.2e})")

        # Attention weights (remember SP model transposes them)
        sp_attn = sp_model.transformer.h[i].attn.c_attn.linear.weight
        gpt2_attn = gpt2_model.transformer.h[i].attn.c_attn.weight
        # SP stores transposed, so compare with transpose
        diff = (sp_attn.t() - gpt2_attn).abs().max().item()
        if diff < 1e-5:
            if i == 0:  # Only print for first layer
                matches.append(f"✅ Layer {i} attention QKV matches (diff={diff:.2e})")
        else:
            mismatches.append(f"❌ Layer {i} attention QKV mismatch (diff={diff:.2e})")

    # Final layer norm
    sp_ln_f = sp_model.transformer.ln_f.weight
    gpt2_ln_f = gpt2_model.transformer.ln_f.weight
    diff = (sp_ln_f - gpt2_ln_f).abs().max().item()
    if diff < 1e-5:
        matches.append(f"✅ Final layer norm matches (diff={diff:.2e})")
    else:
        mismatches.append(f"❌ Final layer norm mismatch (diff={diff:.2e})")

    # LM head (should be tied to embeddings)
    if sp_model.lm_head.weight.data_ptr() == sp_model.transformer.wte.weight.data_ptr():
        matches.append(f"✅ LM head properly tied to token embeddings")
    else:
        mismatches.append(f"❌ LM head NOT tied to token embeddings!")

    # Print results
    print("\nMatches:")
    for match in matches:
        print(f"  {match}")

    if mismatches:
        print("\nMismatches:")
        for mismatch in mismatches:
            print(f"  {mismatch}")

    return len(mismatches) == 0


def test_forward_pass(sp_model, gpt2_model, tokenizer):
    """Test forward pass and compare outputs."""
    print("\n" + "="*60)
    print("Forward Pass Test")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sp_model = sp_model.to(device)
    gpt2_model = gpt2_model.to(device)

    sp_model.eval()
    gpt2_model.eval()

    test_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(test_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)

    with torch.no_grad():
        # SP model in 32-bit mode
        sp_model.set_precision(32)
        sp_outputs = sp_model(input_ids)
        sp_logits = sp_outputs['logits'] if isinstance(sp_outputs, dict) else sp_outputs

        # GPT-2 model
        gpt2_outputs = gpt2_model(input_ids)
        gpt2_logits = gpt2_outputs.logits

        # Compare logits
        diff = (sp_logits - gpt2_logits).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"\nTest text: '{test_text}'")
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {sp_logits.shape}")
        print(f"\nLogits comparison:")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")

        if max_diff < 1e-4:
            print(f"  ✅ Outputs match closely!")
        elif max_diff < 1e-2:
            print(f"  ⚠️ Small differences detected")
        else:
            print(f"  ❌ Significant differences!")

        # Test with labels to get loss
        sp_outputs = sp_model(input_ids, labels=input_ids)
        gpt2_outputs = gpt2_model(input_ids, labels=input_ids)

        sp_loss = sp_outputs['loss'].item()
        gpt2_loss = gpt2_outputs.loss.item()

        print(f"\nLoss comparison:")
        print(f"  SP model loss: {sp_loss:.4f}")
        print(f"  GPT-2 loss: {gpt2_loss:.4f}")
        print(f"  Difference: {abs(sp_loss - gpt2_loss):.4f}")

        sp_ppl = np.exp(sp_loss)
        gpt2_ppl = np.exp(gpt2_loss)

        print(f"\nPerplexity:")
        print(f"  SP model: {sp_ppl:.2f}")
        print(f"  GPT-2: {gpt2_ppl:.2f}")

        return abs(sp_loss - gpt2_loss) < 0.01


def main():
    print("="*80)
    print("MODEL INITIALIZATION VERIFICATION TEST")
    print("="*80)

    # Load configuration
    model_config = ModelConfig()
    print(f"\nConfiguration:")
    print(f"  Layers: {model_config.n_layer}")
    print(f"  Embedding dim: {model_config.n_embd}")
    print(f"  Attention heads: {model_config.n_head}")
    print(f"  Vocab size: {model_config.vocab_size}")
    print(f"  Positions: {model_config.n_positions}")
    print(f"  Bit widths: {model_config.bit_widths}")

    # Create SP model
    print("\n" + "="*60)
    print("Creating SP Model")
    print("="*60)

    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        layer_norm_epsilon=model_config.layer_norm_epsilon,
        embd_pdrop=model_config.embd_pdrop
    )

    # Add SP-specific configs
    gpt2_config.bit_widths = model_config.bit_widths
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    gpt2_config.activation_bits_per_bit = model_config.activation_bits_per_bit

    sp_model = SPLMHeadModel(gpt2_config)

    # Load pretrained weights using the method from main_sp.py
    print("\nLoading pretrained weights into SP model...")
    from part1_switchable_precision.main_sp import load_pretrained_weights
    load_pretrained_weights(sp_model)

    # Count parameters
    sp_param_info = count_parameters_by_type(sp_model, "SP Model (After Loading)")

    # Load standard GPT-2 for comparison
    print("\n" + "="*60)
    print("Loading Standard GPT-2 for Comparison")
    print("="*60)
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_param_info = count_parameters_by_type(gpt2_model, "Standard GPT-2")

    # Compare parameter counts
    print("\n" + "="*60)
    print("Parameter Count Comparison")
    print("="*60)

    # Expected counts for GPT-2 small
    expected = {
        'embeddings': 2,  # wte, wpe
        'layer_norms': 25,  # 12*2 + 1
        'attention_weights': 24,  # 12 * 2 (QKV and proj)
        'mlp_weights': 24,  # 12 * 2 (fc and proj)
        'biases': 73  # All the biases
    }

    print(f"\nExpected for 12-layer GPT-2:")
    print(f"  Embeddings: {expected['embeddings']}")
    print(f"  Layer norms: {expected['layer_norms']}")
    print(f"  Attention weights: {expected['attention_weights']}")
    print(f"  MLP weights: {expected['mlp_weights']}")
    print(f"  Total (approx): ~148 parameters")

    # Verify weight matching
    weights_match = verify_weight_matching(sp_model, gpt2_model)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Test forward pass
    forward_pass_ok = test_forward_pass(sp_model, gpt2_model, tokenizer)

    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if weights_match and forward_pass_ok:
        print("✅ ALL TESTS PASSED!")
        print("  - Weights properly loaded")
        print("  - Forward pass matches GPT-2")
        print("  - Model ready for training")
    else:
        print("⚠️ SOME ISSUES DETECTED")
        if not weights_match:
            print("  - Weight loading issues")
        if not forward_pass_ok:
            print("  - Forward pass mismatch")

    # Test different bit precisions
    print("\n" + "="*60)
    print("Testing Different Bit Precisions")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sp_model = sp_model.to(device)
    sp_model.eval()

    test_text = "Machine learning is transforming technology."
    inputs = tokenizer(test_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)

    for bits in model_config.bit_widths:
        sp_model.set_precision(bits)

        with torch.no_grad():
            outputs = sp_model(input_ids, labels=input_ids)
            loss = outputs['loss'].item()
            ppl = np.exp(loss)

            # Check LoRA status
            lora_enabled = False
            for name, module in sp_model.named_modules():
                if hasattr(module, 'lora_adapters'):
                    bit_key = f'{bits}bit'
                    if bit_key in module.lora_adapters:
                        lora = module.lora_adapters[bit_key]
                        if hasattr(lora, 'enabled') and lora.enabled:
                            lora_enabled = True
                            break

            lora_status = "Disabled (Teacher)" if bits == 32 else ("Enabled" if lora_enabled else "ERROR: Should be enabled!")

            print(f"\n{bits}-bit mode:")
            print(f"  Loss: {loss:.4f}")
            print(f"  Perplexity: {ppl:.2f}")
            print(f"  LoRA: {lora_status}")


if __name__ == "__main__":
    main()
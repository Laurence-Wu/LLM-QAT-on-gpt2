"""
Fix weight loading for QAT models to properly handle transpose and initialization.
"""

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
import gc

def load_pretrained_weights_fixed(model, debug=False):
    """Load pretrained GPT-2 weights with proper transpose and verification."""
    print("Loading pretrained GPT-2 weights with fixed transpose...")
    pretrained = GPT2Model.from_pretrained('gpt2')
    device = next(model.parameters()).device

    # Move pretrained model to same device as target model
    pretrained = pretrained.to(device)

    # Copy embeddings
    model.wte.weight.data = pretrained.wte.weight.data.clone()

    # Handle position embeddings size mismatch
    min_positions = min(model.wpe.weight.shape[0], pretrained.wpe.weight.shape[0])
    model.wpe.weight.data[:min_positions] = pretrained.wpe.weight.data[:min_positions].clone()
    if model.wpe.weight.shape[0] != pretrained.wpe.weight.shape[0]:
        print(f"Adjusted position embeddings from {pretrained.wpe.weight.shape[0]} to {model.wpe.weight.shape[0]}")

    # Copy transformer blocks
    for i in range(min(len(model.h), len(pretrained.h))):
        if debug:
            print(f"Loading block {i}...")

        # Layer normalizations
        model.h[i].ln_1.weight.data = pretrained.h[i].ln_1.weight.data.clone()
        model.h[i].ln_1.bias.data = pretrained.h[i].ln_1.bias.data.clone()
        model.h[i].ln_2.weight.data = pretrained.h[i].ln_2.weight.data.clone()
        model.h[i].ln_2.bias.data = pretrained.h[i].ln_2.bias.data.clone()

        # Attention weights - GPT2 Conv1D has shape (n_in, n_out), Linear has (n_out, n_in)
        # c_attn: combines q, k, v projections
        pretrained_weight = pretrained.h[i].attn.c_attn.weight.data  # Shape: (768, 2304)
        model_weight_shape = model.h[i].attn.c_attn.linear.weight.shape  # Should be (2304, 768)

        if debug:
            print(f"  c_attn - Pretrained shape: {pretrained_weight.shape}, Model shape: {model_weight_shape}")

        # Transpose is needed: Conv1D (768, 2304) -> Linear (2304, 768)
        if pretrained_weight.shape[0] == model_weight_shape[1] and pretrained_weight.shape[1] == model_weight_shape[0]:
            model.h[i].attn.c_attn.linear.weight.data = pretrained_weight.t().contiguous()
        else:
            print(f"  WARNING: Shape mismatch in c_attn at block {i}")

        model.h[i].attn.c_attn.linear.bias.data = pretrained.h[i].attn.c_attn.bias.data.clone()

        # c_proj: attention output projection
        pretrained_weight = pretrained.h[i].attn.c_proj.weight.data  # Shape: (768, 768)
        model_weight_shape = model.h[i].attn.c_proj.linear.weight.shape  # Should be (768, 768)

        if debug:
            print(f"  c_proj - Pretrained shape: {pretrained_weight.shape}, Model shape: {model_weight_shape}")

        # Transpose is needed
        if pretrained_weight.shape[0] == model_weight_shape[1] and pretrained_weight.shape[1] == model_weight_shape[0]:
            model.h[i].attn.c_proj.linear.weight.data = pretrained_weight.t().contiguous()
        else:
            print(f"  WARNING: Shape mismatch in c_proj at block {i}")

        model.h[i].attn.c_proj.linear.bias.data = pretrained.h[i].attn.c_proj.bias.data.clone()

        # MLP weights
        # c_fc: feedforward projection to higher dimension
        pretrained_weight = pretrained.h[i].mlp.c_fc.weight.data  # Shape: (768, 3072)
        model_weight_shape = model.h[i].mlp.c_fc.linear.weight.shape  # Should be (3072, 768)

        if debug:
            print(f"  c_fc - Pretrained shape: {pretrained_weight.shape}, Model shape: {model_weight_shape}")

        if pretrained_weight.shape[0] == model_weight_shape[1] and pretrained_weight.shape[1] == model_weight_shape[0]:
            model.h[i].mlp.c_fc.linear.weight.data = pretrained_weight.t().contiguous()
        else:
            print(f"  WARNING: Shape mismatch in c_fc at block {i}")

        model.h[i].mlp.c_fc.linear.bias.data = pretrained.h[i].mlp.c_fc.bias.data.clone()

        # c_proj: feedforward projection from higher dimension
        pretrained_weight = pretrained.h[i].mlp.c_proj.weight.data  # Shape: (3072, 768)
        model_weight_shape = model.h[i].mlp.c_proj.linear.weight.shape  # Should be (768, 3072)

        if debug:
            print(f"  c_proj - Pretrained shape: {pretrained_weight.shape}, Model shape: {model_weight_shape}")

        if pretrained_weight.shape[0] == model_weight_shape[1] and pretrained_weight.shape[1] == model_weight_shape[0]:
            model.h[i].mlp.c_proj.linear.weight.data = pretrained_weight.t().contiguous()
        else:
            print(f"  WARNING: Shape mismatch in mlp.c_proj at block {i}")

        model.h[i].mlp.c_proj.linear.bias.data = pretrained.h[i].mlp.c_proj.bias.data.clone()

    # Final layer normalization
    model.ln_f.weight.data = pretrained.ln_f.weight.data.clone()
    model.ln_f.bias.data = pretrained.ln_f.bias.data.clone()

    # Initialize LoRA parameters to zero if they exist
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and ('lora_a' in name or 'lora_b' in name):
            if debug:
                print(f"Zeroing LoRA parameter: {name}")
            if 'lora_b' in name:
                # lora_b should be initialized to zero for no initial effect
                param.data.zero_()

    # Delete pretrained model to free memory
    del pretrained
    torch.cuda.empty_cache()
    gc.collect()

    print("Pretrained weights loaded successfully with proper transpose.")


def test_weight_loading():
    """Test the weight loading with a simple forward pass."""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from transformers import GPT2Tokenizer
    from shared.models import QATGPT2
    from transformers import GPT2Config
    import math

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create config
    config = GPT2Config()
    config.n_positions = 256
    config.n_layer = 12
    config.lora_rank = 8
    config.lora_alpha = 16
    config.lora_dropout = 0.1

    # Create model
    model = QATGPT2(config, quantization_bits=8, initialize_weights=False)

    # Load weights with fix
    load_pretrained_weights_fixed(model, debug=True)

    # Move to device
    model = model.to(device)
    model.eval()

    # Test
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    test_text = "The capital of France is"
    inputs = tokenizer(test_text, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss

        if loss is not None:
            perplexity = math.exp(loss.item()) if loss.item() < 20 else float('inf')
            print(f"\nTest Loss: {loss.item():.4f}")
            print(f"Test Perplexity: {perplexity:.1f}")

            if perplexity < 200:
                print("✅ Weight loading appears to be working!")
            else:
                print("❌ Still high perplexity - additional issues to investigate")

    # Compare with reference GPT-2
    print("\nComparing with reference GPT-2...")
    from transformers import GPT2LMHeadModel
    ref_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    ref_model.eval()

    with torch.no_grad():
        ref_outputs = ref_model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
        ref_loss = ref_outputs.loss
        ref_perplexity = math.exp(ref_loss.item()) if ref_loss.item() < 20 else float('inf')
        print(f"Reference Loss: {ref_loss.item():.4f}")
        print(f"Reference Perplexity: {ref_perplexity:.1f}")


if __name__ == "__main__":
    test_weight_loading()
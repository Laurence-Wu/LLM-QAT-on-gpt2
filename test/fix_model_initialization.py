#!/usr/bin/env python3
"""
Model initialization helper for testing
Creates properly initialized SP models with weight loading from GPT-2
"""

import sys
import os
import torch
from transformers import GPT2Config

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models_sp import SPLMHeadModel
from part1_switchable_precision.config_sp import SPConfig


def create_properly_initialized_model(use_pretrained=True, num_layers=None):
    """
    Create and initialize an SP model with proper weight loading.

    Args:
        use_pretrained: If True, load weights from GPT-2
        num_layers: Optional number of layers (for reduced models)

    Returns:
        model: Initialized SPLMHeadModel
        config: SPConfig object
    """

    # Create SP config
    sp_config = SPConfig()

    # Override number of layers if specified (for memory-limited testing)
    if num_layers is not None:
        sp_config.n_layer = num_layers

    # Create model
    sp_model = SPLMHeadModel(sp_config)

    # Load pretrained weights if requested
    if use_pretrained:
        print("Loading weights from pretrained GPT-2...")

        # Load GPT-2 state dict
        from transformers import GPT2LMHeadModel

        # Create a GPT-2 config matching our SP config
        gpt2_config = GPT2Config(
            vocab_size=sp_config.vocab_size,
            n_positions=sp_config.n_positions,
            n_ctx=sp_config.n_ctx,
            n_embd=sp_config.n_embd,
            n_layer=sp_config.n_layer,
            n_head=sp_config.n_head,
            n_inner=sp_config.n_inner,
            activation_function=sp_config.activation_function,
            resid_pdrop=sp_config.resid_pdrop,
            embd_pdrop=sp_config.embd_pdrop,
            attn_pdrop=sp_config.attn_pdrop,
            layer_norm_epsilon=sp_config.layer_norm_epsilon,
        )

        # Load GPT-2 model
        if sp_config.n_layer == 12:  # Full model
            gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        else:
            # For reduced layer models, load full model and copy only needed layers
            gpt2_full = GPT2LMHeadModel.from_pretrained('gpt2')
            gpt2_model = GPT2LMHeadModel(gpt2_config)

            # Copy embeddings and head
            gpt2_model.transformer.wte = gpt2_full.transformer.wte
            gpt2_model.transformer.wpe = gpt2_full.transformer.wpe
            gpt2_model.transformer.ln_f = gpt2_full.transformer.ln_f
            gpt2_model.lm_head = gpt2_full.lm_head

            # Copy only the needed layers
            for i in range(sp_config.n_layer):
                gpt2_model.transformer.h[i] = gpt2_full.transformer.h[i]

            del gpt2_full  # Free memory

        # Load the state dict into SP model
        # This will handle the mapping from GPT2 to SP architecture
        sp_model.load_pretrained_weights(gpt2_model.state_dict())

        print("✅ Weights loaded successfully")

        # Clean up
        del gpt2_model
        torch.cuda.empty_cache()

    # Set default precision to 16-bit (no quantization)
    sp_model.set_precision(16)

    return sp_model, sp_config


def verify_weight_loading(sp_model, device='cpu'):
    """
    Quick verification that weights were loaded correctly.

    Args:
        sp_model: The SPLMHeadModel to verify
        device: Device to run tests on

    Returns:
        bool: True if weights appear to be loaded correctly
    """
    sp_model = sp_model.to(device)
    sp_model.eval()

    # Check that embeddings are not all zeros
    wte_weight = sp_model.transformer.wte.weight
    if torch.all(wte_weight == 0):
        print("❌ Token embeddings are all zeros!")
        return False

    # Check that at least one attention layer has non-zero weights
    first_attn = sp_model.transformer.h[0].attn.c_attn
    if hasattr(first_attn, 'linear'):
        weight = first_attn.linear.weight
    else:
        weight = first_attn.weight

    if torch.all(weight == 0):
        print("❌ Attention weights are all zeros!")
        return False

    print("✅ Basic weight verification passed")
    return True


if __name__ == "__main__":
    # Test the initialization
    print("Testing model initialization...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model with pretrained weights
    model, config = create_properly_initialized_model(use_pretrained=True)

    # Verify weights
    if verify_weight_loading(model, device):
        print("✅ Model initialized successfully with pretrained weights")
    else:
        print("❌ Model initialization failed")
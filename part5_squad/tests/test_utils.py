"""
Shared test utilities for part5_squad tests

Provides helpers that match production behavior in main_squad.py
"""

import torch


def freeze_weights_like_production(model):
    """
    Freeze weights matching production setup in main_squad.py

    In production:
    - Embeddings are frozen
    - All LayerNorm weights/biases are frozen for all precisions
    - All attention and MLP linear weights/biases in base model are frozen
    - ONLY trainable: LoRA adapters and QA heads

    This matches the behavior of load_pretrained_weights() in main_squad.py (lines 78-142)

    Args:
        model: SPQuestionAnsweringModel instance
    """
    # Freeze embeddings (main_squad.py lines 80-82)
    model.transformer.wte.weight.requires_grad = False
    model.transformer.wpe.weight.requires_grad = False

    # Freeze all LayerNorm and base transformer weights
    for i, block in enumerate(model.transformer.h):
        # Freeze LayerNorm weights for all precisions (lines 87-97)
        if hasattr(block.ln_1, 'precision_levels'):
            for precision in block.ln_1.precision_levels:
                block.ln_1.weights[str(precision)].requires_grad = False
                block.ln_1.biases[str(precision)].requires_grad = False

        if hasattr(block.ln_2, 'precision_levels'):
            for precision in block.ln_2.precision_levels:
                block.ln_2.weights[str(precision)].requires_grad = False
                block.ln_2.biases[str(precision)].requires_grad = False

        # Freeze attention weights (lines 100-108)
        block.attn.c_attn.linear.weight.requires_grad = False
        block.attn.c_attn.linear.bias.requires_grad = False
        block.attn.c_proj.linear.weight.requires_grad = False
        block.attn.c_proj.linear.bias.requires_grad = False

        # Freeze MLP weights (lines 111-119)
        block.mlp.c_fc.linear.weight.requires_grad = False
        block.mlp.c_fc.linear.bias.requires_grad = False
        block.mlp.c_proj.linear.weight.requires_grad = False
        block.mlp.c_proj.linear.bias.requires_grad = False

    # Freeze final LayerNorm (lines 122-126)
    if hasattr(model.transformer.ln_f, 'precision_levels'):
        for precision in model.transformer.ln_f.precision_levels:
            model.transformer.ln_f.weights[str(precision)].requires_grad = False
            model.transformer.ln_f.biases[str(precision)].requires_grad = False

    # Enable LoRA adapters (main_squad.py lines 138-142)
    # Count for verification
    lora_count = 0
    for name, module in model.named_modules():
        if not hasattr(module, 'lora_adapters'):
            continue
        for bit_key in module.lora_adapters.keys():
            lora_layer = module.lora_adapters[bit_key]
            if hasattr(lora_layer, 'lora_A'):
                lora_layer.lora_A.requires_grad = True
                lora_layer.lora_B.requires_grad = True
                lora_count += 1

    # QA heads remain trainable (they're initialized randomly, not from pretrained)
    # No need to explicitly set requires_grad=True as they're trainable by default

    return lora_count


def get_trainable_param_count(model):
    """
    Get count of trainable vs frozen parameters

    Returns:
        (trainable_count, frozen_count, total_count)
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + frozen_params

    return trainable_params, frozen_params, total_params


def create_mock_squad_batch(batch_size=2, seq_length=128, device='cpu'):
    """
    Create a mock SQuAD batch for testing

    Args:
        batch_size: Batch size
        seq_length: Sequence length
        device: Device to create tensors on

    Returns:
        Dict with keys: input_ids, attention_mask, start_positions, end_positions
    """
    return {
        'input_ids': torch.randint(0, 50257, (batch_size, seq_length), device=device),
        'attention_mask': torch.ones(batch_size, seq_length, device=device),
        'start_positions': torch.randint(0, seq_length, (batch_size,), device=device),
        'end_positions': torch.randint(0, seq_length, (batch_size,), device=device)
    }


class MockDataLoader:
    """
    Mock DataLoader for testing CalibrationManager

    Yields the same batch multiple times
    """
    def __init__(self, batch_size=2, seq_length=128, num_batches=10, device='cpu'):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_batches = num_batches
        self.device = device

    def __iter__(self):
        for _ in range(self.num_batches):
            yield create_mock_squad_batch(self.batch_size, self.seq_length, self.device)

    def __len__(self):
        return self.num_batches

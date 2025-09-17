"""
Configuration for Quantization-Aware Training (QAT)
Simplified and unified configuration.
"""

class ModelConfig:
    """Model architecture configuration."""
    def __init__(self):
        # GPT-2 architecture
        self.vocab_size = 50257
        self.n_positions = 256
        self.n_embd = 768
        self.n_layer = 6
        self.n_head = 12

        # Regularization
        self.layer_norm_epsilon = 1e-5
        self.embd_pdrop = 0.1

        # Quantization settings
        self.quantization_bits = 8  # Default for single-precision mode
        self.activation_bits = 8  # Bits for activation quantization
        self.kv_cache_bits = 8  # Bits for KV cache quantization (default)
        self.use_gradient_checkpointing = True

        # LoRA settings (used when not in switchable mode)
        self.lora_rank = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.1

        # Switchable precision settings
        self.bit_widths = [4, 8, 16]  # Supported bit-widths

        # IMPORTANT: These dictionaries control the LoRA rank for each bit-width
        # The model will use these exact values - no hardcoded defaults
        # Lower precision uses lower rank for efficiency
        self.lora_rank_per_bit = {4: 8, 8: 16, 16: 32}  # Maps bit-width to LoRA rank
        self.lora_alpha_per_bit = {4: 16, 8: 32, 16: 64}  # Maps bit-width to LoRA alpha
        # Activation and KV cache bits per weight precision
        self.activation_bits_per_bit = {4: 4, 8: 8, 16: 16}  # Match weight precision
        self.kv_cache_bits_per_bit = {4: 4, 8: 8, 16: 16}  # Match weight precision
        self.switch_strategy = 'cyclic'  # Options: 'cyclic', 'random', 'curriculum'
        self.switch_interval = 10  # Switch every N iterations (for cyclic)
        self.curriculum_schedule = [16, 16, 8, 8, 4]  # For curriculum strategy


class TrainingConfig:
    """Training configuration for QAT."""
    def __init__(self):
        # Dataset
        self.train_split = 'train[:5000]'
        self.val_split = 'validation[:1000]'
        self.batch_size = 32
        self.max_seq_length = 256
        self.doc_stride = 128

        # Optimization
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.adam_betas = (0.9, 0.999)
        self.max_grad_norm = 1.0

        # Training schedule
        self.num_iterations = 1500
        self.gradient_accumulation_steps = 8

        # Evaluation
        self.eval_interval = 50
        self.save_interval = 100

        # Memory optimization
        self.use_amp = True
        self.empty_cache_interval = 25
        self.num_workers = 0  # For DataLoader
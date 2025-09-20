"""
Configuration for Switchable Precision Training
Simplified and unified configuration.
"""

class ModelConfig:
    """Model architecture configuration."""
    def __init__(self):
        # GPT-2 architecture - using exact GPT-2 small dimensions
        self.vocab_size = 50257
        self.n_positions = 1024  # Match GPT-2's position embeddings
        self.n_embd = 768        # GPT-2 small embedding dimension
        self.n_layer = 12        # GPT-2 small number of layers (full model)
        self.n_head = 12         # GPT-2 small number of attention heads

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

        # Lower precision uses lower rank for efficiency
        # CRITICAL: 16-bit must have rank=0 to match GPT-2 exactly
        self.lora_rank_per_bit = {4: 8, 8: 16, 16: 0}  # 16-bit has rank=0 (disabled)
        self.lora_alpha_per_bit = {4: 16, 8: 32, 16: 0}  # 16-bit has alpha=0 (disabled)
        
        # Activation and KV cache bits per weight precision
        self.activation_bits_per_bit = {4: 4, 8: 8, 16: 16}  # Match weight precision
        self.kv_cache_bits_per_bit = {4: 4, 8: 8, 16: 16}  # Match weight precision
        self.switch_strategy = 'cyclic'  # Options: 'cyclic', 'random', 'curriculum'
        self.switch_interval = 10  # Switch every N iterations (for cyclic)
        self.curriculum_schedule = [16, 16, 8, 8, 4]  # For curriculum strategy


class TrainingConfig:
    """Training configuration for Switchable Precision."""
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
        self.num_iterations = 150
        self.gradient_accumulation_steps = 8

        # Evaluation
        self.eval_interval = 50
        self.save_interval = 100

        # Memory optimization
        self.use_amp = True
        self.empty_cache_interval = 25
        self.num_workers = 0  # For DataLoader

        # Distillation parameters (following paper specifications)
        self.use_distillation = True
        self.distill_alpha_kl = 1.0  # α₁: Weight for KL divergence
        self.distill_alpha_feature = 1e-7  # α₂: Weight for feature matching
        self.distill_temperature = 3.0  # Temperature for KL softening
        self.teacher_update_interval = 10  # How often to update teacher
        self.distill_warmup_steps = 100  # Steps before starting distillation
        self.feature_layers = None  # Which layers to match (None = all)
        self.cache_size = 32  # Teacher cache size
        self.move_cache_to_cpu = False  # Move cache to CPU to save GPU memory
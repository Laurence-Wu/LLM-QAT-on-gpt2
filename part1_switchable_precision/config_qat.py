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
        self.quantization_bits = 8
        self.use_gradient_checkpointing = True


class TrainingConfig:
    """Training configuration for QAT."""
    def __init__(self):
        # Dataset
        self.train_split = 'train[:20000]'
        self.val_split = 'validation[:2000]'
        self.batch_size = 8
        self.max_seq_length = 256
        self.doc_stride = 128

        # Optimization
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.adam_betas = (0.9, 0.999)
        self.max_grad_norm = 1.0

        # Training schedule
        self.num_iterations = 3500
        self.warmup_steps = 100
        self.gradient_accumulation_steps = 8

        # Evaluation
        self.eval_interval = 50
        self.save_interval = 100

        # Memory optimization
        self.use_amp = True
        self.empty_cache_interval = 25
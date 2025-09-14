"""
Configuration for QAT (Quantization-Aware Training)
Single precision training with fake quantization
"""

import torch


class ModelConfig:
    """
    Model configuration for QAT GPT-2.
    Single precision with fake quantization.
    """
    def __init__(self):
        # GPT-2 architecture
        self.vocab_size = 50257
        self.n_positions = 256
        self.n_embd = 768
        self.n_layer = 6
        self.n_head = 12
        self.layer_norm_epsilon = 1e-5
        self.embd_pdrop = 0.1
        
        # QAT settings (single precision)
        self.quantization_bits = 8  # Fixed precision for QAT
        self.use_gradient_checkpointing = True


class TrainingConfig:
    """
    Training configuration for QAT.
    """
    def __init__(self):
        # Data
        self.train_split = 'train[:2000]'
        self.val_split = 'validation[:200]'
        self.batch_size = 1
        self.max_seq_length = 256
        self.doc_stride = 128
        
        # Optimization
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.adam_betas = (0.9, 0.999)
        
        # Training
        self.num_iterations = 2000
        self.warmup_steps = 5
        self.eval_interval = 50
        self.save_interval = 100
        self.gradient_accumulation_steps = 4  # Increased to allow smaller effective memory per step
        self.max_grad_norm = 1.0

        # Memory optimization
        self.use_amp = True
        self.empty_cache_interval = 25  # Less frequent cache clearing to avoid fragmentation



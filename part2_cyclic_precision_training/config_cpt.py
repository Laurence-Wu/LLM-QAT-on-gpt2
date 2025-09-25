"""
Configuration for Cyclic Precision Training (CPT)
Based on the CPT paper and incorporating successful strategies from Part 1.
"""

import math


class ModelConfig:
    """Model architecture configuration for CPT."""
    def __init__(self):
        # GPT-2 architecture - exact dimensions
        self.vocab_size = 50257
        self.n_positions = 1024
        self.n_embd = 768
        self.n_layer = 12
        self.n_head = 12

        # Regularization
        self.layer_norm_epsilon = 1e-5
        self.embd_pdrop = 0.1

        # Cyclic precision settings
        self.bit_widths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Cycle through these precisions
        self.default_bits = 8  # Default/upper bound precision

        # LoRA settings per bit-width
        # Lower precision needs higher rank to compensate
        self.lora_rank_per_bit = {
            4: 32,  # Highest rank for lowest precision
            6: 24,  # Medium rank
            8: 16,  # Lower rank for higher precision
            32: 0   # No LoRA for FP32 (if used)
        }
        self.lora_alpha_per_bit = {
            4: 64,
            6: 48,
            8: 32,
            32: 0
        }

        # Quantization settings
        self.quantizer_type = 'log'  # Use log quantization
        self.use_per_channel = True  # Always use per-channel calibration
        self.gradient_bits = 8  # Static gradient precision
        self.activation_bits_per_bit = {
            4: 4,
            6: 6,
            8: 8
        }

        # Gradient bifurcation settings
        self.weight_gradient_bits = 16  # Higher precision for weight gradients
        self.activation_gradient_bits = 8  # Lower precision for activation gradients

        # Memory optimization
        self.use_gradient_checkpointing = True


class CPTConfig:
    """Configuration for Cyclic Precision Training."""
    def __init__(self):
        # Cyclic schedule settings
        self.cycle_length = 3  # Number of steps in each cycle (len(bit_widths))
        self.schedule_type = 'cosine'  # 'cosine', 'triangular', or 'linear'

        # Precision Range Test (PRT) settings
        self.prt_start_bits = 2  # Starting precision for PRT
        self.prt_threshold = 0.01  # Accuracy improvement threshold
        self.prt_iterations = 100  # Iterations per precision level



class TrainingConfig:
    """Training configuration for CPT."""
    def __init__(self):
        # Dataset
        self.train_split = 'train[:5000]'
        self.val_split = 'validation[:5000]'
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
        self.num_epochs = 160
        self.num_iterations = None  # Will be calculated based on dataset
        self.gradient_accumulation_steps = 8

        # Cyclic precision schedule
        self.num_cycles = 32  # Total number of complete cycles during training
        self.target_bits = 8  # Targeted precision for training

        # Evaluation
        self.eval_interval = 50
        self.save_interval = 100

        # Memory optimization
        self.empty_cache_interval = 25
        self.num_workers = 0

        # Loss settings
        self.use_distillation = False  # NO distillation for CPT

        # Hardware settings
        self.device = 'cuda'
        self.fp16 = False  # Use FP32 for training stability

        # Logging
        self.log_interval = 10
        self.verbose = True


class SBMConfig:
    """Configuration for SBM-specific components."""
    def __init__(self):
        # Range BatchNorm settings
        self.use_range_bn = True
        self.bn_momentum = 0.1
        self.bn_eps = 1e-5

        # GEMMLOWP quantization settings
        self.use_stochastic_rounding = True
        self.clamp_percentile = 99.9  # For activation clamping

        # Gradient bifurcation
        self.bifurcate_gradients = True
        self.weight_grad_bits = 16
        self.activation_grad_bits = 8


def get_config():
    """Get all configuration objects."""
    return {
        'model': ModelConfig(),
        'cpt': CPTConfig(),
        'training': TrainingConfig(),
        'sbm': SBMConfig()
    }
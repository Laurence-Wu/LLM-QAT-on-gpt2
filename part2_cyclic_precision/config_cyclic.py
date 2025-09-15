"""
Configuration for Cyclic Precision Training (CPT)
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

        # Model settings
        self.use_pretrained = True
        self.use_gradient_checkpointing = True
        self.default_bit_width = 8


class CyclicTrainingConfig:
    """Training configuration for CPT - Step 3 & 5 of Algorithm Test 2."""
    def __init__(self):
        # Dataset - Using SQuAD as specified in Step 3
        self.train_split = 'train[:20000]'  # SQuAD training data
        self.val_split = 'validation[:2000]'  # SQuAD validation data
        self.batch_size = 2
        self.max_seq_length = 256
        self.doc_stride = 128

        # Optimization
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.adam_betas = (0.9, 0.999)
        self.max_grad_norm = 1.0

        # Training schedule - Step 3 specifies 1000 iterations
        self.num_cpt_iterations = 1000  # As per Step 3 requirement
        self.warmup_steps = 50
        self.gradient_accumulation_steps = 4

        # Evaluation
        self.eval_interval = 100  # Evaluate every 100 iterations
        self.save_interval = 200  # Save checkpoint every 200 iterations
        self.log_interval = 10    # Log metrics every 10 iterations

        # Memory optimization
        self.use_amp = True
        self.empty_cache_interval = 10

        # Logging
        self.verbose = True
        self.log_bit_width_changes = True  # Log when bit-width changes


class CyclicPrecisionConfig:
    """Cyclic precision specific configuration for Step 5 of Algorithm Test 2."""
    def __init__(self):
        # Cycling pattern - dynamic bit-width changes during training
        self.cycle_length = 30  # Length of each cycle
        self.bit_width_pattern = [8, 6, 4, 2, 4, 6, 8]  # Pattern to cycle through

        # Different configurations to explore
        self.bit_width_configs = [
            [8, 4, 2, 4, 8],  # Aggressive quantization
            [8, 6, 4, 6, 8],  # Moderate quantization
            [8, 7, 6, 5, 4, 5, 6, 7, 8],  # Gradual quantization
        ]

        # Current config index
        self.current_config_idx = 0

        # Layer-wise cycling - enable different bit-widths per layer
        self.layer_wise_cycling = False
        self.layer_cycle_offset = 2

        # Learning rate adjustment per bit-width
        self.adjust_lr_with_bits = True
        self.lr_scale_factors = {
            2: 0.4,   # 40% LR for 2-bit
            3: 0.5,   # 50% LR for 3-bit
            4: 0.6,   # 60% LR for 4-bit
            5: 0.7,   # 70% LR for 5-bit
            6: 0.8,   # 80% LR for 6-bit
            7: 0.9,   # 90% LR for 7-bit
            8: 1.0,   # 100% LR for 8-bit
        }

        # Progressive cycling - reduce max bit-width over cycles
        self.progressive_cycles = False
        self.progression_rate = 0.9

        # Warm-up for bit transitions
        self.warmup_steps_per_transition = 5

        # Metrics tracking
        self.track_cycle_metrics = True
        self.track_bit_transitions = True
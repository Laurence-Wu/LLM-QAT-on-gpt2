"""
Configuration for Cyclic Precision Training (CPT)
Simplified and unified configuration.
"""

class ModelConfig:
    """Model architecture configuration."""
    def __init__(self):
        self.vocab_size = 50257
        self.n_positions = 1024
        self.n_embd = 768
        self.n_layer = 12
        self.n_head = 12

        self.layer_norm_epsilon = 1e-5
        self.embd_pdrop = 0.1

        self.use_pretrained = True
        self.use_gradient_checkpointing = False
        self.default_bit_width = 32

        # LoRA settings
        self.lora_rank = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.1


class CyclicTrainingConfig:
    """Training configuration for CPT - Step 3 & 5 of Algorithm Test 2."""
    def __init__(self):
        self.train_split = 'train[:3000]'
        self.val_split = 'validation[:100]'
        self.batch_size = 32
        self.max_seq_length = 1024
        self.doc_stride = 512

        self.learning_rate = 2e-4
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.adam_betas = (0.9, 0.999)
        self.max_grad_norm = 1.0

        self.num_cpt_iterations = 1000
        self.warmup_steps = 500
        self.gradient_accumulation_steps = 2

        self.eval_interval = 250
        self.save_interval = 500
        self.log_interval = 50

        self.use_amp = True
        self.empty_cache_interval = 200

        self.verbose = True
        self.log_bit_width_changes = True


class CyclicPrecisionConfig:
    """Cyclic precision specific configuration for Step 5 of Algorithm Test 2."""
    def __init__(self):
        self.bit_widths = [2, 4, 6, 8]  # Available bit widths
        self.cycle_length = 100
        self.bit_width_pattern = [8, 6, 4, 2, 4, 6, 8]

        self.bit_width_configs = [
            [8, 4, 2, 4, 8],
            [8, 6, 4, 6, 8],
            [8, 7, 6, 5, 4, 5, 6, 7, 8],
        ]

        self.current_config_idx = 0

        self.layer_wise_cycling = False
        self.layer_cycle_offset = 2

        self.adjust_lr_with_bits = True
        self.lr_scale_factors = {
            2: 0.4,
            3: 0.5,
            4: 0.6,
            5: 0.7,
            6: 0.8,
            7: 0.9,
            8: 1.0,
        }

        self.progressive_cycles = False
        self.progression_rate = 0.9

        self.warmup_steps_per_transition = 10

        self.track_cycle_metrics = True
        self.track_bit_transitions = True
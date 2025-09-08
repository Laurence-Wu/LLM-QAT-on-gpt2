"""
Configuration for Switchable Precision Training
Defines model architecture, training parameters, and quantization settings.
"""

import torch


class ModelConfig:
    """
    Model configuration for switchable precision GPT-2.
    Controls architecture parameters and quantization bit widths.
    """
    def __init__(self):
        # GPT-2 architecture parameters
        self.vocab_size = 50257
        self.n_positions = 256
        self.n_embd = 768 #d_model
        self.n_layer = 6
        self.n_head = 12
        
        # Regularization
        self.layer_norm_epsilon = 1e-5
        self.embd_pdrop = 0.1 # Embedding dropout
        self.attn_pdrop = 0.1 # Attention dropout
        self.resid_pdrop = 0.1 # Residual dropout
        
        # Quantization
        self.bit_widths = [4, 8, 16]
        self.default_bit_width = 8

        # Memory optimization
        self.use_gradient_checkpointing = True # Enable gradient checkpointing
        self.use_mixed_precision = True # Enable mixed precision training


class TrainingConfig:
    """
    Training configuration for switchable precision training.
    Controls optimization, data loading, and training schedule.
    """
    def __init__(self):
        # Data configuration
        self.train_split = 'train[:20]'# Training data split
        self.val_split = 'validation[:2]'# Validation data split
        self.batch_size = 16
        self.max_seq_length = 256
        self.doc_stride = 128
        # Optimization parameters
        self.learning_rate = 1e-4
        self.weight_decay = 0.01 # L2 regularization
        self.adam_epsilon = 1e-8
        self.adam_betas = (0.9, 0.999)
        # Training schedule
        self.num_iterations = 1000
        self.warmup_steps = 50
        self.eval_interval = 2
        self.save_interval = 100
        
        # Switchable quantization settings
        self.switch_interval = 10               # Iterations between bit-width switches
        self.switch_strategy = 'progressive'    # Strategy: 'cyclic', 'random', or 'progressive'
        self.switch_strategy = 'cyclic'         # Strategy: 'cyclic', 'random', or 'scheduled'
        self.bit_width_schedule = None          # Custom schedule if strategy is 'scheduled'
        
        # Loss and metrics
        self.gradient_accumulation_steps = 4 # Gradient accumulation for larger effective batch
        self.max_grad_norm = 1.0  # Gradient clipping threshold
        
        # Memory optimization
        self.use_amp = True # Use automatic mixed precision
        self.empty_cache_interval = 10 # Clear GPU cache every N iterations


class SwitchableQuantizationConfig:
    """
    Configuration specific to switchable quantization behavior.
    Controls how the model switches between different bit widths.
    """
    def __init__(self):
        # Switching behavior
        self.enable_switching = True            # Enable dynamic bit-width switching
        self.switch_probability = 0.3           # Probability of switching in random mode
        
        # Layer-wise configuration
        self.layer_specific_bits = {}           # Dict mapping layer indices to bit widths
        self.progressive_quantization = False   # Whether to progressively reduce bit widths
        
        # Performance tracking
        self.track_bit_usage = True            # Track bit-width usage statistics
        self.track_switching_impact = True     # Track impact of switching on loss
        
        # Quantization methods
        self.quantization_method = 'symmetric'  # 'symmetric' or 'asymmetric'
        self.calibration_method = 'minmax'     # 'minmax', 'percentile', or 'entropy'
        
        # Fine-tuning after switching
        self.finetune_after_switch = True      # Fine-tune for a few steps after switching
        self.finetune_steps = 5                # Number of fine-tuning steps


class ExperimentConfig:
    """
    Configuration for experimental settings and ablation studies.
    """
    def __init__(self):
        # Experiment identification
        self.experiment_name = "switchable_precision_training"
        self.experiment_id = None               # Will be generated at runtime
        
        # Ablation settings
        self.test_static_baselines = True      # Test static bit-width baselines
        self.test_switching_strategies = True   # Test different switching strategies
        
        # Baseline configurations to test
        self.baseline_bit_widths = [2, 4, 8]   # Static bit widths to test
        
        # Metrics to track
        self.metrics_to_track = [
            'loss',
            'perplexity',
            'accuracy',
            'bit_width_distribution',
            'switching_frequency',
            'memory_usage',
            'inference_speed'
        ]
        
        # Results saving
        self.save_intermediate_results = True   # Save results during training
        self.results_dir = './results/switchable_precision/'
        self.checkpoint_dir = './checkpoints/switchable_precision/'
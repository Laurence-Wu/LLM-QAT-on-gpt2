"""
Configuration for Cyclic Precision Training
Defines model architecture, training parameters, and cyclic precision settings.
"""

import torch


class ModelConfig:
    """
    Model configuration for cyclic precision GPT-2.
    """
    def __init__(self):
        # GPT-2 architecture
        self.vocab_size = 50257  # Standard GPT-2 vocabulary
        self.n_positions = 256   # Maximum sequence length
        self.n_embd = 768        # Embedding dimension
        self.n_layer = 6         # Number of transformer layers
        self.n_head = 12         # Number of attention heads
        
        # Regularization
        self.layer_norm_epsilon = 1e-5
        self.embd_pdrop = 0.1
        self.attn_pdrop = 0.1
        self.resid_pdrop = 0.1
        
        # Quantization settings
        self.bit_widths = [2, 4, 8, 16]  # Available bit widths for cycling
        self.default_bit_width = 8       # Starting bit width
        
        # Model initialization
        self.use_pretrained = True       # Load pretrained GPT-2 weights
        
        # Memory optimization
        self.use_gradient_checkpointing = True
        self.use_mixed_precision = True


class CyclicTrainingConfig:
    """
    Training configuration for cyclic precision training.
    """
    def __init__(self):
        # Data configuration
        self.train_split = 'train[:2000]'
        self.val_split = 'validation[:200]'
        self.batch_size = 2
        self.max_seq_length = 256
        self.doc_stride = 128
        
        # Optimization
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.adam_betas = (0.9, 0.999)
        
        # Training schedule - Phase 1: CPT
        self.num_cpt_iterations = 150      # Iterations for cyclic training
        self.warmup_steps = 10              # Warmup before cycling begins
        
        # Training schedule - Phase 2: Static configurations
        self.num_static_iterations = 50    # Iterations per static configuration
        
        # Evaluation and checkpointing
        self.eval_interval = 20
        self.save_interval = 50
        self.log_interval = 5
        
        # Gradient settings
        self.gradient_accumulation_steps = 4
        self.max_grad_norm = 1.0
        
        # Memory optimization
        self.use_amp = True
        self.empty_cache_interval = 10
        
        # Logging
        self.verbose = True


class CyclicPrecisionConfig:
    """
    Configuration specific to cyclic precision behavior.
    Controls how precision cycles during training.
    """
    def __init__(self):
        # Cyclic pattern configuration
        self.cycle_length = 20              # Number of iterations per cycle
        self.bit_width_pattern = [8, 4, 2, 4, 8]  # Pattern within each cycle
        
        # Advanced cycling options
        self.use_cosine_schedule = False    # Use cosine annealing for bit widths
        self.use_reverse_cycle = True       # Include reverse cycling
        
        # Layer-wise cycling
        self.layer_wise_cycling = False     # Different cycles for different layers
        self.layer_cycle_offset = 2         # Offset between layer cycles
        
        # Transition behavior
        self.smooth_transitions = True      # Smooth transitions between bit widths
        self.transition_steps = 2           # Steps for smooth transition
        
        # Learning rate adjustment
        self.adjust_lr_with_bits = True     # Adjust LR based on current bit width
        self.lr_scale_factors = {
            2: 0.5,   # Lower LR for 2-bit
            4: 0.75,  # Moderate LR for 4-bit
            8: 1.0,   # Normal LR for 8-bit
            16: 1.0   # Normal LR for 16-bit
        }
        
        # Cycle progression
        self.progressive_cycles = True      # Make cycles progressively harder
        self.progression_rate = 0.9         # Rate of progression (multiplier)
        
        # Performance tracking
        self.track_cycle_metrics = True     # Track metrics per cycle
        self.track_bit_transitions = True   # Track transition effects


class StaticPrecisionConfig:
    """
    Configuration for static precision training after CPT.
    """
    def __init__(self):
        # Static configurations to test
        self.test_configurations = {
            'ultra_low': 2,      # 2-bit only
            'low': 4,            # 4-bit only
            'medium': 8,         # 8-bit only
            'high': 16,          # 16-bit only (if available)
            'mixed_low': [2, 4], # Alternating 2-4 bit
            'mixed_med': [4, 8], # Alternating 4-8 bit
            'progressive': 'progressive'  # Special progressive mode
        }
        
        # Fine-tuning settings
        self.finetune_iterations = 50       # Iterations per configuration
        self.finetune_lr_scale = 0.5        # Scale down LR for fine-tuning
        
        # Evaluation settings
        self.evaluate_each_config = True    # Evaluate after each configuration
        self.compare_to_baseline = True     # Compare to non-quantized baseline


class ExperimentConfig:
    """
    Configuration for experimental settings and ablations.
    """
    def __init__(self):
        # Experiment identification
        self.experiment_name = "cyclic_precision_training"
        self.experiment_id = None  # Generated at runtime
        
        # Ablation studies
        self.ablations = {
            'no_cycling': False,           # Train without cycling (baseline)
            'fixed_pattern': False,         # Use fixed pattern instead of cyclic
            'random_cycling': False,        # Random bit width selection
            'layer_independent': False      # Independent cycling per layer
        }
        
        # Metrics to track
        self.metrics = [
            'loss',
            'perplexity',
            'accuracy',
            'cycle_stability',      # Stability across cycles
            'bit_efficiency',       # Efficiency per bit
            'convergence_speed',    # Speed of convergence
            'final_performance'     # Final model performance
        ]
        
        # Analysis settings
        self.analyze_cycles = True          # Detailed cycle analysis
        self.plot_metrics = True            # Generate plots
        self.save_cycle_checkpoints = False # Save model at each cycle
        
        # Results paths
        self.results_dir = './results/cyclic_precision/'
        self.checkpoint_dir = './checkpoints/cyclic_precision/'
        self.plots_dir = './plots/cyclic_precision/'
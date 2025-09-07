from dataclasses import dataclass
from typing import List, Optional
import warnings

@dataclass
class QuantizationConfig:
    weight_bits: int = 16  # Higher precision for better accuracy
    activation_bits: int = 16  # Higher precision for better accuracy
    kv_cache_bits: int = 16  # Higher precision for better accuracy
    symmetric: bool = True
    per_channel: bool = True
    
    def __post_init__(self):
        # Validate bit widths
        for bits_name, bits_value in [
            ('weight_bits', self.weight_bits),
            ('activation_bits', self.activation_bits),
            ('kv_cache_bits', self.kv_cache_bits)
        ]:
            if bits_value not in [2, 4, 8, 16, 32]:
                warnings.warn(f"{bits_name}={bits_value} is not a standard quantization level. "
                            f"Consider using 2, 4, 8, 16, or 32 bits.")
            if bits_value < 4:
                warnings.warn(f"{bits_name}={bits_value} may cause severe accuracy degradation.")
        
        if self.weight_bits == 16 and self.activation_bits == 16:
            warnings.warn("Using 16-bit for both weights and activations. "
                        "Consider 8-bit for better compression with minimal accuracy loss.")
    
@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4  # Increased for faster convergence
    batch_size: int = 16  # Increased for better gradient estimates
    num_iterations: int = 1000  # As specified in algtest2 requirements
    warmup_steps: int = 500  # 10% warmup for stability
    gradient_accumulation_steps: int = 2  # Reduced for more frequent updates
    max_seq_length: int = 512  # Standard sequence length
    doc_stride: int = 256  # 50% overlap for better context
    use_scheduler: bool = True  # Enable learning rate scheduling
    scheduler_type: str = 'cosine'  # Cosine annealing
    weight_decay: float = 0.01  # L2 regularization
    max_grad_norm: float = 1.0  # Gradient clipping
    eval_steps: int = 100  # Evaluate every 100 steps
    save_steps: int = 500  # Save checkpoint every 500 steps
    early_stopping_patience: int = 5  # Stop if no improvement for 5 evals
    min_learning_rate: float = 1e-6  # Minimum LR for scheduler
    
    def __post_init__(self):
        # Validate configuration parameters
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.num_iterations > 0, "Number of iterations must be positive"
        assert self.gradient_accumulation_steps > 0, "Gradient accumulation steps must be positive"
        assert self.max_seq_length > 0, "Max sequence length must be positive"
        assert self.doc_stride > 0 and self.doc_stride <= self.max_seq_length, \
            "Doc stride must be positive and <= max_seq_length"
        assert self.warmup_steps <= self.num_iterations, \
            "Warmup steps cannot exceed total iterations"
        assert self.learning_rate > 0, "Learning rate must be positive"
        
        # Warnings for potential issues
        effective_batch = self.batch_size * self.gradient_accumulation_steps
        if effective_batch < 16:
            warnings.warn(f"Effective batch size ({effective_batch}) is very small. "
                        "This may lead to unstable training.")
        
        if self.max_seq_length < 512:
            warnings.warn(f"Max sequence length ({self.max_seq_length}) is significantly "
                        "reduced from standard GPT-2 (1024). This may impact model performance.")
        
        if self.num_iterations < 1000:
            warnings.warn(f"Training for only {self.num_iterations} iterations. "
                        "This may not be sufficient for convergence.")
        
        # Memory estimation warning (updated for new architecture)
        estimated_memory_gb = (self.batch_size * self.max_seq_length * 768 * 12 * 4) / (1024**3)
        if estimated_memory_gb > 40:
            warnings.warn(f"Estimated memory usage: {estimated_memory_gb:.1f} GB. "
                        "Consider reducing batch_size or max_seq_length if OOM occurs.")

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_positions: int = 512  # Standard context length for training
    n_embd: int = 768  # GPT-2 small embedding dimension
    n_layer: int = 12  # GPT-2 small layer count
    n_head: int = 12  # GPT-2 small attention heads
    layer_norm_epsilon: float = 1e-5
    embd_pdrop: float = 0.1  # Standard dropout for regularization
    bit_widths: List[int] = None
    use_pretrained: bool = True  # Load pretrained GPT-2 weights
    
    def __post_init__(self):
        if self.bit_widths is None:
            self.bit_widths = [8, 16]  # Remove 4-bit until stable, focus on 8 and 16
        
        # Validate model architecture
        assert self.n_embd % self.n_head == 0, \
            f"Embedding dimension ({self.n_embd}) must be divisible by number of heads ({self.n_head})"
        assert self.n_positions > 0, "Number of positions must be positive"
        assert self.n_layer > 0, "Number of layers must be positive"
        assert self.vocab_size > 0, "Vocabulary size must be positive"
        
        # Validate bit widths
        for bits in self.bit_widths:
            if bits not in [2, 4, 8, 16, 32]:
                warnings.warn(f"Bit width {bits} is non-standard. Consider using 2, 4, 8, 16, or 32.")
        
        # Architecture warnings
        if self.n_layer < 12:
            warnings.warn(f"Using only {self.n_layer} layers (standard GPT-2 uses 12-48). "
                        "This will significantly reduce model capacity.")
        
        if self.n_embd < 768:
            warnings.warn(f"Embedding dimension {self.n_embd} is smaller than GPT-2 base (768). "
                        "This will reduce model expressiveness.")
        
        if self.n_positions < 1024:
            warnings.warn(f"Context length {self.n_positions} is much smaller than standard (1024). "
                        "This limits the model's ability to process long sequences.")
        
        # Memory footprint estimation
        param_count = (self.vocab_size * self.n_embd +  # Token embeddings
                      self.n_positions * self.n_embd +  # Position embeddings
                      self.n_layer * (12 * self.n_embd * self.n_embd))  # Transformer layers
        memory_mb = (param_count * 4) / (1024 * 1024)  # 4 bytes per parameter
        
        if memory_mb < 100:
            warnings.warn(f"Model size (~{memory_mb:.1f} MB) is extremely small. "
                        "This configuration is suitable for testing but not production.")

@dataclass
class CyclicPrecisionConfig:
    min_bits: int = 4  # Efficient minimum precision
    max_bits: int = 16  # Balanced maximum precision for H100
    cycle_length: int = 200  # Optimized cycle length
    warmup_steps: int = 50  # Efficient warmup
    
    def __post_init__(self):
        assert self.min_bits > 0 and self.min_bits <= 32, "min_bits must be between 1 and 32"
        assert self.max_bits > 0 and self.max_bits <= 32, "max_bits must be between 1 and 32"
        assert self.min_bits <= self.max_bits, "min_bits must be <= max_bits"
        assert self.cycle_length > 0, "cycle_length must be positive"
        assert self.warmup_steps >= 0, "warmup_steps must be non-negative"
        
        if self.min_bits < 4:
            warnings.warn(f"min_bits={self.min_bits} may cause severe gradient degradation.")
        
        if self.max_bits - self.min_bits > 12:
            warnings.warn(f"Large precision range ({self.min_bits}-{self.max_bits} bits) "
                        "may cause training instability.")

@dataclass
class AdversarialConfig:
    epsilon: float = 0.05  # Increased for realistic attacks
    test_samples: int = 500  # More samples for statistical significance
    use_random_precision: bool = True
    bit_widths: List[int] = None
    attack_steps: int = 10  # PGD attack steps
    attack_type: str = 'pgd'  # Attack type: 'fgsm' or 'pgd'
    
    def __post_init__(self):
        if self.bit_widths is None:
            self.bit_widths = [4, 8, 16]  # Balanced precision for H100 robustness testing
        
        assert self.epsilon > 0 and self.epsilon < 1, "epsilon must be between 0 and 1"
        assert self.test_samples > 0, "test_samples must be positive"
        
        if self.epsilon > 0.1:
            warnings.warn(f"Large epsilon ({self.epsilon}) may produce unrealistic adversarial examples.")
        
        if self.test_samples < 100:
            warnings.warn(f"Only {self.test_samples} test samples may not provide statistically significant results.")

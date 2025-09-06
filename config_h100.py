from dataclasses import dataclass
from typing import List, Optional

@dataclass
class QuantizationConfig:
    weight_bits: int = 16  # Higher precision for better accuracy
    activation_bits: int = 16  # Higher precision for better accuracy
    kv_cache_bits: int = 16  # Higher precision for better accuracy
    symmetric: bool = True
    per_channel: bool = True
    
@dataclass
class TrainingConfig:
    learning_rate: float = 5e-5  # Lower learning rate for more stable training
    batch_size: int = 1  # Ultra-conservative batch size for H100 memory
    num_iterations: int = 200  # Reduced iterations for memory testing
    warmup_steps: int = 20  # Minimal warmup
    gradient_accumulation_steps: int = 8  # Lower accumulation to reduce memory
    max_seq_length: int = 256  # Further reduced sequence length
    doc_stride: int = 128  # Smaller stride

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_positions: int = 256  # Ultra-conservative context length
    n_embd: int = 512  # Smaller embedding for extreme memory efficiency
    n_layer: int = 6  # Minimal layers for memory conservation
    n_head: int = 8  # Fewer attention heads
    layer_norm_epsilon: float = 1e-5
    embd_pdrop: float = 0.1  # Standard dropout for regularization
    bit_widths: List[int] = None
    
    def __post_init__(self):
        if self.bit_widths is None:
            self.bit_widths = [4, 8, 16]  # Balanced precision for H100 efficiency

@dataclass
class CyclicPrecisionConfig:
    min_bits: int = 4  # Efficient minimum precision
    max_bits: int = 16  # Balanced maximum precision for H100
    cycle_length: int = 200  # Optimized cycle length
    warmup_steps: int = 50  # Efficient warmup

@dataclass
class AdversarialConfig:
    epsilon: float = 0.01  # Balanced epsilon for H100 efficiency
    test_samples: int = 200  # Efficient test samples for H100
    use_random_precision: bool = True
    bit_widths: List[int] = None
    
    def __post_init__(self):
        if self.bit_widths is None:
            self.bit_widths = [4, 8, 16]  # Balanced precision for H100 robustness testing

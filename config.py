from dataclasses import dataclass
from typing import List, Optional

@dataclass
class QuantizationConfig:
    weight_bits: int = 8
    activation_bits: int = 8
    kv_cache_bits: int = 8
    symmetric: bool = True
    per_channel: bool = True
    
@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_iterations: int = 100
    warmup_steps: int = 10
    gradient_accumulation_steps: int = 2
    max_seq_length: int = 256
    doc_stride: int = 128

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    layer_norm_epsilon: float = 1e-5
    embd_pdrop: float = 0.1
    bit_widths: List[int] = None
    
    def __post_init__(self):
        if self.bit_widths is None:
            self.bit_widths = [4, 8, 16]

@dataclass
class CyclicPrecisionConfig:
    min_bits: int = 4
    max_bits: int = 8
    cycle_length: int = 200
    warmup_steps: int = 50

@dataclass
class AdversarialConfig:
    epsilon: float = 0.01
    test_samples: int = 100
    use_random_precision: bool = True
    bit_widths: List[int] = None
    
    def __post_init__(self):
        if self.bit_widths is None:
            self.bit_widths = [4, 6, 8]
"""
Part 1: Switchable Precision (SP) Implementation
Multi-precision training with separate LoRA adapters for each bit-width.
"""

from .models_sp import SPModel, SPLMHeadModel
from .config_sp import ModelConfig, TrainingConfig
from .train_sp import train_sp

__all__ = [
    'SPModel',
    'SPLMHeadModel',
    'ModelConfig',
    'TrainingConfig',
    'train_sp'
]
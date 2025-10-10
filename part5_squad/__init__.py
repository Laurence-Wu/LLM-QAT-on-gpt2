"""
Part 5: Switchable Precision for SQuAD Question Answering
Multi-precision training with separate LoRA adapters for each bit-width on SQuAD dataset.
"""

from .models_squad import SPModel, SPQuestionAnsweringModel
from .config_squad import ModelConfig, TrainingConfig
from .train_squad import train_squad

__all__ = [
    'SPModel',
    'SPQuestionAnsweringModel',
    'ModelConfig',
    'TrainingConfig',
    'train_squad'
]
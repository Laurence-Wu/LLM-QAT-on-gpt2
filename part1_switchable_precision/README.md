# Part 1: Switchable Precision Training

This module implements training of GPT-2 models with switchable precision across different bit widths. The model can dynamically switch between different quantization levels during training to explore optimal precision configurations.

## Overview

Switchable precision training allows the model to:
- Dynamically switch between 2-bit, 4-bit, and 8-bit quantization during training
- Learn robust representations that work across multiple precision levels
- Explore optimal bit-width configurations for different layers

## Files Structure

```
part1_switchable_precision/
├── main_switchable.py          # Main training script
├── config_switchable.py        # Configuration classes
├── train_switchable.py         # Training functions
├── evaluate_switchable.py      # Evaluation functions
└── README.md                   # This file
```

## Key Components

### `main_switchable.py`
The main entry point that:
- Initializes the model with switchable quantization
- Loads pretrained GPT-2 weights
- Orchestrates the training process
- Evaluates different quantization configurations
- Saves results and trained models

### `config_switchable.py`
Configuration classes that define:
- **ModelConfig**: Model architecture and quantization settings
- **TrainingConfig**: Optimization and training parameters
- **SwitchableQuantizationConfig**: Switching behavior settings
- **ExperimentConfig**: Experimental and ablation settings

### `train_switchable.py`
Training implementation including:
- Dynamic bit-width switching strategies (cyclic, random, scheduled)
- Knowledge distillation from full-precision teacher
- Layer-wise precision configuration
- Mixed precision training support

### `evaluate_switchable.py`
Evaluation tools for:
- Testing different bit-width configurations
- Analyzing performance vs precision trade-offs
- Measuring inference speed across configurations
- Finding Pareto-optimal configurations

## Usage

### Basic Training
```bash
cd part1_switchable_precision
python main_switchable.py
```

### Configuration Options

The training can be customized through the configuration files:

```python
# Example configuration
model_config = ModelConfig()
model_config.bit_widths = [2, 4, 8]  # Available bit widths
model_config.n_layer = 6             # Number of layers

training_config = TrainingConfig()
training_config.switch_strategy = 'cyclic'    # Switching strategy
training_config.switch_interval = 10         # Iterations between switches
```

### Switching Strategies

1. **Cyclic**: Regular cycling through bit widths
2. **Random**: Random bit width selection
3. **Scheduled**: Custom predefined schedule
4. **Progressive**: Start high precision, gradually decrease

## Key Features

### Dynamic Precision Switching
- **Cycle-based switching**: Switch precision every N iterations
- **Layer-wise control**: Different layers can use different precisions
- **Adaptive learning rates**: Adjust learning rate based on current precision

### Performance Optimization
- **Gradient checkpointing**: Memory-efficient training
- **Mixed precision**: Automatic mixed precision support
- **Memory management**: Automatic GPU cache clearing

### Evaluation and Analysis
- **Configuration comparison**: Test multiple precision configurations
- **Pareto frontier**: Find optimal accuracy/efficiency trade-offs
- **Speed benchmarking**: Measure inference speed for each configuration

## Results

The training produces:
- **Trained model**: Saved as `switchable_model.pt`
- **Results report**: JSON file with detailed metrics
- **Configuration analysis**: Performance across different bit-width setups

### Key Metrics
- **Accuracy**: Token-level prediction accuracy
- **Perplexity**: Language modeling perplexity
- **Effective bits**: Weighted average bit width
- **Compression ratio**: Compression vs full precision
- **Inference speed**: Tokens processed per second

## Advanced Usage

### Custom Switching Patterns
```python
# Define custom bit-width pattern
training_config.switch_strategy = 'scheduled'
training_config.bit_width_schedule = [8, 4, 2, 4, 8]  # Custom pattern
```

### Layer-specific Configuration
```python
# Configure specific layers with different precisions
layer_config = [
    {'attn_bits': 8, 'mlp_bits': 4},  # Layer 0: High attention, low MLP
    {'attn_bits': 4, 'mlp_bits': 8},  # Layer 1: Low attention, high MLP
    # ... more layers
]
model.set_layer_precision(layer_config)
```

## Dependencies

- PyTorch >= 1.9.0
- Transformers >= 4.0.0
- NumPy
- tqdm
- All shared modules from `../shared/`

## Technical Details

### Memory Optimization
- Uses gradient checkpointing to reduce memory usage
- Automatic mixed precision training
- Periodic GPU cache clearing

### Quantization Implementation
- Symmetric quantization for weights and activations
- Per-channel quantization for improved accuracy
- Learnable quantization parameters

### Training Stability
- Knowledge distillation from full-precision teacher
- Gradient clipping for stability
- Warmup learning rate schedule

## Output Files

After training completes, you'll find:
- `switchable_model.pt`: Complete model checkpoint
- `switchable_precision_results.json`: Detailed results
- Training logs with loss progression and bit-width usage statistics
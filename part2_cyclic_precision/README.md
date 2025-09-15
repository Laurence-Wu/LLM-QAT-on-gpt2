# Part 2: Cyclic Precision Training (CPT)

This module implements Cyclic Precision Training followed by evaluation of different bit-width configurations. The approach first trains with systematic precision cycling, then fine-tunes with static configurations to find optimal settings.

## Overview

Cyclic Precision Training (CPT) provides:
- **Systematic cycling**: Structured cycling through precision levels
- **Two-phase training**: Initial CPT followed by static configuration training
- **Comparative evaluation**: Direct comparison between cyclic and static approaches
- **Cycle analysis**: Detailed analysis of cycle progression and stability

## Files Structure

```
part2_cyclic_precision/
├── main_cyclic.py              # Main training script
├── config_cyclic.py            # Configuration classes
├── train_cyclic.py             # Training functions
├── evaluate_cyclic.py          # Evaluation functions
└── README.md                   # This file
```

## Key Components

### `main_cyclic.py`
The main orchestrator that implements a three-phase approach:
1. **Phase 1**: Cyclic Precision Training with structured cycling
2. **Phase 2**: Static bit-width configuration training
3. **Phase 3**: Comparative evaluation and analysis

### `config_cyclic.py`
Configuration classes:
- **ModelConfig**: Model architecture settings
- **CyclicTrainingConfig**: Training parameters for both phases
- **CyclicPrecisionConfig**: Detailed cycling behavior settings
- **StaticPrecisionConfig**: Static configuration settings
- **ExperimentConfig**: Experiment tracking and analysis

### `train_cyclic.py`
Training implementations:
- **CyclicPrecisionScheduler**: Manages cycling patterns and transitions
- **train_with_cpt()**: Main CPT training loop
- **train_with_static_precision()**: Static configuration training
- **Layer configuration utilities**: Helper functions for precision setup

### `evaluate_cyclic.py`
Evaluation and analysis:
- **evaluate_cyclic_training()**: Test CPT model across bit widths
- **compare_bit_configurations()**: Compare CPT vs static approaches
- **analyze_cycle_progression()**: Analyze cycle stability and convergence

## Training Phases

### Phase 1: Cyclic Precision Training

The model trains with systematic precision cycling:

```python
# Example cycle configuration
cyclic_config = CyclicPrecisionConfig()
cyclic_config.cycle_length = 20              # Iterations per cycle
cyclic_config.bit_width_pattern = [8, 4, 2, 4, 8]  # Pattern within cycle
```

**Key Features**:
- **Structured patterns**: Predefined cycling patterns
- **Smooth transitions**: Optional smooth bit-width transitions
- **Learning rate adaptation**: Adjust learning rate based on current precision
- **Cycle tracking**: Monitor performance across cycles

### Phase 2: Static Configuration Training

After CPT, the model is fine-tuned with various static configurations:

```python
static_configs = {
    '2bit': 2,              # Uniform 2-bit
    '4bit': 4,              # Uniform 4-bit
    '8bit': 8,              # Uniform 8-bit
    'mixed_2_4': [2, 4],    # Alternating 2-4 bit
    'mixed_4_8': [4, 8],    # Alternating 4-8 bit
}
```

### Phase 3: Comparative Evaluation

Comprehensive comparison between:
- CPT model performance across different bit widths
- Static configuration models
- Efficiency metrics and trade-offs

## Usage

### Basic Training
```bash
cd part2_cyclic_precision
python main_cyclic.py
```

### Custom Cycle Configuration
```python
# Configure custom cycling pattern
cyclic_config = CyclicPrecisionConfig()
cyclic_config.bit_width_pattern = [16, 8, 4, 2]  # Decreasing precision
cyclic_config.cycle_length = 30                  # Longer cycles
cyclic_config.use_reverse_cycle = True           # Include reverse pattern
```

### Advanced Cycling Options

#### Progressive Cycling
```python
cyclic_config.progressive_cycles = True      # Make cycles progressively harder
cyclic_config.progression_rate = 0.9         # Rate of progression
```

#### Layer-wise Cycling
```python
cyclic_config.layer_wise_cycling = True     # Different cycles per layer
cyclic_config.layer_cycle_offset = 2        # Offset between layer cycles
```

#### Learning Rate Adaptation
```python
cyclic_config.adjust_lr_with_bits = True    # Adjust LR based on bits
cyclic_config.lr_scale_factors = {
    2: 0.5,   # Half LR for 2-bit
    4: 0.75,  # Reduced LR for 4-bit
    8: 1.0,   # Normal LR for 8-bit
}
```

## Key Features

### Cyclic Precision Scheduler
- **Pattern management**: Handle complex cycling patterns
- **Transition control**: Smooth or sharp transitions
- **Layer coordination**: Coordinate layer-wise cycling
- **Metric tracking**: Track cycle performance

### Two-Phase Training
- **CPT initialization**: Train robust initial model
- **Static fine-tuning**: Specialize for specific configurations
- **Comparative analysis**: Direct performance comparison

### Advanced Analysis
- **Cycle stability**: Measure stability across cycles
- **Convergence analysis**: Track convergence patterns
- **Configuration efficiency**: Accuracy vs compression trade-offs

## Results and Metrics

### Training Outputs
- **Best model**: Automatically selected best configuration
- **Comprehensive results**: JSON file with all metrics
- **Cycle analysis**: Detailed cycle progression data

### Key Metrics
- **Per-cycle statistics**: Average, std, min, max loss per cycle
- **Configuration comparison**: Performance across all tested configs
- **Stability score**: Measure of cycle-to-cycle stability
- **Efficiency rankings**: Accuracy per bit rankings

### Analysis Reports
- **Cycle progression**: How cycles improve over time
- **Static vs CPT comparison**: Which approach works better
- **Optimal configuration**: Recommended precision setup

## Configuration Examples

### Conservative Cycling
```python
# Gentle cycling for stable training
cyclic_config.bit_width_pattern = [8, 6, 4, 6, 8]
cyclic_config.cycle_length = 40
cyclic_config.smooth_transitions = True
```

### Aggressive Cycling
```python
# Rapid cycling for robust training
cyclic_config.bit_width_pattern = [8, 2, 8, 2]
cyclic_config.cycle_length = 10
cyclic_config.smooth_transitions = False
```

### Research-focused
```python
# Comprehensive evaluation setup
experiment_config.analyze_cycles = True
experiment_config.save_cycle_checkpoints = True
experiment_config.ablations['random_cycling'] = True
```

## Technical Implementation

### Scheduler Design
The `CyclicPrecisionScheduler` manages:
- Bit width transitions based on iteration count
- Layer-wise coordination for complex patterns
- Learning rate scaling based on current precision
- Cycle boundary detection and metric aggregation

### Memory Management
- Efficient cycle transitions without memory spikes
- Automatic garbage collection at cycle boundaries
- GPU cache management during configuration switches

### Training Stability
- Gradual learning rate warmup
- Knowledge distillation support
- Gradient clipping and mixed precision

## Dependencies

- PyTorch >= 1.9.0
- Transformers >= 4.0.0
- NumPy
- tqdm
- Matplotlib (for cycle analysis plots)
- All shared modules from `../shared/`

## Output Files

After training completes:
- `cyclic_best_model.pt`: Best model across all configurations
- `cyclic_precision_results.json`: Complete results and analysis
- Training logs with cycle progression and comparative metrics

## Advanced Usage Tips

1. **Start Conservative**: Begin with gentle cycling patterns
2. **Monitor Stability**: Watch cycle-to-cycle variance
3. **Compare Approaches**: Always compare CPT vs static methods
4. **Analyze Cycles**: Use cycle analysis to understand convergence
5. **Tune Patterns**: Experiment with different cycling patterns
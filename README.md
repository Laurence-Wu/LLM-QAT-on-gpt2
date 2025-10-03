# LLM Quantization-Aware Training on GPT-2

## Part 1: Switchable Precision Model Architecture

### Overview
Part 1 implements a **Switchable Precision (SP)** architecture for GPT-2 that supports dynamic bit-width switching during inference and training. The model can operate at multiple quantization levels (e.g., 4-bit, 8-bit, 16-bit, 32-bit) with precision-specific parameters to maintain accuracy across different bit-widths.

### Core Components

#### 1. SPLinearWithLoRA ([lora.py](part1_switchable_precision/lora.py))
Linear transformation layer with switchable precision support:
- **Base Linear Layer**: Frozen pretrained weights from GPT-2
- **Per-Precision Quantizers**: Separate weight and input quantizers for each bit-width (e.g., `4bit`, `8bit`)
- **LoRA Adapters**: Low-rank adaptation matrices (A, B) for each precision level to compensate for quantization errors
- **Dynamic Switching**: Runtime precision switching via `set_precision(bits)` method

**Forward Pass Logic**:
- **32-bit mode**: Direct linear computation without quantization
- **Low-bit mode**: `quantize(input) x quantize(weight) + LoRA(input)`

#### 2. SwitchableLayerNorm ([switchable_batchnorm.py](part1_switchable_precision/switchable_batchnorm.py))
Layer normalization with precision-specific parameters:
- **Separate Parameters**: Individual `weight` and `bias` parameters for each precision level (stored in `nn.ParameterDict`)
- **Runtime Selection**: Switches active parameters based on `current_precision`
- **Compatibility Layer**: Provides backward compatibility with standard LayerNorm interface via `ln_layers` attribute

#### 3. LearnableFakeQuantize ([quantization.py](part1_switchable_precision/quantization.py))
Calibration-based fake quantization module:
- **Quantization Methods**:
  - `minmax`: Linear quantization using min-max range
  - `log`: Logarithmic quantization for exponentially distributed values
- **Calibration Process**:
  1. `start_calibration()`: Begin collecting activation statistics
  2. Forward passes: Accumulate min/max values across batches
  3. `finish_calibration()`: Compute scale/zero-point from statistics
- **Granularity**: Supports both per-channel and per-tensor quantization
- **Dynamic Bit-Width**: Can adjust quantization bits via `set_num_bits()`

#### 4. LoRALayer ([lora.py](part1_switchable_precision/lora.py))
Low-rank adaptation for quantization error compensation:
- **Structure**: Two quantized low-rank matrices `A (in_features x rank)` and `B (rank x out_features)`
- **Output**: `x @ quantize(A) @ quantize(B) x (alpha / rank)`
- **Conditional Activation**: Disabled for 32-bit (rank=0) or when `enabled=False`
- **Per-Precision Configuration**: Each bit-width has its own LoRA rank/alpha settings

### Model Hierarchy

```
SPLMHeadModel (Top-level model with language modeling head)
|
+-- SPModel (Core transformer)
    |
    +-- wte (Token embeddings - frozen)
    +-- wpe (Position embeddings - frozen)
    +-- h (ModuleList of SPBlock)
    |   |
    |   +-- SPBlock x n_layer (12 blocks for GPT-2 base)
    |       |
    |       +-- ln_1 (SwitchableLayerNorm - pre-attention)
    |       +-- attn (SPAttention)
    |       |   |
    |       |   +-- c_attn (SPLinearWithLoRA: n_embd -> 3*n_embd for Q,K,V)
    |       |   +-- c_proj (SPLinearWithLoRA: n_embd -> n_embd)
    |       |
    |       +-- ln_2 (SwitchableLayerNorm - pre-MLP)
    |       +-- mlp (SPMLP)
    |           |
    |           +-- c_fc (SPLinearWithLoRA: n_embd -> 4*n_embd)
    |           +-- c_proj (SPLinearWithLoRA: 4*n_embd -> n_embd)
    |
    +-- ln_f (SwitchableLayerNorm - final)
    +-- lm_head (Linear: n_embd -> vocab_size, weight-tied to wte)
```

### Key Features

#### Precision Switching
The entire model can switch precision dynamically:
```python
model.set_precision(4)   # Switch to 4-bit mode
model.set_precision(32)  # Switch to 32-bit (full precision)
```

This propagates through all layers:
- Updates active quantizers in all `SPLinearWithLoRA` layers
- Switches active parameters in all `SwitchableLayerNorm` layers
- Activates/deactivates corresponding LoRA adapters

#### Quantization Configuration
From [config_sp.py](part1_switchable_precision/config_sp.py):
- **Bit Widths**: `[4, 32]` (configurable multi-precision support)
- **Quantizer Mapping**: Different quantizers for different bit-widths
  - Low bits (3-4): `minmax`
  - Higher bits (5-16): `log`
  - 32-bit: No quantization
- **LoRA Configuration**: Precision-specific ranks and alphas
  - Student bits (3-16): rank=64, alpha=64
  - Teacher (32-bit): rank=0 (disabled)

#### Calibration Mode
Special mode for quantizer calibration:
- `disable_lora_for_calibration()`: Use only base quantized weights
- `enable_lora_after_calibration()`: Re-enable LoRA adapters
- Prevents LoRA from interfering with activation range statistics

#### Weight Initialization
Loads pretrained GPT-2 weights:
- **Frozen Base Weights**: Embeddings and all linear layer base weights
- **Trainable Components**:
  - LoRA matrices (A, B) for each precision
  - LayerNorm parameters for each precision
- **Multi-Precision Initialization**: Same pretrained weights copied to all precision-specific LayerNorm layers

### Training Strategy

The architecture supports:
1. **Knowledge Distillation**: 32-bit teacher -> low-bit student
2. **Multi-Precision Training**: Train adaptations for multiple bit-widths simultaneously
3. **Calibration-Based Quantization**: Collect statistics before freezing quantization parameters
4. **Parameter-Efficient Fine-Tuning**: Only train LoRA and LayerNorm, freeze base weights

### Advantages

1. **Dynamic Precision**: Single model supports multiple precision levels
2. **Memory Efficient**: Shared base weights, only precision-specific parameters differ
3. **Accuracy Preservation**: LoRA adapters compensate for quantization errors
4. **Pretrained Compatibility**: Initializes from standard GPT-2 checkpoints
5. **Per-Channel Quantization**: Fine-grained quantization for better accuracy

---

## Part 2: Cyclic Precision Training (CPT)

### Overview
Part 2 implements **Cyclic Precision Training (CPT)**, a dynamic training strategy that cyclically varies quantization bit-widths during training. Unlike Part 1's switchable precision with precision-specific parameters, CPT uses a **single shared LoRA adapter** across all bit-widths and trains the model by cycling through different precision levels according to a schedule (cosine or triangular).

### Core Differences from Part 1

| Aspect | Part 1 (Switchable Precision) | Part 2 (Cyclic Precision Training) |
|--------|-------------------------------|-------------------------------------|
| **LoRA Strategy** | Separate LoRA adapters per bit-width | Single shared LoRA adapter for all bit-widths |
| **LayerNorm** | Precision-specific weights/biases (SwitchableLayerNorm) | Standard LayerNorm (shared across all precisions) |
| **Quantizer Storage** | Per-bit quantizers with individual calibration | Multi-bit quantizers storing parameters for all calibrated bit-widths |
| **Training** | Static precision or manual switching | Cyclic scheduling through precision range |
| **Gradient Quantization** | Not implemented | Quantizes gradients during backward pass via GradientQuantizer |

### Core Components

#### 1. CPTLinear ([cpt_model.py](part2_cyclic_precision_training/cpt_model.py))
Quantized linear layer with shared LoRA:
- **Base Linear Layer**: Standard `nn.Linear` weights (trainable from scratch, not frozen)
- **Shared LoRA Adapter**: Single `LoRAAdapter` used across all bit-widths
- **LoRA Weight Quantizers**: Per-bit quantizers for quantizing LoRA matrices (A, B)
- **Weight/Input Quantizers**: Single quantizer pair that adjusts bit-width dynamically
- **Gradient Quantization**: Quantizes LoRA gradients during backprop using `GradientQuantizer`

**Forward Pass Logic**:
- **32-bit mode**: `x @ weight`
- **Low-bit mode**: `quantize(x) @ quantize(weight) + quantize(x) @ quantize_bits(LoRA_A) @ quantize_bits(LoRA_B)^T x scaling`

#### 2. LoRAAdapter ([cpt_model.py](part2_cyclic_precision_training/cpt_model.py))
Shared low-rank adapter:
- **Single Instance**: One LoRA per linear layer, shared across all precisions
- **Gradient Quantization**: Uses `grad_quantizer_A` and `grad_quantizer_B` for backward pass quantization
- **Structure**: `A (in_features x rank)` and `B (out_features x rank)`
- **Scaling**: `alpha / rank` applied to output

#### 3. LearnableFakeQuantize with Multi-Bit Support ([quantization.py](part2_cyclic_precision_training/quantization.py))
Enhanced quantizer supporting multiple bit-widths:
- **Multi-Bit Calibration**: Stores `scales` and `zero_points` as dictionaries indexed by bit-width
- **Calibrated Bits Tracking**: Maintains `calibrated_bits` set to track which bit-widths have been calibrated
- **Dynamic Bit Switching**: `set_num_bits()` switches active quantization parameters
- **State Persistence**: Custom `state_dict()` and `_load_from_state_dict()` for saving/loading multi-bit parameters

#### 4. GradientQuantizer ([quantization.py](part2_cyclic_precision_training/quantization.py))
Custom autograd function for gradient quantization:
- **Forward**: Pass-through (no-op)
- **Backward**: Quantizes gradients using the associated quantizer
- **Calibration-Aware**: Respects calibration mode and only quantizes if bit-width is calibrated

#### 5. CyclicPrecisionScheduler ([cyclic_scheduler.py](part2_cyclic_precision_training/cyclic_scheduler.py))
Manages precision cycling during training:
- **Schedule Types**:
  - `cosine`: Smooth cosine wave cycling between min and max bits
  - `triangular`: Linear ramp up/down between min and max bits
- **Parameters**:
  - `total_epochs`: Total training epochs
  - `total_cycles`: Number of complete precision cycles
  - `epochs_per_cycle = total_epochs / total_cycles`
- **Precision Mapping**: Rounds continuous precision values to nearest configured bit-width

#### 6. PrecisionRangeTest ([cyclic_scheduler.py](part2_cyclic_precision_training/cyclic_scheduler.py))
Automatically determines optimal precision range:
- **Purpose**: Finds the minimum bit-width (lower bound) where accuracy improvements plateau
- **Method**: Tests incrementing bit-widths from `start_bits`, measures accuracy/loss
- **Threshold-Based**: Stops when accuracy improvement exceeds threshold or plateaus
- **Outputs**: `(lower_bound, upper_bound)` for cyclic training

### Model Hierarchy

```
CPTModel (Top-level model)
|
+-- wte (Token embeddings - trainable)
+-- wpe (Position embeddings - trainable)
+-- h (ModuleList of CPTBlock)
|   |
|   +-- CPTBlock x n_layer (12 blocks for GPT-2 base)
|       |
|       +-- ln_1 (Standard LayerNorm - shared)
|       +-- attn (CPTSelfAttention)
|       |   |
|       |   +-- c_attn (CPTLinear: n_embd -> 3*n_embd for Q,K,V)
|       |   +-- c_proj (CPTLinear: n_embd -> n_embd)
|       |
|       +-- ln_2 (Standard LayerNorm - shared)
|       +-- mlp (ModuleDict)
|           |
|           +-- fc_in (CPTLinear: n_embd -> 4*n_embd)
|           +-- fc_out (CPTLinear: 4*n_embd -> n_embd)
|
+-- ln_f (Standard LayerNorm - shared)
+-- lm_head (CPTLinear: n_embd -> vocab_size, no bias)
```

### Key Features

#### Cyclic Training Schedule
Example with cosine schedule:
```python
scheduler = CyclicPrecisionScheduler(
    bit_widths=[4, 6, 8],
    schedule_type='cosine',
    total_epochs=150,
    total_cycles=15  # 10 epochs per cycle
)

for epoch in range(150):
    precision = scheduler.get_precision_for_epoch(epoch)
    model.set_precision(precision)
    # Train one epoch...
```

Precision varies smoothly: `4 -> 8 -> 4 -> 8 -> ...` over cycles

#### Configuration
From [config_cpt.py](part2_cyclic_precision_training/config_cpt.py):
- **Bit Widths**: `[2, 3, 4, ..., 18, 32]` (18 different precisions supported)
- **Quantizer Type**: Predominantly `log` quantization
- **Shared LoRA**: rank=16, alpha=32 (single adapter for all precisions)
- **Gradient Quantization**: 8-bit gradient quantization enabled
- **Cyclic Parameters**:
  - `total_cycles`: 15
  - `schedule_type`: 'cosine'
  - Target bits: 5

#### Precision Range Testing
Automatic determination of optimal training range:
```python
prt = PrecisionRangeTest(
    model,
    start_bits=2,
    max_bits=18,
    threshold=0.01,
    test_iterations=50,
    target_bits=5
)
lower, upper = prt.find_bounds(dataloader, criterion)
# Train with cyclic schedule between lower and upper bounds
```

### Training Strategy

1. **Calibration Phase**: Calibrate quantizers for all bit-widths in the cycling range
2. **Cyclic Training**:
   - Each epoch uses a different precision based on scheduler
   - Gradients are quantized during backprop
   - Single shared LoRA learns to compensate across all precisions
3. **Multi-Precision Robustness**: Model learns robust representations across precision spectrum
4. **From Scratch Training**: Model is initialized randomly (not from pretrained weights)

### Advantages

1. **Parameter Efficiency**: Single LoRA shared across all bit-widths (vs. separate LoRAs in Part 1)
2. **Robust Quantization**: Cyclic training exposes model to various precision levels, improving generalization
3. **Simplified Architecture**: Standard LayerNorm, no precision-specific parameters
4. **Gradient Quantization**: Reduces memory for gradient storage during training
5. **Automatic Range Selection**: PrecisionRangeTest finds optimal precision bounds
6. **Flexible Scheduling**: Support for cosine and triangular cycling patterns
7. **Wide Bit-Width Support**: Can handle 18 different precision levels (2-18 bits + 32-bit)

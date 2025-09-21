# Test Suite Organization

## Overview
This test folder contains a comprehensive suite for testing the SP Model with multi-precision support, quantization, batch normalization, and distillation capabilities.

## Test Structure

### Main Test Runner
- **run_tests.py** (in parent directory) - Main entry point for running all tests
- **debug_sp_model.py** - Comprehensive test orchestrator

### Core Test Modules

#### 1. test_precision_mismatch.py
Tests for precision-related issues:
- Precision consistency across different bit widths
- Layer-wise precision analysis
- Quantization saturation detection
- Numerical stability checks

#### 2. test_batchnorm_effects.py
Batch normalization behavior tests:
- Statistics tracking per precision
- Gradient flow through BN layers
- Train/eval mode switching
- Small batch size handling
- Precision switching consistency

#### 3. test_training_dynamics.py
Training and optimization tests:
- Multi-batch training with distillation
- Quantization-aware training (QAT)
- Distillation effectiveness with temperature tuning
- Gradient accumulation effects
- Batch norm training dynamics

### Utility Modules

#### Dataset and Calculation
- **dataset_utils.py** - Dataset loading and preprocessing
- **calculate_perplexity_chunked.py** - Efficient perplexity calculation

#### Model Utilities
- **fix_model_initialization.py** - Proper model initialization

#### Standalone Tools
- **test_quantizer_methods.py** - Quantization method testing and comparison

## Running Tests

### From Parent Directory
```bash
# Run all tests
python run_tests.py

# Run specific suite
python run_tests.py --suite basic
python run_tests.py --suite precision
python run_tests.py --suite batchnorm
python run_tests.py --suite training

# Quick mode (reduced samples)
python run_tests.py --quick
```

### From Test Directory
```bash
# Run comprehensive debug suite
python debug_sp_model.py

# Run individual test modules
python test_precision_mismatch.py
python test_batchnorm_effects.py
python test_training_dynamics.py
```

## Test Categories

### Basic Tests
- 32-bit teacher equivalence with GPT-2
- Quantization degradation analysis
- LoRA adapter behavior
- Quantizer activation verification

### Precision Tests
- Cross-precision consistency
- Layer-wise precision effects
- Quantization saturation levels
- Numerical stability monitoring

### Batch Norm Tests
- Per-precision statistics tracking
- Gradient flow analysis
- Mode switching behavior
- Small batch handling

### Training Tests
- Knowledge distillation effectiveness
- Quantization-aware training
- Gradient accumulation strategies
- Multi-precision training dynamics

## Key Features

1. **No hasattr Usage**: All attribute checks use try/except blocks
2. **Modular Design**: Each test module is independent
3. **Comprehensive Coverage**: Tests cover all aspects of multi-precision models
4. **Clean Organization**: Separated concerns with focused test files
5. **Detailed Reporting**: Rich output with status indicators

## Output Interpretation

### Status Indicators
- ✅ Test passed successfully
- ⚠️ Test passed with warnings
- ❌ Test failed
- 📊 Analysis/statistics section
- 🔧 Configuration/setup section
- 🔍 Detailed inspection section

### Metrics
- **Perplexity (PPL)**: Lower is better
- **Cosine Similarity**: Closer to 1.0 is better
- **MSE**: Lower is better
- **Relative Error**: Lower is better

## Notes

- All tests handle GPU/CPU automatically
- Memory cleanup between heavy tests
- Progress indicators for long-running tests
- Detailed error messages for debugging
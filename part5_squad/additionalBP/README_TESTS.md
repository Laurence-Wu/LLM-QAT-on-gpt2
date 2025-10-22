# Test Suite for BF16/FP16 Precision Conversion

Comprehensive test suite for the `set_precision()` method and FP16/BF16 evaluation pipeline.

## Overview

This test suite validates the precision conversion functionality that enables:
- **FP16 mode**: `model.set_precision(16.0)` → Cast to `torch.float16`
- **BF16 mode**: `model.set_precision(16.5)` → Cast to `torch.bfloat16`
- **FP32 mode**: `model.set_precision(32)` → Full precision `torch.float32`
- **INT quantization**: `model.set_precision(4, 8, 16, ...)` → Integer quantization

## Test Files

### 1. `test_precision_conversion.py` (Unit Tests)

**Purpose**: Test the core `set_precision()` functionality across all model components.

**Tests included** (12 tests):

| Test | Description | What it validates |
|------|-------------|-------------------|
| `test_fp16_conversion` | FP16 flag conversion | `set_precision(16.0)` casts model to float16 |
| `test_bf16_conversion` | BF16 flag conversion | `set_precision(16.5)` casts model to bfloat16 |
| `test_fp32_conversion` | FP32 mode | `set_precision(32)` maintains float32 |
| `test_int_quantization` | INT quantization activation | Integer values activate quantizers |
| `test_dtype_consistency` | Dtype uniformity | All components match expected dtype |
| `test_forward_fp16` | FP16 forward pass | Forward pass works in FP16 |
| `test_forward_bf16` | BF16 forward pass | Forward pass works in BF16 |
| `test_backward_fp16` | FP16 backward pass | Gradients computed correctly in FP16 |
| `test_backward_bf16` | BF16 backward pass | Gradients computed correctly in BF16 |
| `test_precision_switching` | Rapid switching | Switch precisions multiple times |
| `test_lora_bypass_fp16_bf16` | LoRA bypass | LoRA correctly bypassed in FP16/BF16 |
| `test_quantizer_access_protection` | Safe access | Quantizers not accessed for 16.0/16.5 |

**Runtime**: ~10-15 seconds

**Usage**:
```bash
python test_precision_conversion.py
```

---

### 2. `test_eval_pipeline.py` (Integration Tests)

**Purpose**: Test the complete evaluation workflow including checkpoint loading, precision conversion, and dataset evaluation.

**Tests included** (10 tests):

| Test | Description | What it validates |
|------|-------------|-------------------|
| `test_load_checkpoint` | Checkpoint loading | Checkpoints load correctly |
| `test_fp32_evaluation_smoke` | FP32 evaluation | FP32 evaluation pipeline works |
| `test_fp16_evaluation_smoke` | FP16 evaluation | FP16 evaluation pipeline works |
| `test_bf16_evaluation_smoke` | BF16 evaluation | BF16 evaluation pipeline works |
| `test_dataset_version_detection` | Dataset version | `dataset.version` attribute works |
| `test_answer_extraction` | Answer extraction | Answer spans extracted correctly |
| `test_dtype_propagation` | Dtype flow | Dtypes propagate through pipeline |
| `test_memory_cleanup` | Memory management | Models properly deleted |
| `test_precision_numerical_similarity` | Numerical similarity | FP16/BF16 close to FP32 |
| `test_evaluation_config_loading` | Config loading | evaluation_config.json loads |

**Runtime**: ~20-30 seconds (may download SQuAD on first run)

**Usage**:
```bash
python test_eval_pipeline.py
```

---

### 3. `test_precision_stress.py` (Stress Tests)

**Purpose**: Test edge cases, numerical stability, and intensive scenarios.

**Tests included** (11 tests):

| Test | Description | What it validates |
|------|-------------|-------------------|
| `test_rapid_precision_switching` | 1000 rapid switches | Switching is fast and stable |
| `test_large_model_bf16` | Large (6-layer) model | BF16 works with larger models |
| `test_gradient_accumulation_fp16` | Multi-step gradients | Gradient accumulation in FP16 |
| `test_nan_detection` | NaN detection | No NaN/Inf in normal operation |
| `test_numerical_stability` | Consistency | Multiple runs produce same results |
| `test_attention_overflow` | Attention stability | Attention doesn't overflow |
| `test_layernorm_precision` | LayerNorm stability | LayerNorm stable in reduced precision |
| `test_checkpoint_save_load_fp16` | FP16 persistence | Save/load FP16 checkpoints |
| `test_checkpoint_save_load_bf16` | BF16 persistence | Save/load BF16 checkpoints |
| `test_precision_with_gradients` | Gradient correctness | FP16 gradients similar to FP32 |
| `test_dtype_mismatch_handling` | Error handling | Dtype mismatches handled gracefully |

**Runtime**: ~30-60 seconds

**Usage**:
```bash
python test_precision_stress.py
```

---

## Quick Start

### Run All Tests

```bash
cd additionalBP
python run_additionalBP_tests.py
```

**Expected output**:
```
======================================================================
                      additionalBP Test Suite
======================================================================

Running 3 test file(s)
Start time: 2025-10-22 14:30:00

======================================================================
Running test_precision_conversion.py
----------------------------------------------------------------------
...
✅ test_precision_conversion.py passed (12.3s)

======================================================================
Running test_eval_pipeline.py
----------------------------------------------------------------------
...
✅ test_eval_pipeline.py passed (25.7s)

======================================================================
Running test_precision_stress.py
----------------------------------------------------------------------
...
✅ test_precision_stress.py passed (45.2s)

======================================================================
                           Test Summary
======================================================================

Results:
  ✅ PASS  test_precision_conversion.py       (12.3s)
  ✅ PASS  test_eval_pipeline.py              (25.7s)
  ✅ PASS  test_precision_stress.py           (45.2s)

Total: 3/3 passed, 0/3 failed
Total duration: 83.2s
End time: 2025-10-22 14:31:23

🎉 All tests passed!
```

---

## Advanced Usage

### Run Specific Test File

```bash
# Run only unit tests
python run_additionalBP_tests.py --test precision_conversion

# Run only integration tests
python run_additionalBP_tests.py --test eval_pipeline

# Run only stress tests
python run_additionalBP_tests.py --test stress
```

### Run with Verbose Output

```bash
python run_additionalBP_tests.py --verbose
```

### Run Individual Test File

```bash
# Run unit tests directly
python test_precision_conversion.py

# Run integration tests directly
python test_eval_pipeline.py

# Run stress tests directly
python test_precision_stress.py
```

---

## Test Coverage

### What is Tested

✅ **Precision Conversion**
- FP16 (16.0 flag) conversion
- BF16 (16.5 flag) conversion
- FP32 (32 flag) mode
- INT quantization (4, 8, 16 flags)

✅ **Model Components**
- SPModel (transformer)
- SPBlock (transformer blocks)
- SPAttention (attention layers)
- SPMLP (MLP layers)
- SPLinearWithLoRA (quantized linear + LoRA)
- SwitchableLayerNorm (layer normalization)
- SPQuestionAnsweringModel (QA heads)

✅ **Forward/Backward Passes**
- Forward pass in FP16/BF16/FP32
- Backward pass with gradient computation
- Loss computation
- Gradient accumulation

✅ **Numerical Properties**
- Dtype consistency
- NaN/Inf detection
- Numerical stability
- Relative error vs FP32

✅ **Evaluation Pipeline**
- Checkpoint loading
- Dataset version detection
- Answer extraction
- Metric computation
- Output formatting

✅ **Edge Cases**
- Rapid precision switching (1000+ switches)
- Large models
- Long sequences
- Checkpoint save/load
- Memory cleanup

### What is NOT Tested

❌ **Actual SQuAD Dataset Evaluation**
- Tests use mock data to avoid long runtimes
- Use `eval_squad_fp.py` for full evaluation

❌ **Multi-GPU / Distributed**
- Tests run on single device (CPU or single GPU)

❌ **Quantization-Aware Training**
- Tests focus on evaluation/inference
- Training with FP16/BF16 not covered

---

## Interpreting Test Results

### All Tests Pass (✅)

**Example output**:
```
✓ FP16 conversion works correctly
✓ BF16 conversion works correctly
...
✅ All tests passed!
```

**Action**: None needed. Implementation is correct.

---

### Some Tests Fail (❌)

**Example output**:
```
❌ Dtype mismatches found:
   Parameter transformer.h.0.attn.c_attn.linear.weight: torch.float32 != torch.float16
```

**Possible causes**:
1. **Incomplete dtype casting**: Model components not fully converted
2. **Buffer dtype**: Some buffers not cast correctly
3. **Precision flag not propagated**: set_precision() not called on all components

**Action**:
1. Check implementation of `set_precision()` in affected component
2. Verify all model components are converted
3. Check for hardcoded dtypes in forward pass

---

### Tests Warn (⚠️)

**Example output**:
```
⚠️ FP16 relative difference = 0.0672 (>5%), might indicate instability
```

**Meaning**: Test passed but found potential issues.

**Common warnings**:
- **High relative error**: FP16/BF16 output differs significantly from FP32
  - Usually acceptable if < 10%
  - Check for numerical instability if > 10%

- **BF16 not supported**: GPU doesn't support BF16
  - Normal on older GPUs
  - Test skipped automatically

- **Model IDs same**: Memory reuse detected
  - Usually not a problem
  - Just indicates Python reused memory address

**Action**: Review warnings but don't necessarily fix if tests pass.

---

### Test Crashes (💥)

**Example output**:
```
❌ Test crashed: KeyError: '16.0bit'
```

**Possible causes**:
1. **Quantizer access**: Code tries to access quantizers for 16.0/16.5
2. **Missing implementation**: set_precision() not implemented for some component
3. **Import error**: Missing dependencies or module path issues

**Action**:
1. Check stack trace for error location
2. Verify implementation matches plan
3. Check for typos in precision flags (16.0 vs "16.0")

---

## Common Issues

### Issue 1: BF16 Tests Skipped

**Symptom**:
```
⚠️ BF16 not supported on this device, skipping
```

**Cause**: GPU doesn't support BF16 (requires Ampere or newer NVIDIA GPU)

**Solution**:
- Normal behavior on older GPUs
- BF16 tests will be skipped
- All other tests should still pass

---

### Issue 2: NaN in FP16

**Symptom**:
```
❌ NaN detected in start_logits
```

**Cause**: Numerical overflow/underflow in FP16

**Solutions**:
1. Check attention scaling (`1/sqrt(head_dim)`)
2. Verify LayerNorm epsilon (should be ~1e-5)
3. Check for very large or very small weight values
4. Consider gradient scaling if training

---

### Issue 3: Import Errors

**Symptom**:
```
ModuleNotFoundError: No module named 'part5_squad'
```

**Cause**: Python path not set correctly

**Solution**:
```bash
# Run from additionalBP directory
cd part5_squad/additionalBP
python run_additionalBP_tests.py

# Or set PYTHONPATH
export PYTHONPATH=/path/to/LLM-QAT-on-gpt2:$PYTHONPATH
python run_additionalBP_tests.py
```

---

### Issue 4: SQuAD Download Fails

**Symptom**:
```
Failed to download SQuAD dataset
```

**Cause**: No internet connection or HuggingFace datasets error

**Solution**:
- Most tests use mock data and will still pass
- Only `test_dataset_version_detection` requires real SQuAD
- Can be skipped if needed

---

## Adding New Tests

### Adding a Test to Existing File

1. Define test function:
```python
def test_my_new_feature():
    """Test description"""
    print("Testing my new feature...")

    # Test implementation
    try:
        # Your test code here
        assert some_condition

        print("  ✓ Test passed")
        return True
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False
```

2. Add to test list in `main()`:
```python
tests = [
    # ... existing tests
    test_my_new_feature,
]
```

### Creating New Test File

1. Create file `test_my_feature.py`
2. Follow structure of existing test files
3. Add to `run_additionalBP_tests.py`:
```python
all_test_files = [
    'test_precision_conversion.py',
    'test_eval_pipeline.py',
    'test_precision_stress.py',
    'test_my_feature.py',  # Add here
]
```

---

## Performance Benchmarks

**Typical runtimes on M1 MacBook Pro (CPU)**:

| Test File | Tests | Duration | Status |
|-----------|-------|----------|--------|
| test_precision_conversion.py | 12 | ~12s | Fast |
| test_eval_pipeline.py | 10 | ~25s | Medium |
| test_precision_stress.py | 11 | ~45s | Slow |
| **Total** | **33** | **~82s** | - |

**On CUDA GPU (RTX 3090)**:

| Test File | Tests | Duration | Status |
|-----------|-------|----------|--------|
| test_precision_conversion.py | 12 | ~8s | Fast |
| test_eval_pipeline.py | 10 | ~15s | Medium |
| test_precision_stress.py | 11 | ~30s | Slow |
| **Total** | **33** | **~53s** | - |

---

## Debugging Failed Tests

### Enable Verbose Mode

```bash
python test_precision_conversion.py --verbose  # If supported
# or
python run_additionalBP_tests.py --verbose
```

### Check Individual Test

```python
# Edit test file, run specific test
if __name__ == '__main__':
    test_fp16_conversion()  # Run just this one
```

### Add Debug Prints

```python
def test_fp16_conversion():
    model.set_precision(16.0)

    # Add debug
    print(f"DEBUG: current_bit_width = {model.transformer.current_bit_width}")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.dtype}")
```

### Use Python Debugger

```bash
python -m pdb test_precision_conversion.py
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Test BF16/FP16 Conversion

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2
    - uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install torch transformers datasets tqdm

    - name: Run tests
      run: |
        cd part5_squad/additionalBP
        python run_additionalBP_tests.py
```

---

## Best Practices

### Before Committing Code

1. **Run all tests**: `python run_additionalBP_tests.py`
2. **Check for warnings**: Review any ⚠️  messages
3. **Verify on both CPU and GPU** (if available)
4. **Test on different Python versions** (3.8, 3.9, 3.10, 3.11)

### When Adding Features

1. **Write test first** (TDD approach)
2. **Add to appropriate test file**:
   - Unit tests → `test_precision_conversion.py`
   - Integration tests → `test_eval_pipeline.py`
   - Stress tests → `test_precision_stress.py`
3. **Update this README** with new test description

### When Debugging

1. **Isolate the failure**: Run only failing test
2. **Add debug prints**: Temporary print statements
3. **Check assumptions**: Verify expected vs actual values
4. **Bisect**: Comment out parts of test to find exact failure point

---

## FAQ

**Q: Why do some tests get skipped?**
A: Tests that require unsupported hardware (e.g., BF16 on old GPUs) are automatically skipped.

**Q: How long should tests take?**
A: Full suite ~1-2 minutes on modern hardware. Individual files 10-60 seconds.

**Q: Can I run tests without GPU?**
A: Yes! All tests work on CPU. GPU just makes them faster.

**Q: Why are there warnings about numerical differences?**
A: FP16/BF16 have less precision than FP32. Small differences are expected and acceptable.

**Q: Do tests require internet?**
A: Mostly no. Only dataset version test tries to download SQuAD, but even that can be skipped.

**Q: Can I disable specific tests?**
A: Yes, comment out the test in the `tests` list in `main()`.

---

## Conclusion

This test suite provides comprehensive coverage of the BF16/FP16 precision conversion functionality. Regular testing ensures that:

✅ Precision conversion works correctly
✅ All model components support FP16/BF16
✅ Forward and backward passes are stable
✅ Evaluation pipeline functions properly
✅ Edge cases are handled gracefully

**For questions or issues**, please check the main README or create an issue.

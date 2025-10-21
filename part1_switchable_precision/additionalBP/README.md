# FP16/BF16 Evaluation for SQuAD QA Models

This directory contains evaluation scripts for comparing the performance of SQuAD Question Answering models across different floating-point precisions: FP32 (baseline), FP16, and BF16.

## Overview

The evaluation script loads trained Part 5 SQuAD QA checkpoints (FP32/32-bit models), casts them to different precisions (FP16 and BF16), and evaluates their performance on both SQuAD v1.1 and v2.0 datasets. This helps understand the impact of reduced precision on QA task performance.

## Purpose

- **Precision Comparison**: Compare F1 and Exact Match scores across FP32, FP16, and BF16 precisions
- **Deployment Insights**: Understand accuracy trade-offs when deploying models at reduced precision
- **Hardware Optimization**: Evaluate potential speedups from FP16/BF16 on compatible hardware
- **Numerical Stability**: Identify potential numerical issues with lower precision formats

## Files

- `eval_squad_fp.py` - Main evaluation script
- `evaluation_config.json` - Configuration for evaluation (dataset splits, max answer length, etc.)
- `README.md` - This documentation file

## Requirements

The script uses the existing project dependencies from Part 5. Ensure you have:

- PyTorch with CUDA support (for GPU evaluation)
- Transformers library
- Datasets library (for SQuAD)
- tqdm for progress bars

For BF16 evaluation, you need:
- A GPU that supports BF16 (e.g., NVIDIA Ampere or newer)
- PyTorch 1.10 or later

## Usage

### Basic Usage

Evaluate a Part 5 SQuAD checkpoint at all precisions (FP32, FP16, BF16):

```bash
cd part1_switchable_precision/additionalBP
python eval_squad_fp.py --checkpoint ../../part5_squad/squad_qa_32bit_FP32_20241021_123456.pth
```

### Advanced Options

Skip FP16 evaluation:
```bash
python eval_squad_fp.py --checkpoint <checkpoint_path> --skip-fp16
```

Skip BF16 evaluation:
```bash
python eval_squad_fp.py --checkpoint <checkpoint_path> --skip-bf16
```

Evaluate only on SQuAD v1.1:
```bash
python eval_squad_fp.py --checkpoint <checkpoint_path> --squad-v1-only
```

Evaluate only on SQuAD v2.0:
```bash
python eval_squad_fp.py --checkpoint <checkpoint_path> --squad-v2-only
```

## Configuration

Edit `evaluation_config.json` to customize evaluation settings:

```json
{
  "device": "cuda",           // "cuda" or "cpu"
  "squad_v1": {
    "split": "validation",     // Dataset split to use
    "max_answer_length": 30,   // Maximum answer span length
    "n_best_size": 20,         // Number of candidate positions
    "max_examples": null       // null for all examples, or integer to limit
  },
  "squad_v2": {
    "split": "validation",
    "max_answer_length": 30,
    "n_best_size": 20,
    "max_examples": null
  }
}
```

## Output

### Console Output

The script prints detailed progress and results:

1. **Loading Phase**: Shows checkpoint loading and model initialization
2. **Evaluation Progress**: Progress bars for each precision and dataset
3. **Individual Results**: F1 and EM scores for each (precision, dataset) pair
4. **Summary Table**: Comparison table showing all results

Example summary:
```
======================================================================
EVALUATION SUMMARY
======================================================================
Model: ../../part5_squad/squad_qa_32bit_FP32_20241021_123456.pth
Device: cuda

SQuAD v1.1:
Precision    Exact Match     F1 Score
----------------------------------------------------------------------
FP32          78.50%          85.20%
FP16          78.45%          85.15%
BF16          78.48%          85.18%

SQuAD v2.0:
Precision    Exact Match     F1 Score
----------------------------------------------------------------------
FP32          75.20%          82.10%
FP16          75.15%          82.05%
BF16          75.18%          82.08%
======================================================================
```

### JSON Output

Results are automatically saved to `fp_comparison_results_<timestamp>.json`:

```json
{
  "model_path": "../../part5_squad/squad_qa_32bit_FP32_20241021_123456.pth",
  "timestamp": "2025-10-21 14:30:00",
  "device": "cuda",
  "squad_v1": {
    "fp32": {
      "f1": 85.2,
      "exact_match": 78.5,
      "total": 10570
    },
    "fp16": {
      "f1": 85.15,
      "exact_match": 78.45,
      "total": 10570
    },
    "bf16": {
      "f1": 85.18,
      "exact_match": 78.48,
      "total": 10570
    }
  },
  "squad_v2": {
    "fp32": {
      "f1": 82.1,
      "exact_match": 75.2,
      "total": 11873
    },
    "fp16": {
      "f1": 82.05,
      "exact_match": 75.15,
      "total": 11873
    },
    "bf16": {
      "f1": 82.08,
      "exact_match": 75.18,
      "total": 11873
    }
  }
}
```

## Technical Details

### Precision Casting

1. **FP32 (Baseline)**: Model loaded at original precision (no casting)
2. **FP16**: Model cast using `.half()` - 16-bit floating point
   - Faster on modern GPUs
   - Reduced memory usage
   - Potential numerical instability
   - Range: ~±65,504, precision: ~3 decimal digits
3. **BF16**: Model cast using `.bfloat16()` - Brain Float 16
   - Similar speed to FP16 on supported hardware
   - Same range as FP32 but reduced precision
   - Better numerical stability than FP16
   - Range: ~±3.4×10^38, precision: ~2 decimal digits

### Evaluation Process

For each precision:
1. Load checkpoint at FP32
2. Cast model to target precision (if not FP32)
3. Set model to eval mode
4. Run forward pass with inputs on same device
5. Cast logits back to FP32 for stable metric computation
6. Extract answer spans using beam search
7. Compute F1 and Exact Match scores

### Memory Management

The script includes automatic memory management:
- Models are deleted after each precision evaluation
- `torch.cuda.empty_cache()` called between evaluations
- Out-of-memory errors are caught and logged

## Interpreting Results

### Expected Behavior

- **FP32**: Baseline performance, highest accuracy
- **FP16**: Usually very close to FP32, possible minor degradation
  - If significant degradation (>1% F1), may indicate numerical instability
- **BF16**: Performance typically between FP32 and FP16
  - Better stability than FP16 due to larger exponent range
  - Slightly lower precision than FP16

### Warning Signs

- **Large F1 drop (>2%)**: Indicates numerical issues, check model architecture
- **Unstable predictions**: May need gradient scaling or loss scaling
- **NaN/Inf values**: Check for overflow in attention computations

### Deployment Recommendations

- **Production**: Use FP32 if accuracy is critical
- **Speed-optimized**: Use BF16 on supported hardware (Ampere+)
- **Memory-constrained**: Use FP16 with careful validation

## Troubleshooting

### BF16 Not Supported

If you see "BF16 not supported on this GPU":
- Your GPU doesn't support BF16 (requires Ampere or newer)
- Use `--skip-bf16` to skip BF16 evaluation

### CUDA Out of Memory

If evaluation fails with OOM:
- Reduce batch size in evaluation loop (currently 1, hardcoded)
- Use CPU evaluation: edit `evaluation_config.json` to set `"device": "cpu"`
- Limit examples: set `"max_examples": 1000` in config

### Import Errors

If you get module import errors:
- Ensure you're running from the correct directory
- Check that Part 5 modules are available
- Verify Python path includes project root

## Example Workflow

1. Train a SQuAD QA model in Part 5:
```bash
cd part5_squad
python main_squad.py
```

2. Locate the checkpoint (e.g., `squad_qa_32bit_FP32_20241021_123456.pth`)

3. Run FP16/BF16 evaluation:
```bash
cd ../part1_switchable_precision/additionalBP
python eval_squad_fp.py --checkpoint ../../part5_squad/squad_qa_32bit_FP32_20241021_123456.pth
```

4. Review results in console output and JSON file

5. Compare across different checkpoints or training configurations

## Notes

- Evaluation is performed on the validation split by default
- The script loads the full SQuAD dataset unless `max_examples` is set
- For reproducibility, use the same random seed across runs
- BF16 evaluation may be skipped automatically if hardware doesn't support it
- All metrics are computed using the official SQuAD evaluation script

## Citation

If you use this evaluation script in your research, please cite the relevant papers:
- SQuAD: Rajpurkar et al., "SQuAD: 100,000+ Questions for Machine Comprehension of Text" (2016)
- SQuAD 2.0: Rajpurkar et al., "Know What You Don't Know: Unanswerable Questions for SQuAD" (2018)

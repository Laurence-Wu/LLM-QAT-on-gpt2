# Part 3: Switchable Precision (SP) Model Evaluation System

## Overview

The Part 3 evaluation system provides a comprehensive framework for evaluating quantized language models trained with Switchable Precision (SP) or Cyclic Precision Training (CPT) methods. It measures model quality across multiple dimensions: perplexity, zero-shot reasoning, model size, and compression ratio.

This evaluation suite is designed for **academic research and model quality assessment**, focusing on comparing different quantization configurations against baseline models.

## What Makes This Evaluation Special

### 1. Sliding Window Perplexity Evaluation
Unlike simple truncation-based perplexity, this uses a **stride-based sliding window** approach:
- Handles sequences longer than model's context window
- Overlapping windows with configurable stride (default: 256 tokens)
- Only counts loss in non-overlapping regions to avoid double-counting
- More accurate perplexity on long-form text (WikiText2, WikiText103, C4)

```
Input: [=====================================]
Window 1:  [--------]
Window 2:       [--------]
Window 3:            [--------]
                ↑ Only count these regions for loss
```

### 2. Quantizer Calibration Verification
Before evaluation, the system **verifies quantizers are properly calibrated**:
- Checks all input/weight quantizers have calibrated scale/zero_point
- Warns if any quantizers are uncalibrated (would cause NaN/Inf)
- Prevents wasted compute time on broken models
- Critical for low-bit (2-4 bit) quantization

### 3. Multi-Dimensional Zero-Shot Suite
Evaluates on **6 diverse common sense reasoning tasks**:

| Task | Description | Metric |
|------|-------------|--------|
| **BoolQ** | Yes/No questions about passages | Accuracy |
| **HellaSwag** | Sentence completion (4 choices) | Accuracy |
| **WinoGrande** | Pronoun resolution | Accuracy|


Average across all 6 tasks provides robust measure of reasoning capability.

### 4. Mixed-Precision Configuration Support

Supports separate quantization for **Weights (W) and Activations (A)**:
- Weight quantization (W): 2-32 bits
- Activation quantization (A): 2-32 bits

Multiple predefined configurations for different compression-accuracy tradeoffs.

### 5. Automatic Publication-Ready Tables
Generates results in **3 formats automatically**:
- **ASCII tables** (console output with tabulate)
- **LaTeX tables** (ready for papers)
- **Markdown tables** (for GitHub/documentation)

### 6. Model Size Calculation

Calculates model size based on weight quantization:
```
Model Size = (# parameters × weight_bits) / 8 / 1024 MB
```

Compression is achieved through weight quantization (2-32 bits).

## Evaluation Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Load Model & Configuration                               │
│    - Load checkpoint (SP or CPT model)                      │
│    - Extract bit_widths, model config                       │
│    - Load evaluation_config.json                            │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Verify Quantizer Calibration (if quantized)             │
│    - Check all quantizers have scale/zero_point            │
│    - Warn if uncalibrated (potential NaN/Inf)              │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. [Optional] Run Diagnostic Tests                          │
│    - Test perplexity on small sample                        │
│    - Check for quantization degradation                     │
│    - Track batch-level metrics                              │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Perplexity Evaluation                                    │
│    - WikiText2 (test set, ~3K samples)                      │
│    - C4 (validation, 5K samples)                            │
│    - [Optional] WikiText103 (test set)                      │
│    - Uses sliding window with stride=256                    │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Zero-Shot Task Evaluation                                │
│    - BoolQ, HellaSwag, WinoGrande                          │
│    - ARC-Easy, ARC-Challenge, OpenBookQA                   │
│    - Max 500 samples per task (configurable)               │
│    - Calculate average across all tasks                     │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Generate Results                                         │
│    - Calculate model size & compression ratio              │
│    - Save JSON results                                      │
│    - Generate ASCII/LaTeX/Markdown tables                   │
│    - Compare with baselines (if available)                  │
└─────────────────────────────────────────────────────────────┘
```

## File Descriptions

### Core Evaluation Files

**main_sp_eval.py**
- Entry point for evaluation
- Orchestrates all evaluation components
- Handles model loading and calibration verification
- Command-line interface with `--model_path`, `--eval_config`, `--diagnose`

**perplexity_eval.py**
- PerplexityEvaluator class
- Sliding window perplexity calculation
- Supports WikiText2, WikiText103, C4 datasets
- Stride-based evaluation to handle long contexts

**zero_shot_tasks.py**
- ZeroShotEvaluator class
- Implements 6 common sense reasoning tasks
- Likelihood-based choice selection (no generation)
- Error-resilient evaluation with configurable max_errors

**sp_metrics.py**
- SPEvaluation class
- Model size calculation (weights + KV cache)
- Compression ratio calculation
- Bit configuration application to models

### Configuration & Utilities

**bit_configurations.py**
- BitConfigurations class
- 13 predefined quantization configs
- Config validation and application
- Compression ratio calculator

**generate_tables.py**
- ResultTableGenerator class
- Generates Table 1 (Zero-Shot Results)
- Generates Table 2 (Perplexity Results)
- Exports to ASCII, LaTeX, Markdown formats

**baseline_comparison.py**
- BaselineComparison class
- Compares results against FP16 baseline
- Calculates degradation metrics
- Accuracy vs bits tradeoff analysis

**evaluation_config.json**
- Centralized configuration file
- Dataset settings (names, splits, max_samples)
- Generation parameters (temperature, max_length)
- Perplexity settings (stride, max_length)
- Output directory and format

## Bit Configurations

### Standard Configurations

| Config Name | W | A | Description | Compression vs FP32 |
|-------------|---|---|-------------|---------------------|
| **FP32** | 32 | 32 | Full precision teacher | 1.0x |
| **FP16** | 16 | 16 | Half precision baseline | 2.0x |
| **INT8** | 8 | 8 | 8-bit quantization | 4.0x |
| **INT6** | 6 | 6 | 6-bit quantization | 5.3x |
| **INT4** | 4 | 4 | 4-bit quantization | 8.0x |
| **INT2** | 2 | 2 | Extreme 2-bit quantization | 16.0x |

### Mixed-Precision Configurations

| Config Name | W | A | Use Case |
|-------------|---|---|----------|
| **W4A8** | 4 | 8 | Aggressive weight compression |
| **W4A16** | 4 | 16 | Weights-only quantization |
| **W8A8** | 8 | 8 | Standard 8-bit quantization |
| **W4A6** | 4 | 6 | Mixed precision experimentation |
| **W2A16** | 2 | 16 | Extreme weight quantization |
| **W3A8** | 3 | 8 | 3-bit weight exploration |

**Note:** W = Weights, A = Activations

## Metrics Explained

### 1. Perplexity (↓ lower is better)

**Definition:** Perplexity = exp(average cross-entropy loss)

**Interpretation:**
- Measures how "surprised" the model is by the text
- Lower perplexity = better language modeling
- Typical ranges:
  - FP32 GPT-2: ~18-22 on WikiText2
  - 8-bit: ~19-25 (minimal degradation)
  - 4-bit: ~25-35 (moderate degradation)
  - 2-bit: ~50-100+ (severe degradation)

**Calculation:**
```python
total_loss = sum(cross_entropy(predictions, targets))
avg_loss = total_loss / num_tokens
perplexity = exp(avg_loss)
```

### 2. Zero-Shot Accuracy (↑ higher is better)

**Definition:** Percentage of correctly answered multiple-choice questions without fine-tuning

**Evaluation Method:**
1. Format question as text prompt
2. Calculate likelihood P(choice | prompt) for each choice
3. Select choice with highest likelihood
4. Compare to ground truth

**Example (BoolQ):**
```
Prompt: "Passage: [text]
         Question: Is X true?
         Answer:"
Choices: [" True", " False"]
Model predicts: P(" True") = 0.73, P(" False") = 0.27
→ Prediction: True
```

### 3. Model Size (MB/GB)

**Calculation:**
```
Model Size = (num_parameters × weight_bits) / 8 / 1024 MB
```

**Example (GPT-2 124M):**
- FP32: ~497 MB
- INT8: ~124 MB
- W4A8: ~62 MB

### 4. Compression Ratio

**Definition:** Size reduction vs baseline (typically FP32 or FP16)

```
Compression Ratio = baseline_bits / quantized_bits
```

**Example:**
- FP32 → INT8: 32/8 = 4.0x compression
- FP32 → W4A8: 32/4 = 8.0x compression (weight-only)

## Usage Examples

### Basic Evaluation

```bash
python main_sp_eval.py \
  --model_path path/to/checkpoint.pth \
  --eval_config evaluation_config.json
```

### With Diagnostics

```bash
python main_sp_eval.py \
  --model_path path/to/checkpoint.pth \
  --eval_config evaluation_config.json \
  --diagnose
```

This runs additional diagnostic tests before evaluation:
- Small-sample perplexity test
- Logits mean/std check
- Quantization health check
- Warns about potential issues (NaN, Inf, extreme values)

### Evaluating Multiple Configurations

To evaluate multiple bit configurations, run separately for each:

```bash
# Evaluate at 8-bit
python main_sp_eval.py --model_path model_8bit.pth --eval_config eval_config.json

# Evaluate at 4-bit
python main_sp_eval.py --model_path model_4bit.pth --eval_config eval_config.json

# Evaluate at 6-bit
python main_sp_eval.py --model_path model_6bit.pth --eval_config eval_config.json
```

Then use `generate_tables.py` to compare results.

## Configuration Guide

### evaluation_config.json Structure

```json
{
  "device": "cuda",

  "perplexity": {
    "stride": 256,           // Sliding window stride
    "max_length": 256,       // Context window size
    "max_samples": 500,      // Max samples per dataset
    "datasets": {
      "WikiText2": {...},
      "C4": {...}
    }
  },

  "zero_shot": {
    "max_samples": 500,      // Max samples per task
    "max_errors": 10,        // Stop if too many errors
    "datasets": {
      "BoolQ": {...},
      "HellaSwag": {...},
      ...
    }
  },

  "output": {
    "directory": "part3_evaluation/results",
    "save_format": "json"
  }
}
```

### Key Parameters

**perplexity.stride**
- How many tokens to advance between windows
- Lower = more accurate but slower
- Default: 256 tokens

**perplexity.max_length**
- Context window size for each evaluation step
- Should match model's n_positions or smaller
- Default: 256 tokens

**zero_shot.max_samples**
- Maximum samples to evaluate per task
- Higher = more accurate but slower
- Default: 500 (balances accuracy and speed)

**zero_shot.max_errors**
- Stop evaluation if too many errors occur
- Prevents wasting time on broken models
- Default: 10 errors

## Output Files

After evaluation, the following files are generated in the output directory:

### 1. JSON Results
```
part3_evaluation/results/results_8bit_20231215_143022.json
```

Contains:
```json
{
  "bit_width": 8,
  "model_size_gb": 0.124,
  "compression_ratio": 4.0,
  "perplexity": {
    "WikiText2": 21.3,
    "C4": 23.7
  },
  "zero_shot": {
    "BoolQ": 62.5,
    "HellaSwag": 31.2,
    "WinoGrande": 51.4,
    "ARC-e": 48.9,
    "ARC-c": 25.1,
    "OBQA": 28.4,
    "Average": 41.3
  }
}
```

### 2. ASCII Tables
```
part3_evaluation/results/table1_zero_shot.txt
part3_evaluation/results/table2_perplexity.txt
```

Console-formatted tables for quick viewing.

### 3. LaTeX Tables
```
part3_evaluation/results/zero_shot_table.tex
part3_evaluation/results/perplexity_table.tex
```

Ready to include in research papers.

### 4. Markdown Tables
```
part3_evaluation/results/results_tables.md
```

GitHub-friendly markdown tables.

## Comparison with Part4

| Aspect | Part 3 Evaluation | Part 4 Evaluation |
|--------|-------------------|-------------------|
| **Purpose** | Model quality assessment | Adversarial robustness defense |
| **Metrics** | Perplexity, Zero-shot accuracy | Defense rate, Attack success rate |
| **Datasets** | WikiText2, C4, 6 zero-shot tasks | WikiText2 (adversarial examples) |
| **Precision** | Fixed precision per run | Random precision switching |
| **Attacks** | None (clean evaluation) | TextFooler, BERT-Attack |
| **Focus** | Compression-accuracy tradeoff | Security and robustness |
| **Output** | Publication tables (LaTeX/MD) | Defense statistics JSON |
| **Use Case** | Academic paper results | Security evaluation |

**Part 3** answers: *"How good is this quantized model compared to baselines?"*

**Part 4** answers: *"Can random precision switching defend against adversarial attacks?"*

## Expected Results

### Typical Performance Ranges (GPT-2 124M on WikiText2)

| Bit Width | Perplexity | Zero-Shot Avg | Model Size | Notes |
|-----------|------------|---------------|------------|-------|
| **FP32** | ~20 | ~42% | 497 MB | Teacher baseline |
| **FP16** | ~20 | ~42% | 249 MB | No degradation |
| **INT8** | ~21-23 | ~40-41% | 124 MB | Minimal degradation |
| **INT6** | ~24-27 | ~38-40% | 93 MB | Moderate degradation |
| **INT4** | ~28-35 | ~35-38% | 62 MB | Noticeable degradation |
| **INT3** | ~40-60 | ~30-35% | 47 MB | Severe degradation |
| **INT2** | ~80-150 | ~25-30% | 31 MB | Extreme degradation |

**Note:** Results depend heavily on training method (SP vs CPT), calibration data, and quantizer type (minmax vs log).

## Troubleshooting

### Issue: "Quantizer not calibrated" error

**Cause:** Model's quantizers don't have calibrated scale/zero_point buffers

**Solution:**
1. Check that checkpoint was saved after calibration
2. Verify `load_state_dict()` loads quantizer buffers
3. Run calibration before evaluation if needed

### Issue: Perplexity = Inf or NaN

**Cause:** Uncalibrated quantizers or numerical instability

**Solution:**
1. Run with `--diagnose` flag to identify issues
2. Check logits mean/std (should be ~0 mean, reasonable std)
3. Verify quantizer calibration status
4. Try higher bit width (e.g., 8-bit instead of 4-bit)

### Issue: Zero-shot accuracy very low (<10%)

**Cause:** Model completely broken or wrong prompt format

**Solution:**
1. Check that model loads correctly
2. Verify bit width is supported by model
3. Test with FP32/FP16 baseline first
4. Check prompt formatting in zero_shot_tasks.py

### Issue: Evaluation too slow

**Solution:**
1. Reduce `max_samples` in config (e.g., 500 → 200)
2. Increase `stride` for perplexity (256 → 512)
3. Skip WikiText103 (very large dataset)
4. Evaluate on subset of zero-shot tasks

## Best Practices

1. **Always verify calibration** - Run with `--diagnose` for first evaluation
2. **Start with high bit width** - Test 8-bit before 4-bit before 2-bit
3. **Use consistent config** - Same eval_config.json across all models for fair comparison
4. **Save results** - Keep all JSON outputs for later analysis
5. **Compare with baseline** - Always evaluate FP16 or FP32 baseline
6. **Check for issues** - Monitor perplexity/accuracy for sudden drops
7. **Document changes** - Note any config changes in results files

## References

- **BitConfigurations:** Predefined quantization configs (bit_configurations.py)
- **Evaluation Config:** JSON schema (evaluation_config.json)
- **Zero-Shot Tasks:** Task implementations (zero_shot_tasks.py)
- **Perplexity:** Sliding window evaluation (perplexity_eval.py)

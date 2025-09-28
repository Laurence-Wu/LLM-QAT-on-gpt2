# Simplified Adversarial Robustness Evaluation with Random Precision Switching

## Objective
Test whether random precision switching improves GPT-2's robustness against TextFooler and gradient-based attacks using WikiText-2 dataset. The implementation reads bit width configurations directly from the trained checkpoint and performs dynamic switching during inference.

## Key Concepts

### Random Precision Switching as Defense
Random switching creates a "moving target" defense by unpredictably changing the model's quantization level during inference. This makes it harder for adversarial examples to consistently fool the model because:
1. Attacks crafted at one precision may not transfer to another
2. Gradient-based attacks become unreliable when the gradient landscape changes
3. The attacker cannot predict which precision will be used at inference time

### Attack Types Evaluated

#### TextFooler (Discrete Attack)
- Operates at the word/token level
- Replaces words with synonyms to fool the model
- Preserves semantic meaning while causing misclassification
- Tests robustness against discrete perturbations

#### Gradient-Based Attack (Continuous Attack)
- Uses model gradients to find minimal perturbations
- Operates in the continuous embedding space
- Similar to HotFlip but adapted for language models
- Tests robustness against continuous perturbations

## Implementation Details

### Model Architecture
- **SPLMHeadModel**: Switchable Precision Language Model with GPT-2 architecture
- **set_precision(bits)**: Method to switch between different bit widths
- **bit_widths**: List of supported precisions (e.g., [4, 8, 16])
- **LoRA adapters**: Each precision has dedicated LoRA parameters

### Random Switching Mechanism
```python
class SimplifiedRandomSwitching:
    - switch_probability: Probability of changing precision (default 0.3)
    - select_next_precision(): Randomly selects from available bit widths
    - forward_with_switching(): Performs inference with random precision
```

### Evaluation Pipeline
1. Load model checkpoint and extract bit width configuration
2. Prepare WikiText-2 test samples
3. Run baseline evaluation at fixed precisions
4. Run random switching defense with different probabilities
5. Compare defense rates and generate report

## Expected Results

Based on the Double-Win Lottery Hypothesis paper, we expect:

### 1. Improved Defense Rates
- **15-25% improvement** over best fixed precision baseline
- Higher improvement for gradient attacks than TextFooler
- Optimal switching probability around 0.3

### 2. Attack-Specific Effectiveness
- **Gradient attacks**: More affected due to changing gradient landscape
- **TextFooler**: Less affected but still shows improvement
- **Transferability**: Attacks at one precision less effective at others

### 3. Precision Distribution
- Uniform distribution when truly random
- No precision dominates in defense effectiveness
- Trade-off between randomness and model performance

## File Structure
```
part4_randomSwitching/
├── CLAUDE.md                          # This documentation
├── simplified_random_switching.py      # Random switching defense
├── adversarial_attacks.py             # TextFooler and gradient attacks
├── wikitext_evaluation.py             # WikiText-2 dataset preparation
├── run_evaluation.py                   # Main evaluation pipeline
└── results/
    └── evaluation_results.json        # Evaluation results
```

## Running the Evaluation

```bash
# Basic evaluation with default settings
python part4_randomSwitching/run_evaluation.py \
    --checkpoint path/to/sp_model.pth \
    --num_samples 100

# Custom evaluation with specific parameters
python part4_randomSwitching/run_evaluation.py \
    --checkpoint path/to/sp_model.pth \
    --num_samples 200 \
    --switch_probs 0.2 0.3 0.4 \
    --output_dir results/
```

## Metrics and Analysis

### Primary Metrics
- **Attack Success Rate**: Percentage of successful adversarial examples
- **Defense Rate**: Percentage of attacks defended
- **Clean Accuracy**: Model accuracy on unperturbed inputs
- **Robustness Gap**: Difference between clean and adversarial accuracy

### Comparative Analysis
- Fixed precision baselines (4, 8, 16-bit)
- Random switching with different probabilities
- Improvement percentages over baselines
- Precision usage distribution

## Theoretical Insights

### Why Random Switching Works
1. **Quantization as Natural Defense**: Different precisions have different rounding behaviors
2. **Non-Differentiable Operations**: Switching disrupts gradient flow
3. **Ensemble Effect**: Multiple precisions act like an ensemble
4. **Unpredictability**: Attackers cannot optimize for all precisions simultaneously

### Trade-offs
- **Performance vs Security**: More switching = better security but potentially lower accuracy
- **Computational Cost**: Switching overhead is minimal compared to defense benefit
- **Precision Selection**: Not all precisions are equally robust

## References
- Double-Win Lottery Hypothesis (2024)
- TextFooler: Jin et al. (2020)
- HotFlip: Ebrahimi et al. (2018)
- Switchable Precision Neural Networks (2023)
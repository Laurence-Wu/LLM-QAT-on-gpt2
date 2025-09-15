# Part 3: Post-Training Analysis

This module provides comprehensive analysis tools for trained GPT-2 models with quantization. It analyzes model performance, bit-width efficiency, robustness, and generates detailed reports with visualizations.

## Overview

Post-training analysis offers:
- **Performance profiling**: Speed, memory, and accuracy analysis
- **Bit-width optimization**: Find optimal precision configurations
- **Robustness testing**: Adversarial and noise robustness evaluation
- **Comprehensive reporting**: Detailed reports with insights and recommendations
- **Visualization**: Rich plots and charts for analysis results

## Files Structure

```
part3_post_training/
├── analyze_model.py           # Main analysis script and ModelAnalyzer class
├── performance_analyzer.py    # Performance metrics analysis
├── bit_width_analyzer.py      # Bit-width optimization analysis
├── robustness_analyzer.py     # Robustness and stability analysis
├── visualization.py           # Plotting and visualization functions
└── README.md                  # This file
```

## Key Components

### `analyze_model.py`
The main analysis coordinator:
- **ModelAnalyzer**: Main class that orchestrates all analysis
- **Model loading**: Load and initialize trained models
- **Report generation**: Create comprehensive analysis reports
- **Insight generation**: Generate actionable insights and recommendations

### `performance_analyzer.py`
Performance profiling tools:
- **Basic metrics**: Loss, perplexity, accuracy, top-k accuracy
- **Speed benchmarking**: Inference latency and throughput measurement
- **Memory profiling**: Model size and runtime memory usage
- **Layer profiling**: Per-layer execution time analysis

### `bit_width_analyzer.py`
Bit-width optimization analysis:
- **Configuration testing**: Evaluate multiple bit-width configurations
- **Layer sensitivity**: Analyze which layers are sensitive to quantization
- **Optimal configuration**: Find best accuracy/efficiency trade-offs
- **Compression analysis**: Calculate compression ratios and efficiency metrics

### `robustness_analyzer.py`
Robustness and stability testing:
- **Adversarial robustness**: FGSM and PGD attack resistance
- **Quantization noise**: Robustness to weight noise
- **Input perturbations**: Resistance to input modifications
- **Stability metrics**: Various robustness measurements

### `visualization.py`
Rich visualization capabilities:
- **Performance plots**: Speed, memory, and accuracy visualizations
- **Bit-width analysis plots**: Configuration comparisons and sensitivity maps
- **Robustness plots**: Attack results and stability metrics
- **Summary dashboard**: Comprehensive overview plot

## Usage

### Basic Analysis
```bash
cd part3_post_training
python analyze_model.py --model_path path/to/model.pt --output_dir ./results/
```

### Command Line Options
```bash
python analyze_model.py \
    --model_path switchable_model.pt \
    --config_path config.json \
    --output_dir ./analysis_results/ \
    --data_split validation[:100]
```

### Programmatic Usage
```python
from analyze_model import ModelAnalyzer

# Initialize analyzer
analyzer = ModelAnalyzer('model.pt')

# Run specific analyses
performance_results = analyzer.analyze_performance(data_loader)
bit_width_results = analyzer.analyze_bit_widths(data_loader)
robustness_results = analyzer.analyze_robustness(data_loader)

# Generate comprehensive report
analyzer.generate_comprehensive_report('./results/')
```

## Analysis Components

### Performance Analysis

#### Basic Metrics
- **Loss and Perplexity**: Language modeling performance
- **Accuracy**: Token-level prediction accuracy
- **Top-k Accuracy**: Top-k prediction performance

#### Speed Benchmarking
- **Latency Distribution**: Mean, P50, P95, P99 latencies
- **Throughput**: Tokens and batches processed per second
- **Memory Efficiency**: Memory usage patterns

```python
speed_metrics = analyzer.performance_analyzer.benchmark_inference_speed(data_loader)
print(f"Throughput: {speed_metrics['tokens_per_second']:.2f} tokens/sec")
```

### Bit-Width Analysis

#### Configuration Testing
The analyzer tests multiple configurations:
- **Uniform configurations**: All layers at same bit-width
- **Mixed precision**: Different bit-widths for different layers
- **Progressive configurations**: Higher precision at boundaries
- **Component-focused**: Attention vs MLP precision focus

#### Layer Sensitivity Analysis
```python
sensitivity = analyzer.bit_width_analyzer.analyze_layer_sensitivity(data_loader)
# Shows which layers are most sensitive to quantization
```

#### Optimization Recommendations
- **Optimal configuration**: Best accuracy/efficiency trade-off
- **Pareto frontier**: Non-dominated configurations
- **Layer-specific advice**: Which layers need higher precision

### Robustness Analysis

#### Adversarial Testing
- **FGSM attacks**: Fast Gradient Sign Method
- **PGD attacks**: Projected Gradient Descent
- **Multiple epsilon values**: Different attack strengths

#### Noise Robustness
- **Weight noise**: Gaussian noise added to model parameters
- **Quantization noise**: Robustness to quantization errors
- **Multiple noise levels**: Different noise intensities

#### Input Perturbations
- **Token dropout**: Random token removal
- **Token substitution**: Random token replacement
- **Multiple perturbation levels**: Different perturbation strengths

## Generated Reports

### Output Structure
```
analysis_results/
├── analysis_results.json      # Raw analysis data
├── summary_report.md          # High-level summary
├── insights.md               # Detailed insights and recommendations
└── plots/                    # Visualization directory
    ├── performance_speed.png
    ├── performance_memory.png
    ├── bit_width_configurations.png
    ├── accuracy_vs_bits.png
    ├── layer_sensitivity.png
    ├── adversarial_robustness.png
    ├── noise_robustness.png
    ├── perturbation_robustness.png
    └── analysis_summary_dashboard.png
```

### Report Contents

#### Summary Report (`summary_report.md`)
- Model configuration overview
- Key performance metrics
- Optimal bit-width configuration
- Robustness summary

#### Insights Report (`insights.md`)
- Performance insights and recommendations
- Quantization optimization advice
- Robustness analysis and suggestions
- Deployment scenario recommendations

#### Analysis Data (`analysis_results.json`)
Complete raw data from all analyses for further processing.

## Key Features

### Comprehensive Coverage
- **Multi-faceted analysis**: Performance, efficiency, robustness
- **Automated insights**: AI-generated recommendations
- **Rich visualizations**: Professional plots and charts
- **Export capabilities**: Multiple output formats

### Efficiency Focus
- **Batch processing**: Efficient evaluation across configurations
- **Memory management**: Careful GPU memory handling
- **Progress tracking**: Real-time progress indicators
- **Result caching**: Avoid redundant computations

### Actionable Results
- **Deployment recommendations**: Best deployment scenarios
- **Optimization suggestions**: How to improve the model
- **Trade-off analysis**: Accuracy vs efficiency insights
- **Configuration guidance**: Optimal bit-width settings

## Advanced Usage

### Custom Analysis
```python
# Custom bit-width configurations
custom_configs = {
    'my_config': [
        {'attn_bits': 8, 'mlp_bits': 4},  # Layer 0
        {'attn_bits': 4, 'mlp_bits': 8},  # Layer 1
        # ... more layers
    ]
}

# Test custom configurations
results = analyzer.bit_width_analyzer.test_configurations(
    data_loader, custom_configs
)
```

### Focused Analysis
```python
# Run only specific analyses
performance_only = analyzer.analyze_performance(data_loader)

# Skip expensive robustness testing
analyzer.analyze_bit_widths(data_loader)
analyzer.generate_comprehensive_report('./results/')
```

### Batch Analysis
```python
# Analyze multiple models
models = ['model1.pt', 'model2.pt', 'model3.pt']
for model_path in models:
    analyzer = ModelAnalyzer(model_path)
    analyzer.analyze_performance(data_loader)
    analyzer.generate_comprehensive_report(f'./results/{model_path}/')
```

## Interpretation Guide

### Performance Metrics
- **High throughput (>1000 tokens/sec)**: Suitable for real-time applications
- **Low memory usage (<100MB)**: Good for edge deployment
- **High accuracy**: Good model quality

### Bit-width Analysis
- **Effective bits**: Average precision across model
- **Compression ratio**: Size reduction vs baseline
- **Pareto frontier**: Optimal accuracy/efficiency configurations

### Robustness Metrics
- **Robustness gap <0.1**: Strong adversarial resistance
- **Noise robustness**: Stability under quantization noise
- **Input robustness**: Stability under input perturbations

## Dependencies

- PyTorch >= 1.9.0
- Transformers >= 4.0.0
- NumPy
- Matplotlib >= 3.0.0
- Seaborn
- Pandas
- tqdm
- psutil (for memory monitoring)
- All shared modules from `../shared/`

## Tips for Effective Analysis

1. **Start with basic analysis**: Get overall performance first
2. **Focus on deployment needs**: Consider your deployment constraints
3. **Compare multiple models**: Analyze different training approaches
4. **Use insights**: Read the generated insights and recommendations
5. **Iterate based on results**: Use analysis to guide further optimization
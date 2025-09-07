import torch
import json
from typing import Dict, Any

def save_checkpoint(model, optimizer, iteration, filename):
    try:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iteration
        }
        # Use _use_new_zipfile_serialization=False to avoid serialization issues
        torch.save(checkpoint, filename, _use_new_zipfile_serialization=False)
        print(f"Checkpoint saved successfully to {filename}")
    except Exception as e:
        error_msg = str(e).lower()
        if 'disk quota exceeded' in error_msg or 'no space left' in error_msg:
            print(f"Warning: Disk quota exceeded. Skipping checkpoint save: {e}")
            print("Consider cleaning up old checkpoints or using a different save location.")
        else:
            print(f"Warning: Failed to save checkpoint to {filename}: {e}")
            # Try alternative save method for non-disk-space errors
            try:
                torch.save(checkpoint, filename + '.backup', _use_new_zipfile_serialization=False)
                print(f"Backup checkpoint saved to {filename}.backup")
            except Exception as e2:
                print(f"Failed to save backup checkpoint: {e2}")
                print("Continuing training without checkpoint...")

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration

def generate_report(results: Dict[str, Any], output_file='results_report.json'):
    report = {
        'quantization_evaluation': results.get('quantization_configs', {}),
        'adversarial_robustness': results.get('robustness', {}),
        'training_metrics': results.get('training', {}),
        'insights': {}
    }
    
    # Add insights if quantization configs exist
    if results.get('quantization_configs') and len(results['quantization_configs']) > 0:
        try:
            best_config = max(results['quantization_configs'].items(), 
                            key=lambda x: x[1].get('efficiency_score', 0))[0]
            report['insights']['best_config'] = best_config
        except (ValueError, KeyError):
            report['insights']['best_config'] = 'Unknown'
    else:
        report['insights']['best_config'] = 'No configs evaluated'
    
    # Add robustness improvement if data exists
    if (results.get('robustness', {}).get('dynamic', {}).get('robustness_ratio') and
        results.get('robustness', {}).get('static', {}).get('robustness_ratio')):
        try:
            dynamic_ratio = results['robustness']['dynamic']['robustness_ratio']
            static_ratio = results['robustness']['static']['robustness_ratio']
            if static_ratio > 0:
                improvement = (dynamic_ratio / static_ratio - 1)
                report['insights']['robustness_improvement'] = improvement
            else:
                report['insights']['robustness_improvement'] = 0
        except (KeyError, ZeroDivisionError):
            report['insights']['robustness_improvement'] = 0
    else:
        report['insights']['robustness_improvement'] = 0
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def print_results_summary(report: Dict[str, Any]):
    print("\n" + "="*60)
    print("Evaluation Results Summary")
    print("="*60)
    
    if 'quantization_evaluation' in report:
        print("\nQuantization Configurations:")
        for config_name, metrics in report['quantization_evaluation'].items():
            print(f"  {config_name}:")
            print(f"    - Perplexity: {metrics['perplexity']}")
            print(f"    - Model Size: {metrics['model_size_mb']:.2f} MB")
            print(f"    - Throughput: {metrics['throughput_tokens_per_sec']:.2f} tokens/sec")
            print(f"    - Efficiency Score: {metrics['efficiency_score']}")
            
            # Add BitOPs metrics if available
            if 'compression_ratio' in metrics:
                print(f"    - Compression Ratio: {metrics['compression_ratio']:.2f}x")
            if 'memory_reduction_pct' in metrics:
                print(f"    - Memory Reduction: {metrics['memory_reduction_pct']:.1f}%")
            
            # Show BitOPs breakdown for mixed precision if available
            if 'bitops_metrics' in metrics and 'mixed_precision' in metrics['bitops_metrics']:
                bitops_data = metrics['bitops_metrics']['mixed_precision']
                total_bitops = bitops_data.get('total_bitops', 0)
                efficiency_score_bitops = bitops_data.get('efficiency_score', 0)
                print(f"    - Total BitOPs: {total_bitops:.2e}")
                print(f"    - BitOPs Efficiency Score: {efficiency_score_bitops:.2f}")
                if 'avg_bits' in bitops_data:
                    print(f"    - Average Precision: {bitops_data['avg_bits']:.1f} bits")
    
    if 'adversarial_robustness' in report:
        print("\nAdversarial Robustness:")
        for precision_type, metrics in report['adversarial_robustness'].items():
            print(f"  {precision_type.title()} Precision:")
            print(f"    - Clean Accuracy: {metrics['clean_accuracy']:.4f}")
            print(f"    - Robust Accuracy: {metrics['robust_accuracy']:.4f}")
            print(f"    - Robustness Gap: {metrics['robustness_gap']:.4f}")
    
    if 'insights' in report:
        print("\nKey Insights:")
        print(f"  - Best Configuration: {report['insights'].get('best_config', 'Unknown')}")
        if 'robustness_improvement' in report['insights']:
            print(f"  - Robustness Improvement with Dynamic Precision: {report['insights']['robustness_improvement']*100:.1f}%")
        else:
            print(f"  - Robustness Improvement: Not available")
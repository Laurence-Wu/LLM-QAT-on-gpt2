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

def generate_report(results: Dict[str, Any], output_file='h100_results_report.json'):
    try:
        report = {
            'quantization_evaluation': results.get('quantization_configs', {}),
            'adversarial_robustness': results.get('robustness', {}),
            'training_metrics': results.get('training', {}),
            'insights': {}
        }
        
        # Add insights if quantization configs exist
        if results.get('quantization_configs'):
            try:
                best_config = max(results['quantization_configs'].items(), 
                                key=lambda x: x[1].get('efficiency_score', 0))[0]
                report['insights']['best_config'] = best_config
            except (ValueError, KeyError):
                report['insights']['best_config'] = 'Unknown'
        
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
        
        # Try to save report, but don't fail if disk quota exceeded
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {output_file}")
        except Exception as e:
            if 'disk quota exceeded' in str(e).lower():
                print(f"Warning: Could not save report due to disk quota: {e}")
            else:
                print(f"Warning: Could not save report: {e}")
        
        return report
    except Exception as e:
        print(f"Error generating report: {e}")
        return {'error': str(e)}

def print_results_summary(report: Dict[str, Any]):
    print("\n" + "="*60)
    print("H100 Evaluation Results Summary")
    print("="*60)
    
    if report.get('quantization_evaluation'):
        print("\nQuantization Configurations:")
        for config_name, metrics in report['quantization_evaluation'].items():
            print(f"  {config_name}:")
            if isinstance(metrics, dict):
                print(f"    - Perplexity: {metrics.get('perplexity', 'N/A')}")
                print(f"    - Model Size: {metrics.get('model_size_mb', 0):.2f} MB")
                print(f"    - Throughput: {metrics.get('throughput_tokens_per_sec', 0):.2f} tokens/sec")
                print(f"    - Efficiency Score: {metrics.get('efficiency_score', 'N/A')}")
    
    if report.get('adversarial_robustness'):
        print("\nAdversarial Robustness:")
        for precision_type, metrics in report['adversarial_robustness'].items():
            if isinstance(metrics, dict):
                print(f"  {precision_type.title()} Precision:")
                print(f"    - Clean Accuracy: {metrics.get('clean_accuracy', 0):.4f}")
                print(f"    - Robust Accuracy: {metrics.get('robust_accuracy', 0):.4f}")
                print(f"    - Robustness Gap: {metrics.get('robustness_gap', 0):.4f}")
    
    if report.get('insights'):
        print("\nKey Insights:")
        if 'best_config' in report['insights']:
            print(f"  - Best Configuration: {report['insights']['best_config']}")
        if 'robustness_improvement' in report['insights']:
            improvement = report['insights']['robustness_improvement']
            print(f"  - Robustness Improvement with Dynamic Precision: {improvement*100:.1f}%")
    
    print(f"\n✅ H100-optimized training summary completed!")

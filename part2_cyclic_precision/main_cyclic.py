#!/usr/bin/env python3
"""
Part 2: Cyclic Precision Training (CPT)
This module implements cyclic precision training followed by different bit-width configurations.
First trains with cyclic precision, then evaluates various bit-width setups.
"""

import os
import sys
import torch
import gc
from transformers import GPT2Config, GPT2TokenizerFast, GPT2Model

# Add shared folder to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

# Memory optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# Import shared components
from shared.models import SwitchableQuantizedGPT2
from shared.dataset import create_dataloaders

# Import local configurations and training functions
from config_cyclic import ModelConfig, CyclicTrainingConfig, CyclicPrecisionConfig
from train_cyclic import train_with_cpt, train_with_static_precision
from evaluate_cyclic import evaluate_cyclic_training, compare_bit_configurations


def initialize_model(model_config, device):
    """
    Initialize GPT-2 model for cyclic precision training.
    
    Args:
        model_config: Configuration object with model parameters
        device: torch.device for model placement
    
    Returns:
        Initialized model ready for cyclic precision training
    """
    print("\nInitializing model for cyclic precision training...")
    
    # Create GPT-2 configuration
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        layer_norm_epsilon=model_config.layer_norm_epsilon,
        embd_pdrop=model_config.embd_pdrop,
        bit_widths=model_config.bit_widths
    )
    
    # Initialize model
    model = SwitchableQuantizedGPT2(gpt2_config).to(device)
    
    # Load pretrained weights if configured
    if model_config.use_pretrained:
        load_pretrained_weights(model)
    
    print(f"Model initialized: {model_config.n_layer} layers, {model_config.n_embd} embedding dim")
    print(f"Cyclic bit widths: {model_config.bit_widths}")
    
    return model


def load_pretrained_weights(model):
    """
    Load pretrained GPT-2 weights into the model.
    
    Args:
        model: SwitchableQuantizedGPT2 model
    """
    try:
        print("Loading pretrained GPT-2 weights...")
        pretrained = GPT2Model.from_pretrained('gpt2')
        
        # Copy embeddings
        model.wte.weight.data = pretrained.wte.weight.data.clone()
        model.wpe.weight.data = pretrained.wpe.weight.data.clone()
        
        # Copy transformer blocks with proper weight transposition
        for i in range(min(len(model.h), len(pretrained.h))):
            # Layer norms
            model.h[i].ln_1.weight.data = pretrained.h[i].ln_1.weight.data.clone()
            model.h[i].ln_1.bias.data = pretrained.h[i].ln_1.bias.data.clone()
            model.h[i].ln_2.weight.data = pretrained.h[i].ln_2.weight.data.clone()
            model.h[i].ln_2.bias.data = pretrained.h[i].ln_2.bias.data.clone()
            
            # Attention layers
            if hasattr(model.h[i].attn.c_attn, 'quantized_linear'):
                model.h[i].attn.c_attn.quantized_linear.weight.data = \
                    pretrained.h[i].attn.c_attn.weight.data.t().contiguous()
                if pretrained.h[i].attn.c_attn.bias is not None:
                    model.h[i].attn.c_attn.quantized_linear.bias.data = \
                        pretrained.h[i].attn.c_attn.bias.data.clone()
            
            if hasattr(model.h[i].attn.c_proj, 'quantized_linear'):
                model.h[i].attn.c_proj.quantized_linear.weight.data = \
                    pretrained.h[i].attn.c_proj.weight.data.t().contiguous()
                if pretrained.h[i].attn.c_proj.bias is not None:
                    model.h[i].attn.c_proj.quantized_linear.bias.data = \
                        pretrained.h[i].attn.c_proj.bias.data.clone()
            
            # MLP layers
            if hasattr(model.h[i].mlp.c_fc, 'quantized_linear'):
                model.h[i].mlp.c_fc.quantized_linear.weight.data = \
                    pretrained.h[i].mlp.c_fc.weight.data.t().contiguous()
                if pretrained.h[i].mlp.c_fc.bias is not None:
                    model.h[i].mlp.c_fc.quantized_linear.bias.data = \
                        pretrained.h[i].mlp.c_fc.bias.data.clone()
            
            if hasattr(model.h[i].mlp.c_proj, 'quantized_linear'):
                model.h[i].mlp.c_proj.quantized_linear.weight.data = \
                    pretrained.h[i].mlp.c_proj.weight.data.t().contiguous()
                if pretrained.h[i].mlp.c_proj.bias is not None:
                    model.h[i].mlp.c_proj.quantized_linear.bias.data = \
                        pretrained.h[i].mlp.c_proj.bias.data.clone()
        
        # Final layer norm
        model.ln_f.weight.data = pretrained.ln_f.weight.data.clone()
        model.ln_f.bias.data = pretrained.ln_f.bias.data.clone()
        
        print("Pretrained weights loaded successfully")
        
    except Exception as e:
        print(f"Warning: Could not load pretrained weights: {e}")
        print("Continuing with random initialization...")


def main():
    """
    Main function for cyclic precision training.
    Implements two-phase training: cyclic precision followed by static configurations.
    """
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load configurations
    model_config = ModelConfig()
    training_config = CyclicTrainingConfig()
    cyclic_config = CyclicPrecisionConfig()
    
    # Initialize model
    model = initialize_model(model_config, device)
    
    if torch.cuda.is_available():
        print(f"Model GPU Memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
    
    # Setup tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create data loaders
    print("\nLoading datasets...")
    train_loader, val_loader = create_dataloaders(
        tokenizer,
        train_split=training_config.train_split,
        val_split=training_config.val_split,
        batch_size=training_config.batch_size,
        max_length=training_config.max_seq_length,
        doc_stride=training_config.doc_stride
    )
    
    # PHASE 1: Cyclic Precision Training
    print("\n" + "="*60)
    print("PHASE 1: CYCLIC PRECISION TRAINING (CPT)")
    print("="*60)
    print(f"\nCyclic Configuration:")
    print(f"  Cycle length: {cyclic_config.cycle_length} iterations")
    print(f"  Bit width pattern: {cyclic_config.bit_width_pattern}")
    print(f"  Total cycles: {training_config.num_cpt_iterations // cyclic_config.cycle_length}")
    
    cpt_model, cpt_stats = train_with_cpt(
        model,
        train_loader,
        val_loader,
        training_config,
        cyclic_config,
        n_layers=model_config.n_layer
    )
    
    print("\nCyclic Precision Training completed!")
    print(f"  Final loss: {cpt_stats['final_loss']:.4f}")
    print(f"  Best validation loss: {cpt_stats['best_val_loss']:.4f}")
    
    # PHASE 2: Static Bit-Width Configuration Training
    print("\n" + "="*60)
    print("PHASE 2: STATIC BIT-WIDTH CONFIGURATION TRAINING")
    print("="*60)
    print("\nTraining with different static bit-width configurations...")
    
    static_results = {}
    
    # Test different static configurations
    static_configs = {
        '2bit': 2,
        '4bit': 4,
        '8bit': 8,
        'mixed_2_4': [2, 4],  # Alternating
        'mixed_4_8': [4, 8],  # Alternating
    }
    
    for config_name, bit_config in static_configs.items():
        print(f"\n--- Training with {config_name} configuration ---")
        
        # Reset model to CPT checkpoint
        static_model = initialize_model(model_config, device)
        static_model.load_state_dict(cpt_model.state_dict())
        
        # Train with static configuration
        trained_model, train_stats = train_with_static_precision(
            static_model,
            train_loader,
            val_loader,
            training_config,
            bit_config,
            n_layers=model_config.n_layer
        )
        
        static_results[config_name] = {
            'model': trained_model,
            'stats': train_stats
        }
        
        print(f"  {config_name} - Final loss: {train_stats['final_loss']:.4f}")
    
    # PHASE 3: Comparative Evaluation
    print("\n" + "="*60)
    print("PHASE 3: COMPARATIVE EVALUATION")
    print("="*60)
    
    # Evaluate CPT model
    print("\nEvaluating Cyclic Precision Training model...")
    cpt_evaluation = evaluate_cyclic_training(cpt_model, val_loader, cyclic_config)
    
    # Compare all configurations
    print("\nComparing bit-width configurations...")
    comparison_results = compare_bit_configurations(
        cpt_model,
        static_results,
        val_loader,
        n_layers=model_config.n_layer
    )
    
    # Generate comprehensive report
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*60)
    
    results = {
        'model_config': {
            'n_layers': model_config.n_layer,
            'n_embd': model_config.n_embd,
            'bit_widths': model_config.bit_widths
        },
        'cyclic_training': {
            'config': cyclic_config.__dict__,
            'stats': cpt_stats,
            'evaluation': cpt_evaluation
        },
        'static_configurations': {
            name: {
                'stats': res['stats'],
                'evaluation': comparison_results.get(name, {})
            }
            for name, res in static_results.items()
        },
        'comparison': comparison_results.get('summary', {})
    }
    
    # Save results
    report_path = 'cyclic_precision_results.json'
    report = generate_report(results, report_path)
    print_results_summary(report)
    
    # Save best model
    best_config = comparison_results.get('best_configuration', 'cpt')
    if best_config == 'cpt':
        best_model = cpt_model
    else:
        best_model = static_results.get(best_config, {}).get('model', cpt_model)
    
    model_save_path = 'cyclic_best_model.pt'
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'best_configuration': best_config,
        'model_config': model_config.__dict__,
        'training_config': training_config.__dict__,
        'cyclic_config': cyclic_config.__dict__,
        'results': results
    }, model_save_path)
    print(f"\nBest model saved to: {model_save_path}")
    print(f"Best configuration: {best_config}")
    
    # Print final summary
    print("\n" + "="*60)
    print("CYCLIC PRECISION TRAINING COMPLETED")
    print("="*60)
    print(f"\nKey Achievements:")
    print(f"- Cyclic training iterations: {training_config.num_cpt_iterations}")
    print(f"- Static configurations tested: {len(static_configs)}")
    print(f"- Best configuration: {best_config}")
    
    if 'summary' in comparison_results:
        summary = comparison_results['summary']
        if 'best_perplexity' in summary:
            print(f"- Best perplexity: {summary['best_perplexity']:.2f}")
        if 'best_accuracy' in summary:
            print(f"- Best accuracy: {summary['best_accuracy']:.4f}")
    
    if torch.cuda.is_available():
        print(f"\nMemory Statistics:")
        print(f"- Final GPU Memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
        print(f"- Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
    
    return best_model, results


if __name__ == "__main__":
    try:
        model, results = main()
        print("\n✅ Cyclic precision training completed successfully!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        if torch.cuda.is_available():
            print(f"GPU Memory at failure: {torch.cuda.memory_allocated() / 1e9:.3f} GB")

import os
import sys
import torch
import gc
import json
from datetime import datetime
from transformers import GPT2Config, GPT2TokenizerFast

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from models_sp import SPLMHeadModel
from dataset import create_dataloaders
from deploy import save_sp_checkpoints
from config_sp import ModelConfig, TrainingConfig
from train_sp import train_sp

def initialize_model(model_config, device):

    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        activation_function='gelu_new',
        layer_norm_epsilon=model_config.layer_norm_epsilon,
        embd_pdrop=model_config.embd_pdrop
    )

    gpt2_config.quantization_bits = model_config.quantization_bits
    gpt2_config.lora_rank = model_config.lora_rank
    gpt2_config.lora_alpha = model_config.lora_alpha
    gpt2_config.lora_rank_per_bit = model_config.lora_rank_per_bit
    gpt2_config.lora_alpha_per_bit = model_config.lora_alpha_per_bit
    gpt2_config.activation_bits_per_bit = model_config.activation_bits_per_bit
    gpt2_config.quantizer_per_bit = model_config.quantizer_per_bit
    gpt2_config.bit_widths = model_config.bit_widths

    model = SPLMHeadModel(gpt2_config)

    load_pretrained_weights(model)

    model = model.to(device)

    model.transformer.unfreeze_weights(32)
    return model

def load_pretrained_weights(model):
    print("Loading pretrained GPT-2 weights")
    # use the exact weight from hugginggace gpt2 https://huggingface.co/openai-community/gpt2/blob/main/config.json
    from transformers import GPT2LMHeadModel # have to do this base on the projection layer.
    pretrained = GPT2LMHeadModel.from_pretrained('gpt2')

    model.transformer.wte.weight.data = pretrained.transformer.wte.weight.data.clone()
    model.transformer.wte.weight.requires_grad = False
    model.transformer.wpe.weight.data = pretrained.transformer.wpe.weight.data.clone()
    model.transformer.wpe.weight.requires_grad = False

    model.lm_head.weight.requires_grad = False

    for i in range(len(pretrained.transformer.h)):
        for precision in model.transformer.h[i].ln_1.precision_levels:
            model.transformer.h[i].ln_1.weights[str(precision)].data = pretrained.transformer.h[i].ln_1.weight.data.clone()
            model.transformer.h[i].ln_1.biases[str(precision)].data = pretrained.transformer.h[i].ln_1.bias.data.clone()
            model.transformer.h[i].ln_1.weights[str(precision)].requires_grad = False
            model.transformer.h[i].ln_1.biases[str(precision)].requires_grad = False

        for precision in model.transformer.h[i].ln_2.precision_levels:
            model.transformer.h[i].ln_2.weights[str(precision)].data = pretrained.transformer.h[i].ln_2.weight.data.clone()
            model.transformer.h[i].ln_2.biases[str(precision)].data = pretrained.transformer.h[i].ln_2.bias.data.clone()
            model.transformer.h[i].ln_2.weights[str(precision)].requires_grad = False
            model.transformer.h[i].ln_2.biases[str(precision)].requires_grad = False

        model.transformer.h[i].attn.c_attn.linear.weight.data = pretrained.transformer.h[i].attn.c_attn.weight.data.t().contiguous()
        model.transformer.h[i].attn.c_attn.linear.bias.data = pretrained.transformer.h[i].attn.c_attn.bias.data.clone()
        model.transformer.h[i].attn.c_attn.linear.weight.requires_grad = False
        model.transformer.h[i].attn.c_attn.linear.bias.requires_grad = False

        model.transformer.h[i].attn.c_proj.linear.weight.data = pretrained.transformer.h[i].attn.c_proj.weight.data.t().contiguous()
        model.transformer.h[i].attn.c_proj.linear.bias.data = pretrained.transformer.h[i].attn.c_proj.bias.data.clone()
        model.transformer.h[i].attn.c_proj.linear.weight.requires_grad = False
        model.transformer.h[i].attn.c_proj.linear.bias.requires_grad = False

        model.transformer.h[i].mlp.c_fc.linear.weight.data = pretrained.transformer.h[i].mlp.c_fc.weight.data.t().contiguous()
        model.transformer.h[i].mlp.c_fc.linear.bias.data = pretrained.transformer.h[i].mlp.c_fc.bias.data.clone()
        model.transformer.h[i].mlp.c_fc.linear.weight.requires_grad = False
        model.transformer.h[i].mlp.c_fc.linear.bias.requires_grad = False

        model.transformer.h[i].mlp.c_proj.linear.weight.data = pretrained.transformer.h[i].mlp.c_proj.weight.data.t().contiguous()
        model.transformer.h[i].mlp.c_proj.linear.bias.data = pretrained.transformer.h[i].mlp.c_proj.bias.data.clone()
        model.transformer.h[i].mlp.c_proj.linear.weight.requires_grad = False
        model.transformer.h[i].mlp.c_proj.linear.bias.requires_grad = False

    for precision in model.transformer.ln_f.precision_levels:
        model.transformer.ln_f.weights[str(precision)].data = pretrained.transformer.ln_f.weight.data.clone()
        model.transformer.ln_f.biases[str(precision)].data = pretrained.transformer.ln_f.bias.data.clone()
        model.transformer.ln_f.weights[str(precision)].requires_grad = False
        model.transformer.ln_f.biases[str(precision)].requires_grad = False

    lora_count = 0

    target_modules = [
        'c_attn',
        'c_proj',
        'c_fc',
    ]

    for name, module in model.named_modules():
        if not any(target in name for target in target_modules):
            # these do not have LoRA
            continue
        if not hasattr(module, 'lora_adapters'):
            continue
        for bit_key in module.lora_adapters.keys():
            lora_layer = module.lora_adapters[bit_key]
            if hasattr(lora_layer, 'lora_A'):
                lora_layer.lora_A.requires_grad = True
                lora_layer.lora_B.requires_grad = True
                lora_count += 1

    print(f"Enabled {lora_count} LoRA adapter pairs for training")

    del pretrained 
    torch.cuda.empty_cache()
    gc.collect()
    # nice summary to print out
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + frozen_params

    print(f"Pretrained weights loaded and frozen successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    print(f"  Trainable (LoRA) parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Switchable Precision Training Script')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Training requires CUDA.")

    device = torch.device('cuda')
    print(f"Using device: {device}")
    torch.cuda.empty_cache()
    gc.collect()

    model_config = ModelConfig()
    training_config = TrainingConfig()

    model = initialize_model(model_config, device)

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print("\nDatasets")
    train_loader, val_loader = create_dataloaders(
        tokenizer,
        train_split=training_config.train_split,
        val_split=training_config.val_split,
        batch_size=training_config.batch_size,
        max_length=training_config.max_seq_length,
        doc_stride=training_config.doc_stride
    )

    trained_model, training_stats = train_sp(
        model,
        train_loader,
        val_loader,
        training_config,
        model_config
    )
    
    print("Training complete")

    # Save training statistics with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_filename = f"training_stats_{timestamp}.json"

    # Add configs to training_stats
    try:
        training_stats['model_config'] = {
            attr: getattr(model_config, attr)
            for attr in dir(model_config)
            if not attr.startswith('_') and not callable(getattr(model_config, attr))
        }
        training_stats['training_config'] = {
            attr: getattr(training_config, attr)
            for attr in dir(training_config)
            if not attr.startswith('_') and not callable(getattr(training_config, attr))
        }
    except Exception as e:
        print(f"Warning: Could not add configs to stats: {e}")

    # Save to file
    try:
        with open(stats_filename, 'w') as f:
            json.dump(training_stats, f, indent=2)
        print(f"Training statistics saved to {stats_filename}")
    except Exception as e:
        print(f"Error saving training statistics: {e}")

    try:
        saved_checkpoints = save_sp_checkpoints(
            trained_model,
            base_filename="sp_gpt2",
            model_config=model_config,
            training_config=training_config
        )

        print("\nCheckpoint Summary:")
        for bits, filepath in saved_checkpoints.items():
            print(f"  {bits}-bit: {filepath}")

    except Exception as e:
        print(f"Error saving models: {e}")
        raise

    return trained_model

if __name__ == "__main__":
    main()

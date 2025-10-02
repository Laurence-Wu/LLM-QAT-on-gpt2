import torch
import sys
import os
from types import SimpleNamespace
from pathlib import Path

# Add part2 directory to path for CPTModel import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
part2_dir = os.path.join(parent_dir, 'part2_cyclic_precision_training')
if part2_dir not in sys.path:
    sys.path.insert(0, part2_dir)

from cpt_model import CPTModel


def load_cpt_model(model_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    print(f"Loading CPT model from {model_path}")

    checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)

    model_config_dict = checkpoint['model_config']
    training_config_dict = checkpoint['training_config']

    try:
        cpt_config_dict = checkpoint['cpt_config']
    except KeyError:
        cpt_config_dict = {}
        print("Warning: No cpt_config in checkpoint")

    try:
        checkpoint_bit_width = checkpoint['bit_width']
        print(f"Checkpoint at {checkpoint_bit_width}-bit precision")
    except KeyError:
        checkpoint_bit_width = None
        print("Warning: No bit_width in checkpoint")

    model_config = SimpleNamespace(**model_config_dict)
    training_config = SimpleNamespace(**training_config_dict)

    if checkpoint_bit_width is not None:
        model_config.bit_widths = [checkpoint_bit_width]
        print(f"Overriding bit_widths to [{checkpoint_bit_width}] for evaluation")

    if cpt_config_dict:
        cpt_config = SimpleNamespace(**cpt_config_dict)
    else:
        cpt_config = SimpleNamespace(
            cycle_length=3,
            schedule_type='cosine',
            prt_start_bits=2,
            prt_threshold=0.01,
            prt_iterations=100
        )

    config = {
        'model': model_config,
        'training': training_config,
        'cpt': cpt_config
    }

    print("\nCreating CPT model...")
    model = CPTModel(config)
    model.config = config

    if checkpoint_bit_width:
        model.set_precision(checkpoint_bit_width)

    override_count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'CPTLinear':
            module.quantizer_weight.per_channel = False
            module.quantizer_input.per_channel = False
            override_count += 1

    print(f"Set {override_count} quantizers to per-tensor mode")

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("Model weights loaded")

    model = model.cuda()
    model.eval()
    model.enable_lora_after_calibration()

    lora_enabled_count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'LoRAAdapter' and hasattr(module, 'calibration_mode'):
            if not module.calibration_mode:
                lora_enabled_count += 1

    print(f"LoRA adapters enabled for evaluation ({lora_enabled_count} adapters active)")

    print("\nDEBUG: Model configuration:")
    print(f"  Current precision: {checkpoint_bit_width}-bit")
    print(f"  LoRA enabled: {lora_enabled_count > 0}")
    print(f"  Model on device: {next(model.parameters()).device}")

    return model, checkpoint_bit_width, model_config, training_config
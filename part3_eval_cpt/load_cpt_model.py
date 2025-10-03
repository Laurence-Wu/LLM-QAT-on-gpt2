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

    print("Loading pretrained GPT-2 base weights...")
    from transformers import GPT2LMHeadModel
    import gc

    pretrained = GPT2LMHeadModel.from_pretrained('gpt2')

    model.wte.weight.data = pretrained.transformer.wte.weight.data.clone()
    model.wpe.weight.data = pretrained.transformer.wpe.weight.data.clone()

    for i in range(len(pretrained.transformer.h)):
        model.h[i].ln_1.weight.data = pretrained.transformer.h[i].ln_1.weight.data.clone()
        model.h[i].ln_1.bias.data = pretrained.transformer.h[i].ln_1.bias.data.clone()
        model.h[i].ln_2.weight.data = pretrained.transformer.h[i].ln_2.weight.data.clone()
        model.h[i].ln_2.bias.data = pretrained.transformer.h[i].ln_2.bias.data.clone()

        model.h[i].attn.c_attn.linear.weight.data = pretrained.transformer.h[i].attn.c_attn.weight.data.t().contiguous()
        model.h[i].attn.c_attn.linear.bias.data = pretrained.transformer.h[i].attn.c_attn.bias.data.clone()
        model.h[i].attn.c_proj.linear.weight.data = pretrained.transformer.h[i].attn.c_proj.weight.data.t().contiguous()
        model.h[i].attn.c_proj.linear.bias.data = pretrained.transformer.h[i].attn.c_proj.bias.data.clone()

        model.h[i].mlp['fc_in'].linear.weight.data = pretrained.transformer.h[i].mlp.c_fc.weight.data.t().contiguous()
        model.h[i].mlp['fc_in'].linear.bias.data = pretrained.transformer.h[i].mlp.c_fc.bias.data.clone()
        model.h[i].mlp['fc_out'].linear.weight.data = pretrained.transformer.h[i].mlp.c_proj.weight.data.t().contiguous()
        model.h[i].mlp['fc_out'].linear.bias.data = pretrained.transformer.h[i].mlp.c_proj.bias.data.clone()

    model.ln_f.weight.data = pretrained.transformer.ln_f.weight.data.clone()
    model.ln_f.bias.data = pretrained.transformer.ln_f.bias.data.clone()
    model.lm_head.linear.weight.data = pretrained.lm_head.weight.data.clone()

    del pretrained
    gc.collect()
    print("Pretrained base weights loaded")

    if checkpoint_bit_width:
        model.set_precision(checkpoint_bit_width)

    print("\nDEBUG: Checking checkpoint state_dict for calibration data...")
    state_dict = checkpoint['model_state_dict']

    # Check for old format (scale/zero_point buffers)
    old_format_keys = [k for k in state_dict.keys() if k.endswith('.scale') or k.endswith('.zero_point')]
    if old_format_keys:
        print(f"  Found {len(old_format_keys)} old-format calibration keys (scale/zero_point buffers)")
        print(f"  Sample keys: {old_format_keys[:3]}")
    else:
        print("  No old-format calibration keys found")

    # Check for new format (per-precision scales/zero_points)
    new_format_keys = [k for k in state_dict.keys() if '_scales_' in k or '_zero_points_' in k or k.endswith('_calibrated_bits')]
    if new_format_keys:
        print(f"  Found {len(new_format_keys)} new-format calibration keys (per-precision)")
        print(f"  Sample keys: {new_format_keys[:3]}")
    else:
        print("  No new-format calibration keys found")

    # Check quantizer keys
    quantizer_keys = [k for k in state_dict.keys() if 'quantizer' in k]
    print(f"  Total quantizer-related keys in checkpoint: {len(quantizer_keys)}")

    # Check specifically for lora_weight_quantizers
    lora_wq_keys = [k for k in state_dict.keys() if 'lora_weight_quantizers' in k]
    if lora_wq_keys:
        print(f"  LoRA weight quantizer keys: {len(lora_wq_keys)}")
        print(f"    Sample: {lora_wq_keys[:5]}")
    else:
        print(f"  WARNING: No lora_weight_quantizers calibration in checkpoint!")

    # Check for LoRA weight quantizer CALIBRATION (not just buffers)
    lora_wq_calib_keys = [k for k in state_dict.keys() if 'lora_weight_quantizers' in k and ('_scales_' in k or '_zero_points_' in k)]
    if lora_wq_calib_keys:
        print(f"  LoRA weight quantizer CALIBRATION keys: {len(lora_wq_calib_keys)}")
        print(f"    Sample: {lora_wq_calib_keys[:5]}")
    else:
        print(f"  WARNING: No LoRA weight quantizer calibration (_scales_/_zero_points_) in checkpoint!")
        print(f"  Only found running_min/max buffers, which means LoRA quantizers were NEVER calibrated during training")

    # Also check grad_quantizer vs lora_weight_quantizers
    grad_q_keys = [k for k in state_dict.keys() if 'grad_quantizer' in k]
    print(f"  Gradient quantizer keys: {len(grad_q_keys)}")

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("Checkpoint weights loaded (LoRA + trained LayerNorms)")

    first_linear = None
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'CPTLinear':
            first_linear = (name, module)
            break

    if first_linear:
        name, module = first_linear
        print(f"\nDEBUG: Quantizer status for {name}:")
        print(f"  Weight quantizer:")
        current_bits_w = module.quantizer_weight.num_bits
        print(f"    - Current precision: {current_bits_w}-bit")
        print(f"    - Calibrated bits: {module.quantizer_weight.calibrated_bits}")
        print(f"    - Calibrated at {current_bits_w}-bit: {current_bits_w in module.quantizer_weight.calibrated_bits}")
        print(f"    - per_channel: {module.quantizer_weight.per_channel}")
        if current_bits_w in module.quantizer_weight.scales:
            print(f"    - Scale shape: {module.quantizer_weight.scales[current_bits_w].shape}")
            print(f"    - Scale mean: {module.quantizer_weight.scales[current_bits_w].mean().item():.6f}")
            print(f"    - Scale std: {module.quantizer_weight.scales[current_bits_w].std().item():.6f}")
        else:
            print(f"    - Scale: Not calibrated at {current_bits_w}-bit")
        print(f"  Input quantizer:")
        current_bits_i = module.quantizer_input.num_bits
        print(f"    - Current precision: {current_bits_i}-bit")
        print(f"    - Calibrated bits: {module.quantizer_input.calibrated_bits}")
        print(f"    - Calibrated at {current_bits_i}-bit: {current_bits_i in module.quantizer_input.calibrated_bits}")
        print(f"    - per_channel: {module.quantizer_input.per_channel}")
        if current_bits_i in module.quantizer_input.scales:
            print(f"    - Scale shape: {module.quantizer_input.scales[current_bits_i].shape}")
            print(f"    - Scale mean: {module.quantizer_input.scales[current_bits_i].mean().item():.6f}")
        else:
            print(f"    - Scale: Not calibrated at {current_bits_i}-bit")

        # Check LoRA weight quantizers specifically
        print(f"  LoRA weight quantizers:")
        for bits_key, lora_wq in module.lora_weight_quantizers.items():
            print(f"    {bits_key}:")
            print(f"      - Calibrated bits: {lora_wq.calibrated_bits}")
            print(f"      - Scales: {list(lora_wq.scales.keys())}")
            print(f"      - Zero points: {list(lora_wq.zero_points.keys())}")

    model = model.cuda()
    model.eval()

    # CRITICAL: Enable LoRA for evaluation (disable_lora_for_calibration was turning it off!)
    model.enable_lora_after_calibration()

    lora_calibration_count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'LoRAAdapter' and hasattr(module, 'calibration_mode'):
            if module.calibration_mode:
                lora_calibration_count += 1

    print(f"LoRA in calibration mode (should be 0): {lora_calibration_count} adapters")

    print("\nDEBUG: Model configuration:")
    print(f"  Current precision: {checkpoint_bit_width}-bit")
    print(f"  LoRA enabled: {not bool(lora_calibration_count)} (calibration_mode=False)")
    print(f"  Model on device: {next(model.parameters()).device}")

    return model, checkpoint_bit_width, model_config, training_config
import torch
import torch.nn as nn
import os
import time
from typing import Dict, Optional
from cpt_model import CPTModel

def save_target_model(model: CPTModel, config: dict, target_bits: int, output_dir: str):
    import traceback
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    model.set_precision(target_bits)
    state_dict = model.state_dict()

    filtered_state_dict = {}
    for key, value in state_dict.items():
        filtered_state_dict[key] = value

    state_dict_size = sum(
        p.numel() * p.element_size()
        for p in filtered_state_dict.values()
        if isinstance(p, torch.Tensor)
    )

    filename = os.path.join(output_dir, f"cpt_model_{target_bits}bit_target_{timestamp}.pth")

    checkpoint = {
        'model_state_dict': filtered_state_dict,
        'model_config': config['model'].__dict__,
        'training_config': config['training'].__dict__,
        'cpt_config': config['cpt'].__dict__,
        'bit_width': target_bits,
        'target_precision': target_bits,
        'timestamp': timestamp,
        'lora_rank': config['model'].shared_lora_rank,
        'lora_alpha': config['model'].shared_lora_alpha,
        'checkpoint_version': '1.2',
        'pytorch_version': torch.__version__,
        'model_type': 'CPT_TARGET'
    }

    max_retries = 3
    retry_count = 0
    save_successful = False

    while retry_count < max_retries and not save_successful:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            torch.save(checkpoint, filename, pickle_protocol=4)

            time.sleep(0.5)
            file_size = os.path.getsize(filename)

            test_load = torch.load(filename, map_location='cpu', weights_only=False)

            assert 'model_state_dict' in test_load, "Missing model_state_dict"
            assert 'bit_width' in test_load, "Missing bit_width"
            assert test_load['bit_width'] == target_bits, f"Bit width mismatch: {test_load['bit_width']} != {target_bits}"

            save_successful = True

            del test_load

        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(1.0)
            else:
                traceback.print_exc()

                if os.path.exists(filename):
                    try:
                        os.remove(filename)
                    except:
                        pass

    if save_successful:
        summary_file = os.path.join(output_dir, f"cpt_target_model_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write("CPT Target Model Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Target Precision: {target_bits}-bit\n")
            f.write(f"Model Type: Cyclic Precision Training (CPT)\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model Path: {filename}\n")
            f.write(f"File Size: {file_size / (1024*1024):.2f} MB\n")
            f.write(f"Total Parameters: {sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor)):,}\n")
            f.write(f"LoRA Rank: {config['model'].shared_lora_rank}\n")
            f.write(f"LoRA Alpha: {config['model'].shared_lora_alpha}\n")
            f.write(f"Training Config:\n")
            f.write(f"  - Learning Rate: {config['training'].learning_rate}\n")
            f.write(f"  - Batch Size: {config['training'].batch_size}\n")
            f.write(f"  - Num Epochs: {config['training'].num_epochs}\n")
            f.write(f"CPT Config:\n")
            f.write(f"  - Cycle Length: {config['cpt'].total_cycles}\n")
            f.write(f"  - Schedule Type: {config['cpt'].schedule_type}\n")
            f.write("=" * 50 + "\n")

        return filename
    else:
        return None


import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import gc
import json
import time
import random

from part5_squad.distillation_manager_qa import DistillationManagerQA

def cleanup_memory():

    torch.cuda.empty_cache()
    gc.collect()

def get_next_batch(train_iter, train_loader):

    try:
        return next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        return next(train_iter)

class CalibrationManager:

    def __init__(self, model, train_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.calibrated_bits = set()

    def calibrate_all_precisions(self, bit_widths, num_batches=10):
        for bits in bit_widths:
            if bits < 32 and bits not in self.calibrated_bits:
                self.model.set_precision(bits)
                self._calibrate_precision(bits, num_batches)
                self.calibrated_bits.add(bits)

    def _calibrate_precision(self, bits, num_batches):

        bits_key = f'{bits}bit'

        if bits >= 32:
            print(f"  Skipping calibration for {bits}-bit (no quantization needed)")
            return

        weight_calibrated = 0
        weight_errors = []

        for name, module in self.model.named_modules():
            if not hasattr(module, 'quantizers_weight'):
                continue

            if bits_key not in module.quantizers_weight:
                continue

            weight_quantizer = module.quantizers_weight[bits_key]

            if hasattr(module, 'linear') and hasattr(module.linear, 'weight'):
                weight = module.linear.weight.data
            elif hasattr(module, 'weight'):
                weight = module.weight.data
            else:
                weight_errors.append(f"{name}: No weight tensor found")
                continue

            try:
                weight_quantizer.start_calibration()
                with torch.no_grad():
                    _ = weight_quantizer(weight)
                weight_quantizer.finish_calibration(debug=False)
                weight_calibrated += 1
            except Exception as e:
                weight_errors.append(f"{name}: {str(e)}")

        if weight_errors:
            print(f"    ⚠️ {len(weight_errors)} warnings (showing first 3):")
            for err in weight_errors[:3]:
                print(f"      - {err}")

        input_started = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizers_input') and bits_key in module.quantizers_input:
                module.quantizers_input[bits_key].start_calibration()
                input_started += 1

        print(f"    Started calibration for {input_started} input quantizers")

        self.model.disable_lora_for_calibration()

        train_iter = iter(self.train_loader)
        with torch.no_grad():
            for i in range(num_batches):
                try:
                    batch = next(train_iter)
                    input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                    _ = self.model(input_ids)
                    del batch, input_ids
                except StopIteration:
                    print(f"    Only {i} batches available")
                    break

        self.model.enable_lora_after_calibration()

        input_calibrated = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizers_input') and bits_key in module.quantizers_input:
                module.quantizers_input[bits_key].finish_calibration(debug=False)
                input_calibrated += 1

        print(f"    ✓ Calibrated {input_calibrated} input quantizers")

        torch.cuda.empty_cache()
        gc.collect()

    def calibrate_lora_only(self, bits, num_batches=10):

        if bits >= 32:
            return

        bits_key = f'{bits}bit'

        lora_calibrated = 0
        for name, module in self.model.named_modules():
            if not hasattr(module, 'lora_adapters'):
                continue
            if bits_key not in module.lora_adapters:
                continue

            lora_layer = module.lora_adapters[bits_key]

            if hasattr(lora_layer, 'quantize_A') and hasattr(lora_layer, 'lora_A') and lora_layer.enabled:
                try:
                    lora_layer.quantize_A.start_calibration()
                    with torch.no_grad():
                        _ = lora_layer.quantize_A(lora_layer.lora_A)
                    lora_layer.quantize_A.finish_calibration(debug=False)
                    lora_calibrated += 1
                except Exception as e:
                    print(f"    Warning calibrating {name} quantize_A: {e}")

            if hasattr(lora_layer, 'quantize_B') and hasattr(lora_layer, 'lora_B') and lora_layer.enabled:
                try:
                    lora_layer.quantize_B.start_calibration()
                    with torch.no_grad():
                        _ = lora_layer.quantize_B(lora_layer.lora_B)
                    lora_layer.quantize_B.finish_calibration(debug=False)
                    lora_calibrated += 1
                except Exception as e:
                    print(f"    Warning calibrating {name} quantize_B: {e}")

        if lora_calibrated > 0:
            torch.cuda.empty_cache()
            gc.collect()

    def ensure_calibrated(self, bits):

        if bits >= 32:
            return

        if bits not in self.calibrated_bits:
            print(f"  ⚠️ {bits}-bit not calibrated, calibrating now...")
            self.model.set_precision(bits)
            self._calibrate_precision(bits, num_batches=10)
            self.calibrated_bits.add(bits)

def should_evaluate(iteration, config):

    return iteration % config.eval_interval == 0 and iteration > 0

def setup_optimizer(model, config):

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Optimizing {len(trainable_params)} trainable parameter tensors")

    return AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.adam_betas,
        eps=config.adam_epsilon
    )

class StatsTracker:

    def __init__(self, bit_widths=None):
        self.iteration_losses = []
        self.validation_losses = []
        self.bit_width_usage = []
        self.learning_rates = []
        self.memory_usage = []

        if bit_widths:
            self.losses_per_bit = {bits: [] for bits in bit_widths}
            self.precision_counts = {bits: 0 for bits in bit_widths}

    def update(self, iteration, loss, bits, optimizer):

        self.iteration_losses.append(loss)
        self.bit_width_usage.append(bits)
        self.learning_rates.append(optimizer.param_groups[0]['lr'])
        self.memory_usage.append(torch.cuda.memory_allocated() / 1024**2)

        if bits in self.losses_per_bit:
            self.losses_per_bit[bits].append(loss)

    def add_validation(self, val_loss):

        self.validation_losses.append(val_loss)

    def record_precision_usage(self, precision):

        if precision in self.precision_counts:
            self.precision_counts[precision] += 1

    def to_dict(self):

        return {
            'iteration_losses': self.iteration_losses,
            'validation_losses': self.validation_losses,
            'bit_width_usage': self.bit_width_usage,
            'learning_rates': self.learning_rates,
            'memory_usage': self.memory_usage,
            'losses_per_bit': self.losses_per_bit,
            'precision_counts': self.precision_counts
        }

def compute_loss_single_precision_qa(model, batch, precision, teacher_bits, distill_mgr, config, iteration):
    """
    Compute QA loss for single precision

    If precision == teacher_bits: Compute task loss + cache teacher outputs
    Else (student): Compute task loss + distillation loss

    Args:
        model: SPQuestionAnsweringModel
        batch: Dict with 'input_ids', 'attention_mask', 'start_positions', 'end_positions'
        precision: Current bit-width
        teacher_bits: Teacher precision (usually 32)
        distill_mgr: DistillationManagerQA
        config: Training configuration
        iteration: Current iteration

    Returns:
        loss: Scalar loss (normalized by gradient accumulation steps)
    """
    device = next(model.parameters()).device
    input_ids = batch['input_ids'].to(device, non_blocking=True)
    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
    start_positions = batch['start_positions'].to(device, non_blocking=True)
    end_positions = batch['end_positions'].to(device, non_blocking=True)

    model.set_precision(precision)

    with torch.amp.autocast('cuda'):
        if precision == teacher_bits:
            # ===== TEACHER MODE (32-bit) =====
            # Compute task loss (ground truth)
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
                output_hidden_states=True,
                return_dict=True
            )
            loss = outputs['loss']

            # Cache teacher outputs for student distillation
            with torch.no_grad():
                distill_mgr.update_teacher_qa(input_ids, outputs)

        else:
            # ===== STUDENT MODE (low-bit) =====
            # Forward pass to get outputs
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            # Task loss (ground truth)
            task_loss = model._compute_qa_loss(
                outputs['start_logits'],
                outputs['end_logits'],
                start_positions,
                end_positions
            )

            # Distillation loss (learn from teacher)
            distill_loss = distill_mgr.compute_distillation_loss_qa(
                student_outputs=outputs,
                input_ids=input_ids,
                config=config
            )

            # Combined loss
            loss = task_loss + distill_loss

    del outputs, input_ids, attention_mask, start_positions, end_positions

    return loss / config.gradient_accumulation_steps

def train_step(model, train_iter, train_loader, optimizer, scaler,
               available_precisions, distill_mgr, config, iteration, stats_tracker, calib_mgr, scheduler, batch=None):
    """
    Single training step with multi-precision training

    Args:
        model: SPQuestionAnsweringModel
        train_iter: Iterator over training data
        train_loader: DataLoader (for reset)
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        available_precisions: List of bit-widths to train
        distill_mgr: DistillationManagerQA
        config: Training configuration
        iteration: Current iteration
        stats_tracker: StatsTracker
        calib_mgr: CalibrationManager
        scheduler: Learning rate scheduler
        batch: Pre-fetched batch (optional)

    Returns:
        (total_loss, next_batch)
    """
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0
    precisions_used = []
    model.train()
    torch.set_grad_enabled(True)

    if batch is None:
        batch = get_next_batch(train_iter, train_loader)

    for bit_step in range(config.gradient_accumulation_steps):
        teacher_bit = max(available_precisions)
        student_bits = [b for b in available_precisions if b != teacher_bit]
        if bit_step == 0:
            precision = teacher_bit
        else:
            precision = random.choice(student_bits)
        precisions_used.append(precision)

        if calib_mgr and iteration >= 0 and precision < 32:
            model.set_precision(precision)
            calib_mgr.calibrate_lora_only(precision, num_batches=2)

        if stats_tracker:
            stats_tracker.record_precision_usage(precision)

        loss = compute_loss_single_precision_qa(
            model, batch, precision,
            teacher_bits=teacher_bit,
            distill_mgr=distill_mgr,
            config=config,
            iteration=iteration + bit_step
        )

        total_loss += loss.detach().item()

        scaler.scale(loss).backward()
        scheduler.step()

        del loss

    if iteration % 100 == 0:
        precision_counts = {p: precisions_used.count(p) for p in set(precisions_used)}
        current_lr = scheduler.get_last_lr()[0]
        scheduler_step = iteration * config.gradient_accumulation_steps
        print(f"Step {iteration} | LR: {current_lr:.6f} | Scheduler steps: {scheduler_step}/{config.num_iterations * config.gradient_accumulation_steps} | Precision dist: {precision_counts}")

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    scaler.step(optimizer)
    scaler.update()

    next_batch = get_next_batch(train_iter, train_loader)

    return total_loss, next_batch

def evaluate(model, val_loader, device):
    """
    Evaluate model on validation set

    Args:
        model: SPQuestionAnsweringModel
        val_loader: Validation DataLoader
        device: Device

    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    max_batches = 5

    with torch.no_grad():
        for batch in val_loader:
            if num_batches >= max_batches:
                break

            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            start_positions = batch['start_positions'].to(device, non_blocking=True)
            end_positions = batch['end_positions'].to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                loss = outputs['loss']

            total_loss += loss.item()
            num_batches += 1

            del outputs, loss, input_ids, attention_mask, start_positions, end_positions, batch

    cleanup_memory()
    return total_loss / max(num_batches, 1)

def train_squad(model, train_loader, val_loader, config, model_config):
    """
    Main training loop for SQuAD QA task

    Args:
        model: SPQuestionAnsweringModel
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: TrainingConfig
        model_config: ModelConfig

    Returns:
        (trained_model, training_stats)
    """
    device = torch.device('cuda')
    model = model.to(device)
    cleanup_memory()

    calib_mgr = CalibrationManager(model, train_loader, device)
    student_bits = [b for b in model_config.bit_widths if b < 32]
    calib_mgr.calibrate_all_precisions(student_bits)

    teacher_bits = model_config.teacher_bits
    distill_mgr = DistillationManagerQA(
        model=model,
        full_precision_bits=teacher_bits,
        config=config
    )

    optimizer = setup_optimizer(model, config)
    total_lr_steps = config.num_iterations * config.gradient_accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_lr_steps)
    print(f"Created LR scheduler with {total_lr_steps:,} total steps")
    print(f"  ({config.num_iterations} iterations * {config.gradient_accumulation_steps} accumulation steps)")
    scaler = torch.amp.GradScaler('cuda')

    stats = StatsTracker(bit_widths=model_config.bit_widths)

    print(f"Iterations: {config.num_iterations}, Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")

    train_iter = iter(train_loader)
    progress_bar = tqdm(range(config.num_iterations), desc="SQuAD Training")

    available_precisions = model_config.bit_widths

    for bits in student_bits:
        calib_mgr.ensure_calibrated(bits)

    current_batch = None

    for iteration in progress_bar:
        total_loss, current_batch = train_step(
            model, train_iter, train_loader, optimizer, scaler,
            available_precisions, distill_mgr, config, iteration, stats, calib_mgr, scheduler, current_batch
        )

        distill_mgr.step()

        stats.update(iteration, total_loss, 32, optimizer)

        if iteration % 20 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            current_lr = scheduler.get_last_lr()[0]
            precision_str = ', '.join([f"{b}:{stats.precision_counts[b]}" for b in sorted(stats.precision_counts.keys())])
            progress_bar.set_postfix({
                'loss': f'{total_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'precisions': precision_str,
                'gpu_alloc': f'{allocated:.1f}GB',
                'gpu_res': f'{reserved:.1f}GB'
            })

        if should_evaluate(iteration, config):
            current_lr = scheduler.get_last_lr()[0]
            print(f"\n[Iter {iteration}] LR: {current_lr:.6f} | Evaluating all precisions...")
            val_losses = {}
            for bits in available_precisions:
                model.set_precision(bits)
                val_loss = evaluate(model, val_loader, device)
                val_losses[bits] = val_loss
            avg_val_loss = sum(val_losses.values()) / len(val_losses)
            stats.add_validation(avg_val_loss)
            val_str = ', '.join([f"{b}b:{v:.4f}" for b, v in sorted(val_losses.items())])
            print(f"[Iter {iteration}] Train: {total_loss:.4f}, Val: [{val_str}]")

            if distill_mgr:
                cache_stats = distill_mgr.get_cache_stats()
                print(f"[Iter {iteration}] Cache stats - Size: {cache_stats['cache_size']}, "
                      f"Misses: {cache_stats['cache_misses']}")

            model.train()

        if iteration % 10 == 0:
            cleanup_memory()

    print("\nTraining complete.")

    return model, stats.to_dict()

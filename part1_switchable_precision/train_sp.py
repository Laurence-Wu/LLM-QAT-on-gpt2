"""
Simplified Switchable Precision Training Module with Self-Distillation

This refactored version provides cleaner separation of concerns and better maintainability.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import gc
import json
import time
import random

from distillation_manager import DistillationManager

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def cleanup_memory():
    """Centralized memory cleanup."""
    torch.cuda.empty_cache()
    gc.collect()


def get_next_batch(train_iter, train_loader):
    """Get next batch from iterator, reset if needed."""
    try:
        return next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        return next(train_iter)

class CalibrationManager:
    """Manages quantizer calibration for different bit-widths."""

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
        """Internal calibration logic with separate weight and input calibration."""
        
        bits_key = f'{bits}bit'

        # CRITICAL: Skip 32-bit as it doesn't need calibration
        if bits >= 32:
            print(f"  Skipping calibration for {bits}-bit (no quantization needed)")
            return

        # Step 1: Calibrate WEIGHT quantizers on actual weight tensors
        print(f"  Step 1: Calibrating weight quantizers for {bits}-bit...")
        weight_calibrated = 0
        weight_errors = []

        for name, module in self.model.named_modules():
            if not hasattr(module, 'quantizers_weight'):
                continue

            if bits_key not in module.quantizers_weight:
                print(f"{name}: Missing {bits_key} in quantizers_weight")
                continue

            weight_quantizer = module.quantizers_weight[bits_key]

            # Get the weight tensor
            if hasattr(module, 'linear') and hasattr(module.linear, 'weight'):
                weight = module.linear.weight.data
            elif hasattr(module, 'weight'):
                weight = module.weight.data
            else:
                weight_errors.append(f"{name}: No weight tensor found")
                continue

            # Calibrate weight quantizer
            try:
                weight_quantizer.start_calibration()
                with torch.no_grad():
                    _ = weight_quantizer(weight)
                weight_quantizer.finish_calibration(debug=False)
                weight_calibrated += 1
            except Exception as e:
                weight_errors.append(f"{name}: {str(e)}")

        print(f"    ✓ Calibrated {weight_calibrated} weight quantizers")
        if weight_errors:
            print(f"    ⚠️ {len(weight_errors)} warnings (showing first 3):")
            for err in weight_errors[:3]:
                print(f"      - {err}")

        # Step 2: Calibrate INPUT quantizers via forward passes (skip LoRA initially)
        print(f"  Step 2: Calibrating input quantizers for {bits}-bit...")

        # Start input quantizer calibration
        input_started = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizers_input') and bits_key in module.quantizers_input:
                module.quantizers_input[bits_key].start_calibration()
                input_started += 1

        print(f"    Started calibration for {input_started} input quantizers")

        # Disable LoRA during calibration forward passes
        self.model.disable_lora_for_calibration()

        # Collect statistics via forward passes
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

        # Re-enable LoRA after calibration
        self.model.enable_lora_after_calibration()

        # Finish input quantizer calibration
        input_calibrated = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizers_input') and bits_key in module.quantizers_input:
                module.quantizers_input[bits_key].finish_calibration(debug=False)
                input_calibrated += 1

        print(f"    ✓ Calibrated {input_calibrated} input quantizers")

        torch.cuda.empty_cache()
        gc.collect()

    def calibrate_lora_only(self, bits, num_batches=10):
        """Calibrate only LoRA quantizers for given bit-width by directly calibrating on weight tensors."""
        if bits >= 32:
            return

        bits_key = f'{bits}bit'

        # Directly calibrate LoRA quantizers on the actual weight tensors
        lora_calibrated = 0
        for name, module in self.model.named_modules():
            if not hasattr(module, 'lora_adapters'):
                continue
            if bits_key not in module.lora_adapters:
                continue

            lora_layer = module.lora_adapters[bits_key]

            # Calibrate quantize_A directly on lora_A weights
            if hasattr(lora_layer, 'quantize_A') and hasattr(lora_layer, 'lora_A') and lora_layer.enabled:
                try:
                    lora_layer.quantize_A.start_calibration()
                    with torch.no_grad():
                        _ = lora_layer.quantize_A(lora_layer.lora_A)
                    lora_layer.quantize_A.finish_calibration(debug=False)
                    lora_calibrated += 1
                except Exception as e:
                    print(f"    Warning calibrating {name} quantize_A: {e}")

            # Calibrate quantize_B directly on lora_B weights
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
        """Ensure the given bit-width is calibrated, calibrate if not."""
        if bits >= 32:
            # 32-bit doesn't need calibration
            return

        if bits not in self.calibrated_bits:
            print(f"  ⚠️ {bits}-bit not calibrated, calibrating now...")
            self.model.set_precision(bits)
            self._calibrate_precision(bits, num_batches=10)
            self.calibrated_bits.add(bits)

        # Print calibration statistics for debugging
        self._print_calibration_stats(bits)

    def _print_calibration_stats(self, bits):
        """Print detailed calibration statistics for debugging."""
        bits_key = f'{bits}bit'
        print(f"\n  📊 Calibration Statistics for {bits}-bit:")

        # Collect statistics for weight quantizers
        weight_stats = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizers_weight') and bits_key in module.quantizers_weight:
                q = module.quantizers_weight[bits_key]
                if q.calibrated:
                    weight_stats.append({
                        'name': f"{name}.weight",
                        'scale': q.scale.mean().item() if q.scale.numel() > 1 else q.scale.item(),
                        'zero_point': q.zero_point.mean().item() if q.zero_point.numel() > 1 else q.zero_point.item(),
                        'min': q.running_min.min().item() if hasattr(q, 'running_min') else 0,
                        'max': q.running_max.max().item() if hasattr(q, 'running_max') else 0,
                        'type': q.quantizer_type
                    })
        # Print weight quantizer statistics
        if weight_stats:
            print(f"  Weight Quantizers ({len(weight_stats)} total):")
            # Check for duplicate scales
            scales = [s['scale'] for s in weight_stats]
            unique_scales = len(set(scales))
            if unique_scales < len(scales):
                print(f"    ⚠️ WARNING: Only {unique_scales} unique scale values out of {len(scales)} quantizers!")

            # Show distribution
            scale_min = min(scales)
            scale_max = max(scales)
            scale_mean = sum(scales) / len(scales)
            print(f"    Scale range: [{scale_min:.6f}, {scale_max:.6f}], mean: {scale_mean:.6f}")

            # Show sample of quantizers with same scale (if any)
            from collections import Counter
            scale_counts = Counter(scales)
            duplicates = [(scale, count) for scale, count in scale_counts.items() if count > 1]
            if duplicates:
                print(f"    Duplicate scales found:")
                for scale, count in duplicates[:3]:  # Show first 3
                    print(f"      Scale {scale:.6f}: {count} quantizers")


# Removed get_next_bitwidth - using fixed precision for distillation-based training


def should_evaluate(iteration, config):
    """Check if evaluation should be performed."""
    return iteration % config.eval_interval == 0 and iteration > 0


def setup_optimizer(model, config):
    """Setup optimizer with only trainable parameters."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Optimizing {len(trainable_params)} trainable parameter tensors")

    return AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.adam_betas,
        eps=config.adam_epsilon
    )


# ============================================================================
# CALIBRATION MANAGER
# ============================================================================

# ============================================================================
# STATISTICS TRACKER
# ============================================================================

class StatsTracker:
    """Tracks and manages training statistics."""

    def __init__(self, bit_widths=None):
        self.iteration_losses = []
        self.validation_losses = []
        self.bit_width_usage = []
        self.learning_rates = []
        self.memory_usage = []

        # Initialize losses_per_bit and precision_counts based on configured bit widths
        if bit_widths:
            self.losses_per_bit = {bits: [] for bits in bit_widths}
            self.precision_counts = {bits: 0 for bits in bit_widths}

    def update(self, iteration, loss, bits, optimizer):
        """Update statistics for current iteration."""
        self.iteration_losses.append(loss)
        self.bit_width_usage.append(bits)
        self.learning_rates.append(optimizer.param_groups[0]['lr'])
        self.memory_usage.append(torch.cuda.memory_allocated() / 1024**2)  # MB

        if bits in self.losses_per_bit:
            self.losses_per_bit[bits].append(loss)

    def add_validation(self, val_loss):
        """Add validation loss."""
        self.validation_losses.append(val_loss)

    def record_precision_usage(self, precision):
        """Record that a precision was used in training."""
        if precision in self.precision_counts:
            self.precision_counts[precision] += 1

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'iteration_losses': self.iteration_losses,
            'validation_losses': self.validation_losses,
            'bit_width_usage': self.bit_width_usage,
            'learning_rates': self.learning_rates,
            'memory_usage': self.memory_usage,
            'losses_per_bit': self.losses_per_bit,
            'precision_counts': self.precision_counts
        }

    def save(self, filepath, model_config=None, training_config=None):
        """Save statistics to JSON file with COMPLETE configuration."""
        data = self.to_dict()

        # Save ALL model configuration attributes
        if model_config:
            model_config_dict = {}
            # Iterate through all attributes of model_config
            for attr_name in dir(model_config):
                if not attr_name.startswith('_'):  # Skip private attributes
                    attr_value = getattr(model_config, attr_name)
                    # Skip methods
                    if not callable(attr_value):
                        model_config_dict[attr_name] = attr_value

            data['model_config'] = model_config_dict

            # Log what was saved for debugging
            print(f"Saved {len(model_config_dict)} model config attributes")

        # Save ALL training configuration attributes
        if training_config:
            training_config_dict = {}
            # Iterate through all attributes of training_config
            for attr_name in dir(training_config):
                if not attr_name.startswith('_'):  # Skip private attributes
                    attr_value = getattr(training_config, attr_name)
                    # Skip methods
                    if not callable(attr_value):
                        training_config_dict[attr_name] = attr_value

            data['training_config'] = training_config_dict

            # Log what was saved for debugging
            print(f"Saved {len(training_config_dict)} training config attributes")

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Training statistics saved to {filepath}")


# ============================================================================
# LOSS COMPUTATION
# ============================================================================

def compute_loss_single_precision(model, batch, precision, teacher_bits, distill_mgr, config, iteration):
    """
    Compute loss for a SINGLE precision following the paper's methodology.
    Each batch trains at ONE randomly selected precision only.
    """
    device = next(model.parameters()).device
    input_ids = batch['input_ids'].to(device, non_blocking=True)
    attention_mask = batch.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to(device, non_blocking=True)

    # Set the model to the selected precision
    model.set_precision(precision)

    with torch.amp.autocast('cuda'):  # Always use AMP
        if precision == teacher_bits:
            # teacher and cache the outputs
            outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True, return_dict=True)
            loss = outputs['loss']

            # Cache teacher outputs for future student training
            with torch.no_grad():
                distill_mgr.update_teacher(input_ids, attention_mask)

        else:
            # student
            # First ensure teacher outputs are cached
            if distill_mgr._get_from_cache(input_ids) is None:
                raise ValueError(f"Teacher outputs not cached for student precision {precision}-bit")
                # # Generate and cache teacher outputs first
                # with torch.no_grad():
                #     model.set_precision(teacher_bits)
                #     distill_mgr.update_teacher(input_ids, attention_mask)
                #     model.set_precision(precision)  # Switch back to student precision

            # Now compute student outputs and distillation loss
            outputs = model(input_ids, output_hidden_states=True, return_dict=True)
            loss = distill_mgr.compute_distillation_loss(outputs, input_ids)

    # Clean up
    del outputs, input_ids
    if attention_mask is not None:
        del attention_mask

    return loss / config.gradient_accumulation_steps


# ============================================================================
# TRAINING STEP
# ============================================================================

def train_step(model, train_iter, train_loader, optimizer, scaler,
               available_precisions, distill_mgr, config, iteration, stats_tracker,calib_mgr,scheduler):

    optimizer.zero_grad(set_to_none=True)
    total_loss = 0
    precisions_used = []
    # Ensure training mode
    model.train()
    torch.set_grad_enabled(True)

    # Process gradient accumulation steps
    for bit_step in range(config.gradient_accumulation_steps):
        # Get batch
        batch = get_next_batch(train_iter, train_loader)

        # CRITICAL: Randomly sample ONE precision for this batch
        # Over many iterations, all precisions will be trained
        teacher_bit = max(available_precisions)
        student_bits = [b for b in available_precisions if b != teacher_bit]
        if bit_step % len(available_precisions) == 0:
            precision = random.choice(student_bits)
        else:
            precision = teacher_bit
        precisions_used.append(precision)

        if calib_mgr and iteration > 0 and precision < 32:
            model.set_precision(precision)
            calib_mgr.calibrate_lora_only(precision, num_batches=2)

        # Track precision usage
        if stats_tracker:
            stats_tracker.record_precision_usage(precision)

        # Compute loss for the selected precision only
        loss = compute_loss_single_precision(
            model, batch, precision,
            teacher_bits=32,  # 32-bit is always the teacher
            distill_mgr=distill_mgr,
            config=config,
            iteration=iteration + bit_step
        )

        total_loss += loss.detach().item()

        # Backward pass - gradients accumulate from the selected precision
        scaler.scale(loss).backward()
        # Update learning rate
        scheduler.step()

        # Clean up
        del batch, loss

    # Print precision distribution for this step (for debugging)
    if iteration % 100 == 0:
        precision_counts = {p: precisions_used.count(p) for p in set(precisions_used)}
        print(f"Step {iteration} precision distribution: {precision_counts}")

    # Optimizer step happens after gradient accumulation, regardless of which precisions were used
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    scaler.step(optimizer)
    scaler.update()

    return total_loss


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, val_loader, device):
    """Quick evaluation on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    max_batches = 5  # Limit for memory efficiency

    with torch.no_grad():
        for batch in val_loader:
            if num_batches >= max_batches:
                break

            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):  # Always use AMP
                outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
                loss = outputs['loss']

            total_loss += loss.item()
            num_batches += 1

            # Clean up
            del outputs, loss, input_ids, batch
            if attention_mask is not None:
                del attention_mask

    cleanup_memory()
    return total_loss / max(num_batches, 1)


def train_sp(model, train_loader, val_loader, config, model_config):

    device = torch.device('cuda')
    model = model.to(device)
    cleanup_memory()

    # Initialize calibration manager and calibrate all student precisions
    calib_mgr = CalibrationManager(model, train_loader, device)
    student_bits = [b for b in model_config.bit_widths if b < 32]  # Calibrate all students (4, 8, 16)
    calib_mgr.calibrate_all_precisions(student_bits)

    # Initialize distillation manager if enabled
    distill_mgr = None
    if config.use_distillation:
        teacher_bits = model_config.teacher_bits if hasattr(model_config, 'teacher_bits') else 32
        distill_mgr = DistillationManager(
            model=model,
            full_precision_bits=teacher_bits,  # Use FP32 as teacher
            config=config
        )

    # Setup optimizer and scheduler
    optimizer = setup_optimizer(model, config)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_iterations)
    scaler = torch.amp.GradScaler('cuda')  # Always use AMP for mixed precision training

    # Initialize statistics tracker with configured bit widths
    stats = StatsTracker(bit_widths=model_config.bit_widths)

    # Training info
    print(f"\n{'Starting SP training with distillation' if config.use_distillation else 'Starting SP training'}")
    print(f"Iterations: {config.num_iterations}, Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")

    # Main training loop
    train_iter = iter(train_loader)
    progress_bar = tqdm(range(config.num_iterations), desc="SP Training")

    # All available precisions for random sampling (including teacher)

    available_precisions = [model_config.bit_widths]

    for bits in student_bits:
        calib_mgr.ensure_calibrated(bits)

    for iteration in progress_bar:
        # Execute training steps (cycles all the bit widths) with random precision sampling
        total_loss = train_step(
            model, train_iter, train_loader, optimizer, scaler,
            available_precisions, distill_mgr, config, iteration, stats, calib_mgr,scheduler
        )

        # Update distillation manager
        distill_mgr.step()

        # Track statistics (use max precision for tracking)
        stats.update(iteration, total_loss, 32, optimizer)  # Use 32 as default for tracking

        # Update progress bar with precision distribution
        if iteration % 20 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            # Get precision distribution from stats
            precision_str = ', '.join([f"{b}:{stats.precision_counts[b]}" for b in sorted(stats.precision_counts.keys())])
            progress_bar.set_postfix({
                'loss': f'{total_loss:.4f}',
                'precisions': precision_str,
                'gpu_alloc': f'{allocated:.1f}GB',
                'gpu_res': f'{reserved:.1f}GB'
            })

        # Periodic evaluation
        if should_evaluate(iteration, config):
            print(f"\n[Iter {iteration}] Evaluating all precisions...")
            val_losses = {}
            for bits in available_precisions:
                model.set_precision(bits)
                val_loss = evaluate(model, val_loader, device)
                val_losses[bits] = val_loss
            avg_val_loss = sum(val_losses.values()) / len(val_losses)
            stats.add_validation(avg_val_loss)
            val_str = ', '.join([f"{b}b:{v:.4f}" for b, v in sorted(val_losses.items())])
            print(f"[Iter {iteration}] Train: {total_loss:.4f}, Val: [{val_str}]")

            # Display cache statistics if using distillation
            if distill_mgr:
                cache_stats = distill_mgr.get_cache_stats()
                print(f"[Iter {iteration}] Cache stats - Size: {cache_stats['cache_size']}, "
                      f"Hit rate: {cache_stats['hit_rate']:.2%}, "
                      f"Hits: {cache_stats['cache_hits']}, Misses: {cache_stats['cache_misses']}")

            model.train()  # Return to training mode

        # Periodic memory cleanup
        if iteration % 10 == 0:
            cleanup_memory()

    print("\nTraining complete.")

    # Save statistics
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    stats.save(
        f'sp_training_stats_{timestamp}.json',
        model_config=model_config,
        training_config=config
    )

    return model, stats.to_dict()
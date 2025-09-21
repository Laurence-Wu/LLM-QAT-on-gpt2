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
try:
    from .distillation_manager import DistillationManager
except ImportError:
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

class CalibrationManager:
    """Manages quantizer calibration for different bit-widths."""

    def __init__(self, model, train_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.calibrated_bits = set()

    def calibrate_all_precisions(self, bit_widths, num_batches=10):
        """Calibrate all bit-widths at once during initialization."""
        print("\n📊 INITIAL CALIBRATION PHASE")
        print("=" * 50)

        for bits in bit_widths:
            if bits < 32 and bits not in self.calibrated_bits:  # Calibrate all students (4, 8, 16)
                print(f"\nCalibrating {bits}-bit precision...")
                self.model.set_precision(bits)
                self._calibrate_precision(bits, num_batches)
                self.calibrated_bits.add(bits)

        print("\n✅ Initial calibration complete for all precisions")
        print("=" * 50)

    def ensure_calibrated(self, bits, num_batches=5):
        """Ensure a specific bit-width is calibrated."""
        if bits < 32 and bits not in self.calibrated_bits:  # Calibrate all students
            print(f"\n⚠️ {bits}-bit not calibrated, calibrating now...")
            self._calibrate_precision(bits, num_batches)
            self.calibrated_bits.add(bits)
            print(f"✅ Calibration complete for {bits}-bit")

    def _calibrate_precision(self, bits, num_batches):
        """Internal calibration logic."""
        # Ensure model is in training mode for calibration
        self.model.train()

        # Start calibration for all quantizers at this precision
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizers_weight') and hasattr(module, 'quantizers_input'):
                bits_key = f'{bits}bit'
                if bits_key in module.quantizers_weight:
                    module.quantizers_weight[bits_key].start_calibration()
                if bits_key in module.quantizers_input:
                    module.quantizers_input[bits_key].start_calibration()

        # Collect statistics
        train_iter = iter(self.train_loader)
        with torch.no_grad():
            for i in range(num_batches):
                try:
                    batch = next(train_iter)
                    input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                    _ = self.model(input_ids)
                    del batch, input_ids
                except StopIteration:
                    break

        # Finish calibration
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizers_weight') and hasattr(module, 'quantizers_input'):
                bits_key = f'{bits}bit'
                if bits_key in module.quantizers_weight:
                    module.quantizers_weight[bits_key].finish_calibration()
                if bits_key in module.quantizers_input:
                    module.quantizers_input[bits_key].finish_calibration()

        cleanup_memory()


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
        """Save statistics to JSON file."""
        data = self.to_dict()

        # Add configuration information if provided
        if model_config:
            data['model_config'] = {
                'quantization_bits': getattr(model_config, 'quantization_bits', None),
                'bit_widths': getattr(model_config, 'bit_widths', None),
                'n_layer': getattr(model_config, 'n_layer', None),
                'n_embd': getattr(model_config, 'n_embd', None),
                'n_head': getattr(model_config, 'n_head', None)
            }

        if training_config:
            data['training_config'] = {
                'batch_size': training_config.batch_size,
                'learning_rate': training_config.learning_rate,
                'num_iterations': training_config.num_iterations,
                'gradient_accumulation_steps': training_config.gradient_accumulation_steps,
            }

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
            # TEACHER MODE (32-bit): Train with ground truth and cache outputs
            outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True, return_dict=True)
            loss = outputs['loss']

            # Cache teacher outputs for future student training
            with torch.no_grad():
                if distill_mgr:
                    # Store the teacher outputs in cache
                    distill_mgr.update_teacher(input_ids, attention_mask)

        else:
            # STUDENT MODE (4/8/16-bit): Train with distillation
            outputs = model(input_ids, output_hidden_states=True, return_dict=True)

            # Try to get cached teacher outputs
            if distill_mgr and distill_mgr._get_from_cache(input_ids) is not None:
                # Use distillation loss if teacher outputs are available
                loss = distill_mgr.compute_distillation_loss(outputs, input_ids)
            else:
                print(f"Warning: No cached teacher for {precision}-bit, using standard loss")

    # Clean up
    del outputs, input_ids
    if attention_mask is not None:
        del attention_mask

    return loss / config.gradient_accumulation_steps


# ============================================================================
# TRAINING STEP
# ============================================================================

def train_step(model, train_iter, train_loader, optimizer, scaler,
               available_precisions, distill_mgr, config, iteration, stats_tracker=None):
    """
    Execute a single training step with random precision sampling.
    Each batch in the gradient accumulation uses a randomly selected precision.
    The optimizer steps after all gradient accumulation steps, regardless of which precisions were used.
    """
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0
    precisions_used = []

    # Ensure training mode
    model.train()
    torch.set_grad_enabled(True)

    # Process gradient accumulation steps
    for step in range(config.gradient_accumulation_steps):
        # Get batch
        batch = get_next_batch(train_iter, train_loader)

        # CRITICAL: Randomly sample ONE precision for this batch
        # Over many iterations, all precisions will be trained
        precision = random.choice(available_precisions)
        precisions_used.append(precision)

        # Track precision usage
        if stats_tracker:
            stats_tracker.record_precision_usage(precision)

        # Compute loss for the selected precision only
        loss = compute_loss_single_precision(
            model, batch, precision,
            teacher_bits=32,  # 32-bit is always the teacher
            distill_mgr=distill_mgr,
            config=config,
            iteration=iteration + step
        )

        total_loss += loss.detach().item()

        # Backward pass - gradients accumulate from the selected precision
        scaler.scale(loss).backward()

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


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_sp(model, train_loader, val_loader, config, model_config):
    """Simplified SP training with clean separation of concerns."""

    # Setup
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
    available_precisions = model_config.bit_widths  # [6, 8, 16, 32]

    # Ensure all student precisions are calibrated
    student_bits = [b for b in available_precisions if b < 32]
    for bits in student_bits:
        calib_mgr.ensure_calibrated(bits)

    for iteration in progress_bar:
        # Execute training step with random precision sampling
        total_loss = train_step(
            model, train_iter, train_loader, optimizer, scaler,
            available_precisions, distill_mgr, config, iteration, stats
        )

        # Update learning rate
        scheduler.step()

        # Update distillation manager
        if distill_mgr:
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

    # Clean up distillation manager
    if distill_mgr:
        distill_mgr.clear_cache()

    # Save statistics
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    stats.save(
        f'sp_training_stats_{timestamp}.json',
        model_config=model_config,
        training_config=config
    )

    return model, stats.to_dict()
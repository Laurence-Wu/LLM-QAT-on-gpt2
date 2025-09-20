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
            if bits < 16 and bits not in self.calibrated_bits:
                print(f"\nCalibrating {bits}-bit precision...")
                self.model.set_precision(bits)
                self._calibrate_precision(bits, num_batches)
                self.calibrated_bits.add(bits)

        print("\n✅ Initial calibration complete for all precisions")
        print("=" * 50)

    def ensure_calibrated(self, bits, num_batches=5):
        """Ensure a specific bit-width is calibrated."""
        if bits < 16 and bits not in self.calibrated_bits:
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

    def __init__(self):
        self.iteration_losses = []
        self.validation_losses = []
        self.bit_width_usage = []
        self.learning_rates = []
        self.memory_usage = []
        self.losses_per_bit = {4: [], 8: [], 16: []}

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

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'iteration_losses': self.iteration_losses,
            'validation_losses': self.validation_losses,
            'bit_width_usage': self.bit_width_usage,
            'learning_rates': self.learning_rates,
            'memory_usage': self.memory_usage,
            'losses_per_bit': self.losses_per_bit
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

def compute_loss_for_all_students(model, batch, teacher_bits, student_bits_list, distill_mgr, config, iteration):
    """Compute loss for teacher and all student precisions."""
    device = next(model.parameters()).device
    input_ids = batch['input_ids'].to(device, non_blocking=True)
    attention_mask = batch.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to(device, non_blocking=True)

    total_loss = 0
    num_students = 0

    # First, compute teacher outputs and update cache (using FP32)
    model.set_precision(teacher_bits)
    with torch.amp.autocast('cuda', enabled=config.use_amp):
        # Teacher forward pass to update cache
        if distill_mgr and distill_mgr.should_update_teacher(teacher_bits, iteration):
            distill_mgr.update_teacher(input_ids, attention_mask)

    # Now compute distillation loss for each student precision
    for student_bits in student_bits_list:
        model.set_precision(student_bits)

        with torch.amp.autocast('cuda', enabled=config.use_amp):
            if student_bits == teacher_bits:
                # Teacher trains with standard loss
                outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
                loss = outputs['loss']
            else:
                # Students train with distillation loss
                outputs = model(input_ids, output_hidden_states=True, return_dict=True)
                loss = distill_mgr.compute_distillation_loss(outputs, input_ids)

            total_loss += loss
            num_students += 1

    # Average loss across all precisions
    avg_loss = total_loss / num_students

    # Clean up
    del outputs, input_ids
    if attention_mask is not None:
        del attention_mask

    return avg_loss / config.gradient_accumulation_steps


# ============================================================================
# TRAINING STEP
# ============================================================================

def train_step(model, train_iter, train_loader, optimizer, scaler,
               teacher_bits, student_bits_list, distill_mgr, config, iteration):
    """Execute a single training step for all student precisions with distillation."""
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0

    # Ensure training mode
    model.train()
    torch.set_grad_enabled(True)

    # Process gradient accumulation steps
    for step in range(config.gradient_accumulation_steps):
        # Get batch
        batch = get_next_batch(train_iter, train_loader)

        # Compute loss for teacher and all students
        loss = compute_loss_for_all_students(
            model, batch, teacher_bits, student_bits_list,
            distill_mgr, config, iteration
        )
        total_loss += loss.detach().item()

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Clean up
        del batch, loss

    # Optimizer step
    if scaler:
        # Check if any gradients were computed
        has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)

        if has_gradients:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            print(f"ERROR: No gradients computed at iteration {iteration}")
            print(f"  current_bits: {current_bits}")
            print(f"  total_loss: {total_loss}")
            # Skip optimizer step but update scaler to prevent assertion error
            scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

    return total_loss


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, val_loader, device, use_amp):
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

            with torch.amp.autocast('cuda', enabled=use_amp):
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

    # Initialize calibration manager and calibrate all precisions
    calib_mgr = CalibrationManager(model, train_loader, device)
    calib_mgr.calibrate_all_precisions(model_config.bit_widths)

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
    scaler = torch.amp.GradScaler('cuda') if config.use_amp else None

    # Initialize statistics tracker
    stats = StatsTracker()

    # Training info
    print(f"\n{'Starting SP training with distillation' if config.use_distillation else 'Starting SP training'}")
    print(f"Iterations: {config.num_iterations}, Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")

    # Main training loop
    train_iter = iter(train_loader)
    progress_bar = tqdm(range(config.num_iterations), desc="SP Training")

    # Get teacher and student bit-widths from config
    teacher_bits = model_config.teacher_bits if hasattr(model_config, 'teacher_bits') else 32
    student_bits_list = model_config.bit_widths  # [4, 8, 16]

    # Ensure all student precisions are calibrated
    for bits in student_bits_list:
        calib_mgr.ensure_calibrated(bits)

    for iteration in progress_bar:
        # Execute training step for all precisions
        total_loss = train_step(
            model, train_iter, train_loader, optimizer, scaler,
            teacher_bits, student_bits_list, distill_mgr, config, iteration
        )

        # Update learning rate
        scheduler.step()

        # Update distillation manager
        if distill_mgr:
            distill_mgr.step()

        # Track statistics (use teacher bits for tracking)
        stats.update(iteration, total_loss, teacher_bits, optimizer)

        # Update progress bar
        if iteration % 20 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            progress_bar.set_postfix({
                'loss': f'{total_loss:.4f}',
                'bits': f'{student_bits_list}',
                'gpu_alloc': f'{allocated:.1f}GB',
                'gpu_res': f'{reserved:.1f}GB'
            })

        # Periodic evaluation
        if should_evaluate(iteration, config):
            val_loss = evaluate(model, val_loader, device, config.use_amp)
            stats.add_validation(val_loss)
            print(f"\n[Iter {iteration}] Train: {total_loss:.4f}, Val: {val_loss:.4f}, Training bits: {student_bits_list}")
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
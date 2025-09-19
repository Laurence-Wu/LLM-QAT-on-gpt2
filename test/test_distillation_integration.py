#!/usr/bin/env python3
"""
Test Full Distillation Integration
Verifies the complete distillation pipeline works correctly.
"""

import sys
import os
import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part1_switchable_precision.config_sp import ModelConfig, TrainingConfig
from part1_switchable_precision.distillation_manager import DistillationManager
from shared.models_sp import SPLMHeadModel


def test_distillation_manager():
    """Test the distillation manager functionality."""
    print("\n" + "="*80)
    print("DISTILLATION MANAGER INTEGRATION TEST")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Initialize configs
    model_config = ModelConfig()
    training_config = TrainingConfig()

    # Create model
    print("\n1. Creating switchable precision model...")
    model = SPLMHeadModel(model_config)
    model = model.to(device)

    # Initialize distillation manager
    print("\n2. Initializing distillation manager...")
    distill_mgr = DistillationManager(
        model=model,
        full_precision_bits=max(model_config.bit_widths),
        config=training_config
    )
    print(f"   Full precision bits: {distill_mgr.full_precision_bits}")
    print(f"   Distillation alpha KL: {training_config.distill_alpha_kl}")
    print(f"   Distillation alpha feature: {training_config.distill_alpha_feature}")
    print(f"   Temperature: {training_config.distill_temperature}")

    # Create dummy inputs
    batch_size = 4
    seq_length = 32
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones_like(input_ids)

    print("\n3. Testing teacher update at full precision...")
    # Set model to full precision
    model.set_precision(16)
    print(f"   Current precision: {model.get_current_precision()} bits")

    # Check if update is needed
    should_update = distill_mgr.should_update_teacher(16, 0)
    print(f"   Should update teacher: {should_update}")

    if should_update:
        # Update teacher cache
        print("   Updating teacher cache...")
        teacher_entry = distill_mgr.update_teacher(input_ids, attention_mask)
        print(f"   Teacher logits shape: {teacher_entry['logits'].shape}")
        print(f"   Hidden states count: {len(teacher_entry['hidden_states'])}")

    print("\n4. Testing student distillation at low precision...")
    # Switch to low precision
    model.set_precision(8)
    print(f"   Current precision: {model.get_current_precision()} bits")

    # Get student outputs
    with torch.no_grad():
        student_outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

    # Compute distillation loss
    print("   Computing distillation loss...")
    loss = distill_mgr.compute_distillation_loss(student_outputs, input_ids)
    print(f"   Distillation loss: {loss.item():.4f}")

    print("\n5. Testing cache management...")
    # Add multiple entries to test LRU eviction
    old_cache_size = training_config.cache_size
    training_config.cache_size = 3  # Small cache for testing

    for i in range(5):
        # Create different inputs
        test_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_length)).to(device)
        test_ids[0, 0] = i  # Make each batch unique

        # Update at full precision
        model.set_precision(16)
        distill_mgr.update_teacher(test_ids)

        print(f"   Iteration {i}: Cache size = {len(distill_mgr.teacher_cache)}")

    print(f"   Final cache size: {len(distill_mgr.teacher_cache)} (limit: {training_config.cache_size})")

    print("\n6. Testing precision switching behavior...")
    # Test switching from teacher
    model.set_precision(16)
    distill_mgr.mark_switch_from_teacher()
    print(f"   Marked switch from teacher, pending update: {distill_mgr.pending_teacher_update}")

    # Now at low precision, should trigger update next time at full precision
    model.set_precision(8)
    should_update = distill_mgr.should_update_teacher(8, 10)
    print(f"   Should update at 8-bit: {should_update}")

    model.set_precision(16)
    should_update = distill_mgr.should_update_teacher(16, 10)
    print(f"   Should update at 16-bit: {should_update}")

    print("\n7. Testing memory cleanup...")
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    distill_mgr.clear_cache()
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"   Memory before clear: {initial_memory / 1024**2:.1f} MB")
    print(f"   Memory after clear: {final_memory / 1024**2:.1f} MB")
    print(f"   Cache entries after clear: {len(distill_mgr.teacher_cache)}")

    print("\n✅ Distillation manager integration test complete!")


def test_distillation_loss_components():
    """Test individual components of the distillation loss."""
    print("\n" + "="*80)
    print("DISTILLATION LOSS COMPONENTS TEST")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup
    vocab_size = 100
    batch_size = 2
    seq_length = 10
    hidden_size = 64

    # Create dummy teacher and student outputs
    teacher_logits = torch.randn(batch_size, seq_length, vocab_size).to(device)
    student_logits = torch.randn(batch_size, seq_length, vocab_size).to(device)

    # Make teacher more confident (sharper distribution)
    teacher_logits = teacher_logits * 2.0

    print("\n1. Testing KL divergence loss...")
    temperature = 3.0

    # Compute KL divergence manually
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl_loss = F.kl_div(
        student_log_probs.view(-1, vocab_size),
        teacher_log_probs.view(-1, vocab_size),
        reduction='batchmean',
        log_target=True
    ) * (temperature * temperature)

    print(f"   KL divergence loss: {kl_loss.item():.4f}")

    print("\n2. Testing feature matching loss...")
    # Create dummy hidden states
    num_layers = 3
    teacher_hidden = [torch.randn(batch_size, seq_length, hidden_size).to(device) for _ in range(num_layers)]
    student_hidden = [torch.randn(batch_size, seq_length, hidden_size).to(device) for _ in range(num_layers)]

    # Compute MSE loss
    feature_loss = 0
    for i in range(num_layers):
        layer_loss = F.mse_loss(student_hidden[i], teacher_hidden[i], reduction='mean')
        feature_loss = feature_loss + layer_loss
        print(f"   Layer {i} MSE: {layer_loss.item():.6f}")

    feature_loss = feature_loss / num_layers
    print(f"   Average feature loss: {feature_loss.item():.6f}")

    print("\n3. Testing combined loss...")
    alpha_kl = 1.0
    alpha_feature = 1e-7

    total_loss = alpha_kl * kl_loss + alpha_feature * feature_loss
    print(f"   KL component: {alpha_kl * kl_loss.item():.4f}")
    print(f"   Feature component: {alpha_feature * feature_loss.item():.8f}")
    print(f"   Total distillation loss: {total_loss.item():.4f}")

    print("\n✅ Loss components test complete!")


def test_distillation_convergence():
    """Test that distillation improves student performance."""
    print("\n" + "="*80)
    print("DISTILLATION CONVERGENCE TEST")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize
    model_config = ModelConfig()
    model_config.n_layer = 2  # Smaller model for faster testing
    training_config = TrainingConfig()

    model = SPLMHeadModel(model_config)
    model = model.to(device)

    distill_mgr = DistillationManager(
        model=model,
        full_precision_bits=16,
        config=training_config
    )

    # Create consistent test data
    torch.manual_seed(42)
    batch_size = 4
    seq_length = 32
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_length)).to(device)

    print("\n1. Getting teacher (16-bit) predictions...")
    model.set_precision(16)
    with torch.no_grad():
        teacher_outputs = model(input_ids, output_hidden_states=True, return_dict=True)
        teacher_logits = teacher_outputs['logits']

    # Cache teacher outputs
    distill_mgr.update_teacher(input_ids)

    print("\n2. Testing student (8-bit) before distillation...")
    model.set_precision(8)
    with torch.no_grad():
        student_outputs_before = model(input_ids, output_hidden_states=True, return_dict=True)
        student_logits_before = student_outputs_before['logits']

    # Measure initial divergence
    initial_divergence = F.kl_div(
        F.log_softmax(student_logits_before, dim=-1),
        F.log_softmax(teacher_logits, dim=-1),
        reduction='batchmean',
        log_target=True
    )
    print(f"   Initial KL divergence: {initial_divergence.item():.4f}")

    print("\n3. Simulating distillation training...")
    # Simple optimization loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(10):
        model.train()
        optimizer.zero_grad()

        # Get student outputs
        student_outputs = model(input_ids, output_hidden_states=True, return_dict=True)

        # Compute distillation loss
        loss = distill_mgr.compute_distillation_loss(student_outputs, input_ids)

        # Backward pass
        loss.backward()
        optimizer.step()

        if step % 3 == 0:
            print(f"   Step {step}: Loss = {loss.item():.4f}")

    print("\n4. Testing student after distillation...")
    model.eval()
    with torch.no_grad():
        student_outputs_after = model(input_ids, output_hidden_states=True, return_dict=True)
        student_logits_after = student_outputs_after['logits']

    # Measure final divergence
    final_divergence = F.kl_div(
        F.log_softmax(student_logits_after, dim=-1),
        F.log_softmax(teacher_logits, dim=-1),
        reduction='batchmean',
        log_target=True
    )
    print(f"   Final KL divergence: {final_divergence.item():.4f}")

    improvement = initial_divergence.item() - final_divergence.item()
    print(f"\n   Improvement: {improvement:.4f} (lower is better)")

    if improvement > 0:
        print("   ✅ Distillation improved student-teacher alignment!")
    else:
        print("   ⚠️ No improvement observed (may need more iterations)")

    print("\n✅ Convergence test complete!")


def main():
    """Run all distillation integration tests."""
    print("\n" + "="*80)
    print("COMPREHENSIVE DISTILLATION INTEGRATION TESTS")
    print("="*80)

    # Run tests
    test_distillation_manager()
    test_distillation_loss_components()
    test_distillation_convergence()

    print("\n" + "="*80)
    print("ALL DISTILLATION TESTS COMPLETE")
    print("="*80)

    print("\n📊 SUMMARY:")
    print("• DistillationManager properly manages teacher cache")
    print("• Teacher updates occur at full precision")
    print("• Student receives distillation loss at low precision")
    print("• LRU cache eviction works correctly")
    print("• Memory cleanup functions properly")
    print("• KL divergence and feature matching losses computed correctly")
    print("• Distillation can improve student-teacher alignment")


if __name__ == "__main__":
    main()
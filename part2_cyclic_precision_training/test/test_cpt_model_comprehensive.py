#!/usr/bin/env python3
"""
Comprehensive test suite for CPT (Cyclic Precision Training) model validation.
Tests model structure, quantization, LoRA adapters, and overall correctness.
"""

import torch
import torch.nn as nn
import sys
import os
import time
import gc
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from cpt_model import CPTModel, CPTLinear, CPTSelfAttention, CPTBlock
from config_cpt import ModelConfig, TrainingConfig, CPTConfig
from deploy import save_target_model
from quantization import LearnableFakeQuantize
from switchable_batchnorm import SwitchableLayerNorm
from calibration import CalibrationManager


class CPTModelValidator:
    """Comprehensive validation suite for CPT models."""

    def __init__(self, model_path=None):
        """Initialize validator with optional model path."""
        self.model_path = model_path
        self.model = None
        self.config = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_results = {'passed': 0, 'failed': 0, 'warnings': 0}
        self.failed_tests = []
        self.calibrated = False

    def load_model(self):
        """Load CPT model from checkpoint or create new one."""
        if self.model_path and os.path.exists(self.model_path):
            print(f"\n=== Loading model from {self.model_path} ===")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            # Reconstruct config
            model_config = ModelConfig()
            for key, value in checkpoint['model_config'].items():
                setattr(model_config, key, value)

            training_config = TrainingConfig()
            cpt_config = CPTConfig()

            self.config = {
                'model': model_config,
                'training': training_config,
                'cpt': cpt_config
            }

            # Create and load model
            self.model = CPTModel(self.config)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.model = self.model.to(self.device)

            # Get bit width
            self.current_bits = checkpoint['bit_width']
            print(f"Model loaded at {self.current_bits}-bit precision")
        else:
            print("\n=== Creating new CPT model for testing ===")
            model_config = ModelConfig()
            training_config = TrainingConfig()
            cpt_config = CPTConfig()

            self.config = {
                'model': model_config,
                'training': training_config,
                'cpt': cpt_config
            }

            self.model = CPTModel(self.config)
            self.model = self.model.to(self.device)
            self.current_bits = model_config.default_bits

    def test_model_structure(self):
        """Test 1: Validate model structure and components."""
        print("\n=== Test 1: Model Structure Validation ===")

        # Check main components
        components_to_check = [
            ('wte', 'Word token embeddings'),
            ('wpe', 'Position embeddings'),
            ('drop', 'Dropout layer'),
            ('h', 'Transformer blocks'),
            ('ln_f', 'Final layer norm'),
            ('lm_head', 'Language model head')
        ]

        for attr_name, description in components_to_check:
            if hasattr(self.model, attr_name):
                component = getattr(self.model, attr_name)
                print(f"[PASS] {description} ({attr_name}): {type(component).__name__}")
                self.test_results['passed'] += 1
            else:
                print(f"[FAIL] Missing {description} ({attr_name})")
                self.test_results['failed'] += 1
                self.failed_tests.append(f"Missing {attr_name}")

        # Check transformer blocks
        num_blocks = len(self.model.h)
        expected_blocks = self.config['model'].n_layer
        if num_blocks == expected_blocks:
            print(f"[PASS] Transformer blocks count: {num_blocks}")
            self.test_results['passed'] += 1
        else:
            print(f"[FAIL] Transformer blocks: expected {expected_blocks}, got {num_blocks}")
            self.test_results['failed'] += 1
            self.failed_tests.append(f"Wrong transformer block count")

        # Check each transformer block structure
        for i, block in enumerate(self.model.h):
            has_ln1 = hasattr(block, 'ln_1')
            has_attn = hasattr(block, 'attn')
            has_ln2 = hasattr(block, 'ln_2')
            has_mlp = hasattr(block, 'mlp')

            if has_ln1 and has_attn and has_ln2 and has_mlp:
                if i == 0:  # Only print for first block
                    print(f"[PASS] Block structure validated (ln_1, attn, ln_2, mlp)")
                self.test_results['passed'] += 1
            else:
                print(f"[FAIL] Block {i} missing components")
                self.test_results['failed'] += 1
                self.failed_tests.append(f"Block {i} structure issue")

    def test_quantization_setup(self):
        """Test 2: Validate quantization configuration."""
        print("\n=== Test 2: Quantization Configuration ===")

        bit_widths = self.config['model'].bit_widths
        student_bits = [b for b in bit_widths if b < 32]

        # Check CPTLinear modules for quantizers
        cpt_linear_count = 0
        quantizer_issues = []

        for name, module in self.model.named_modules():
            if isinstance(module, CPTLinear):
                cpt_linear_count += 1

                # Check weight quantizers
                if not hasattr(module, 'quantizers_weight'):
                    quantizer_issues.append(f"{name}: missing quantizers_weight")
                    continue

                # Check input quantizers
                if not hasattr(module, 'quantizers_input'):
                    quantizer_issues.append(f"{name}: missing quantizers_input")
                    continue

                # Validate quantizers for each bit width
                for bits in student_bits:
                    key = f'{bits}bit'

                    # Check weight quantizer
                    if key not in module.quantizers_weight:
                        quantizer_issues.append(f"{name}: missing weight quantizer for {bits}bit")
                    else:
                        wq = module.quantizers_weight[key]
                        if not isinstance(wq, LearnableFakeQuantize):
                            quantizer_issues.append(f"{name}: wrong type for weight quantizer {bits}bit")
                        elif wq.num_bits != bits:
                            quantizer_issues.append(f"{name}: weight quantizer {bits}bit has wrong num_bits: {wq.num_bits}")

                    # Check input quantizer
                    if key not in module.quantizers_input:
                        quantizer_issues.append(f"{name}: missing input quantizer for {bits}bit")
                    else:
                        iq = module.quantizers_input[key]
                        if not isinstance(iq, LearnableFakeQuantize):
                            quantizer_issues.append(f"{name}: wrong type for input quantizer {bits}bit")
                        elif iq.num_bits != bits:
                            quantizer_issues.append(f"{name}: input quantizer {bits}bit has wrong num_bits: {iq.num_bits}")

        print(f"Found {cpt_linear_count} CPTLinear modules")

        if not quantizer_issues:
            print(f"[PASS] All quantizers properly configured for bits: {student_bits}")
            self.test_results['passed'] += 1
        else:
            print(f"[FAIL] Quantizer issues found:")
            for issue in quantizer_issues[:5]:  # Show first 5 issues
                print(f"  - {issue}")
            if len(quantizer_issues) > 5:
                print(f"  ... and {len(quantizer_issues) - 5} more issues")
            self.test_results['failed'] += 1
            self.failed_tests.append("Quantizer configuration issues")

    def test_lora_adapters(self):
        """Test 3: Validate LoRA adapter configuration."""
        print("\n=== Test 3: LoRA Adapter Validation ===")

        bit_widths = self.config['model'].bit_widths
        lora_rank_per_bit = self.config['model'].lora_rank_per_bit
        lora_alpha_per_bit = self.config['model'].lora_alpha_per_bit

        lora_issues = []
        modules_checked = 0

        for name, module in self.model.named_modules():
            if isinstance(module, CPTLinear):
                modules_checked += 1

                if not hasattr(module, 'lora_adapters'):
                    lora_issues.append(f"{name}: missing lora_adapters")
                    continue

                for bits in bit_widths:
                    if bits == 32:  # Skip FP32
                        continue

                    key = f'{bits}bit'
                    expected_rank = lora_rank_per_bit.get(bits, 16)
                    expected_alpha = lora_alpha_per_bit.get(bits, 32)

                    if key not in module.lora_adapters:
                        lora_issues.append(f"{name}: missing LoRA adapter for {bits}bit")
                    else:
                        adapter = module.lora_adapters[key]

                        # Check rank
                        if adapter.rank != expected_rank:
                            lora_issues.append(f"{name}: {bits}bit LoRA rank is {adapter.rank}, expected {expected_rank}")

                        # Check alpha
                        if adapter.alpha != expected_alpha:
                            lora_issues.append(f"{name}: {bits}bit LoRA alpha is {adapter.alpha}, expected {expected_alpha}")

                        # Check parameter shapes
                        if expected_rank > 0:
                            if adapter.lora_A is None or adapter.lora_B is None:
                                lora_issues.append(f"{name}: {bits}bit LoRA parameters are None")
                            else:
                                # Check shapes
                                expected_A_shape = (module.in_features, expected_rank)
                                expected_B_shape = (expected_rank, module.out_features)

                                if adapter.lora_A.shape != expected_A_shape:
                                    lora_issues.append(f"{name}: {bits}bit lora_A shape is {adapter.lora_A.shape}, expected {expected_A_shape}")
                                if adapter.lora_B.shape != expected_B_shape:
                                    lora_issues.append(f"{name}: {bits}bit lora_B shape is {adapter.lora_B.shape}, expected {expected_B_shape}")

        print(f"Checked {modules_checked} CPTLinear modules for LoRA adapters")

        if not lora_issues:
            print(f"[PASS] All LoRA adapters properly configured")
            self.test_results['passed'] += 1
        else:
            print(f"[FAIL] LoRA adapter issues found:")
            for issue in lora_issues[:5]:
                print(f"  - {issue}")
            if len(lora_issues) > 5:
                print(f"  ... and {len(lora_issues) - 5} more issues")
            self.test_results['failed'] += 1
            self.failed_tests.append("LoRA adapter configuration issues")

    def test_weight_shapes(self):
        """Test 4: Validate weight shapes and dimensions."""
        print("\n=== Test 4: Weight Shape Validation ===")

        config = self.config['model']
        shape_checks = []

        # Check embeddings
        wte_shape = self.model.wte.weight.shape
        expected_wte = (config.vocab_size, config.n_embd)
        if wte_shape == expected_wte:
            print(f"[PASS] Token embeddings shape: {wte_shape}")
            self.test_results['passed'] += 1
        else:
            print(f"[FAIL] Token embeddings: expected {expected_wte}, got {wte_shape}")
            self.test_results['failed'] += 1
            self.failed_tests.append("Token embedding shape mismatch")

        wpe_shape = self.model.wpe.weight.shape
        expected_wpe = (config.n_positions, config.n_embd)
        if wpe_shape == expected_wpe:
            print(f"[PASS] Position embeddings shape: {wpe_shape}")
            self.test_results['passed'] += 1
        else:
            print(f"[FAIL] Position embeddings: expected {expected_wpe}, got {wpe_shape}")
            self.test_results['failed'] += 1
            self.failed_tests.append("Position embedding shape mismatch")

        # Check attention layers
        for i, block in enumerate(self.model.h):
            attn = block.attn

            # Check c_attn (combined QKV)
            c_attn_weight = attn.c_attn.linear.weight
            expected_c_attn = (3 * config.n_embd, config.n_embd)
            if c_attn_weight.shape == expected_c_attn:
                if i == 0:  # Only print for first block
                    print(f"[PASS] Attention c_attn shape: {c_attn_weight.shape}")
                self.test_results['passed'] += 1
            else:
                print(f"[FAIL] Block {i} c_attn: expected {expected_c_attn}, got {c_attn_weight.shape}")
                self.test_results['failed'] += 1
                self.failed_tests.append(f"Block {i} c_attn shape mismatch")

            # Check c_proj
            c_proj_weight = attn.c_proj.linear.weight
            expected_c_proj = (config.n_embd, config.n_embd)
            if c_proj_weight.shape == expected_c_proj:
                if i == 0:
                    print(f"[PASS] Attention c_proj shape: {c_proj_weight.shape}")
                self.test_results['passed'] += 1
            else:
                print(f"[FAIL] Block {i} c_proj: expected {expected_c_proj}, got {c_proj_weight.shape}")
                self.test_results['failed'] += 1
                self.failed_tests.append(f"Block {i} c_proj shape mismatch")

        # Check MLP layers
        block = self.model.h[0]
        mlp = block.mlp

        fc_in_weight = mlp.fc_in.linear.weight
        expected_fc_in = (4 * config.n_embd, config.n_embd)
        if fc_in_weight.shape == expected_fc_in:
            print(f"[PASS] MLP fc_in shape: {fc_in_weight.shape}")
            self.test_results['passed'] += 1
        else:
            print(f"[FAIL] MLP fc_in: expected {expected_fc_in}, got {fc_in_weight.shape}")
            self.test_results['failed'] += 1
            self.failed_tests.append("MLP fc_in shape mismatch")

        fc_out_weight = mlp.fc_out.linear.weight
        expected_fc_out = (config.n_embd, 4 * config.n_embd)
        if fc_out_weight.shape == expected_fc_out:
            print(f"[PASS] MLP fc_out shape: {fc_out_weight.shape}")
            self.test_results['passed'] += 1
        else:
            print(f"[FAIL] MLP fc_out: expected {expected_fc_out}, got {fc_out_weight.shape}")
            self.test_results['failed'] += 1
            self.failed_tests.append("MLP fc_out shape mismatch")

        # Check LM head
        lm_head_weight = self.model.lm_head.linear.weight
        expected_lm_head = (config.vocab_size, config.n_embd)
        if lm_head_weight.shape == expected_lm_head:
            print(f"[PASS] LM head shape: {lm_head_weight.shape}")
            self.test_results['passed'] += 1
        else:
            print(f"[FAIL] LM head: expected {expected_lm_head}, got {lm_head_weight.shape}")
            self.test_results['failed'] += 1
            self.failed_tests.append("LM head shape mismatch")

    def test_precision_switching(self):
        """Test 5: Validate precision switching functionality."""
        print("\n=== Test 5: Precision Switching Tests ===")

        bit_widths = self.config['model'].bit_widths
        test_bits = [2, 4, 6, 8, 16]  # Test subset of bit widths

        for bits in test_bits:
            if bits not in bit_widths:
                continue

            # Set precision
            self.model.set_precision(bits)

            # Check current_bits
            if self.model.current_precision == bits:
                print(f"[PASS] Set precision to {bits} bits")
                self.test_results['passed'] += 1
            else:
                print(f"[FAIL] Failed to set precision to {bits} bits (current: {self.model.current_precision})")
                self.test_results['failed'] += 1
                self.failed_tests.append(f"Precision switching failed for {bits} bits")

            # Check if correct quantizers are active
            if bits < 32:
                # Sample a CPTLinear module
                for name, module in self.model.named_modules():
                    if isinstance(module, CPTLinear):
                        if module.current_bits == bits:
                            self.test_results['passed'] += 1
                        else:
                            print(f"[FAIL] Module {name} not updated to {bits} bits")
                            self.test_results['failed'] += 1
                            self.failed_tests.append(f"Module precision mismatch")
                        break  # Check only first CPTLinear

    def calibrate_model(self):
        """Calibrate model quantizers for testing."""
        if self.calibrated:
            return

        print("\n=== Calibrating Model Quantizers ===")

        # Create dummy data for calibration
        batch_size = 4
        seq_len = 128
        num_batches = 5

        bit_widths = self.config['model'].bit_widths
        student_bits = [b for b in bit_widths if b < 32]

        for bits in student_bits:
            print(f"Calibrating {bits}-bit precision...")
            self.model.set_precision(bits)
            bits_key = f'{bits}bit'

            # Step 1: Calibrate weight quantizers
            weight_calibrated = 0
            for name, module in self.model.named_modules():
                if isinstance(module, CPTLinear):
                    if bits_key in module.quantizers_weight:
                        weight_quantizer = module.quantizers_weight[bits_key]
                        weight = module.linear.weight.data

                        weight_quantizer.start_calibration()
                        with torch.no_grad():
                            _ = weight_quantizer(weight)
                        weight_quantizer.finish_calibration(debug=False)
                        weight_calibrated += 1

            print(f"  Calibrated {weight_calibrated} weight quantizers")

            # Step 2: Start calibration for input quantizers
            input_started = 0
            for name, module in self.model.named_modules():
                if isinstance(module, CPTLinear):
                    if bits_key in module.quantizers_input:
                        module.quantizers_input[bits_key].start_calibration()
                        input_started += 1

            print(f"  Started calibration for {input_started} input quantizers")

            # Step 3: Disable LoRA during input calibration
            self.model.disable_lora_for_calibration()

            # Step 4: Run forward passes with dummy data
            with torch.no_grad():
                for i in range(num_batches):
                    dummy_input = torch.randint(0, self.config['model'].vocab_size,
                                               (batch_size, seq_len)).to(self.device)
                    _ = self.model(dummy_input)
                    del dummy_input

            # Step 5: Re-enable LoRA
            self.model.enable_lora_after_calibration()

            # Step 6: Finish calibration for input quantizers
            input_calibrated = 0
            for name, module in self.model.named_modules():
                if isinstance(module, CPTLinear):
                    if bits_key in module.quantizers_input:
                        module.quantizers_input[bits_key].finish_calibration(debug=False)
                        input_calibrated += 1

            print(f"  Calibrated {input_calibrated} input quantizers")

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

        self.calibrated = True
        print("✅ Model calibration complete")

    def test_forward_pass(self):
        """Test 6: Validate forward pass and output format."""
        print("\n=== Test 6: Forward Pass Validation ===")

        # Ensure model is calibrated
        self.calibrate_model()

        # Create sample input
        batch_size = 2
        seq_len = 128
        input_ids = torch.randint(0, self.config['model'].vocab_size,
                                 (batch_size, seq_len)).to(self.device)

        # Test at different bit widths
        test_bits = [6, 8]

        for bits in test_bits:
            if bits not in self.config['model'].bit_widths:
                continue

            self.model.set_precision(bits)
            self.model.eval()

            with torch.no_grad():
                outputs = self.model(input_ids)

            # Check output type
            from transformers.modeling_outputs import CausalLMOutputWithPast
            if isinstance(outputs, CausalLMOutputWithPast):
                print(f"[PASS] {bits}-bit forward pass returns CausalLMOutputWithPast")
                self.test_results['passed'] += 1
            else:
                print(f"[FAIL] {bits}-bit forward pass returns wrong type: {type(outputs)}")
                self.test_results['failed'] += 1
                self.failed_tests.append(f"Wrong output type at {bits} bits")

            # Check logits shape
            expected_shape = (batch_size, seq_len, self.config['model'].vocab_size)
            if outputs.logits.shape == expected_shape:
                print(f"[PASS] {bits}-bit logits shape: {outputs.logits.shape}")
                self.test_results['passed'] += 1
            else:
                print(f"[FAIL] {bits}-bit logits shape: expected {expected_shape}, got {outputs.logits.shape}")
                self.test_results['failed'] += 1
                self.failed_tests.append(f"Wrong logits shape at {bits} bits")

            # Check for NaN or Inf
            if torch.isnan(outputs.logits).any():
                print(f"[FAIL] {bits}-bit forward pass contains NaN values")
                self.test_results['failed'] += 1
                self.failed_tests.append(f"NaN values at {bits} bits")
            elif torch.isinf(outputs.logits).any():
                print(f"[FAIL] {bits}-bit forward pass contains Inf values")
                self.test_results['failed'] += 1
                self.failed_tests.append(f"Inf values at {bits} bits")
            else:
                print(f"[PASS] {bits}-bit forward pass has valid values")
                self.test_results['passed'] += 1

    def test_layer_norms(self):
        """Test 7: Validate SwitchableLayerNorm configuration."""
        print("\n=== Test 7: SwitchableLayerNorm Validation ===")

        bit_widths = self.config['model'].bit_widths
        ln_issues = []

        # Check final layer norm
        ln_f = self.model.ln_f
        if isinstance(ln_f, SwitchableLayerNorm):
            print(f"[PASS] Final layer norm is SwitchableLayerNorm")
            self.test_results['passed'] += 1

            # Check bit-specific parameters
            for bits in bit_widths:
                if bits == 32:
                    continue
                key = f'{bits}bit'
                if key in ln_f.ln_dict:
                    self.test_results['passed'] += 1
                else:
                    ln_issues.append(f"ln_f missing {key}")
        else:
            print(f"[FAIL] Final layer norm is not SwitchableLayerNorm: {type(ln_f)}")
            self.test_results['failed'] += 1
            self.failed_tests.append("Wrong final layer norm type")

        # Check block layer norms
        for i, block in enumerate(self.model.h):
            # Check ln_1
            if isinstance(block.ln_1, SwitchableLayerNorm):
                if i == 0:
                    print(f"[PASS] Block layer norms are SwitchableLayerNorm")
                self.test_results['passed'] += 1
            else:
                ln_issues.append(f"Block {i} ln_1 is not SwitchableLayerNorm")

            # Check ln_2
            if isinstance(block.ln_2, SwitchableLayerNorm):
                self.test_results['passed'] += 1
            else:
                ln_issues.append(f"Block {i} ln_2 is not SwitchableLayerNorm")

        if ln_issues:
            print(f"[FAIL] Layer norm issues:")
            for issue in ln_issues[:5]:
                print(f"  - {issue}")
            self.test_results['failed'] += 1
            self.failed_tests.append("Layer norm configuration issues")

    def test_calibration_mode(self):
        """Test 8: Validate calibration mode functionality."""
        print("\n=== Test 8: Calibration Mode Tests ===")

        # Test enabling calibration mode
        for name, module in self.model.named_modules():
            if isinstance(module, CPTLinear):
                module.calibration_mode = True

                if module.calibration_mode:
                    print(f"[PASS] Calibration mode enabled for CPTLinear modules")
                    self.test_results['passed'] += 1
                else:
                    print(f"[FAIL] Failed to enable calibration mode")
                    self.test_results['failed'] += 1
                    self.failed_tests.append("Calibration mode toggle failed")
                break

        # Test LoRA disabling during calibration
        self.model.disable_lora_for_calibration()
        lora_disabled = True

        for name, module in self.model.named_modules():
            if isinstance(module, CPTLinear):
                if module.lora_disabled:
                    continue
                else:
                    lora_disabled = False
                    break

        if lora_disabled:
            print(f"[PASS] LoRA successfully disabled for calibration")
            self.test_results['passed'] += 1
        else:
            print(f"[FAIL] LoRA not properly disabled")
            self.test_results['failed'] += 1
            self.failed_tests.append("LoRA disable failed")

        # Re-enable LoRA
        self.model.enable_lora_after_calibration()

    def test_memory_and_performance(self):
        """Test 9: Memory usage and performance benchmarks."""
        print("\n=== Test 9: Memory and Performance ===")

        if not torch.cuda.is_available():
            print("[SKIP] CUDA not available for memory tests")
            return

        # Ensure model is calibrated
        self.calibrate_model()

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

        # Test at different bit widths
        test_configs = [
            (8, 128),  # 8-bit, seq_len=128
            (6, 128),  # 6-bit, seq_len=128
            (4, 128),  # 4-bit, seq_len=128
        ]

        for bits, seq_len in test_configs:
            if bits not in self.config['model'].bit_widths:
                continue

            self.model.set_precision(bits)
            self.model.eval()

            # Measure memory
            torch.cuda.reset_peak_memory_stats()
            input_ids = torch.randint(0, 50257, (1, seq_len)).to(self.device)

            # Warmup
            with torch.no_grad():
                _ = self.model(input_ids)

            # Time inference
            torch.cuda.synchronize()
            start_time = time.time()

            num_runs = 10
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = self.model(input_ids)

            torch.cuda.synchronize()
            avg_time = (time.time() - start_time) / num_runs

            # Get memory stats
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB

            print(f"[INFO] {bits}-bit, seq_len={seq_len}:")
            print(f"  - Avg inference time: {avg_time*1000:.2f} ms")
            print(f"  - Peak memory: {peak_memory:.2f} GB")
            print(f"  - Throughput: {seq_len / avg_time:.1f} tokens/sec")

            if avg_time < 1.0:  # Should complete within 1 second
                self.test_results['passed'] += 1
            else:
                print(f"[WARN] Slow inference at {bits} bits")
                self.test_results['warnings'] += 1

    def test_checkpoint_saving(self):
        """Test 10: Validate checkpoint saving functionality."""
        print("\n=== Test 10: Checkpoint Saving Tests ===")

        import tempfile

        # Test saving at target precision (6-bit)
        target_bits = 6
        if target_bits in self.config['model'].bit_widths:
            self.model.set_precision(target_bits)

            with tempfile.TemporaryDirectory() as tmpdir:
                saved_path = save_target_model(
                    self.model,
                    self.config,
                    target_bits,
                    tmpdir
                )

                if os.path.exists(saved_path):
                    print(f"[PASS] Model saved successfully to {saved_path}")
                    self.test_results['passed'] += 1

                    # Try loading it back
                    checkpoint = torch.load(saved_path, map_location='cpu', weights_only=False)

                    # Check required keys
                    required_keys = [
                        'model_state_dict',
                        'model_config',
                        'training_config',
                        'cpt_config',
                        'bit_width'
                    ]

                    missing_keys = []
                    for key in required_keys:
                        if key not in checkpoint:
                            missing_keys.append(key)

                    if not missing_keys:
                        print(f"[PASS] Checkpoint contains all required keys")
                        self.test_results['passed'] += 1
                    else:
                        print(f"[FAIL] Checkpoint missing keys: {missing_keys}")
                        self.test_results['failed'] += 1
                        self.failed_tests.append("Incomplete checkpoint")

                    # Check bit width
                    saved_bits = checkpoint['bit_width']
                    if saved_bits == target_bits:
                        print(f"[PASS] Checkpoint saved at correct precision: {saved_bits} bits")
                        self.test_results['passed'] += 1
                    else:
                        print(f"[FAIL] Wrong bit width in checkpoint: {saved_bits}, expected {target_bits}")
                        self.test_results['failed'] += 1
                        self.failed_tests.append("Wrong checkpoint bit width")
                else:
                    print(f"[FAIL] Failed to save model")
                    self.test_results['failed'] += 1
                    self.failed_tests.append("Model save failed")

    def run_all_tests(self):
        """Run complete test suite."""
        print("="*70)
        print("CPT MODEL COMPREHENSIVE VALIDATION SUITE")
        print("="*70)

        # Load or create model
        self.load_model()

        # Run all tests
        self.test_model_structure()
        self.test_quantization_setup()
        self.test_lora_adapters()
        self.test_weight_shapes()
        self.test_precision_switching()
        self.test_forward_pass()
        self.test_layer_norms()
        self.test_calibration_mode()
        self.test_memory_and_performance()
        self.test_checkpoint_saving()

        # Print summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total Tests Run: {self.test_results['passed'] + self.test_results['failed']}")
        print(f"Passed: {self.test_results['passed']}")
        print(f"Failed: {self.test_results['failed']}")
        print(f"Warnings: {self.test_results['warnings']}")

        if self.failed_tests:
            print(f"\nFailed Tests:")
            for test in self.failed_tests:
                print(f"  - {test}")

        if self.test_results['failed'] == 0:
            print("\n✅ ALL TESTS PASSED! Model is correctly configured.")
        else:
            print(f"\n❌ {self.test_results['failed']} TESTS FAILED. Please review the issues above.")

        return self.test_results['failed'] == 0


def main():
    """Main test execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive CPT Model Validation')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to saved CPT model checkpoint')
    args = parser.parse_args()

    # Run validator
    validator = CPTModelValidator(args.model_path)
    success = validator.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
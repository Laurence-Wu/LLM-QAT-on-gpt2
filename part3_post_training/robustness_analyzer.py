"""
Robustness Analysis Module
Analyzes model robustness to adversarial attacks, noise, and perturbations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
import gc


class RobustnessAnalyzer:
    """
    Analyzes various aspects of model robustness.
    """
    
    def __init__(self, model):
        """
        Initialize the robustness analyzer.
        
        Args:
            model: The trained model to analyze
        """
        self.model = model
        self.device = next(model.parameters()).device
        
    def test_adversarial_robustness(self, data_loader, max_batches: int = 20) -> Dict:
        """
        Test model robustness against adversarial attacks.
        
        Args:
            data_loader: DataLoader for evaluation
            max_batches: Maximum batches to test
        
        Returns:
            Dictionary of adversarial robustness metrics
        """
        print("Testing adversarial robustness...")
        
        # Test different attack methods
        attacks = {
            'fgsm_eps_0.01': {'method': 'fgsm', 'epsilon': 0.01},
            'fgsm_eps_0.1': {'method': 'fgsm', 'epsilon': 0.1},
            'pgd_eps_0.01': {'method': 'pgd', 'epsilon': 0.01, 'steps': 5},
            'pgd_eps_0.1': {'method': 'pgd', 'epsilon': 0.1, 'steps': 5}
        }
        
        results = {}
        
        # Baseline clean accuracy
        clean_accuracy = self._evaluate_clean_accuracy(data_loader, max_batches)
        results['clean_accuracy'] = clean_accuracy
        
        # Test each attack
        for attack_name, attack_params in attacks.items():
            print(f"  Testing {attack_name}...")
            
            if attack_params['method'] == 'fgsm':
                robust_acc = self._test_fgsm_attack(
                    data_loader, attack_params['epsilon'], max_batches
                )
            elif attack_params['method'] == 'pgd':
                robust_acc = self._test_pgd_attack(
                    data_loader, 
                    attack_params['epsilon'], 
                    attack_params['steps'],
                    max_batches
                )
            
            results[attack_name] = robust_acc
            
            print(f"    Clean: {clean_accuracy:.4f}, Robust: {robust_acc:.4f}")
        
        # Calculate robustness metrics
        worst_case_accuracy = min(results[k] for k in results if k != 'clean_accuracy')
        robustness_gap = clean_accuracy - worst_case_accuracy
        robustness_ratio = worst_case_accuracy / max(clean_accuracy, 1e-8)
        
        results.update({
            'worst_case_accuracy': worst_case_accuracy,
            'robustness_gap': robustness_gap,
            'robustness_ratio': robustness_ratio
        })
        
        return results
    
    def _evaluate_clean_accuracy(self, data_loader, max_batches: int) -> float:
        """
        Evaluate clean accuracy without any perturbations.
        
        Args:
            data_loader: DataLoader for evaluation
            max_batches: Maximum batches to evaluate
        
        Returns:
            Clean accuracy
        """
        self.model.eval()
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                outputs = self.model(input_ids, labels=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                # Calculate accuracy
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                predictions = torch.argmax(shift_logits, dim=-1)
                correct = (predictions == shift_labels)
                
                if attention_mask is not None:
                    shift_mask = attention_mask[..., 1:].contiguous()
                    correct = correct * shift_mask
                    valid_tokens = shift_mask.sum().item()
                else:
                    valid_tokens = shift_labels.numel()
                
                total_correct += correct.sum().item()
                total_tokens += valid_tokens
        
        return total_correct / max(total_tokens, 1)
    
    def _test_fgsm_attack(self, data_loader, epsilon: float, max_batches: int) -> float:
        """
        Test Fast Gradient Sign Method (FGSM) attack.
        
        Args:
            data_loader: DataLoader for evaluation
            epsilon: Attack strength
            max_batches: Maximum batches to test
        
        Returns:
            Robust accuracy under FGSM attack
        """
        self.model.eval()
        total_correct = 0
        total_tokens = 0
        
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Get embeddings and enable gradients
            embeddings = self.model.wte(input_ids).requires_grad_(True)
            
            # Forward pass with embeddings
            outputs = self.model(inputs_embeds=embeddings, labels=input_ids, attention_mask=attention_mask)
            loss = outputs['loss']
            
            # Compute gradients
            loss.backward()
            
            # Generate adversarial embeddings using FGSM
            adv_embeddings = embeddings + epsilon * embeddings.grad.sign()
            
            # Forward pass with adversarial embeddings
            with torch.no_grad():
                adv_outputs = self.model(inputs_embeds=adv_embeddings, attention_mask=attention_mask)
                adv_logits = adv_outputs['logits']
                
                # Calculate accuracy
                shift_logits = adv_logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                predictions = torch.argmax(shift_logits, dim=-1)
                correct = (predictions == shift_labels)
                
                if attention_mask is not None:
                    shift_mask = attention_mask[..., 1:].contiguous()
                    correct = correct * shift_mask
                    valid_tokens = shift_mask.sum().item()
                else:
                    valid_tokens = shift_labels.numel()
                
                total_correct += correct.sum().item()
                total_tokens += valid_tokens
            
            # Clear gradients and cache
            self.model.zero_grad()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        
        return total_correct / max(total_tokens, 1)
    
    def _test_pgd_attack(self, data_loader, epsilon: float, steps: int, max_batches: int) -> float:
        """
        Test Projected Gradient Descent (PGD) attack.
        
        Args:
            data_loader: DataLoader for evaluation
            epsilon: Attack strength
            steps: Number of PGD steps
            max_batches: Maximum batches to test
        
        Returns:
            Robust accuracy under PGD attack
        """
        self.model.eval()
        total_correct = 0
        total_tokens = 0
        alpha = epsilon / steps * 2  # Step size
        
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Get original embeddings
            orig_embeddings = self.model.wte(input_ids)
            
            # Initialize adversarial embeddings with random noise
            adv_embeddings = orig_embeddings.clone().detach()
            adv_embeddings += torch.empty_like(adv_embeddings).uniform_(-epsilon, epsilon)
            
            # PGD iterations
            for _ in range(steps):
                adv_embeddings.requires_grad_(True)
                
                outputs = self.model(inputs_embeds=adv_embeddings, labels=input_ids, attention_mask=attention_mask)
                loss = outputs['loss']
                
                # Compute gradients
                loss.backward()
                
                # Update adversarial embeddings
                with torch.no_grad():
                    adv_embeddings = adv_embeddings + alpha * adv_embeddings.grad.sign()
                    
                    # Project back to epsilon ball
                    perturbation = adv_embeddings - orig_embeddings
                    perturbation = torch.clamp(perturbation, -epsilon, epsilon)
                    adv_embeddings = orig_embeddings + perturbation
                
                # Clear gradients
                self.model.zero_grad()
                adv_embeddings = adv_embeddings.detach()
            
            # Evaluate with final adversarial embeddings
            with torch.no_grad():
                adv_outputs = self.model(inputs_embeds=adv_embeddings, attention_mask=attention_mask)
                adv_logits = adv_outputs['logits']
                
                # Calculate accuracy
                shift_logits = adv_logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                predictions = torch.argmax(shift_logits, dim=-1)
                correct = (predictions == shift_labels)
                
                if attention_mask is not None:
                    shift_mask = attention_mask[..., 1:].contiguous()
                    correct = correct * shift_mask
                    valid_tokens = shift_mask.sum().item()
                else:
                    valid_tokens = shift_labels.numel()
                
                total_correct += correct.sum().item()
                total_tokens += valid_tokens
            
            # Clear cache
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        
        return total_correct / max(total_tokens, 1)
    
    def test_quantization_noise(self, data_loader, max_batches: int = 20) -> Dict:
        """
        Test robustness to quantization noise.
        
        Args:
            data_loader: DataLoader for evaluation
            max_batches: Maximum batches to test
        
        Returns:
            Dictionary of quantization noise robustness results
        """
        print("Testing quantization noise robustness...")
        
        results = {}
        noise_levels = [0.001, 0.01, 0.1]  # Different noise levels
        
        # Baseline without noise
        baseline_accuracy = self._evaluate_clean_accuracy(data_loader, max_batches)
        results['baseline'] = baseline_accuracy
        
        # Test different noise levels
        for noise_level in noise_levels:
            print(f"  Testing noise level {noise_level}...")
            
            noisy_accuracy = self._test_weight_noise(data_loader, noise_level, max_batches)
            results[f'noise_{noise_level}'] = noisy_accuracy
            
            print(f"    Baseline: {baseline_accuracy:.4f}, Noisy: {noisy_accuracy:.4f}")
        
        # Calculate noise robustness metrics
        worst_noise_accuracy = min(results[k] for k in results if k != 'baseline')
        noise_robustness_gap = baseline_accuracy - worst_noise_accuracy
        
        results.update({
            'worst_noise_accuracy': worst_noise_accuracy,
            'noise_robustness_gap': noise_robustness_gap
        })
        
        return results
    
    def _test_weight_noise(self, data_loader, noise_level: float, max_batches: int) -> float:
        """
        Test model with weight noise added.
        
        Args:
            data_loader: DataLoader for evaluation
            noise_level: Standard deviation of Gaussian noise
            max_batches: Maximum batches to test
        
        Returns:
            Accuracy with weight noise
        """
        self.model.eval()
        
        # Store original weights
        original_weights = {}
        for name, param in self.model.named_parameters():
            original_weights[name] = param.data.clone()
        
        # Add noise to weights
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                noise = torch.randn_like(param) * noise_level
                param.data += noise
        
        # Evaluate with noisy weights
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                outputs = self.model(input_ids, labels=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                # Calculate accuracy
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                predictions = torch.argmax(shift_logits, dim=-1)
                correct = (predictions == shift_labels)
                
                if attention_mask is not None:
                    shift_mask = attention_mask[..., 1:].contiguous()
                    correct = correct * shift_mask
                    valid_tokens = shift_mask.sum().item()
                else:
                    valid_tokens = shift_labels.numel()
                
                total_correct += correct.sum().item()
                total_tokens += valid_tokens
        
        # Restore original weights
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = original_weights[name]
        
        return total_correct / max(total_tokens, 1)
    
    def test_input_perturbations(self, data_loader, max_batches: int = 20) -> Dict:
        """
        Test robustness to input perturbations.
        
        Args:
            data_loader: DataLoader for evaluation
            max_batches: Maximum batches to test
        
        Returns:
            Dictionary of input perturbation robustness results
        """
        print("Testing input perturbation robustness...")
        
        results = {}
        
        # Baseline
        baseline_accuracy = self._evaluate_clean_accuracy(data_loader, max_batches)
        results['baseline'] = baseline_accuracy
        
        # Test different perturbation types
        perturbations = {
            'token_dropout_0.1': {'type': 'dropout', 'prob': 0.1},
            'token_dropout_0.2': {'type': 'dropout', 'prob': 0.2},
            'token_substitution_0.1': {'type': 'substitution', 'prob': 0.1},
            'token_substitution_0.2': {'type': 'substitution', 'prob': 0.2}
        }
        
        for pert_name, pert_params in perturbations.items():
            print(f"  Testing {pert_name}...")
            
            if pert_params['type'] == 'dropout':
                accuracy = self._test_token_dropout(data_loader, pert_params['prob'], max_batches)
            elif pert_params['type'] == 'substitution':
                accuracy = self._test_token_substitution(data_loader, pert_params['prob'], max_batches)
            
            results[pert_name] = accuracy
            print(f"    Baseline: {baseline_accuracy:.4f}, Perturbed: {accuracy:.4f}")
        
        # Calculate perturbation robustness metrics
        worst_pert_accuracy = min(results[k] for k in results if k != 'baseline')
        perturbation_robustness_gap = baseline_accuracy - worst_pert_accuracy
        
        results.update({
            'worst_perturbation_accuracy': worst_pert_accuracy,
            'perturbation_robustness_gap': perturbation_robustness_gap
        })
        
        return results
    
    def _test_token_dropout(self, data_loader, dropout_prob: float, max_batches: int) -> float:
        """
        Test with random token dropout.
        
        Args:
            data_loader: DataLoader for evaluation
            dropout_prob: Probability of dropping each token
            max_batches: Maximum batches to test
        
        Returns:
            Accuracy with token dropout
        """
        self.model.eval()
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Apply token dropout
                dropout_mask = torch.rand_like(input_ids.float()) > dropout_prob
                perturbed_input_ids = input_ids * dropout_mask.long()
                
                outputs = self.model(perturbed_input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                # Calculate accuracy (use original labels)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                predictions = torch.argmax(shift_logits, dim=-1)
                correct = (predictions == shift_labels)
                
                if attention_mask is not None:
                    shift_mask = attention_mask[..., 1:].contiguous()
                    correct = correct * shift_mask
                    valid_tokens = shift_mask.sum().item()
                else:
                    valid_tokens = shift_labels.numel()
                
                total_correct += correct.sum().item()
                total_tokens += valid_tokens
        
        return total_correct / max(total_tokens, 1)
    
    def _test_token_substitution(self, data_loader, substitution_prob: float, max_batches: int) -> float:
        """
        Test with random token substitution.
        
        Args:
            data_loader: DataLoader for evaluation
            substitution_prob: Probability of substituting each token
            max_batches: Maximum batches to test
        
        Returns:
            Accuracy with token substitution
        """
        self.model.eval()
        total_correct = 0
        total_tokens = 0
        vocab_size = self.model.config.vocab_size
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Apply token substitution
                substitution_mask = torch.rand_like(input_ids.float()) < substitution_prob
                random_tokens = torch.randint(0, vocab_size, input_ids.shape, device=self.device)
                perturbed_input_ids = torch.where(substitution_mask, random_tokens, input_ids)
                
                outputs = self.model(perturbed_input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                # Calculate accuracy (use original labels)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                predictions = torch.argmax(shift_logits, dim=-1)
                correct = (predictions == shift_labels)
                
                if attention_mask is not None:
                    shift_mask = attention_mask[..., 1:].contiguous()
                    correct = correct * shift_mask
                    valid_tokens = shift_mask.sum().item()
                else:
                    valid_tokens = shift_labels.numel()
                
                total_correct += correct.sum().item()
                total_tokens += valid_tokens
        
        return total_correct / max(total_tokens, 1)
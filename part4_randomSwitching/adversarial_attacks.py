"""
Simplified Adversarial Attack Implementation
Focuses on TextFooler, BERT-Attack, and Gradient-based attacks for evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import random
from nltk.corpus import wordnet
import nltk
from transformers import BertTokenizer, BertForMaskedLM

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextFoolerAttack:
    """
    TextFooler attack implementation for language models.
    Performs word-level substitutions to create adversarial examples.
    """

    def __init__(self, model, tokenizer, device='cuda'):
        """
        Initialize TextFooler attacker.

        Args:
            model: Target model to attack
            tokenizer: Tokenizer for the model
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    def get_synonyms(self, word: str, pos_tag: Optional[str] = None) -> List[str]:
        """
        Get synonyms for a word using WordNet.

        Args:
            word: Word to find synonyms for
            pos_tag: Part-of-speech tag

        Returns:
            List of synonym words
        """
        synonyms = set()

        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)

        return list(synonyms)[:10]

    def compute_importance_scores(self, input_ids: torch.Tensor,
                                 labels: torch.Tensor) -> torch.Tensor:
        """
        Compute importance scores for each token using gradients.

        Args:
            input_ids: Input token IDs
            labels: Target labels

        Returns:
            Importance scores for each token
        """
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        self.model.zero_grad()

        input_embeds = self.model.transformer.wte(input_ids).clone().detach()
        input_embeds.requires_grad = True

        outputs = self.model(
            inputs_embeds=input_embeds.unsqueeze(0) if input_embeds.dim() == 2 else input_embeds,
            labels=labels.unsqueeze(0) if labels.dim() == 1 else labels
        )

        loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
        loss.backward()

        grad_norm = torch.norm(input_embeds.grad, dim=-1)

        return grad_norm

    def generate_adversarial(self, text: str, target_label: Optional[int] = None,
                           max_perturb_ratio: float = 0.3) -> Dict:
        """
        Generate adversarial example using TextFooler.

        Args:
            text: Input text to perturb
            target_label: Optional target label for targeted attack
            max_perturb_ratio: Maximum ratio of words to perturb

        Returns:
            Dictionary with adversarial example and metrics
        """
        # Encode text to get input_ids
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)

        if input_ids.shape[1] < 3:
            return {
                'success': False,
                'original_text': text,
                'adversarial_text': text,
                'num_changes': 0,
                'perturb_ratio': 0.0
            }

        labels = input_ids.clone()

        # Get original model predictions
        with torch.no_grad():
            orig_outputs = self.model(input_ids, labels=labels)
            orig_loss = orig_outputs['loss'].item()
            orig_perplexity = np.exp(orig_loss)
            orig_logits = orig_outputs['logits']

            # Compute original token-level accuracy (next-token prediction)
            orig_predictions = orig_logits[0, :-1].argmax(dim=-1)
            orig_labels = labels[0, 1:]
            orig_mask = orig_labels != -100
            orig_correct = (orig_predictions[orig_mask] == orig_labels[orig_mask]).sum().item()
            orig_total = orig_mask.sum().item()
            orig_accuracy = orig_correct / max(orig_total, 1)

        # Compute importance scores for each token position
        importance_scores = self.compute_importance_scores(input_ids[0], labels[0])

        # Split text into words for word-level perturbation
        words = text.split()
        if len(words) < 2:
            return {
                'success': False,
                'original_text': text,
                'adversarial_text': text,
                'num_changes': 0,
                'perturb_ratio': 0.0
            }

        # Map token positions to word positions
        word_token_map = []
        current_pos = 0
        for word_idx, word in enumerate(words):
            word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
            word_token_map.append((word_idx, current_pos, len(word_tokens)))
            current_pos += len(word_tokens)

        # Compute word-level importance by averaging token importance
        word_importance = []
        for word_idx, start_pos, num_tokens in word_token_map:
            if start_pos + num_tokens <= len(importance_scores):
                avg_importance = importance_scores[start_pos:start_pos + num_tokens].mean().item()
                word_importance.append((word_idx, avg_importance))

        # Sort words by importance
        word_importance.sort(key=lambda x: x[1], reverse=True)

        max_changes = int(len(words) * max_perturb_ratio)
        num_changes = 0
        perturbed_words = words.copy()

        # Get original embeddings for semantic similarity
        with torch.no_grad():
            orig_embeds = self.model.transformer.wte(input_ids[0])
            orig_embed_mean = orig_embeds.mean(dim=0)

        for word_idx, importance in word_importance[:max_changes]:
            if word_idx >= len(words):
                continue

            original_word = words[word_idx]

            # Skip very short words or special tokens
            if len(original_word) < 3:
                continue

            synonyms = self.get_synonyms(original_word)

            if not synonyms:
                continue

            best_synonym = None
            best_loss = orig_loss

            for synonym in synonyms:
                temp_words = perturbed_words.copy()
                temp_words[word_idx] = synonym

                temp_text = ' '.join(temp_words)
                temp_ids = self.tokenizer.encode(temp_text, return_tensors='pt').to(self.device)

                # Check semantic similarity via embeddings
                with torch.no_grad():
                    temp_embeds = self.model.transformer.wte(temp_ids[0])
                    temp_embed_mean = temp_embeds.mean(dim=0)
                    similarity = F.cosine_similarity(orig_embed_mean.unsqueeze(0),
                                                     temp_embed_mean.unsqueeze(0)).item()

                # Skip if semantic similarity is too low
                if similarity < 0.4:
                    continue

                temp_labels = temp_ids.clone()

                with torch.no_grad():
                    temp_outputs = self.model(temp_ids, labels=temp_labels)
                    temp_loss = temp_outputs['loss'].item()

                if temp_loss > best_loss:
                    best_loss = temp_loss
                    best_synonym = synonym

            if best_synonym:
                perturbed_words[word_idx] = best_synonym
                num_changes += 1

                # Early stopping if perplexity increases significantly
                if best_loss > orig_loss * 1.5:
                    break

        adversarial_text = ' '.join(perturbed_words)
        adv_ids = self.tokenizer.encode(adversarial_text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            adv_outputs = self.model(adv_ids, labels=adv_ids)
            adv_loss = adv_outputs['loss'].item()
            adv_perplexity = np.exp(adv_loss)
            adv_logits = adv_outputs['logits']

            # Compute adversarial token-level accuracy
            adv_predictions = adv_logits[0, :-1].argmax(dim=-1)
            adv_labels = adv_ids[0, 1:]
            adv_mask = adv_labels != -100
            adv_correct = (adv_predictions[adv_mask] == adv_labels[adv_mask]).sum().item()
            adv_total = adv_mask.sum().item()
            adv_accuracy = adv_correct / max(adv_total, 1)

        # Success criterion: accuracy drop > 5% (primary metric)
        accuracy_drop = orig_accuracy - adv_accuracy
        success = accuracy_drop > 0.05

        return {
            'success': success,
            'original_text': text,
            'adversarial_text': adversarial_text,
            'num_changes': num_changes,
            'perturb_ratio': num_changes / len(words) if len(words) > 0 else 0.0,

            # PRIMARY: Accuracy metrics
            'original_accuracy': orig_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'accuracy_drop': accuracy_drop,

            # SECONDARY: Perplexity (for filtering nonsensical attacks)
            'original_perplexity': orig_perplexity,
            'adversarial_perplexity': adv_perplexity,
            'perplexity_increase': (adv_perplexity - orig_perplexity) / orig_perplexity,

            # Keep for compatibility
            'original_loss': orig_loss,
            'adversarial_loss': adv_loss,
            'loss_increase': (adv_loss - orig_loss) / orig_loss,

            'original_input_ids': input_ids.cpu(),
            'adversarial_input_ids': adv_ids.cpu(),
            'original_predictions': orig_predictions.cpu(),
            'original_labels': labels.cpu()
        }


class BERTAttack:
    """
    BERT-based adversarial attack using masked language model predictions.
    More semantically-aware than TextFooler by using BERT for word substitutions.
    """

    def __init__(self, model, tokenizer, device='cuda', bert_model_name='bert-base-uncased'):
        """
        Initialize BERT-Attack.

        Args:
            model: Target model to attack
            tokenizer: Tokenizer for the target model
            device: Device for computation
            bert_model_name: BERT model to use for word substitutions
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Load BERT for masked language modeling
        print(f"Loading BERT model: {bert_model_name} for word substitution...")
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_mlm = BertForMaskedLM.from_pretrained(bert_model_name).to(device)
        self.bert_mlm.eval()

    def get_masked_lm_predictions(self, text: str, word_idx: int, top_k: int = 50) -> List[str]:
        """
        Get top-k BERT MLM predictions for a masked word.

        Args:
            text: Original text
            word_idx: Index of word to mask
            top_k: Number of predictions to return

        Returns:
            List of candidate words
        """
        words = text.split()
        if word_idx >= len(words):
            return []

        # Create masked text
        masked_words = words.copy()
        masked_words[word_idx] = '[MASK]'
        masked_text = ' '.join(masked_words)

        # Tokenize for BERT
        bert_tokens = self.bert_tokenizer.encode(masked_text, return_tensors='pt').to(self.device)

        # Find mask token position
        mask_token_id = self.bert_tokenizer.mask_token_id
        mask_positions = (bert_tokens == mask_token_id).nonzero(as_tuple=True)[1]

        if len(mask_positions) == 0:
            return []

        mask_pos = mask_positions[0].item()

        # Get predictions
        with torch.no_grad():
            outputs = self.bert_mlm(bert_tokens)
            predictions = outputs.logits[0, mask_pos]

        # Get top-k tokens
        top_k_tokens = predictions.topk(top_k).indices.tolist()

        # Convert to words and filter
        candidates = []
        original_word = words[word_idx].lower()

        for token_id in top_k_tokens:
            word = self.bert_tokenizer.decode([token_id]).strip()

            # Filter criteria
            if (word.lower() != original_word and
                word.isalpha() and
                len(word) > 2 and
                not word.startswith('##')):
                candidates.append(word)

            if len(candidates) >= 10:
                break

        return candidates

    def compute_importance_scores(self, text: str, input_ids: torch.Tensor,
                                  labels: torch.Tensor) -> List[Tuple[int, float]]:
        """
        Compute importance score for each word by masking.

        Args:
            text: Original text
            input_ids: Input token IDs
            labels: Labels for loss computation

        Returns:
            List of (word_idx, importance_score) tuples
        """
        words = text.split()
        word_importance = []

        # Get original loss
        with torch.no_grad():
            orig_outputs = self.model(input_ids, labels=labels)
            orig_loss = orig_outputs['loss'].item()

        # Compute importance by masking each word
        for word_idx in range(len(words)):
            # Create masked version
            masked_words = words.copy()
            masked_words[word_idx] = '[UNK]'
            masked_text = ' '.join(masked_words)

            masked_ids = self.tokenizer.encode(masked_text, return_tensors='pt').to(self.device)
            masked_labels = masked_ids.clone()

            # Compute loss with word masked
            with torch.no_grad():
                masked_outputs = self.model(masked_ids, labels=masked_labels)
                masked_loss = masked_outputs['loss'].item()

            # Importance = loss increase when word is masked
            importance = abs(masked_loss - orig_loss)
            word_importance.append((word_idx, importance))

        # Sort by importance
        word_importance.sort(key=lambda x: x[1], reverse=True)
        return word_importance

    def check_semantic_similarity(self, orig_ids: torch.Tensor,
                                  adv_ids: torch.Tensor,
                                  threshold: float = 0.4) -> bool:
        """
        Check semantic similarity using embeddings.

        Args:
            orig_ids: Original input IDs
            adv_ids: Adversarial input IDs
            threshold: Minimum similarity threshold

        Returns:
            True if similarity >= threshold
        """
        with torch.no_grad():
            orig_embeds = self.model.transformer.wte(orig_ids)
            adv_embeds = self.model.transformer.wte(adv_ids)

            orig_mean = orig_embeds.mean(dim=0)
            adv_mean = adv_embeds.mean(dim=0)

            similarity = F.cosine_similarity(orig_mean.unsqueeze(0),
                                            adv_mean.unsqueeze(0)).item()

        return similarity >= threshold

    def generate_adversarial(self, text: str, max_perturb_ratio: float = 0.3) -> Dict:
        """
        Generate adversarial example using BERT-Attack.

        Args:
            text: Input text to perturb
            max_perturb_ratio: Maximum ratio of words to perturb

        Returns:
            Dictionary with adversarial example and metrics
        """
        # Encode text
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)

        if input_ids.shape[1] < 3:
            return {
                'success': False,
                'original_text': text,
                'adversarial_text': text,
                'num_changes': 0,
                'perturb_ratio': 0.0
            }

        labels = input_ids.clone()

        # Get original predictions
        with torch.no_grad():
            orig_outputs = self.model(input_ids, labels=labels)
            orig_loss = orig_outputs['loss'].item()
            orig_perplexity = np.exp(orig_loss)
            orig_logits = orig_outputs['logits']

            # Compute original token-level accuracy
            orig_predictions = orig_logits[0, :-1].argmax(dim=-1)
            orig_labels = labels[0, 1:]
            orig_mask = orig_labels != -100
            orig_correct = (orig_predictions[orig_mask] == orig_labels[orig_mask]).sum().item()
            orig_total = orig_mask.sum().item()
            orig_accuracy = orig_correct / max(orig_total, 1)

        words = text.split()
        if len(words) < 2:
            return {
                'success': False,
                'original_text': text,
                'adversarial_text': text,
                'num_changes': 0,
                'perturb_ratio': 0.0
            }

        # Compute word importance
        word_importance = self.compute_importance_scores(text, input_ids, labels)

        max_changes = int(len(words) * max_perturb_ratio)
        num_changes = 0
        perturbed_words = words.copy()

        for word_idx, importance in word_importance[:max_changes]:
            if word_idx >= len(words):
                continue

            original_word = words[word_idx]

            # Skip short words
            if len(original_word) < 3:
                continue

            # Get BERT MLM predictions
            candidates = self.get_masked_lm_predictions(text, word_idx, top_k=50)

            if not candidates:
                continue

            best_substitute = None
            best_loss = orig_loss

            for candidate in candidates:
                temp_words = perturbed_words.copy()
                temp_words[word_idx] = candidate

                temp_text = ' '.join(temp_words)
                temp_ids = self.tokenizer.encode(temp_text, return_tensors='pt').to(self.device)

                # Check semantic similarity
                if not self.check_semantic_similarity(input_ids[0], temp_ids[0], threshold=0.4):
                    continue

                temp_labels = temp_ids.clone()

                with torch.no_grad():
                    temp_outputs = self.model(temp_ids, labels=temp_labels)
                    temp_loss = temp_outputs['loss'].item()

                if temp_loss > best_loss:
                    best_loss = temp_loss
                    best_substitute = candidate

            if best_substitute:
                perturbed_words[word_idx] = best_substitute
                num_changes += 1

                # Early stopping
                if best_loss > orig_loss * 1.5:
                    break

        adversarial_text = ' '.join(perturbed_words)
        adv_ids = self.tokenizer.encode(adversarial_text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            adv_outputs = self.model(adv_ids, labels=adv_ids)
            adv_loss = adv_outputs['loss'].item()
            adv_perplexity = np.exp(adv_loss)
            adv_logits = adv_outputs['logits']

            # Compute adversarial token-level accuracy
            adv_predictions = adv_logits[0, :-1].argmax(dim=-1)
            adv_labels = adv_ids[0, 1:]
            adv_mask = adv_labels != -100
            adv_correct = (adv_predictions[adv_mask] == adv_labels[adv_mask]).sum().item()
            adv_total = adv_mask.sum().item()
            adv_accuracy = adv_correct / max(adv_total, 1)

        # Success criterion: accuracy drop > 5% (primary metric)
        accuracy_drop = orig_accuracy - adv_accuracy
        success = accuracy_drop > 0.05

        return {
            'success': success,
            'original_text': text,
            'adversarial_text': adversarial_text,
            'num_changes': num_changes,
            'perturb_ratio': num_changes / len(words) if len(words) > 0 else 0.0,

            # PRIMARY: Accuracy metrics
            'original_accuracy': orig_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'accuracy_drop': accuracy_drop,

            # SECONDARY: Perplexity (for filtering nonsensical attacks)
            'original_perplexity': orig_perplexity,
            'adversarial_perplexity': adv_perplexity,
            'perplexity_increase': (adv_perplexity - orig_perplexity) / orig_perplexity,

            # Keep for compatibility
            'original_loss': orig_loss,
            'adversarial_loss': adv_loss,
            'loss_increase': (adv_loss - orig_loss) / orig_loss,

            'original_input_ids': input_ids.cpu(),
            'adversarial_input_ids': adv_ids.cpu(),
            'original_predictions': orig_predictions.cpu(),
            'original_labels': labels.cpu()
        }


class AttackEvaluator:
    """
    Evaluates adversarial attacks against models.
    """

    def __init__(self, model, tokenizer, device='cuda'):
        """
        Initialize attack evaluator.

        Args:
            model: Target model
            tokenizer: Tokenizer
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.textfooler = TextFoolerAttack(model, tokenizer, device)
        self.bert_attack = None  # Lazy initialization due to BERT loading time

    def evaluate_textfooler(self, test_samples: List[Dict],
                           max_samples: int = 50) -> Dict:
        """
        Evaluate TextFooler attack on test samples.

        Args:
            test_samples: List of test samples
            max_samples: Maximum samples to evaluate

        Returns:
            Evaluation results with 'adversarial_examples' list
        """
        results = {
            'total_samples': 0,
            'successful_attacks': 0,
            'avg_num_changes': 0,
            'avg_perturb_ratio': 0,

            # PRIMARY: Accuracy metrics
            'avg_original_accuracy': 0,
            'avg_adversarial_accuracy': 0,
            'avg_accuracy_drop': 0,

            # SECONDARY: For reference
            'avg_perplexity_increase': 0,
            'attack_success_rate': 0,

            # Store successful adversarial examples for defense evaluation
            'adversarial_examples': []
        }

        num_samples = min(len(test_samples), max_samples)

        for i, sample in enumerate(test_samples[:num_samples]):
            if 'text' in sample:
                text = sample['text']
            else:
                input_ids = sample['input_ids']
                if input_ids.dim() > 1:
                    input_ids = input_ids[0]
                text = self.tokenizer.decode(input_ids, skip_special_tokens=True)

            attack_result = self.textfooler.generate_adversarial(text)

            results['total_samples'] += 1
            if attack_result['success']:
                results['successful_attacks'] += 1
                # Store successful adversarial example for defense evaluation
                results['adversarial_examples'].append({
                    'original_text': text,
                    'adversarial_text': attack_result['adversarial_text'],
                    'adversarial_accuracy': attack_result.get('adversarial_accuracy', 0),
                    'original_accuracy': attack_result.get('original_accuracy', 0),
                    'original_input_ids': attack_result.get('original_input_ids'),
                    'adversarial_input_ids': attack_result.get('adversarial_input_ids'),
                    'original_predictions': attack_result.get('original_predictions'),
                    'original_labels': attack_result.get('original_labels')
                })

            results['avg_num_changes'] += attack_result['num_changes']
            results['avg_perturb_ratio'] += attack_result['perturb_ratio']

            # Accuracy metrics
            if 'original_accuracy' in attack_result:
                results['avg_original_accuracy'] += attack_result['original_accuracy']
            if 'adversarial_accuracy' in attack_result:
                results['avg_adversarial_accuracy'] += attack_result['adversarial_accuracy']
            if 'accuracy_drop' in attack_result:
                results['avg_accuracy_drop'] += attack_result['accuracy_drop']
            if 'perplexity_increase' in attack_result:
                results['avg_perplexity_increase'] += attack_result['perplexity_increase']

        if results['total_samples'] > 0:
            results['avg_num_changes'] /= results['total_samples']
            results['avg_perturb_ratio'] /= results['total_samples']
            results['avg_original_accuracy'] /= results['total_samples']
            results['avg_adversarial_accuracy'] /= results['total_samples']
            results['avg_accuracy_drop'] /= results['total_samples']
            results['avg_perplexity_increase'] /= results['total_samples']
            results['attack_success_rate'] = results['successful_attacks'] / results['total_samples']

        return results

    def evaluate_bert_attack(self, test_samples: List[Dict],
                            max_samples: int = 50) -> Dict:
        """
        Evaluate BERT-Attack on test samples.

        Args:
            test_samples: List of test samples
            max_samples: Maximum samples to evaluate

        Returns:
            Evaluation results with 'adversarial_examples' list
        """
        # Lazy initialization of BERT-Attack
        if self.bert_attack is None:
            print("Initializing BERT-Attack (this may take a moment)...")
            self.bert_attack = BERTAttack(self.model, self.tokenizer, self.device)

        results = {
            'total_samples': 0,
            'successful_attacks': 0,
            'avg_num_changes': 0,
            'avg_perturb_ratio': 0,

            # PRIMARY: Accuracy metrics
            'avg_original_accuracy': 0,
            'avg_adversarial_accuracy': 0,
            'avg_accuracy_drop': 0,

            # SECONDARY: For reference
            'avg_perplexity_increase': 0,
            'attack_success_rate': 0,

            # Store successful adversarial examples for defense evaluation
            'adversarial_examples': []
        }

        num_samples = min(len(test_samples), max_samples)

        for i, sample in enumerate(test_samples[:num_samples]):
            if 'text' in sample:
                text = sample['text']
            else:
                input_ids = sample['input_ids']
                if input_ids.dim() > 1:
                    input_ids = input_ids[0]
                text = self.tokenizer.decode(input_ids, skip_special_tokens=True)

            attack_result = self.bert_attack.generate_adversarial(text)

            results['total_samples'] += 1
            if attack_result['success']:
                results['successful_attacks'] += 1
                # Store successful adversarial example for defense evaluation
                results['adversarial_examples'].append({
                    'original_text': text,
                    'adversarial_text': attack_result['adversarial_text'],
                    'adversarial_accuracy': attack_result.get('adversarial_accuracy', 0),
                    'original_accuracy': attack_result.get('original_accuracy', 0),
                    'original_input_ids': attack_result.get('original_input_ids'),
                    'adversarial_input_ids': attack_result.get('adversarial_input_ids'),
                    'original_predictions': attack_result.get('original_predictions'),
                    'original_labels': attack_result.get('original_labels')
                })

            results['avg_num_changes'] += attack_result['num_changes']
            results['avg_perturb_ratio'] += attack_result['perturb_ratio']

            # Accuracy metrics
            if 'original_accuracy' in attack_result:
                results['avg_original_accuracy'] += attack_result['original_accuracy']
            if 'adversarial_accuracy' in attack_result:
                results['avg_adversarial_accuracy'] += attack_result['adversarial_accuracy']
            if 'accuracy_drop' in attack_result:
                results['avg_accuracy_drop'] += attack_result['accuracy_drop']
            if 'perplexity_increase' in attack_result:
                results['avg_perplexity_increase'] += attack_result['perplexity_increase']

        if results['total_samples'] > 0:
            results['avg_num_changes'] /= results['total_samples']
            results['avg_perturb_ratio'] /= results['total_samples']
            results['avg_original_accuracy'] /= results['total_samples']
            results['avg_adversarial_accuracy'] /= results['total_samples']
            results['avg_accuracy_drop'] /= results['total_samples']
            results['avg_perplexity_increase'] /= results['total_samples']
            results['attack_success_rate'] = results['successful_attacks'] / results['total_samples']

        return results
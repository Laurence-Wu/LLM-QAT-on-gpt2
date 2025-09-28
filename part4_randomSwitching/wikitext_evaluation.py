"""
WikiText-2 Dataset Preparation and Evaluation Utilities
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm


class WikiText2Dataset(Dataset):
    """
    WikiText-2 dataset for language modeling evaluation.
    """

    def __init__(self, tokenizer, split: str = 'test',
                 max_length: int = 128,
                 num_samples: Optional[int] = None):
        """
        Initialize WikiText-2 dataset.

        Args:
            tokenizer: Tokenizer for the model
            split: Dataset split ('train', 'validation', 'test')
            max_length: Maximum sequence length
            num_samples: Limit number of samples (None for all)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        self.samples = self._prepare_samples(dataset, num_samples)

        print(f"Loaded {len(self.samples)} samples from WikiText-2 {split} split")

    def _prepare_samples(self, dataset, num_samples: Optional[int]) -> List[Dict]:
        """
        Prepare samples from raw WikiText data.

        Args:
            dataset: Raw WikiText dataset
            num_samples: Number of samples to prepare

        Returns:
            List of prepared samples
        """
        samples = []
        count = 0

        for item in tqdm(dataset, desc="Preparing WikiText-2 samples"):
            text = item['text'].strip()

            if len(text) < 50:
                continue

            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            sample = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'text': text[:500]
            }

            samples.append(sample)
            count += 1

            if num_samples and count >= num_samples:
                break

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def prepare_wikitext2_samples(tokenizer, num_samples: int = 100,
                             split: str = 'test',
                             max_length: int = 128) -> List[Dict]:
    """
    Prepare WikiText-2 samples for evaluation.

    Args:
        tokenizer: Model tokenizer
        num_samples: Number of samples to prepare
        split: Dataset split to use
        max_length: Maximum sequence length

    Returns:
        List of prepared samples
    """
    dataset = WikiText2Dataset(
        tokenizer,
        split=split,
        max_length=max_length,
        num_samples=num_samples
    )

    return dataset.samples


def create_evaluation_batches(samples: List[Dict],
                            batch_size: int = 8) -> List[List[Dict]]:
    """
    Create batches from samples for evaluation.

    Args:
        samples: List of samples
        batch_size: Batch size

    Returns:
        List of batches
    """
    batches = []

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        batches.append(batch)

    return batches


def evaluate_perplexity(model, samples: List[Dict],
                       device: str = 'cuda') -> Dict:
    """
    Evaluate model perplexity on samples.

    Args:
        model: Language model
        samples: Test samples
        device: Device for computation

    Returns:
        Dictionary with perplexity metrics
    """
    model.eval()
    model.to(device)

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for sample in tqdm(samples, desc="Calculating perplexity"):
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
            labels = sample['labels'].unsqueeze(0).to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss

            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(avg_loss)

    return {
        'perplexity': perplexity,
        'avg_loss': avg_loss,
        'total_tokens': total_tokens,
        'num_samples': len(samples)
    }


def evaluate_generation_quality(model, tokenizer, samples: List[Dict],
                              num_generate: int = 20,
                              device: str = 'cuda') -> Dict:
    """
    Evaluate generation quality of the model.

    Args:
        model: Language model
        tokenizer: Tokenizer
        samples: Test samples
        num_generate: Number of samples to generate from
        device: Device for computation

    Returns:
        Dictionary with generation metrics
    """
    model.eval()
    model.to(device)

    generation_results = []

    for i, sample in enumerate(samples[:num_generate]):
        text = sample.get('text', '')
        if not text:
            input_ids = sample['input_ids']
            text = tokenizer.decode(input_ids, skip_special_tokens=True)

        prompt = ' '.join(text.split()[:20])

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 30,
                num_beams=1,
                temperature=1.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)

        generation_results.append({
            'prompt': prompt,
            'generated': generated_text,
            'continuation': generated_text[len(prompt):]
        })

    avg_continuation_length = np.mean([
        len(r['continuation'].split()) for r in generation_results
    ])

    return {
        'num_generated': len(generation_results),
        'avg_continuation_length': avg_continuation_length,
        'examples': generation_results[:3]
    }


class WikiTextEvaluator:
    """
    Comprehensive evaluator for WikiText-2 dataset.
    """

    def __init__(self, model, tokenizer, device: str = 'cuda'):
        """
        Initialize evaluator.

        Args:
            model: Language model
            tokenizer: Tokenizer
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate_clean_performance(self, samples: List[Dict]) -> Dict:
        """
        Evaluate model performance on clean samples.

        Args:
            samples: Test samples

        Returns:
            Performance metrics
        """
        perplexity_results = evaluate_perplexity(
            self.model, samples, self.device
        )

        accuracy_results = self._evaluate_accuracy(samples)

        return {
            'perplexity': perplexity_results['perplexity'],
            'avg_loss': perplexity_results['avg_loss'],
            'accuracy': accuracy_results['accuracy'],
            'top5_accuracy': accuracy_results['top5_accuracy'],
            'num_samples': len(samples)
        }

    def _evaluate_accuracy(self, samples: List[Dict]) -> Dict:
        """
        Evaluate prediction accuracy.

        Args:
            samples: Test samples

        Returns:
            Accuracy metrics
        """
        self.model.eval()

        correct = 0
        correct_top5 = 0
        total = 0

        with torch.no_grad():
            for sample in samples:
                input_ids = sample['input_ids'].unsqueeze(0).to(self.device)
                labels = sample['labels'].unsqueeze(0).to(self.device)

                outputs = self.model(input_ids=input_ids)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits

                predictions = logits.argmax(dim=-1)
                top5_preds = logits.topk(5, dim=-1).indices

                mask = labels != -100

                correct += (predictions[mask] == labels[mask]).sum().item()

                for label, top5 in zip(labels[mask], top5_preds[0][mask[0]]):
                    if label in top5:
                        correct_top5 += 1

                total += mask.sum().item()

        accuracy = correct / max(total, 1)
        top5_accuracy = correct_top5 / max(total, 1)

        return {
            'accuracy': accuracy,
            'top5_accuracy': top5_accuracy,
            'total_predictions': total
        }

    def evaluate_under_attack(self, samples: List[Dict],
                            attacker,
                            attack_type: str = 'textfooler') -> Dict:
        """
        Evaluate model under adversarial attack.

        Args:
            samples: Test samples
            attacker: Attack instance
            attack_type: Type of attack

        Returns:
            Evaluation results under attack
        """
        clean_results = self.evaluate_clean_performance(samples[:50])

        if attack_type == 'textfooler':
            attack_results = attacker.evaluate_textfooler(samples[:50])
        else:
            attack_results = attacker.evaluate_gradient(samples[:50], attack_type)

        adversarial_samples = self._generate_adversarial_samples(
            samples[:50], attacker, attack_type
        )

        adv_results = self.evaluate_clean_performance(adversarial_samples)

        robustness_gap = clean_results['accuracy'] - adv_results['accuracy']

        return {
            'clean_performance': clean_results,
            'adversarial_performance': adv_results,
            'attack_statistics': attack_results,
            'robustness_gap': robustness_gap,
            'attack_type': attack_type
        }

    def _generate_adversarial_samples(self, samples: List[Dict],
                                     attacker,
                                     attack_type: str) -> List[Dict]:
        """
        Generate adversarial samples.

        Args:
            samples: Original samples
            attacker: Attack instance
            attack_type: Type of attack

        Returns:
            List of adversarial samples
        """
        adversarial_samples = []

        for sample in samples:
            if attack_type == 'textfooler':
                text = sample.get('text', '')
                if not text:
                    text = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)

                attack_result = attacker.textfooler.generate_adversarial(text)

                adv_text = attack_result['adversarial_text']
                adv_tokens = self.tokenizer(
                    adv_text,
                    max_length=128,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )

                adv_sample = {
                    'input_ids': adv_tokens['input_ids'].squeeze(0),
                    'attention_mask': adv_tokens['attention_mask'].squeeze(0),
                    'labels': adv_tokens['input_ids'].squeeze(0).clone(),
                    'text': adv_text,
                    'is_adversarial': True
                }

            else:
                input_ids = sample['input_ids'].to(self.device)
                labels = sample.get('labels', input_ids.clone())

                if attack_type == 'hotflip':
                    attack_result = attacker.gradient.hotflip_attack(input_ids, labels)
                else:
                    attack_result = attacker.gradient.pgd_attack(input_ids, labels)

                adv_sample = {
                    'input_ids': attack_result['perturbed_ids'].cpu().squeeze(0),
                    'attention_mask': sample['attention_mask'],
                    'labels': labels.cpu().squeeze(0) if labels is not None else None,
                    'is_adversarial': True
                }

            adversarial_samples.append(adv_sample)

        return adversarial_samples


if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    print("Testing WikiText-2 evaluation module...")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    samples = prepare_wikitext2_samples(tokenizer, num_samples=10)
    print(f"Prepared {len(samples)} samples")

    if samples:
        first_sample = samples[0]
        print(f"First sample text (truncated): {first_sample['text'][:100]}...")
        print(f"Input IDs shape: {first_sample['input_ids'].shape}")
        print(f"Labels shape: {first_sample['labels'].shape}")
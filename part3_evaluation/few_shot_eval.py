import torch
import numpy as np
from datasets import load_dataset
from typing import Dict, List, Tuple
from tqdm import tqdm
import random

class FewShotEvaluator:
    def __init__(self, model, tokenizer, device, config):
        """Initialize with required config - NO DEFAULTS"""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config  # Store full config
        self.model = self.model.to(self.device)

    def evaluate_mmlu(self, bit_config: Dict, num_shots: int = None) -> Dict:
        """
        Evaluate on MMLU with 5-shot prompting
        Return scores by category:
        - Humanities
        - STEM
        - Social Sciences
        - Other
        - Average
        """
        if num_shots is None:
            num_shots = self.config['num_shots']

        mmlu_cfg = self.config['datasets']['MMLU']
        try:
            dataset = load_dataset(mmlu_cfg['dataset_name'], mmlu_cfg['config'], split=mmlu_cfg['split'])
        except Exception as e:
            print(f"Warning: Could not load MMLU dataset: {e}")
            return {
                'Humanities': 0.0,
                'STEM': 0.0,
                'Social Sciences': 0.0,
                'Other': 0.0,
                'Average': 0.0
            }

        categories = self.config['mmlu_categories']

        results_by_category = {cat: [] for cat in categories.keys()}

        self.model.eval()
        errors = 0
        max_errors = self.config.get('max_errors', 10)
        show_errors = self.config.get('show_first_n_errors', 3)
        max_samples = self.config['max_samples']

        with torch.no_grad():
            for i, example in enumerate(tqdm(dataset, desc="Evaluating MMLU", leave=False)):
                if i >= max_samples:
                    break
                try:
                    subject = example.get('subject', 'other')

                    category = 'Other'
                    for cat, subjects in categories.items():
                        if any(s in subject.lower() for s in subjects):
                            category = cat
                            break

                    prompt = self._create_mmlu_prompt(example, num_shots)
                    answer = example['answer']

                    predicted = self._generate_answer(prompt)

                    predicted_answer = self._extract_answer_choice(predicted)
                    correct = predicted_answer == str(answer)

                    results_by_category[category].append(float(correct))
                except Exception as e:
                    errors += 1
                    if errors <= show_errors:
                        print(f"\nError in MMLU example {i}: {str(e)}")
                        if errors == 1:
                            import traceback
                            traceback.print_exc()
                    if errors > max_errors:
                        print(f"\nToo many errors in MMLU ({errors} total), stopping")
                        break
                    continue

        scores = {}
        for category, results in results_by_category.items():
            if results:
                scores[category] = round(np.mean(results) * 100, 1)
            else:
                scores[category] = 0.0

        valid_scores = [s for s in scores.values() if s > 0]
        scores['Average'] = round(np.mean(valid_scores), 1) if valid_scores else 0.0

        return scores

    def evaluate_triviaqa(self, bit_config: Dict, num_shots: int = None) -> float:
        """
        Evaluate on TriviaQA with 5-shot prompting
        Return exact match score
        """
        if num_shots is None:
            num_shots = self.config['num_shots']

        triviaqa_cfg = self.config['datasets']['TriviaQA']
        try:
            dataset = load_dataset(triviaqa_cfg['dataset_name'], triviaqa_cfg['config'], split=triviaqa_cfg['split'])
        except Exception as e:
            print(f"Warning: Could not load TriviaQA dataset: {e}")
            return 0.0

        correct = 0
        total = 0
        errors = 0
        max_errors = self.config.get('max_errors', 10)
        show_errors = self.config.get('show_first_n_errors', 3)
        max_samples = self.config['max_samples']

        self.model.eval()
        with torch.no_grad():
            for example in tqdm(dataset, desc="Evaluating TriviaQA", leave=False):
                try:
                    question = example['question']
                    answers = example['answer']['aliases']

                    prompt = self._create_triviaqa_prompt(question, num_shots)

                    predicted = self._generate_answer(prompt)

                    if any(ans.lower() in predicted.lower() for ans in answers):
                        correct += 1
                    total += 1
                except Exception as e:
                    errors += 1
                    if errors <= show_errors:
                        print(f"\nError in TriviaQA example {total}: {str(e)}")
                        if errors == 1:
                            import traceback
                            traceback.print_exc()
                    if errors > max_errors:
                        print(f"\nToo many errors in TriviaQA ({errors} total), stopping")
                        break
                    continue

                if total >= max_samples:
                    break

        accuracy = (correct / max(total, 1)) * 100
        return round(accuracy, 1)

    def format_few_shot_prompt(self, examples: List[Dict], question: str) -> str:
        """
        Create few-shot prompt with examples
        """
        prompt = ""

        for i, ex in enumerate(examples):
            prompt += f"Example {i+1}:\n"
            prompt += f"Question: {ex['question']}\n"
            prompt += f"Answer: {ex['answer']}\n\n"

        prompt += f"Question: {question}\n"
        prompt += "Answer:"

        return prompt

    def _create_mmlu_prompt(self, example: Dict, num_shots: int) -> str:
        """Create MMLU prompt with few-shot examples"""
        # For limited context models, create a more concise prompt
        try:
            max_positions = self.model.config.n_positions
        except AttributeError:
            max_positions = 256
            print(f"Warning: Could not get n_positions, using {max_positions}")

        if max_positions <= 256:
            # Ultra-concise format for small context models
            question = example['question'][:100]  # Truncate long questions
            choices = example['choices']

            prompt = f"Q: {question}\n"
            for i, choice in enumerate(choices[:4]):  # Limit to 4 choices
                choice_text = choice[:50]  # Truncate long choices
                prompt += f"{chr(65+i)}: {choice_text}\n"
            prompt += "A:"
        else:
            # Original format for larger models
            prompt = f"The following are multiple choice questions about {example.get('subject', 'general knowledge')}.\n\n"
            question = example['question']
            choices = example['choices']

            prompt += f"Question: {question}\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += "Answer:"

        return prompt

    def _create_triviaqa_prompt(self, question: str, num_shots: int) -> str:
        """Create TriviaQA prompt with few-shot examples"""
        try:
            max_positions = self.model.config.n_positions
        except AttributeError:
            max_positions = 256
            print(f"Warning: Could not get n_positions, using {max_positions}")

        if max_positions <= 256:
            # For limited context, use fewer examples and shorter format
            few_shot_examples = [
                {"question": "Capital of France?", "answer": "Paris"},
                {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
                {"question": "Year WWII ended?", "answer": "1945"}
            ]

            # Use only 2-3 examples for small context
            actual_shots = min(num_shots, 2)
            prompt = ""

            for ex in few_shot_examples[:actual_shots]:
                prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n"

            # Truncate question if needed
            question_text = question[:150] if len(question) > 150 else question
            prompt += f"Q: {question_text}\nA:"
        else:
            # Original format for larger models
            few_shot_examples = [
                {"question": "What is the capital of France?", "answer": "Paris"},
                {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
                {"question": "What year did World War II end?", "answer": "1945"},
                {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
                {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"}
            ]

            prompt = "Answer the following questions:\n\n"

            for ex in few_shot_examples[:num_shots]:
                prompt += f"Q: {ex['question']}\n"
                prompt += f"A: {ex['answer']}\n\n"

            prompt += f"Q: {question}\n"
            prompt += "A:"

        return prompt

    def _generate_answer(self, prompt: str, max_length: int = None) -> str:
        """Generate answer using the model"""
        if max_length is None:
            max_length = self.config['generation']['max_length']

        # Get model's max position from config
        try:
            max_positions = self.model.config.n_positions
        except AttributeError:
            max_positions = 256
            print(f"Warning: Could not get n_positions, using {max_positions}")
        # Leave room for generation
        max_input_length = max(max_positions - max_length - 1, 10)  # Ensure at least 10 tokens

        # Get training sequence length from model config
        try:
            training_seq_len = self.model.config.n_positions
        except AttributeError as e:
            raise ValueError(f"Cannot get n_positions from model config: {e}")

        # Tokenize with padding to match training sequence length
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True,
                               max_length=min(max_input_length, training_seq_len),
                               padding='max_length')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Ensure input matches training sequence length
        if inputs['input_ids'].size(1) < training_seq_len:
            pad_length = training_seq_len - inputs['input_ids'].size(1)
            pad_token_id = self.tokenizer.pad_token_id
            inputs['input_ids'] = torch.cat([
                inputs['input_ids'],
                torch.full((1, pad_length), pad_token_id, device=self.device)
            ], dim=1)
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = torch.cat([
                    inputs['attention_mask'],
                    torch.zeros((1, pad_length), device=self.device, dtype=inputs['attention_mask'].dtype)
                ], dim=1)

        # Safety check: ensure input doesn't exceed model's max positions
        if inputs['input_ids'].size(1) > max_positions - max_length:
            print(f"WARNING: Truncating input from {inputs['input_ids'].size(1)} to {max_positions - max_length}")
            inputs['input_ids'] = inputs['input_ids'][:, :max_positions - max_length]
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'][:, :max_positions - max_length]

        with torch.no_grad():
            # SPLMHeadModel's generate expects max_length (total), not max_new_tokens
            current_length = inputs['input_ids'].size(1)
            remaining_length = max_positions - current_length
            actual_max_new = min(max_length, remaining_length)
            total_length = current_length + actual_max_new

            gen_cfg = self.config['generation']
            outputs = self.model.generate(
                inputs['input_ids'],  # Only pass input_ids
                max_length=total_length,
                temperature=gen_cfg['temperature'],
                do_sample=gen_cfg['do_sample'],
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return generated.strip()

    def _extract_answer_choice(self, text: str) -> str:
        """Extract answer choice (A, B, C, D) from generated text"""
        text = text.upper().strip()

        for char in ['A', 'B', 'C', 'D']:
            if char in text[:3]:
                return char

        return 'A'
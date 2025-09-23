import torch
import numpy as np
from datasets import load_dataset
from typing import Dict, List, Tuple
from tqdm import tqdm
import random

class FewShotEvaluator:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model = self.model.to(self.device)

    def evaluate_mmlu(self, bit_config: Dict, num_shots: int = 5) -> Dict:
        """
        Evaluate on MMLU with 5-shot prompting
        Return scores by category:
        - Humanities
        - STEM
        - Social Sciences
        - Other
        - Average
        """
        try:
            dataset = load_dataset('cais/mmlu', 'all', split='test[:2000]')
        except Exception as e:
            print(f"Warning: Could not load MMLU dataset: {e}")
            return {
                'Humanities': 0.0,
                'STEM': 0.0,
                'Social Sciences': 0.0,
                'Other': 0.0,
                'Average': 0.0
            }

        categories = {
            'Humanities': ['history', 'philosophy', 'law'],
            'STEM': ['physics', 'chemistry', 'biology', 'computer_science', 'math', 'engineering'],
            'Social Sciences': ['politics', 'sociology', 'psychology', 'economics'],
            'Other': ['other', 'business', 'health']
        }

        results_by_category = {cat: [] for cat in categories.keys()}

        self.model.eval()
        with torch.no_grad():
            for i, example in enumerate(tqdm(dataset, desc="Evaluating MMLU", leave=False)):
                if i >= 1000:
                    break
                subject = example.get('subject', 'other')

                category = 'Other'
                for cat, subjects in categories.items():
                    if any(s in subject.lower() for s in subjects):
                        category = cat
                        break

                prompt = self._create_mmlu_prompt(example, num_shots)
                answer = example['answer']

                predicted = self._generate_answer(prompt, max_length=5)

                predicted_answer = self._extract_answer_choice(predicted)
                correct = predicted_answer == str(answer)

                results_by_category[category].append(float(correct))

        scores = {}
        for category, results in results_by_category.items():
            if results:
                scores[category] = round(np.mean(results) * 100, 1)
            else:
                scores[category] = 0.0

        valid_scores = [s for s in scores.values() if s > 0]
        scores['Average'] = round(np.mean(valid_scores), 1) if valid_scores else 0.0

        return scores

    def evaluate_triviaqa(self, bit_config: Dict, num_shots: int = 5) -> float:
        """
        Evaluate on TriviaQA with 5-shot prompting
        Return exact match score
        """
        try:
            dataset = load_dataset('trivia_qa', 'rc.nocontext', split='validation[:2000]')
        except Exception as e:
            print(f"Warning: Could not load TriviaQA dataset: {e}")
            return 0.0

        correct = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for example in tqdm(dataset, desc="Evaluating TriviaQA", leave=False):
                question = example['question']
                answers = example['answer']['aliases']

                prompt = self._create_triviaqa_prompt(question, num_shots)

                predicted = self._generate_answer(prompt, max_length=20)

                if any(ans.lower() in predicted.lower() for ans in answers):
                    correct += 1
                total += 1

                if total >= 1000:
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
        max_positions = self.model.config.n_positions if hasattr(self.model.config, 'n_positions') else 256

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
        max_positions = self.model.config.n_positions if hasattr(self.model.config, 'n_positions') else 256

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

    def _generate_answer(self, prompt: str, max_length: int = 10) -> str:
        """Generate answer using the model"""
        # Get model's max position from config
        max_positions = self.model.config.n_positions if hasattr(self.model.config, 'n_positions') else 256
        # Leave room for generation
        max_input_length = max(max_positions - max_length - 1, 10)  # Ensure at least 10 tokens

        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_input_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Safety check: ensure input doesn't exceed model's max positions
        if inputs['input_ids'].size(1) > max_positions - max_length:
            print(f"WARNING: Truncating input from {inputs['input_ids'].size(1)} to {max_positions - max_length}")
            inputs['input_ids'] = inputs['input_ids'][:, :max_positions - max_length]
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'][:, :max_positions - max_length]

        with torch.no_grad():
            # Ensure we don't exceed model's max positions
            remaining_length = max_positions - inputs['input_ids'].size(1)
            actual_max_new = min(max_length, remaining_length)

            # SPLMHeadModel's generate doesn't take attention_mask, only input_ids
            outputs = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=actual_max_new,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
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
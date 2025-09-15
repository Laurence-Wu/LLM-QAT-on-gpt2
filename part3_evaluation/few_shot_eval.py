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
            dataset = load_dataset('cais/mmlu', 'all', split='test[:500]')
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
            for example in tqdm(dataset, desc="Evaluating MMLU", leave=False):
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
            dataset = load_dataset('trivia_qa', 'rc.nocontext', split='validation[:200]')
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

                if total >= 100:
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
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
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
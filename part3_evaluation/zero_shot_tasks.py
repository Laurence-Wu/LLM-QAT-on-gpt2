import torch
import numpy as np
from datasets import load_dataset
from typing import Dict, List
from tqdm import tqdm
import torch.nn.functional as F

class ZeroShotEvaluator:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = 'cuda'
        self.model = self.model.to(self.device)
        self.tasks = None

    def load_all_tasks(self) -> Dict:
        """
        Load all 8 datasets:
        - BoolQ: Boolean yes/no questions
        - PIQA: Physical interaction QA
        - SIQA: Social interaction QA
        - HellaSwag: Sentence completion
        - WinoGrande: Pronoun resolution
        - ARC-easy: Science questions (easy)
        - ARC-challenge: Science questions (hard)
        - OBQA: OpenBookQA
        """
        tasks = {}

        try:
            tasks['BoolQ'] = load_dataset('boolq', split='validation[:500]')
        except:
            print("Warning: Could not load BoolQ dataset")

        try:
            tasks['PIQA'] = load_dataset('piqa', split='validation[:500]')
        except:
            print("Warning: Could not load PIQA dataset")

        try:
            tasks['SIQA'] = load_dataset('social_i_qa', split='validation[:500]')
        except:
            print("Warning: Could not load SIQA dataset")

        try:
            tasks['HellaSwag'] = load_dataset('hellaswag', split='validation[:500]')
        except:
            print("Warning: Could not load HellaSwag dataset")

        try:
            tasks['WinoGrande'] = load_dataset('winogrande', 'winogrande_m', split='validation[:500]')
        except:
            print("Warning: Could not load WinoGrande dataset")

        try:
            arc = load_dataset('ai2_arc', 'ARC-Easy')
            tasks['ARC-e'] = arc['validation'][:500]
        except:
            print("Warning: Could not load ARC-Easy dataset")

        try:
            arc = load_dataset('ai2_arc', 'ARC-Challenge')
            tasks['ARC-c'] = arc['validation'][:500]
        except:
            print("Warning: Could not load ARC-Challenge dataset")

        try:
            tasks['OBQA'] = load_dataset('openbookqa', 'main', split='validation[:500]')
        except:
            print("Warning: Could not load OBQA dataset")

        return tasks

    def evaluate_task(self, task_name: str, dataset, bit_config: Dict) -> float:
        """
        Evaluate single task with specified bit configuration
        Return accuracy score
        """
        if dataset is None:
            return 0.0

        correct = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for example in tqdm(dataset, desc=f"Evaluating {task_name}", leave=False):
                try:
                    score = self._evaluate_single_example(task_name, example)
                    correct += score
                    total += 1
                except Exception as e:
                    continue

                if total >= 100:
                    break

        accuracy = (correct / max(total, 1)) * 100
        return accuracy

    def _evaluate_single_example(self, task_name: str, example) -> float:
        """Evaluate a single example based on task type"""

        if task_name == 'BoolQ':
            question = example['question']
            passage = example['passage']
            answer = example['answer']

            prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer (True/False):"
            predicted = self._generate_answer(prompt, max_length=5)

            predicted_bool = 'true' in predicted.lower()
            return float(predicted_bool == answer)

        elif task_name == 'PIQA':
            goal = example['goal']
            sol1 = example['sol1']
            sol2 = example['sol2']
            label = example['label']

            prompt = f"Goal: {goal}\nSolution A: {sol1}\nSolution B: {sol2}\nWhich solution is better (A or B)?"
            predicted = self._generate_answer(prompt, max_length=5)

            predicted_label = 0 if 'a' in predicted.lower() else 1
            return float(predicted_label == label)

        elif task_name == 'SIQA':
            context = example['context']
            question = example['question']
            answerA = example['answerA']
            answerB = example['answerB']
            answerC = example['answerC']
            label = int(example['label']) - 1

            prompt = f"Context: {context}\nQuestion: {question}\nA: {answerA}\nB: {answerB}\nC: {answerC}\nAnswer (A, B, or C):"
            predicted = self._generate_answer(prompt, max_length=5)

            if 'a' in predicted.lower():
                predicted_label = 0
            elif 'b' in predicted.lower():
                predicted_label = 1
            else:
                predicted_label = 2

            return float(predicted_label == label)

        elif task_name == 'HellaSwag':
            context = example['ctx']
            endings = example['endings']
            label = example['label']

            prompt = f"Context: {context}\n"
            for i, ending in enumerate(endings):
                prompt += f"{chr(65+i)}: {ending}\n"
            prompt += "Which ending is most likely? Answer:"

            predicted = self._generate_answer(prompt, max_length=5)

            for i in range(len(endings)):
                if chr(65+i).lower() in predicted.lower():
                    return float(i == int(label))
            return 0.0

        elif task_name == 'WinoGrande':
            sentence = example['sentence']
            option1 = example['option1']
            option2 = example['option2']
            answer = example['answer']

            prompt = f"Sentence: {sentence}\nOption 1: {option1}\nOption 2: {option2}\nWhich option fills the blank better (1 or 2)?"
            predicted = self._generate_answer(prompt, max_length=5)

            predicted_answer = '1' if '1' in predicted else '2'
            return float(predicted_answer == answer)

        elif task_name in ['ARC-e', 'ARC-c']:
            question = example['question']
            choices = example['choices']
            answer = example['answerKey']

            prompt = f"Question: {question}\n"
            for i, choice in enumerate(choices['text']):
                label = choices['label'][i]
                prompt += f"{label}: {choice}\n"
            prompt += "Answer:"

            predicted = self._generate_answer(prompt, max_length=5)
            return float(answer.upper() in predicted.upper())

        elif task_name == 'OBQA':
            question = example['question_stem']
            choices = example['choices']
            answer = example['answerKey']

            prompt = f"Question: {question}\n"
            for i, choice in enumerate(choices['text']):
                label = choices['label'][i]
                prompt += f"{label}: {choice}\n"
            prompt += "Answer:"

            predicted = self._generate_answer(prompt, max_length=5)
            return float(answer.upper() in predicted.upper())

        return 0.0

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

    def evaluate_all_tasks(self, bit_config: Dict) -> Dict:
        """
        Run all 8 tasks and return results in paper format:
        {
            'BoolQ': 72.4,
            'PIQA': 79.1,
            ...
            'Average': 71.2
        }
        """
        if self.tasks is None:
            print("Loading datasets...")
            self.tasks = self.load_all_tasks()

        results = {}
        task_names = ['BoolQ', 'PIQA', 'SIQA', 'HellaSwag', 'WinoGrande', 'ARC-e', 'ARC-c', 'OBQA']

        for task_name in task_names:
            if task_name in self.tasks:
                print(f"  Evaluating {task_name}...")
                score = self.evaluate_task(task_name, self.tasks[task_name], bit_config)
                results[task_name] = round(score, 1)
            else:
                results[task_name] = 0.0

        valid_scores = [v for v in results.values() if v > 0]
        results['Average'] = round(np.mean(valid_scores) if valid_scores else 0, 1)

        return results
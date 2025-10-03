import torch
import numpy as np
from datasets import load_dataset
from typing import Dict, List
from tqdm import tqdm
import torch.nn.functional as F

class ZeroShotEvaluator:

    def __init__(self, model, tokenizer, device, config):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.model = self.model.to(self.device)
        self.tasks = None

    def load_all_tasks(self) -> Dict:
        tasks = {}
        dataset_configs = self.config['datasets']
        try:
            boolq_cfg = dataset_configs['BoolQ']
            tasks['BoolQ'] = load_dataset(boolq_cfg['dataset_name'], split=boolq_cfg['split'])
        except Exception as e:
            print(f'Warning: Could not load BoolQ: {e}')
        try:
            hellaswag_cfg = dataset_configs['HellaSwag']
            tasks['HellaSwag'] = load_dataset(hellaswag_cfg['dataset_name'], split=hellaswag_cfg['split'])
        except Exception as e:
            print(f'Warning: Could not load HellaSwag: {e}')
            tasks['HellaSwag'] = None
        try:
            wino_cfg = dataset_configs['WinoGrande']
            tasks['WinoGrande'] = load_dataset(wino_cfg['dataset_name'], wino_cfg['config'], split=wino_cfg['split'])
        except Exception as e:
            print(f'Warning: Could not load WinoGrande: {e}')
            tasks['WinoGrande'] = None
        return tasks

    def evaluate_task(self, task_name: str, dataset, bit_config: Dict) -> float:
        if dataset is None:
            return 0.0
        correct = 0
        total = 0
        errors = 0
        max_errors = self.config.get('max_errors', 10)
        self.model.eval()
        with torch.no_grad():
            dataset_iter = tqdm(dataset, desc=f'Evaluating {task_name}', leave=False)
            for example in dataset_iter:
                try:
                    score = self._evaluate_single_example(task_name, example)
                    correct += score
                    total += 1
                except KeyboardInterrupt:
                    print(f'\nEvaluation interrupted for {task_name}')
                    break
                except Exception as e:
                    errors += 1
                    if errors > max_errors:
                        print(f'\nToo many errors in {task_name}, stopping')
                        break
                    continue
                if total >= self.config['max_samples']:
                    break
        accuracy = correct / max(total, 1) * 100
        return accuracy

    def _evaluate_single_example(self, task_name: str, example) -> float:
        if task_name == 'BoolQ':
            question = example['question']
            passage = example['passage']
            answer = example['answer']
            if len(passage) > 150:
                passage = passage[:150] + '...'
            context_text = f'Passage: {passage}\nQuestion: {question}\nAnswer:'
            choices = [' True', ' False']
            predicted_idx = self._compute_choice_likelihood(context_text, choices)
            predicted_bool = predicted_idx == 0
            return float(predicted_bool == answer)
        elif task_name == 'HellaSwag':
            context = example['ctx']
            endings = example['endings']
            label = int(example['label'])
            context_text = context.strip()
            if not context_text.endswith(' '):
                context_text += ' '
            predicted_idx = self._compute_choice_likelihood(context_text, endings)
            return float(predicted_idx == label)
        elif task_name == 'WinoGrande':
            sentence = example['sentence']
            option1 = example['option1']
            option2 = example['option2']
            answer = example['answer']
            if '_' in sentence:
                parts = sentence.split('_')
                if len(parts) == 2:
                    context = parts[0]
                    suffix = parts[1]
                    choices = [option1 + suffix, option2 + suffix]
                else:
                    choices = [sentence.replace('_', option1), sentence.replace('_', option2)]
                    context = ''
            else:
                context = sentence + ' '
                choices = [option1, option2]
            predicted_idx = self._compute_choice_likelihood(context, choices)
            predicted_answer = str(predicted_idx + 1)
            return float(predicted_answer == answer)
        return 0.0

    def _compute_choice_likelihood(self, context: str, choices: List[str]) -> int:
        log_probs = []
        for choice in choices:
            if not context or len(context.strip()) == 0:
                full_text = choice
                context_length = 0
            else:
                full_text = context + choice
                context_tokens = self.tokenizer(context, return_tensors='pt', padding=False, truncation=True, max_length=200)
                context_length = context_tokens['input_ids'].shape[1]
            full_tokens = self.tokenizer(full_text, return_tensors='pt', padding=False, truncation=True, max_length=256)
            input_ids = full_tokens['input_ids'].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                if logits.dim() == 2:
                    logits = logits.unsqueeze(0)
                log_probs_all = F.log_softmax(logits, dim=-1)
                if context_length == 0:
                    choice_start = 0
                    choice_end = input_ids.shape[1] - 1
                else:
                    choice_start = max(0, context_length - 1)
                    choice_end = input_ids.shape[1] - 1
                if choice_start < choice_end and choice_end > 0:
                    predicted_tokens = input_ids[:, choice_start + 1:choice_end + 1]
                    relevant_log_probs = log_probs_all[:, choice_start:choice_end, :]
                    if predicted_tokens.shape[1] > 0 and relevant_log_probs.shape[1] > 0:
                        token_log_probs = relevant_log_probs.gather(2, predicted_tokens.unsqueeze(-1)).squeeze(-1)
                        total_log_prob = token_log_probs.sum().item()
                        avg_log_prob = total_log_prob / max(1, predicted_tokens.shape[1])
                    else:
                        avg_log_prob = float('-inf')
                else:
                    avg_log_prob = float('-inf')
                log_probs.append(avg_log_prob)
        return max(range(len(log_probs)), key=lambda i: log_probs[i]) if log_probs else 0

    def evaluate_all_tasks(self, bit_config: Dict) -> Dict:
        if self.tasks is None:
            print('Loading datasets...')
            self.tasks = self.load_all_tasks()
        results = {}
        available_tasks = [name for name in self.tasks.keys() if self.tasks[name] is not None]
        print(f"Available tasks: {', '.join(available_tasks)}")
        for task_name in available_tasks:
            print(f'  Evaluating {task_name}...')
            score = self.evaluate_task(task_name, self.tasks[task_name], bit_config)
            results[task_name] = round(score, 1)
        if results:
            results['Average'] = round(np.mean(list(results.values())), 1)
        else:
            results['Average'] = 0.0
        return results
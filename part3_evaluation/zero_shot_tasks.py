import torch
import numpy as np
from datasets import load_dataset
from typing import Dict, List
from tqdm import tqdm
import torch.nn.functional as F

class ZeroShotEvaluator:
    def __init__(self, model, tokenizer, device, config):
        """Initialize with required config - NO DEFAULTS"""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config  # Store full config
        self.model = self.model.to(self.device)
        self.tasks = None

    def load_all_tasks(self) -> Dict:
        """
        Load available datasets:
        - BoolQ: Boolean yes/no questions
        - HellaSwag: Sentence completion
        - WinoGrande: Pronoun resolution
        - ARC-easy: Science questions (easy)
        - ARC-challenge: Science questions (hard)
        - OBQA: OpenBookQA
        """
        tasks = {}

        dataset_configs = self.config['datasets']

        try:
            boolq_cfg = dataset_configs['BoolQ']
            tasks['BoolQ'] = load_dataset(boolq_cfg['dataset_name'], split=boolq_cfg['split'])
        except Exception as e:
            print(f"Warning: Could not load BoolQ dataset: {e}")

        try:
            hellaswag_cfg = dataset_configs['HellaSwag']
            tasks['HellaSwag'] = load_dataset(hellaswag_cfg['dataset_name'], split=hellaswag_cfg['split'])
        except Exception as e:
            print(f"Warning: Could not load HellaSwag dataset: {e}")
            tasks['HellaSwag'] = None

        try:
            wino_cfg = dataset_configs['WinoGrande']
            tasks['WinoGrande'] = load_dataset(wino_cfg['dataset_name'], wino_cfg['config'], split=wino_cfg['split'])
        except:
            print("Warning: Could not load WinoGrande dataset")

        try:
            arce_cfg = dataset_configs['ARC-e']
            tasks['ARC-e'] = load_dataset(arce_cfg['dataset_name'], arce_cfg['config'], split=arce_cfg['split'])
        except Exception as e:
            print(f"Warning: Could not load ARC-Easy dataset: {e}")
            tasks['ARC-e'] = None

        try:
            arcc_cfg = dataset_configs['ARC-c']
            tasks['ARC-c'] = load_dataset(arcc_cfg['dataset_name'], arcc_cfg['config'], split=arcc_cfg['split'])
        except Exception as e:
            print(f"Warning: Could not load ARC-Challenge dataset: {e}")
            tasks['ARC-c'] = None

        try:
            obqa_cfg = dataset_configs['OBQA']
            tasks['OBQA'] = load_dataset(obqa_cfg['dataset_name'], obqa_cfg['config'], split=obqa_cfg['split'])
        except Exception as e:
            print(f"Warning: Could not load OBQA dataset: {e}")
            tasks['OBQA'] = None

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
        errors = 0
        max_errors = self.config.get('max_errors', 10)
        show_errors = self.config.get('show_first_n_errors', 3)

        self.model.eval()
        with torch.no_grad():
            dataset_iter = tqdm(dataset, desc=f"Evaluating {task_name}", leave=False)
            for example in dataset_iter:
                try:
                    score = self._evaluate_single_example(task_name, example)
                    correct += score
                    total += 1
                except KeyboardInterrupt:
                    print(f"\nEvaluation interrupted for {task_name}")
                    break
                except Exception as e:
                    errors += 1
                    if errors <= show_errors:
                        print(f"\nError in {task_name} example {total}: {str(e)}")
                        import traceback
                        if errors == 1:  # Show full traceback for first error
                            traceback.print_exc()
                    if errors > max_errors:
                        print(f"\nToo many errors in {task_name} ({errors} total), stopping evaluation")
                        break
                    continue

                if total >= self.config['max_samples']:
                    break

        accuracy = (correct / max(total, 1)) * 100
        return accuracy

    def _evaluate_single_example(self, task_name: str, example) -> float:
        """Evaluate a single example based on task type"""
        # Get max positions for prompt construction
        try:
            max_positions = self.model.config.n_positions
        except AttributeError:
            max_positions = 256
            print(f"Warning: Could not get n_positions from model config, using {max_positions}")

        truncation = self.config['prompt_truncation']
        is_limited_context = max_positions <= truncation['limited_context_threshold']

        if task_name == 'BoolQ':
            question = example['question']
            passage = example['passage']
            answer = example['answer']

            if is_limited_context:
                # Truncate passage for limited context models
                passage = passage[:truncation['passage_limit']]
                question = question[:truncation['question_limit']]
                prompt = f"Passage: {passage}\nQ: {question}\nTrue/False:"
            else:
                prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer (True/False):"
            gen_cfg = self.config['generation']
            predicted = self._generate_answer(prompt, max_length=gen_cfg['max_length'])

            predicted_bool = 'true' in predicted.lower()
            return float(predicted_bool == answer)

        elif task_name == 'HellaSwag':
            context = example['ctx']
            endings = example['endings']
            label = example['label']

            if is_limited_context:
                # Truncate for limited context
                context = context[:truncation['context_limit']]
                prompt = f"Context: {context}\n"
                for i, ending in enumerate(endings[:4]):  # Limit endings
                    ending_text = ending[:truncation['endings_limit']]  # Truncate each ending
                    prompt += f"{chr(65+i)}: {ending_text}\n"
                prompt += "Best:"
            else:
                prompt = f"Context: {context}\n"
                for i, ending in enumerate(endings):
                    prompt += f"{chr(65+i)}: {ending}\n"
                prompt += "Which ending is most likely? Answer:"

            gen_cfg = self.config['generation']
            predicted = self._generate_answer(prompt, max_length=gen_cfg['max_length'])

            for i in range(len(endings)):
                if chr(65+i).lower() in predicted.lower():
                    return float(i == int(label))
            return 0.0

        elif task_name == 'WinoGrande':
            sentence = example['sentence']
            option1 = example['option1']
            option2 = example['option2']
            answer = example['answer']

            if is_limited_context:
                # More concise format
                sentence = sentence[:truncation['context_limit']]
                prompt = f"S: {sentence}\n1: {option1}\n2: {option2}\nBest:"
            else:
                prompt = f"Sentence: {sentence}\nOption 1: {option1}\nOption 2: {option2}\nWhich option fills the blank better (1 or 2)?"
            gen_cfg = self.config['generation']
            predicted = self._generate_answer(prompt, max_length=gen_cfg['max_length'])

            predicted_answer = '1' if '1' in predicted else '2'
            return float(predicted_answer == answer)

        elif task_name in ['ARC-e', 'ARC-c']:
            question = example['question']
            choices = example['choices']
            answer = example['answerKey']

            if is_limited_context:
                # Truncate question and choices
                question = question[:truncation['context_limit']]
                prompt = f"Q: {question}\n"
                for i, choice in enumerate(choices['text'][:4]):  # Limit to 4 choices
                    label = choices['label'][i]
                    choice_text = choice[:truncation['choice_limit']]  # Truncate long choices
                    prompt += f"{label}: {choice_text}\n"
                prompt += "A:"
            else:
                prompt = f"Question: {question}\n"
                for i, choice in enumerate(choices['text']):
                    label = choices['label'][i]
                    prompt += f"{label}: {choice}\n"
                prompt += "Answer:"

            gen_cfg = self.config['generation']
            predicted = self._generate_answer(prompt, max_length=gen_cfg['max_length'])
            return float(answer.upper() in predicted.upper())

        elif task_name == 'OBQA':
            question = example['question_stem']
            choices = example['choices']
            answer = example['answerKey']

            if is_limited_context:
                # Truncate for limited context
                question = question[:truncation['context_limit']]
                prompt = f"Q: {question}\n"
                for i, choice in enumerate(choices['text'][:4]):
                    label = choices['label'][i]
                    choice_text = choice[:truncation['choice_limit']]
                    prompt += f"{label}: {choice_text}\n"
                prompt += "A:"
            else:
                prompt = f"Question: {question}\n"
                for i, choice in enumerate(choices['text']):
                    label = choices['label'][i]
                    prompt += f"{label}: {choice}\n"
                prompt += "Answer:"

            gen_cfg = self.config['generation']
            predicted = self._generate_answer(prompt, max_length=gen_cfg['max_length'])
            return float(answer.upper() in predicted.upper())

        return 0.0

    def _generate_answer(self, prompt: str, max_length: int) -> str:
        """Generate answer using the model with strict 256-token limit"""
        # Force max context to 256 for compatibility with training
        MAX_CONTEXT = 256

        # Reserve space for generation
        max_generation = min(max_length, 50)  # Cap generation length
        max_prompt_length = MAX_CONTEXT - max_generation

        # Tokenize and truncate prompt to fit within context
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=max_prompt_length,
            padding=False  # Don't pad, let model handle variable length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Validate input length
        actual_length = inputs['input_ids'].size(1)
        if actual_length > max_prompt_length:
            print(f"WARNING: Input truncated from {actual_length} to {max_prompt_length}")
            inputs['input_ids'] = inputs['input_ids'][:, :max_prompt_length]
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'][:, :max_prompt_length]

        # Generate with remaining space
        with torch.no_grad():
            current_length = inputs['input_ids'].size(1)
            remaining_space = MAX_CONTEXT - current_length
            actual_max_gen = min(max_generation, remaining_space)
            total_length = current_length + actual_max_gen

            gen_cfg = self.config['generation']
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=total_length,
                temperature=gen_cfg['temperature'],
                do_sample=gen_cfg['do_sample'],
                eos_token_id=self.tokenizer.eos_token_id,
                attention_mask=inputs.get('attention_mask', None)
            )

        generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return generated.strip()

    def evaluate_all_tasks(self, bit_config: Dict) -> Dict:
        """
        Run available tasks and return results in paper format:
        Only evaluate datasets that successfully loaded
        """
        if self.tasks is None:
            print("Loading datasets...")
            self.tasks = self.load_all_tasks()

        results = {}
        # Only use task names that actually loaded
        available_tasks = [name for name in self.tasks.keys()
                          if self.tasks[name] is not None]

        print(f"Available tasks: {', '.join(available_tasks)}")

        for task_name in available_tasks:
            print(f"  Evaluating {task_name}...")
            score = self.evaluate_task(task_name, self.tasks[task_name], bit_config)
            results[task_name] = round(score, 1)

        # Calculate average only from evaluated tasks
        if results:
            results['Average'] = round(np.mean(list(results.values())), 1)
        else:
            results['Average'] = 0.0

        return results
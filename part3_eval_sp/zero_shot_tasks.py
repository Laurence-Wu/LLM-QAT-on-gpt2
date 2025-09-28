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
        (QA tasks removed: ARC-e, ARC-c, OBQA)
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
        except Exception as e:
            print(f"Warning: Could not load WinoGrande dataset: {e}")
            tasks['WinoGrande'] = None

        # QA tasks (ARC-e, ARC-c, OBQA) removed - focusing on classification tasks only

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

            # Use likelihood-based scoring
            # Truncate passage if needed to fit in context
            max_passage_len = 150  # Conservative to leave room for question and answer
            if len(passage) > max_passage_len:
                passage = passage[:max_passage_len] + "..."

            context_text = f"Passage: {passage}\nQuestion: {question}\nAnswer:"

            # Compare likelihood of "True" vs "False"
            choices = [" True", " False"]
            predicted_idx = self._compute_choice_likelihood(context_text, choices)

            # predicted_idx: 0 = True, 1 = False
            predicted_bool = (predicted_idx == 0)
            return float(predicted_bool == answer)

        elif task_name == 'HellaSwag':
            context = example['ctx']
            endings = example['endings']
            label = int(example['label'])

            # Use likelihood-based scoring
            # Format: context should naturally continue with one of the endings
            context_text = context.strip()
            if not context_text.endswith(' '):
                context_text += ' '

            # Compute likelihood for each ending
            predicted_idx = self._compute_choice_likelihood(context_text, endings)
            return float(predicted_idx == label)

        elif task_name == 'WinoGrande':
            sentence = example['sentence']
            option1 = example['option1']
            option2 = example['option2']
            answer = example['answer']

            # Use likelihood-based scoring
            if '_' in sentence:
                # Split at the blank to create context and choices
                parts = sentence.split('_')
                if len(parts) == 2:
                    # Context is everything before the blank
                    context = parts[0]
                    # Choices are option + everything after blank
                    suffix = parts[1]
                    choices = [option1 + suffix, option2 + suffix]
                else:
                    # Multiple blanks or edge case - score full sentences
                    choices = [sentence.replace('_', option1), sentence.replace('_', option2)]
                    context = ""  # Will score full sentences
            else:
                # No blank marker - this shouldn't happen in WinoGrande
                # But handle it by comparing sentence + option
                context = sentence + " "
                choices = [option1, option2]

            # Get most likely completion
            predicted_idx = self._compute_choice_likelihood(context, choices)

            # Answer is '1' or '2' (1-indexed)
            predicted_answer = str(predicted_idx + 1)
            return float(predicted_answer == answer)

        elif task_name in ['ARC-e', 'ARC-c']:
            question = example['question']
            choices = example['choices']
            answer = example['answerKey']

            # Use likelihood-based scoring
            # Format context as question with "Answer:" prompt
            context_text = f"Question: {question}\nAnswer:"

            # Create choice completions with their labels
            choice_texts = []
            label_to_idx = {}
            for i, (choice_text, choice_label) in enumerate(zip(choices['text'], choices['label'])):
                # Each choice completion includes the label and text
                choice_completion = f" {choice_label}. {choice_text}"
                choice_texts.append(choice_completion)
                label_to_idx[choice_label] = i

            # Get predicted index
            predicted_idx = self._compute_choice_likelihood(context_text, choice_texts)

            # Check if predicted label matches answer
            if answer in label_to_idx:
                correct_idx = label_to_idx[answer]
                return float(predicted_idx == correct_idx)
            return 0.0

        elif task_name == 'OBQA':
            question = example['question_stem']
            choices = example['choices']
            answer = example['answerKey']

            # Use likelihood-based scoring (same as ARC)
            context_text = f"Question: {question}\nAnswer:"

            # Create choice completions with their labels
            choice_texts = []
            label_to_idx = {}
            for i, (choice_text, choice_label) in enumerate(zip(choices['text'], choices['label'])):
                choice_completion = f" {choice_label}. {choice_text}"
                choice_texts.append(choice_completion)
                label_to_idx[choice_label] = i

            # Get predicted index
            predicted_idx = self._compute_choice_likelihood(context_text, choice_texts)

            # Check if predicted label matches answer
            if answer in label_to_idx:
                correct_idx = label_to_idx[answer]
                return float(predicted_idx == correct_idx)
            return 0.0

        return 0.0

    def _compute_choice_likelihood(self, context: str, choices: List[str]) -> int:
        """
        Compute log probabilities for each choice and return index of most likely.
        Uses proper likelihood scoring for multiple-choice evaluation.
        """
        log_probs = []

        for choice in choices:
            # Handle empty context case
            if not context or len(context.strip()) == 0:
                # Score the entire choice text
                full_text = choice
                context_length = 0
            else:
                # Normal case with context
                full_text = context + choice
                context_tokens = self.tokenizer(context, return_tensors='pt', padding=False, truncation=True, max_length=200)
                context_length = context_tokens['input_ids'].shape[1]

            # Tokenize full text
            full_tokens = self.tokenizer(full_text, return_tensors='pt', padding=False, truncation=True, max_length=256)

            # Move to device
            input_ids = full_tokens['input_ids'].to(self.device)

            # Get model outputs
            with torch.no_grad():
                outputs = self.model(input_ids)
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs.logits

                # Compute log probabilities for the choice tokens only
                if logits.dim() == 2:
                    logits = logits.unsqueeze(0)  # Add batch dimension if needed

                # Get log probabilities
                log_probs_all = F.log_softmax(logits, dim=-1)

                # Determine which tokens to score
                if context_length == 0:
                    # Score all tokens except the first one (no previous token to condition on)
                    choice_start = 0
                    choice_end = input_ids.shape[1] - 1
                else:
                    # Score tokens after context
                    choice_start = max(0, context_length - 1)  # Ensure non-negative
                    choice_end = input_ids.shape[1] - 1

                if choice_start < choice_end and choice_end > 0:
                    # Get predicted token positions
                    predicted_tokens = input_ids[:, choice_start+1:choice_end+1]
                    # Get corresponding log probs
                    relevant_log_probs = log_probs_all[:, choice_start:choice_end, :]

                    # Ensure dimensions match
                    if predicted_tokens.shape[1] > 0 and relevant_log_probs.shape[1] > 0:
                        # Gather log probs for actual tokens
                        token_log_probs = relevant_log_probs.gather(2, predicted_tokens.unsqueeze(-1)).squeeze(-1)

                        # Sum log probabilities (or take mean to normalize for length)
                        total_log_prob = token_log_probs.sum().item()
                        # Normalize by number of tokens to avoid length bias
                        avg_log_prob = total_log_prob / max(1, predicted_tokens.shape[1])
                    else:
                        avg_log_prob = float('-inf')
                else:
                    avg_log_prob = float('-inf')

                log_probs.append(avg_log_prob)

        # Return index of highest probability choice
        if log_probs:
            return max(range(len(log_probs)), key=lambda i: log_probs[i])
        return 0  # Default to first choice if something goes wrong

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
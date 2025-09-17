#!/usr/bin/env python3
"""
Test script to identify and fix prompt truncation issues in evaluation.
This specifically tests why ARC/OBQA tasks get 0% accuracy.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import SwitchableQATGPT2
from part1_switchable_precision.main_qat import load_pretrained_weights
from transformers import GPT2Tokenizer, GPT2Config
from datasets import load_dataset


class TruncationTester:
    """Test and fix truncation issues in evaluation."""
    
    def __init__(self, model_path, config_path=None):
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)

            # Load config from checkpoint or use default
            if 'config' in checkpoint:
                config = checkpoint['config']
            else:
                config = GPT2Config()
                config.n_layer = 12
                config.lora_rank = 8
                config.lora_alpha = 16
                config.lora_dropout = 0.0

            # Create model and load state dict
            if 'bit_widths' in checkpoint:
                bit_widths = checkpoint['bit_widths']
            else:
                bit_widths = [4, 8, 16]

            self.model = SwitchableQATGPT2(config, bit_widths=bit_widths, initialize_weights=False)

            # Load model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                # Checkpoint might be just the state dict
                self.model.load_state_dict(checkpoint)

            self.model = self.model.to(self.device)
        else:
            config = GPT2Config()
            config.n_layer = 12
            config.lora_rank = 8
            config.lora_alpha = 16
            config.lora_dropout = 0.0
            self.model = SwitchableQATGPT2(config, bit_widths=[4, 8, 16], initialize_weights=False)
            load_pretrained_weights(self.model)
            self.model = self.model.to(self.device)

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        
    def test_arc_truncation(self):
        """Test if ARC prompts are being truncated."""
        print("="*70)
        print("TESTING ARC PROMPT TRUNCATION")
        print("="*70)
        
        # Load a real ARC example
        try:
            arc = load_dataset('ai2_arc', 'ARC-Easy')
            example = arc['validation'][0]
        except:
            # Create a mock example
            example = {
                'question': 'Which gas makes up most of the Earth\'s atmosphere?',
                'choices': {
                    'text': ['Oxygen', 'Carbon dioxide', 'Nitrogen', 'Hydrogen'],
                    'label': ['A', 'B', 'C', 'D']
                },
                'answerKey': 'C'
            }
        
        print(f"\nOriginal question: {example['question']}")
        print(f"Correct answer: {example['answerKey']}")
        print(f"Choices: {example['choices']['text']}")
        
        # Create the full prompt as the evaluator does
        prompt = f"Question: {example['question']}\n"
        for i, choice in enumerate(example['choices']['text']):
            label = example['choices']['label'][i]
            prompt += f"{label}: {choice}\n"
        prompt += "Answer:"
        
        # Check token lengths
        full_tokens = self.tokenizer.encode(prompt)
        print(f"\nFull prompt token length: {len(full_tokens)}")
        print(f"Model max positions: {self.model.config.n_positions}")
        
        # Test with different truncation settings
        truncation_tests = [
            ('No truncation', None),
            ('Truncate at 250', 250),
            ('Truncate at 200', 200),
            ('Truncate at 150', 150),
            ('Truncate at 100', 100),
        ]
        
        for test_name, max_length in truncation_tests:
            print(f"\n[{test_name}]")
            
            if max_length:
                truncated_tokens = self.tokenizer(
                    prompt, 
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_length
                )
                truncated_text = self.tokenizer.decode(truncated_tokens['input_ids'][0])
            else:
                truncated_text = prompt
                truncated_tokens = self.tokenizer(prompt, return_tensors='pt')
            
            # Check if all answer choices are present
            choices_present = []
            for label in example['choices']['label']:
                present = f"{label}:" in truncated_text
                choices_present.append(present)
                if not present:
                    print(f"  WARNING: Choice {label} is MISSING!")
            
            if all(choices_present):
                print(f"  ✓ All choices preserved")
            else:
                print(f"  ✗ Only {sum(choices_present)}/4 choices preserved")
            
            # Try to generate an answer
            if max_length is None or max_length >= 100:
                inputs = truncated_tokens.to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs['input_ids'],
                        max_new_tokens=5,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                print(f"  Generated: {generated}")
                
                # Check if it's a valid answer
                valid_answer = any(label in generated.upper() for label in ['A', 'B', 'C', 'D'])
                print(f"  Valid answer format: {valid_answer}")
    
    def test_fixed_evaluation(self):
        """Test evaluation with fixed prompt handling."""
        print("\n" + "="*70)
        print("TESTING FIXED EVALUATION APPROACH")
        print("="*70)
        
        # Create a modified generate function that handles truncation better
        def generate_answer_fixed(prompt, max_length=5):
            """Modified generation that preserves answer choices."""
            
            # If it's a multiple choice question, extract essentials
            if all(marker in prompt for marker in ["A:", "B:", "C:", "D:"]):
                # Extract question and choices
                lines = prompt.split('\n')
                question_line = None
                choice_lines = []
                
                for line in lines:
                    if 'Question:' in line:
                        question_line = line
                    elif any(line.strip().startswith(f"{l}:") for l in ['A', 'B', 'C', 'D']):
                        choice_lines.append(line)
                
                # Reconstruct minimal prompt
                if question_line and len(choice_lines) >= 4:
                    minimal_prompt = question_line + '\n'
                    minimal_prompt += '\n'.join(choice_lines[:4])
                    minimal_prompt += '\nAnswer:'
                    
                    print(f"  Original length: {len(self.tokenizer.encode(prompt))} tokens")
                    print(f"  Minimal length: {len(self.tokenizer.encode(minimal_prompt))} tokens")
                    
                    prompt = minimal_prompt
            
            # Now generate with the (possibly modified) prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=250
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=max_length,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated
        
        # Test on multiple examples
        test_questions = [
            {
                'question': 'What color is the sky?',
                'choices': ['Red', 'Green', 'Blue', 'Yellow'],
                'labels': ['A', 'B', 'C', 'D'],
                'correct': 'C'
            },
            {
                'question': 'How many days are in a week?',
                'choices': ['5', '6', '7', '8'],
                'labels': ['A', 'B', 'C', 'D'],
                'correct': 'C'
            }
        ]
        
        print("\nTesting with minimal prompts:")
        correct = 0
        total = 0
        
        for test in test_questions:
            # Create full prompt
            prompt = f"Question: {test['question']}\n"
            for i, choice in enumerate(test['choices']):
                prompt += f"{test['labels'][i]}: {choice}\n"
            prompt += "Answer:"
            
            # Generate answer with fixed method
            answer = generate_answer_fixed(prompt)
            
            # Check if correct
            is_correct = test['correct'] in answer.upper()
            if is_correct:
                correct += 1
            total += 1
            
            print(f"\nQ: {test['question']}")
            print(f"Generated: {answer}")
            print(f"Correct answer: {test['correct']}")
            print(f"Result: {'✓' if is_correct else '✗'}")
        
        print(f"\nAccuracy with fixed approach: {correct}/{total} = {correct/total*100:.1f}%")
    
    def suggest_fixes(self):
        """Suggest fixes based on findings."""
        print("\n" + "="*70)
        print("RECOMMENDED FIXES")
        print("="*70)
        
        print("""
1. IMMEDIATE FIX (for testing):
   - Modify zero_shot_tasks.py to use minimal prompts for multiple choice
   - Extract only question + choices, skip long contexts
   
2. BETTER FIX:
   - Increase model's n_positions to 512 or 1024
   - Retrain with longer sequences
   
3. EVALUATION FIX:
   - Modify _generate_answer() in zero_shot_tasks.py:
     ```python
     def _generate_answer(self, prompt: str, max_length: int = 10) -> str:
         # For multiple choice, ensure choices are preserved
         if "A:" in prompt and "D:" in prompt:
             # Extract and preserve choices...
     ```
   
4. QUICK WORKAROUND:
   - Set model to FP16 (no quantization) for evaluation
   - Disable LoRA during evaluation
   - Use beam search instead of greedy decoding
        """)


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, default=None)
    
    args = parser.parse_args()
    
    tester = TruncationTester(args.model_path, args.config_path)
    tester.test_arc_truncation()
    tester.test_fixed_evaluation()
    tester.suggest_fixes()


if __name__ == "__main__":
    main()

#!/bin/bash

# Master script to diagnose model issues
# Run this to systematically identify problems with your QAT model

echo "========================================================================="
echo "QAT MODEL DIAGNOSTIC SUITE"
echo "========================================================================="
echo ""
echo "This script will run multiple tests to identify why your model has poor"
echo "performance (0% on ARC tasks, high perplexity, etc.)"
echo ""

# Check if model path is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model_path> [config_path]"
    echo "Example: $0 qat_gpt2_8bit_fp32_20250916_211603.pth qat_training_stats_20250916_211603.json"
    exit 1
fi

MODEL_PATH=$1
CONFIG_PATH=${2:-""}

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Model path: $MODEL_PATH${NC}"
if [ -n "$CONFIG_PATH" ]; then
    echo -e "${GREEN}Config path: $CONFIG_PATH${NC}"
fi
echo ""

# Function to run a test
run_test() {
    local test_name=$1
    local script_name=$2
    local args=$3
    
    echo ""
    echo -e "${YELLOW}=========================================================================${NC}"
    echo -e "${YELLOW}Running: $test_name${NC}"
    echo -e "${YELLOW}=========================================================================${NC}"
    
    if [ -n "$CONFIG_PATH" ]; then
        python3 $script_name --model_path $MODEL_PATH --config_path $CONFIG_PATH $args
    else
        python3 $script_name --model_path $MODEL_PATH $args
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $test_name completed successfully${NC}"
    else
        echo -e "${RED}✗ $test_name failed${NC}"
    fi
}

# Run comprehensive diagnostics
echo -e "${YELLOW}STEP 1: COMPREHENSIVE DIAGNOSTICS${NC}"
echo "This will test all major systems and identify issues..."
run_test "Comprehensive Diagnostics" "diagnose_model_issues.py" ""

# Check for output file
if [ -f "model_diagnostic_results.json" ]; then
    echo ""
    echo -e "${GREEN}Diagnostic results saved to model_diagnostic_results.json${NC}"
    echo "Key findings:"
    python3 -c "
import json
with open('model_diagnostic_results.json', 'r') as f:
    results = json.load(f)
    if 'basic_generation' in results:
        print(f\"  - Basic generation success: {results['basic_generation']['success_rate']*100:.1f}%\")
    if 'quantization_stability' in results:
        print(f\"  - Quantization stable: {results['quantization_stability']['stable']}\")
    if 'prompt_truncation' in results:
        print(f\"  - Prompts being truncated: {results['prompt_truncation']['was_truncated']}\")
        print(f\"  - Answer choices preserved: {results['prompt_truncation']['choices_preserved']}\")
    if 'simple_zero_shot' in results:
        print(f\"  - Simple task accuracy: {results['simple_zero_shot']['accuracy']*100:.1f}%\")
"
fi

# Run truncation test
echo ""
echo -e "${YELLOW}STEP 2: TRUNCATION ANALYSIS${NC}"
echo "Testing if evaluation prompts are being truncated..."
run_test "Truncation Analysis" "test_truncation_issue.py" ""

# Run fix tests
echo ""
echo -e "${YELLOW}STEP 3: TESTING POTENTIAL FIXES${NC}"
echo "Trying various fixes to see what improves performance..."
run_test "Fix Testing" "test_model_fixes.py" ""

# Summary
echo ""
echo -e "${YELLOW}=========================================================================${NC}"
echo -e "${YELLOW}DIAGNOSTIC SUMMARY${NC}"
echo -e "${YELLOW}=========================================================================${NC}"

echo ""
echo "Based on the tests, here are the most likely issues:"
echo ""

# Parse results and provide recommendations
python3 -c "
import json
import os

if os.path.exists('model_diagnostic_results.json'):
    with open('model_diagnostic_results.json', 'r') as f:
        results = json.load(f)
    
    issues = []
    fixes = []
    
    # Check for truncation
    if 'prompt_truncation' in results:
        if results['prompt_truncation']['was_truncated'] and not results['prompt_truncation']['choices_preserved']:
            issues.append('CRITICAL: Evaluation prompts are being truncated, cutting off answer choices')
            fixes.append('Modify evaluation to use shorter prompts or increase model context length')
    
    # Check for quantization issues
    if 'quantization_stability' in results:
        if not results['quantization_stability']['stable']:
            issues.append('Quantization scales are unstable during generation')
            fixes.append('Fix quantization calibration or use FP16 for evaluation')
    
    # Check for LoRA issues
    if 'lora_interference' in results:
        if results['lora_interference']['lora_makes_worse']:
            issues.append('LoRA adapters are interfering with pretrained knowledge')
            fixes.append('Disable LoRA for evaluation or reduce LoRA rank')
    
    # Check basic generation
    if 'basic_generation' in results:
        if results['basic_generation']['success_rate'] < 0.5:
            issues.append('Model cannot generate coherent text for basic prompts')
            fixes.append('Model may need retraining with fixed quantization')
    
    # Print issues
    if issues:
        print('IDENTIFIED ISSUES:')
        for i, issue in enumerate(issues, 1):
            print(f'  {i}. {issue}')
        
        print('\\nRECOMMENDED FIXES:')
        for i, fix in enumerate(fixes, 1):
            print(f'  {i}. {fix}')
    else:
        print('No critical issues identified. Model may just need more training.')

print('\\nNext steps:')
print('  1. Review model_diagnostic_results.json for detailed findings')
print('  2. Apply the recommended fixes')
print('  3. Re-run evaluation after fixes')
"

echo ""
echo -e "${GREEN}Diagnostic complete!${NC}"

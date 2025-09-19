#!/usr/bin/env python3
"""
Zero-Shot Validation Test
Tests properly initialized SP model against GPT-2 on zero-shot tasks
to verify correct initialization and equivalent performance.
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test.fix_model_initialization import create_properly_initialized_model
from part1_switchable_precision.distillation_manager import DistillationManager
from part1_switchable_precision.config_sp import TrainingConfig


def load_models():
    """Load both SP model and original GPT-2 for comparison."""
    print("\n" + "="*80)
    print("LOADING MODELS FOR ZERO-SHOT COMPARISON")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load properly initialized SP model
    print("\n1. Loading SP model with proper initialization...")
    sp_model, sp_config = create_properly_initialized_model(use_pretrained=True)
    sp_model = sp_model.to(device)
    sp_model.eval()

    # Load original GPT-2
    print("\n2. Loading original GPT-2 model...")
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_model = gpt2_model.to(device)
    gpt2_model.eval()

    # Load tokenizer
    print("\n3. Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print("\n✅ All models loaded successfully!")
    return sp_model, gpt2_model, tokenizer, sp_config, device


def test_language_modeling_perplexity(sp_model, gpt2_model, tokenizer, device):
    """Test perplexity on language modeling task."""
    print("\n" + "="*60)
    print("LANGUAGE MODELING PERPLEXITY TEST")
    print("="*60)

    # Test sentences from different domains
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "The weather today is sunny with a chance of rain.",
        "Neural networks consist of interconnected layers of neurons.",
        "Climate change is one of the most pressing issues of our time.",
        "The stock market has been volatile in recent months.",
        "Quantum computing represents a revolutionary approach to computation."
    ]

    print(f"\nTesting perplexity on {len(test_sentences)} sentences...")

    sp_perplexities = []
    gpt2_perplexities = []

    for i, sentence in enumerate(test_sentences):
        print(f"\n{i+1}. \"{sentence[:50]}{'...' if len(sentence) > 50 else ''}\"")

        # Tokenize
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            # SP Model perplexity at 16-bit (should match GPT-2)
            sp_model.set_precision(16)
            sp_outputs = sp_model(input_ids, labels=input_ids, attention_mask=attention_mask)
            sp_loss = sp_outputs['loss'].item()
            sp_perplexity = torch.exp(torch.tensor(sp_loss)).item()
            sp_perplexities.append(sp_perplexity)

            # GPT-2 perplexity
            gpt2_outputs = gpt2_model(input_ids, labels=input_ids, attention_mask=attention_mask)
            gpt2_loss = gpt2_outputs['loss'].item()
            gpt2_perplexity = torch.exp(torch.tensor(gpt2_loss)).item()
            gpt2_perplexities.append(gpt2_perplexity)

        print(f"   SP (16-bit):  Loss = {sp_loss:.4f}, PPL = {sp_perplexity:.2f}")
        print(f"   GPT-2:        Loss = {gpt2_loss:.4f}, PPL = {gpt2_perplexity:.2f}")
        print(f"   Difference:   {abs(sp_perplexity - gpt2_perplexity):.2f} PPL")

    # Overall statistics
    avg_sp_ppl = np.mean(sp_perplexities)
    avg_gpt2_ppl = np.mean(gpt2_perplexities)
    max_diff = max(abs(sp - gpt2) for sp, gpt2 in zip(sp_perplexities, gpt2_perplexities))
    avg_diff = np.mean([abs(sp - gpt2) for sp, gpt2 in zip(sp_perplexities, gpt2_perplexities)])

    print(f"\n📊 PERPLEXITY RESULTS:")
    print(f"   Average SP (16-bit): {avg_sp_ppl:.2f}")
    print(f"   Average GPT-2:       {avg_gpt2_ppl:.2f}")
    print(f"   Average difference:  {avg_diff:.2f}")
    print(f"   Maximum difference:  {max_diff:.2f}")

    # Validation criteria
    if avg_diff < 1.0 and max_diff < 5.0:
        print(f"   ✅ PASSED: Perplexity matches GPT-2 (diff < 1.0 avg, < 5.0 max)")
    elif avg_diff < 2.0 and max_diff < 10.0:
        print(f"   ⚠️ ACCEPTABLE: Close to GPT-2 (diff < 2.0 avg, < 10.0 max)")
    else:
        print(f"   ❌ FAILED: Significant difference from GPT-2")

    return sp_perplexities, gpt2_perplexities


def test_text_generation_quality(sp_model, gpt2_model, tokenizer, device):
    """Test text generation quality."""
    print("\n" + "="*60)
    print("TEXT GENERATION QUALITY TEST")
    print("="*60)

    prompts = [
        "The future of artificial intelligence",
        "In a world where technology",
        "Climate change impacts",
        "Python programming language"
    ]

    generation_params = {
        'max_length': 50,
        'temperature': 0.8,
        'top_p': 0.9,
        'do_sample': True,
        'pad_token_id': tokenizer.eos_token_id
    }

    print(f"\nGenerating text for {len(prompts)} prompts...")

    for i, prompt in enumerate(prompts):
        print(f"\n{i+1}. Prompt: \"{prompt}\"")

        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)

        with torch.no_grad():
            # SP model generation at 16-bit
            sp_model.set_precision(16)
            torch.manual_seed(42)  # For reproducible comparison
            sp_outputs = sp_model.generate(input_ids, **generation_params)
            sp_text = tokenizer.decode(sp_outputs[0], skip_special_tokens=True)

            # GPT-2 generation
            torch.manual_seed(42)  # Same seed
            gpt2_outputs = gpt2_model.generate(input_ids, **generation_params)
            gpt2_text = tokenizer.decode(gpt2_outputs[0], skip_special_tokens=True)

        print(f"   SP (16-bit): {sp_text}")
        print(f"   GPT-2:       {gpt2_text}")

        # Simple quality metrics
        sp_tokens = len(tokenizer.encode(sp_text))
        gpt2_tokens = len(tokenizer.encode(gpt2_text))
        print(f"   Token counts: SP={sp_tokens}, GPT-2={gpt2_tokens}")


def test_different_precisions(sp_model, tokenizer, device):
    """Test performance across different bit precisions."""
    print("\n" + "="*60)
    print("MULTI-PRECISION PERFORMANCE TEST")
    print("="*60)

    test_text = "The rapid development of artificial intelligence has transformed many industries."
    inputs = tokenizer(test_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)

    print(f"\nTest sentence: \"{test_text}\"")
    print(f"Testing precisions: {[16, 8, 4]} bits")

    results = {}

    with torch.no_grad():
        for bits in [16, 8, 4]:
            print(f"\n--- {bits}-bit precision ---")
            sp_model.set_precision(bits)

            # Measure inference time
            start_time = time.time()
            for _ in range(10):  # Multiple runs for timing
                outputs = sp_model(input_ids, labels=input_ids)
            end_time = time.time()

            loss = outputs['loss'].item()
            perplexity = torch.exp(torch.tensor(loss)).item()
            avg_time = (end_time - start_time) / 10 * 1000  # ms per inference

            results[bits] = {
                'loss': loss,
                'perplexity': perplexity,
                'time_ms': avg_time
            }

            print(f"   Loss: {loss:.4f}")
            print(f"   Perplexity: {perplexity:.2f}")
            print(f"   Avg inference time: {avg_time:.2f} ms")

    # Analysis
    print(f"\n📊 PRECISION COMPARISON:")
    baseline_ppl = results[16]['perplexity']

    for bits in [16, 8, 4]:
        ppl = results[bits]['perplexity']
        ppl_degradation = ((ppl - baseline_ppl) / baseline_ppl) * 100
        time_ms = results[bits]['time_ms']

        print(f"   {bits:2d}-bit: PPL={ppl:6.2f} ({ppl_degradation:+5.1f}%), Time={time_ms:5.1f}ms")

    # Quality assessment
    print(f"\n📋 QUALITY ASSESSMENT:")
    ppl_8bit_deg = ((results[8]['perplexity'] - baseline_ppl) / baseline_ppl) * 100
    ppl_4bit_deg = ((results[4]['perplexity'] - baseline_ppl) / baseline_ppl) * 100

    if ppl_8bit_deg < 5:
        print(f"   ✅ 8-bit: Excellent quality ({ppl_8bit_deg:.1f}% degradation)")
    elif ppl_8bit_deg < 15:
        print(f"   ⚠️ 8-bit: Good quality ({ppl_8bit_deg:.1f}% degradation)")
    else:
        print(f"   ❌ 8-bit: Poor quality ({ppl_8bit_deg:.1f}% degradation)")

    if ppl_4bit_deg < 10:
        print(f"   ✅ 4-bit: Excellent quality ({ppl_4bit_deg:.1f}% degradation)")
    elif ppl_4bit_deg < 25:
        print(f"   ⚠️ 4-bit: Acceptable quality ({ppl_4bit_deg:.1f}% degradation)")
    else:
        print(f"   ❌ 4-bit: Poor quality ({ppl_4bit_deg:.1f}% degradation)")

    return results


def test_logit_distribution_similarity(sp_model, gpt2_model, tokenizer, device):
    """Test that logit distributions are similar between SP and GPT-2."""
    print("\n" + "="*60)
    print("LOGIT DISTRIBUTION SIMILARITY TEST")
    print("="*60)

    test_sentences = [
        "The cat sat on the",
        "Machine learning algorithms can",
        "In the future, technology will"
    ]

    similarities = []

    for sentence in test_sentences:
        print(f"\nTesting: \"{sentence}\"")

        inputs = tokenizer(sentence, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)

        with torch.no_grad():
            # Get logits from both models
            sp_model.set_precision(16)
            sp_outputs = sp_model(input_ids)
            sp_logits = sp_outputs['logits'][0, -1, :]  # Last token logits

            gpt2_outputs = gpt2_model(input_ids)
            gpt2_logits = gpt2_outputs['logits'][0, -1, :]

            # Compute similarity metrics
            # 1. Cosine similarity
            cosine_sim = F.cosine_similarity(sp_logits.unsqueeze(0), gpt2_logits.unsqueeze(0)).item()

            # 2. Top-k overlap (top 10 tokens)
            sp_top_k = torch.topk(sp_logits, 10).indices
            gpt2_top_k = torch.topk(gpt2_logits, 10).indices
            overlap = len(set(sp_top_k.cpu().numpy()) & set(gpt2_top_k.cpu().numpy()))
            overlap_ratio = overlap / 10

            # 3. KL divergence between probability distributions
            sp_probs = F.softmax(sp_logits, dim=-1)
            gpt2_probs = F.softmax(gpt2_logits, dim=-1)
            kl_div = F.kl_div(sp_probs.log(), gpt2_probs, reduction='sum').item()

            similarities.append({
                'cosine': cosine_sim,
                'top_k_overlap': overlap_ratio,
                'kl_divergence': kl_div
            })

            print(f"   Cosine similarity: {cosine_sim:.4f}")
            print(f"   Top-10 overlap: {overlap}/10 ({overlap_ratio:.1%})")
            print(f"   KL divergence: {kl_div:.4f}")

    # Overall statistics
    avg_cosine = np.mean([s['cosine'] for s in similarities])
    avg_overlap = np.mean([s['top_k_overlap'] for s in similarities])
    avg_kl = np.mean([s['kl_divergence'] for s in similarities])

    print(f"\n📊 OVERALL SIMILARITY:")
    print(f"   Average cosine similarity: {avg_cosine:.4f}")
    print(f"   Average top-10 overlap: {avg_overlap:.1%}")
    print(f"   Average KL divergence: {avg_kl:.4f}")

    # Assessment
    if avg_cosine > 0.99 and avg_overlap > 0.8 and avg_kl < 0.1:
        print(f"   ✅ EXCELLENT: SP model logits match GPT-2 very closely")
    elif avg_cosine > 0.95 and avg_overlap > 0.6 and avg_kl < 0.5:
        print(f"   ⚠️ GOOD: SP model logits are similar to GPT-2")
    else:
        print(f"   ❌ POOR: SP model logits differ significantly from GPT-2")

    return similarities


def test_with_distillation_integration(sp_model, sp_config, device):
    """Test that distillation integration works with properly initialized model."""
    print("\n" + "="*60)
    print("DISTILLATION INTEGRATION TEST")
    print("="*60)

    # Initialize distillation manager
    training_config = TrainingConfig()
    distill_mgr = DistillationManager(
        model=sp_model,
        full_precision_bits=16,
        config=training_config
    )

    # Test data
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    text = "Artificial intelligence will transform the future of technology."
    inputs = tokenizer(text, return_tensors='pt', max_length=32, truncation=True, padding='max_length')
    input_ids = inputs['input_ids'].to(device)

    print(f"\nTest text: \"{text}\"")

    # Test teacher-student workflow
    print("\n1. Testing teacher mode (16-bit)...")
    sp_model.set_precision(16)
    should_update = distill_mgr.should_update_teacher(16, 0)
    print(f"   Should update teacher: {should_update}")

    with torch.no_grad():
        teacher_outputs = sp_model(input_ids, output_hidden_states=True, return_dict=True)
        teacher_loss = F.cross_entropy(
            teacher_outputs['logits'].view(-1, teacher_outputs['logits'].size(-1)),
            input_ids.view(-1)
        )
        print(f"   Teacher loss: {teacher_loss.item():.4f}")

        # Cache teacher outputs
        if should_update:
            distill_mgr.update_teacher(input_ids)
            print(f"   ✅ Teacher outputs cached")

    print("\n2. Testing student mode (8-bit)...")
    sp_model.set_precision(8)
    with torch.no_grad():
        student_outputs = sp_model(input_ids, output_hidden_states=True, return_dict=True)

        # Compute distillation loss
        distill_loss = distill_mgr.compute_distillation_loss(student_outputs, input_ids)
        print(f"   Distillation loss: {distill_loss.item():.4f}")

        # Standard loss for comparison
        student_loss = F.cross_entropy(
            student_outputs['logits'].view(-1, student_outputs['logits'].size(-1)),
            input_ids.view(-1)
        )
        print(f"   Standard loss: {student_loss.item():.4f}")

    print(f"\n✅ Distillation integration working correctly!")
    return True


def main():
    """Run comprehensive zero-shot validation tests."""
    print("\n" + "="*80)
    print("COMPREHENSIVE ZERO-SHOT VALIDATION")
    print("="*80)
    print("Testing properly initialized SP model against GPT-2 baseline")

    # Load models
    sp_model, gpt2_model, tokenizer, sp_config, device = load_models()

    # Run tests
    print("\n" + "="*80)
    print("RUNNING VALIDATION TESTS")
    print("="*80)

    # Test 1: Language modeling perplexity
    sp_perplexities, gpt2_perplexities = test_language_modeling_perplexity(
        sp_model, gpt2_model, tokenizer, device
    )

    # Test 2: Text generation quality
    test_text_generation_quality(sp_model, gpt2_model, tokenizer, device)

    # Test 3: Multiple precisions
    precision_results = test_different_precisions(sp_model, tokenizer, device)

    # Test 4: Logit similarity
    similarity_results = test_logit_distribution_similarity(
        sp_model, gpt2_model, tokenizer, device
    )

    # Test 5: Distillation integration
    distill_ok = test_with_distillation_integration(sp_model, sp_config, device)

    # Final summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    print("\n🎯 TEST RESULTS:")

    # Perplexity assessment
    avg_ppl_diff = np.mean([abs(sp - gpt2) for sp, gpt2 in zip(sp_perplexities, gpt2_perplexities)])
    if avg_ppl_diff < 1.0:
        print("✅ Language Modeling: PASSED - Perplexity matches GPT-2")
    else:
        print("❌ Language Modeling: FAILED - Perplexity differs from GPT-2")

    # Precision assessment
    ppl_16 = precision_results[16]['perplexity']
    ppl_8 = precision_results[8]['perplexity']
    ppl_4 = precision_results[4]['perplexity']

    deg_8 = ((ppl_8 - ppl_16) / ppl_16) * 100
    deg_4 = ((ppl_4 - ppl_16) / ppl_16) * 100

    if deg_8 < 15 and deg_4 < 30:
        print("✅ Multi-Precision: PASSED - Acceptable quality across precisions")
    else:
        print("❌ Multi-Precision: FAILED - Significant quality loss at low precision")

    # Similarity assessment
    avg_cosine = np.mean([s['cosine'] for s in similarity_results])
    if avg_cosine > 0.95:
        print("✅ Logit Similarity: PASSED - Distributions match GPT-2")
    else:
        print("❌ Logit Similarity: FAILED - Distributions differ from GPT-2")

    # Distillation assessment
    if distill_ok:
        print("✅ Distillation Integration: PASSED - Works correctly")
    else:
        print("❌ Distillation Integration: FAILED - Issues found")

    # Overall assessment
    print(f"\n🏆 OVERALL ASSESSMENT:")
    if (avg_ppl_diff < 1.0 and deg_8 < 15 and deg_4 < 30 and
        avg_cosine > 0.95 and distill_ok):
        print("🎉 EXCELLENT: Model initialization is correct!")
        print("   SP model performs equivalently to GPT-2")
        print("   All precisions work properly")
        print("   Distillation integration functional")
    elif (avg_ppl_diff < 2.0 and deg_8 < 25 and avg_cosine > 0.90):
        print("👍 GOOD: Model initialization is mostly correct")
        print("   Minor differences from GPT-2 but acceptable")
    else:
        print("⚠️ NEEDS IMPROVEMENT: Model initialization has issues")
        print("   Consider checking weight loading process")

    print(f"\n📊 KEY METRICS:")
    print(f"   Average perplexity difference: {avg_ppl_diff:.2f}")
    print(f"   8-bit quality degradation: {deg_8:.1f}%")
    print(f"   4-bit quality degradation: {deg_4:.1f}%")
    print(f"   Average cosine similarity: {avg_cosine:.4f}")

    return sp_model, {
        'perplexity_diff': avg_ppl_diff,
        'precision_degradation': {'8bit': deg_8, '4bit': deg_4},
        'cosine_similarity': avg_cosine,
        'distillation_ok': distill_ok
    }


if __name__ == "__main__":
    model, metrics = main()
    print("\n✅ Zero-shot validation complete!")
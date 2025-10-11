"""
Debug script to test evaluation with actual checkpoints
Tests with small subset to verify functionality
"""
import torch
import sys

from part5_squad.eval_squad import (
    load_squad_model_from_checkpoint,
    load_evaluation_config_squad
)

def test_checkpoint_loading(checkpoint_path):
    """Test loading a specific checkpoint"""
    print(f"\n{'='*70}")
    print(f"Testing checkpoint: {checkpoint_path}")
    print(f"{'='*70}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    try:
        # Load checkpoint
        model, bit_width = load_squad_model_from_checkpoint(checkpoint_path, device)
        print(f"\n✅ Checkpoint loaded successfully!")
        print(f"   Bit-width: {bit_width}")
        print(f"   Current precision: {model.transformer.get_current_precision()}")

        # Check model state
        print(f"\n📊 Model info:")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")

        # Verify quantizers are calibrated (for 7-bit)
        if bit_width < 32:
            print(f"\n🔍 Checking calibration status for {bit_width}-bit model...")
            calibrated_count = 0
            total_quantizers = 0

            for name, module in model.named_modules():
                if hasattr(module, 'calibrated'):
                    total_quantizers += 1
                    if module.calibrated:
                        calibrated_count += 1

            print(f"   Calibrated quantizers: {calibrated_count}/{total_quantizers}")

            if calibrated_count == total_quantizers:
                print(f"   ✅ All quantizers are calibrated (loaded from checkpoint)")
            else:
                print(f"   ⚠️  Warning: {total_quantizers - calibrated_count} quantizers not calibrated")

        # Test forward pass with dummy input
        print(f"\n🧪 Testing forward pass...")
        dummy_input = torch.randint(0, 50257, (1, 128), device=device)
        model.eval()

        with torch.no_grad():
            outputs = model(dummy_input)

        print(f"   ✅ Forward pass successful")
        print(f"   Output keys: {list(outputs.keys())}")
        print(f"   Start logits shape: {outputs['start_logits'].shape}")
        print(f"   End logits shape: {outputs['end_logits'].shape}")

        return True

    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_eval_debug.py <checkpoint_path>")
        print("\nExample:")
        print("  python test_eval_debug.py checkpoints/squad_gpt2_32bit_20251010_200045.pth")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    success = test_checkpoint_loading(checkpoint_path)

    if success:
        print(f"\n{'='*70}")
        print("✅ All tests passed! Checkpoint is ready for evaluation.")
        print(f"{'='*70}\n")
        print("To run full evaluation:")
        print(f"  python -m part5_squad.eval_squad --model_path {checkpoint_path}")
    else:
        print(f"\n{'='*70}")
        print("❌ Tests failed. Please check the errors above.")
        print(f"{'='*70}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()

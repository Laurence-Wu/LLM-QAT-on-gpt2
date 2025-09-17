# PyTorch CUDA Setup Instructions

## Current Issue
The current PyTorch installation is CPU-only (`torch 2.8.0+cpu`), but you have a CUDA-capable GPU (NVIDIA GeForce RTX 4070 with CUDA 12.7).

## Solution: Install PyTorch with CUDA Support

### Step 1: Uninstall CPU-only PyTorch
```bash
source venv/bin/activate
pip uninstall -y torch torchvision torchaudio
```

### Step 2: Install PyTorch with CUDA support
For CUDA 12.x (which you have), use:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Or for the latest stable version:
```bash
pip install torch torchvision torchaudio
```

### Step 3: Verify CUDA is working
```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output:
```
PyTorch version: 2.x.x+cu121
CUDA available: True
CUDA device: NVIDIA GeForce RTX 4070
```

## After CUDA is Set Up

Once PyTorch with CUDA is installed, you can run the diagnostic tests:

```bash
cd test
source ../venv/bin/activate
./run_all_diagnostics.sh ../qat_gpt2_8bit_fp32_20250917_113554.pth
```

Or run individual diagnostic scripts:
```bash
python diagnose_model_issues.py --model_path ../qat_gpt2_8bit_fp32_20250917_113554.pth
```

## Note
The model was trained with these configurations (from the checkpoint):
- n_positions: 256 (not 1024)
- n_layer: 6
- LoRA ranks: {4-bit: 8, 8-bit: 16, 16-bit: 32}
- KV cache bits: 8

All test scripts have been updated to load these configurations directly from the checkpoint file.
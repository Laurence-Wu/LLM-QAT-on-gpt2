# Efficient LLMs via Switchable and Dynamic Quantization

Implementation of quantization-aware training with LoRA adaptation for GPT-2 models.

## Requirements

- **OS**: Windows 10/11, Ubuntu 22.04 LTS, or WSL2
- **Python**: 3.10 or higher
- **CUDA**: 11.8+ (optional, for GPU support)
- **RAM**: 16GB+ recommended
- **Disk Space**: 10GB+ free

## Installation

### Windows Installation

#### Method 1: Quick Setup Script

```powershell
# Clone the repository
git clone <repository-url>
cd EIC_lab

# Run setup script (PowerShell or Command Prompt)
setup_windows.bat

# Activate virtual environment
venv\Scripts\activate

# Run the main program
python main.py
```

#### Method 2: Manual Installation (Windows)

```powershell
# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch (check CUDA availability)
# For GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision torchaudio

# Install other requirements
pip install -r requirements.txt

# Run the program
python main.py
```

### Ubuntu/Linux Installation

#### Method 1: Quick Setup Script

```bash
# Clone the repository
git clone <repository-url>
cd EIC_lab

# Fix line endings if coming from Windows
dos2unix setup_ubuntu.sh  # Install dos2unix if needed: sudo apt-get install dos2unix

# Make executable and run
chmod +x setup_ubuntu.sh
./setup_ubuntu.sh

# Activate virtual environment
source venv/bin/activate

# Run the main program
python main.py
```

#### Method 2: Manual Installation (Ubuntu)

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and dependencies
sudo apt-get install -y python3.10 python3.10-dev python3.10-venv python3-pip git build-essential

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (GPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or install PyTorch (CPU version)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install requirements
pip install -r requirements.txt

# Run the main program
python main.py
```

### WSL2 Installation (Windows Subsystem for Linux)

```bash
# In WSL2 terminal
cd /mnt/c/Users/<YourUsername>/Desktop/WinCoding/EIC_lab

# Fix Windows line endings
sed -i 's/\r$//' setup_ubuntu.sh

# Run setup
bash setup_ubuntu.sh

# Activate environment
source venv/bin/activate

# Run program
python main.py
```

### Docker Installation

```bash
# Build Docker image
docker build -t quantized-gpt2 .

# Run with GPU support
docker run --gpus all -it quantized-gpt2

# Run CPU-only
docker run -it quantized-gpt2
```

### Conda Installation

```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate environment
conda activate quantized-gpt2

# Run program
python main.py
```

## Project Structure

```
EIC_lab/
├── config.py           # Configuration dataclasses
├── quantization.py     # Quantization modules and schedulers
├── lora.py            # LoRA adaptation modules
├── models.py          # Quantized GPT-2 architecture
├── training.py        # Training functions (switchable, CPT)
├── evaluation.py      # Evaluation and robustness testing
├── dataset.py         # SQuAD dataset handling
├── utils.py           # Utility functions
├── main.py            # Main entry point
├── requirements.txt   # Python dependencies
├── setup_ubuntu.sh    # Ubuntu setup script
├── Dockerfile         # Docker configuration
└── README.md          # This file
```

## Features

1. **Quantization-Aware Training (QAT)**
   - Learnable fake quantization
   - Per-channel quantization support
   - Symmetric/asymmetric modes

2. **LoRA Integration**
   - Multi-precision LoRA modules
   - Adaptive rank selection
   - Switchable bit-width configurations

3. **Training Methods**
   - Joint training with switchable quantization
   - Cyclic Precision Training (CPT)
   - Knowledge distillation from teacher model

4. **Evaluation Framework**
   - Multiple quantization configurations
   - Perplexity and throughput metrics
   - Adversarial robustness testing (FGSM)

## Configuration

Edit `config.py` to modify:
- Training hyperparameters
- Model architecture settings
- Quantization bit-widths
- Dataset parameters

## GPU Support

The code automatically detects CUDA availability. For GPU acceleration:

1. Install NVIDIA drivers:
```bash
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
```

2. Install CUDA 11.8:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-11-8
```

3. Verify installation:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## Memory Requirements

- **Minimum**: 8GB RAM (CPU-only, reduced batch size)
- **Recommended**: 16GB RAM + 8GB GPU VRAM
- **Optimal**: 32GB RAM + 16GB GPU VRAM

To reduce memory usage:
- Decrease `batch_size` in `config.py`
- Reduce `max_seq_length`
- Use gradient accumulation

## Troubleshooting

### Windows Issues

#### PowerShell Script Execution
```powershell
# If you get "cannot be loaded because running scripts is disabled"
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Python Not Found
```powershell
# Install Python from Microsoft Store or python.org
# Or use:
winget install Python.Python.3.10
```

### Linux/WSL Issues

#### Line Ending Issues (WSL/Linux)
```bash
# Fix Windows line endings
dos2unix *.sh *.py
# Or use sed:
sed -i 's/\r$//' setup_ubuntu.sh
```

#### Permission Denied
```bash
chmod +x setup_ubuntu.sh
# Or run with bash:
bash setup_ubuntu.sh
```

### Common Issues

#### Out of Memory
```python
# In config.py, reduce:
batch_size = 8  # or 4
max_seq_length = 256  # instead of 384
gradient_accumulation_steps = 8  # increase this
```

#### CUDA Not Found
```bash
# Check CUDA installation
nvidia-smi  # Windows/Linux
nvcc --version  # Linux

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CPU only:
pip install torch torchvision torchaudio
```

#### Import Errors
```bash
# Windows:
set PYTHONPATH=%PYTHONPATH%;%cd%

# Linux/WSL:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### SSL Certificate Errors (Corporate Networks)
```bash
# Temporary fix (not recommended for production)
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <package_name>

# Or set environment variable
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
```

## Performance Tips

1. **Enable Mixed Precision Training**:
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

2. **Use DataLoader Workers**:
```python
DataLoader(..., num_workers=4, pin_memory=True)
```

3. **Gradient Checkpointing** (for large models):
```python
model.gradient_checkpointing_enable()
```

## Citation

This implementation is based on:
- LoRA: Low-Rank Adaptation of Large Language Models
- LLM-QAT: Data-Free Quantization Aware Training for LLMs
- Cyclic Precision Training (CPT) from ICLR 2021

## License

MIT License
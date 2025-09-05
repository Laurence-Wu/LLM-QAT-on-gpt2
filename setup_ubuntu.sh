#!/bin/bash

# Setup script for Ubuntu 22.04 LTS
# Efficient LLMs via Switchable and Dynamic Quantization

set -e

echo "=========================================="
echo "Setting up environment on Ubuntu 22.04"
echo "=========================================="

# Update system packages
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python 3.10 (default in Ubuntu 22.04) and essential tools
echo "Installing Python and essential tools..."
sudo apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    pkg-config

# Install CUDA dependencies (optional, for GPU support)
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Setting up CUDA..."
    
    # Install CUDA 11.8 (compatible with PyTorch 2.0+)
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-11-8
    
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    
    # Install cuDNN (for deep learning)
    echo "Please install cuDNN manually from NVIDIA website if needed"
else
    echo "No NVIDIA GPU detected. Installing CPU-only version..."
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (with CUDA support if available)
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    # GPU version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    # CPU version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install project dependencies
echo "Installing project dependencies..."
pip install transformers>=4.30.0
pip install datasets>=2.14.0
pip install numpy>=1.24.0
pip install tqdm>=4.65.0
pip install accelerate>=0.20.0
pip install sentencepiece>=0.1.99
pip install protobuf>=3.20.0

# Download GPT-2 model and tokenizer (cache for faster startup)
echo "Pre-downloading GPT-2 model and tokenizer..."
python3 -c "
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
print('Downloading GPT-2 model and tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
lm_model = GPT2LMHeadModel.from_pretrained('gpt2')
print('GPT-2 model cached successfully!')
"

# Download SQuAD dataset (cache for faster startup)
echo "Pre-downloading SQuAD dataset..."
python3 -c "
from datasets import load_dataset
print('Downloading SQuAD dataset...')
dataset = load_dataset('squad', split='train[:100]')
print('SQuAD dataset cached successfully!')
"

# Create directories for outputs
echo "Creating output directories..."
mkdir -p checkpoints
mkdir -p results
mkdir -p logs

# Set environment variables
echo "Setting environment variables..."
echo 'export PYTHONPATH="${PYTHONPATH}:$(pwd)"' >> ~/.bashrc
echo 'export OMP_NUM_THREADS=4' >> ~/.bashrc
echo 'export MKL_NUM_THREADS=4' >> ~/.bashrc

# System optimization for deep learning
echo "Optimizing system for deep learning..."
# Increase file descriptor limits
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Disable swap for better performance (optional)
# sudo swapoff -a

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the main program:"
echo "  python main.py"
echo ""
echo "System Information:"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
#!/bin/bash
# GPU Setup Script using pyenv (works on any Ubuntu version)
# This fixes the Python 3.13 compatibility issue

set -e  # Exit on error

echo "================================="
echo "GPU Setup Script (using pyenv)"
echo "================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
   echo "❌ Do not run as root. Run as: ./setup_gpu_pyenv.sh"
   exit 1
fi

# Install dependencies for building Python
echo "Step 1: Installing build dependencies..."
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl git \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev

# Install pyenv
echo ""
echo "Step 2: Installing pyenv..."
if [ ! -d "$HOME/.pyenv" ]; then
    curl https://pyenv.run | bash

    # Add pyenv to bashrc if not already there
    if ! grep -q 'export PYENV_ROOT="$HOME/.pyenv"' ~/.bashrc; then
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
        echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    fi

    # Set up pyenv for current session
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
else
    echo "✅ pyenv already installed"
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
fi

# Install Python 3.11
echo ""
echo "Step 3: Installing Python 3.11.9 with pyenv..."
pyenv install -s 3.11.9

# Backup old venv
echo ""
echo "Step 4: Backing up current venv..."
cd /home/hrsh/MEGA_PROJECTS/research_paper
if [ -d "venv" ]; then
    mv venv venv_old_python313_backup
    echo "✅ Old venv backed up to venv_old_python313_backup/"
fi

# Create new venv with Python 3.11
echo ""
echo "Step 5: Creating new venv with Python 3.11..."
~/.pyenv/versions/3.11.9/bin/python -m venv venv

# Activate and install packages
echo ""
echo "Step 6: Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip

echo ""
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

# Test GPU
echo ""
echo "================================="
echo "Step 7: Testing GPU..."
echo "================================="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'Python: {torch.version.__version__ if hasattr(torch.version, \"__version__\") else \"N/A\"}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device: {torch.cuda.get_device_name(0)}')
    x = torch.randn(100, 100).to('cuda')
    print(f'✅ GPU WORKING! Tensor on: {x.device}')
    print(f'Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB')
    del x
    torch.cuda.empty_cache()
else:
    print('❌ GPU still not working')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================="
    echo "✅ SUCCESS! GPU is working!"
    echo "================================="
    echo ""
    echo "IMPORTANT: Activate the new environment with:"
    echo "  source venv/bin/activate"
    echo ""
    echo "Then run experiments:"
    echo "  python experiments/baseline.py --model gpt2-medium --num_examples 10 --compression_type none"
    echo ""
    echo "Note: pyenv is now installed. To use Python 3.11 globally:"
    echo "  pyenv global 3.11.9"
    echo ""
else
    echo ""
    echo "❌ Setup completed but GPU test failed"
    echo "Check GPU_FIX_INSTRUCTIONS.md for troubleshooting"
    exit 1
fi

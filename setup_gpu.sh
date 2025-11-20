#!/bin/bash
# Automatic Python 3.11 Setup Script for GPU Support
# This fixes the Python 3.13 compatibility issue

set -e  # Exit on error

echo "================================="
echo "GPU Setup Script"
echo "================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
   echo "❌ Do not run as root. Run as: ./setup_gpu.sh"
   exit 1
fi

echo "Step 1: Adding deadsnakes PPA for Python 3.11..."
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update

echo ""
echo "Step 2: Installing Python 3.11..."
sudo apt install -y python3.11 python3.11-venv python3.11-dev

echo ""
echo "Step 3: Backing up current venv..."
if [ -d "venv" ]; then
    mv venv venv_old_python313_backup
    echo "✅ Old venv backed up to venv_old_python313_backup/"
fi

echo ""
echo "Step 4: Creating new venv with Python 3.11..."
python3.11 -m venv venv

echo ""
echo "Step 5: Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip

echo ""
echo "Installing PyTorch with CUDA..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

echo ""
echo "================================="
echo "Step 6: Testing GPU..."
echo "================================="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    x = torch.randn(100, 100).to('cuda')
    print(f'✅ GPU WORKING! Tensor on: {x.device}')
    print(f'Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB')
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
    echo "You can now run experiments:"
    echo "  source venv/bin/activate"
    echo "  python experiments/baseline.py --model gpt2-medium --num_examples 10 --compression_type none"
    echo ""
else
    echo ""
    echo "❌ Setup completed but GPU test failed"
    echo "Check GPU_FIX_INSTRUCTIONS.md for troubleshooting"
fi

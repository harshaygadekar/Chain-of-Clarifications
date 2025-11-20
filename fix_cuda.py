#!/usr/bin/env python
"""
Workaround for Python 3.13 + PyTorch CUDA initialization issue.
This must be run BEFORE any torch imports.
"""
import os
import sys

# Set environment variables BEFORE importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Now try importing torch
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ CUDA WORKING!")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

    # Try allocation
    try:
        x = torch.randn(1000, 1000, device='cuda')
        print(f"✅ GPU allocation successful!")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        del x
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Allocation failed: {e}")
else:
    print("❌ CUDA not available - this is a Python 3.13 compatibility issue")
    print("\nSOLUTION: Use Python 3.11 or run on CPU")
    sys.exit(1)

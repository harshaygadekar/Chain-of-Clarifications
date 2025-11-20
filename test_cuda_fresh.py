#!/usr/bin/env python
"""Fresh CUDA test without any prior imports"""

if __name__ == "__main__":
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        try:
            x = torch.randn(100, 100).to('cuda')
            print(f"✅ SUCCESS! Tensor on {x.device}")
        except Exception as e:
            print(f"❌ Failed: {e}")
    else:
        print("CUDA not detected")

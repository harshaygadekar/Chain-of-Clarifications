#!/bin/bash

# Fix NVIDIA Driver Issue
# This script purges existing NVIDIA drivers and installs the stable 550 version.

set -e  # Exit on error

echo "=== NVIDIA Driver Fix Script ==="
echo "This script will remove current NVIDIA drivers and install version 550."
echo "You will need to enter your sudo password."
echo ""

# 1. Remove existing drivers
echo "[1/4] Purging existing NVIDIA drivers..."
sudo apt-get purge -y nvidia-* libnvidia-*
sudo apt-get autoremove -y

# 2. Update package list
echo "[2/4] Updating package lists..."
sudo apt-get update

# 3. Install stable driver
echo "[3/4] Installing nvidia-driver-550..."
sudo apt-get install -y nvidia-driver-550

# 4. Final instructions
echo ""
echo "=== SUCCESS ==="
echo "Driver installation complete."
echo "IMPORTANT: You MUST reboot your computer now for changes to take effect."
echo "Please run: sudo reboot"

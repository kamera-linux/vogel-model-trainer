#!/usr/bin/env python3
"""
Auto-detect CUDA availability and install the appropriate onnxruntime package.

This script checks if CUDA is available and installs:
- onnxruntime-gpu if CUDA is detected
- onnxruntime (CPU) otherwise (e.g., Raspberry Pi)
"""

import subprocess
import sys
import os


def has_cuda():
    """Check if CUDA is available on this system."""
    try:
        # Method 1: Check if nvidia-smi exists and works
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    try:
        # Method 2: Check via torch if available
        import torch
        if torch.cuda.is_available():
            return True
    except ImportError:
        pass
    
    # Method 3: Check CUDA environment variables
    if os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH'):
        return True
    
    return False


def get_installed_onnxruntime():
    """Check which onnxruntime package is currently installed."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        for line in result.stdout.split('\n'):
            if 'onnxruntime-gpu' in line.lower():
                return 'onnxruntime-gpu'
            elif 'onnxruntime' in line.lower() and 'gpu' not in line.lower():
                return 'onnxruntime'
    except (subprocess.TimeoutExpired, Exception):
        pass
    return None


def install_onnxruntime():
    """Install the appropriate onnxruntime package based on hardware."""
    current = get_installed_onnxruntime()
    cuda_available = has_cuda()
    
    if cuda_available:
        target = 'onnxruntime-gpu'
        print("🎮 CUDA detected! Installing onnxruntime-gpu for GPU acceleration...")
    else:
        target = 'onnxruntime'
        print("💻 No CUDA detected. Installing onnxruntime (CPU) for compatibility...")
    
    # Check if we need to change
    if current == target:
        print(f"✅ {target} is already installed.")
        return True
    
    # Uninstall wrong version if present
    if current:
        print(f"🔄 Removing {current}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", current],
            stdout=subprocess.DEVNULL
        )
    
    # Install correct version
    print(f"📦 Installing {target}>=1.15.0...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", f"{target}>=1.15.0"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✅ Successfully installed {target}")
        
        # Verify installation
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            print(f"📊 Available execution providers: {', '.join(providers)}")
            if cuda_available and 'CUDAExecutionProvider' in providers:
                print("✨ GPU acceleration is ready!")
            elif not cuda_available and 'CPUExecutionProvider' in providers:
                print("✨ CPU execution is ready!")
        except ImportError:
            print("⚠️  Warning: Could not verify onnxruntime installation")
        
        return True
    else:
        print(f"❌ Failed to install {target}")
        print(result.stderr)
        return False


def main():
    """Main entry point."""
    # Suppress ONNX Runtime GPU device discovery warnings
    # These occur when checking for integrated GPUs (DRM devices) on systems with discrete NVIDIA GPUs
    os.environ['ORT_DISABLE_DEVICE_DISCOVERY_WARNING'] = '1'
    
    print("🔍 Detecting hardware configuration...")
    success = install_onnxruntime()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

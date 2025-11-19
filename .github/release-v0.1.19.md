# Release v0.1.19 - Hardware Auto-Detection

**Release Date:** November 19, 2025

## 🔧 New Features

### Hardware Auto-Detection Script
- **Automatic ONNX Runtime Installation** (`scripts/setup_onnxruntime.py`)
  - Detects CUDA GPU availability using multiple methods:
    - `nvidia-smi` command check
    - `torch.cuda.is_available()` check
    - Environment variables (CUDA_HOME, CUDA_PATH)
  - Automatically installs correct version:
    - `onnxruntime-gpu` for NVIDIA CUDA systems (4-6x faster background removal)
    - `onnxruntime` for CPU-only systems (Raspberry Pi, ARM64, etc.)
  - Removes conflicting versions automatically
  - Verifies installation and displays available execution providers
  - Cross-platform support: Works on all hardware configurations

### Installation Methods

**Automatic (Recommended):**
```bash
# One-liner from GitHub (downloads and runs script)
python -c "$(curl -fsSL https://raw.githubusercontent.com/kamera-linux/vogel-model-trainer/main/scripts/setup_onnxruntime.py)"

# Or from source
python scripts/setup_onnxruntime.py
```

**Manual (Advanced Users):**
```bash
# For CUDA systems
pip install vogel-model-trainer[gpu]

# For CPU-only systems
pip install vogel-model-trainer[cpu]
```

## 📦 Dependency Changes

### Removed from Core Dependencies
- `onnxruntime` is no longer automatically installed with `pip install vogel-model-trainer`
- Prevents hardware conflicts (wrong version on incompatible systems)

### Added Optional Dependencies
- `[gpu]` group: Installs `onnxruntime-gpu>=1.15.0`
- `[cpu]` group: Installs `onnxruntime>=1.15.0`

## 📚 Documentation Updates

### All README Files Updated (EN, DE, JA)
- Added hardware auto-detection installation instructions
- Added curl one-liner for automatic setup
- Added manual installation options
- Added hardware support section explaining GPU vs CPU

### Scripts Documentation
- New comprehensive documentation in `scripts/README.md`
- Explains setup_onnxruntime.py functionality
- Documents expected behavior for different hardware
- Includes troubleshooting tips

## 🚀 Performance Impact

### Background Removal Speed (with --remove-background flag)
- **GPU (RTX 2070 SUPER)**: ~0.3-0.5 seconds per image
- **CPU (Desktop i7)**: ~1-3 seconds per image
- **Speedup**: 4-6x faster with GPU acceleration

### Example Performance for 1000 Images
- **GPU**: 5-8 minutes
- **CPU**: 33-50 minutes

## 🔍 Technical Details

### Files Modified
- `pyproject.toml` - Removed onnxruntime from dependencies, added optional groups
- `README.md`, `README.de.md`, `README.ja.md` - Updated installation sections
- `scripts/README.md` - Added setup_onnxruntime.py documentation

### Files Added
- `scripts/setup_onnxruntime.py` - Hardware detection and installation script

### Warning Explanation
Users may see this warning with onnxruntime-gpu (harmless):
```
[W:onnxruntime:, env.cc:241 ThreadMain] pthread_setaffinity_np failed for thread
```

This occurs when ONNX Runtime checks for integrated GPUs on systems with discrete NVIDIA GPUs. It does not affect functionality - CUDAExecutionProvider works correctly.

## 💡 Why This Matters

### Problem Solved
Previously, users had to manually:
1. Determine if their system has CUDA
2. Choose correct onnxruntime version
3. Install manually (error-prone)
4. Debug conflicts if wrong version installed

### Solution
Now users just run one command:
```bash
python scripts/setup_onnxruntime.py
```

The script handles everything automatically!

## 🎯 Use Cases

### CUDA GPU Systems (Workstations)
- Automatically installs `onnxruntime-gpu`
- Enables GPU acceleration for background removal
- 4-6x performance improvement

### CPU-Only Systems (Raspberry Pi, ARM64)
- Automatically installs `onnxruntime` (CPU version)
- Ensures compatibility on all platforms
- No manual intervention needed

### Mixed Environments
Perfect for users with multiple machines:
- Development workstation with GPU
- Production Raspberry Pi deployment
- Same installation command works everywhere

## 📊 Migration Guide

### For New Users
Just follow updated README installation instructions:
```bash
pip install vogel-model-trainer
python scripts/setup_onnxruntime.py
```

### For Existing Users
If you already have onnxruntime installed, you can:
1. Keep current installation (no action needed)
2. Or run setup script to optimize for your hardware:
   ```bash
   python scripts/setup_onnxruntime.py
   ```

The script will detect and upgrade/downgrade as needed.

## 🔗 Related Issues
- Resolves GPU device discovery warning confusion
- Fixes hardware-specific installation problems
- Streamlines cross-platform deployment

## 📝 Notes
- This is a feature release with breaking changes to dependency management
- Core functionality unchanged - only installation process improved
- Backward compatible - existing installations continue to work
- Recommended: Run setup script to verify optimal configuration

---

**Full Changelog**: https://github.com/kamera-linux/vogel-model-trainer/blob/main/CHANGELOG.md

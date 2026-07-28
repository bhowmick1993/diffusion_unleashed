# Step 2: Environment Setup

## Prerequisites

Before we begin, ensure you have:

- **Python 3.8+** installed
- **CUDA-capable GPU** (recommended) or CPU
- **Git** for cloning repositories
- **pip** or **conda** for package management

## Option 1: Using pip (Recommended)

### 1. Create a Virtual Environment

```bash
# Create virtual environment
python -m venv diffusion-env

# Activate it
# On Windows:
diffusion-env\Scripts\activate
# On Linux/Mac:
source diffusion-env/bin/activate
```

### 2. Install PyTorch

First, install PyTorch based on your system:

**For CUDA (GPU support):**
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio
```

### 3. Install Other Dependencies

```bash
pip install -r requirements.txt
```

## Option 2: Using Conda

### 1. Create Conda Environment

```bash
conda create -n diffusion-env python=3.10
conda activate diffusion-env
```

### 2. Install PyTorch

```bash
# For CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CPU
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 3. Install Other Dependencies

```bash
pip install -r requirements.txt
```

## Verify Installation

Create a test script `test_setup.py`:

```python
import torch
import torchvision
import numpy as np
from PIL import Image

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test imports
print("✓ All imports successful!")
```

Run it:
```bash
python test_setup.py
```

Expected output:
```
PyTorch version: 2.x.x
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce RTX 3090
✓ All imports successful!
```

## Project Structure Setup

Create the necessary directories:

```bash
mkdir -p outputs/checkpoints
mkdir -p outputs/samples
mkdir -p outputs/logs
mkdir -p data/raw
mkdir -p data/processed
```

## IDE Setup (Optional but Recommended)

### VS Code
- Install Python extension
- Install Jupyter extension (for notebooks)
- Set Python interpreter to your virtual environment

### PyCharm
- Configure project interpreter to your virtual environment
- Enable Jupyter notebook support

## Common Issues and Solutions

### Issue: CUDA out of memory
**Solution**: Reduce batch size in config or use gradient accumulation

### Issue: Import errors
**Solution**: Ensure virtual environment is activated and dependencies are installed

### Issue: Slow training on CPU
**Solution**: Consider using Google Colab (free GPU) or reduce image resolution

## Next Steps

Once your environment is set up, proceed to [Step 3: Data Preparation](./03-data-preparation.md) to learn how to prepare your dataset.


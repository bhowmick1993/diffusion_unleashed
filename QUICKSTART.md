# Quick Start Guide

Get up and running with diffusion model training in 5 minutes!

## Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv diffusion-env

# Activate it
# Windows:
diffusion-env\Scripts\activate
# Linux/Mac:
source diffusion-env/bin/activate

# Install PyTorch (choose based on your system)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

## Step 2: Prepare Your Data

```bash
# Preprocess your images
python scripts/preprocess_data.py \
    --input_dir path/to/your/images \
    --output_dir data/processed \
    --size 64
```

Or use CIFAR-10 for testing:

```python
from torchvision import datasets
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.CIFAR10(root='./data', download=True, transform=transform)
```

## Step 3: Train Your Model

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data_dir data/processed \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 1e-4
```

## Step 4: Generate Samples

```bash
python scripts/sample.py \
    --checkpoint outputs/checkpoints/checkpoint_epoch_50.pt \
    --num_samples 16 \
    --image_size 64 \
    --output_dir outputs/samples
```

## Step 5: Explore with Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/01-introduction.ipynb
```

## Tips for First Training

1. **Start Small**: Use 64x64 images and a small dataset (1000-5000 images)
2. **Monitor Training**: Check TensorBoard logs: `tensorboard --logdir=outputs/logs`
3. **Save Regularly**: Checkpoints are saved every 10 epochs by default
4. **GPU Recommended**: Training on CPU is very slow - consider using Google Colab

## Troubleshooting

**Out of Memory?**
- Reduce batch size in config
- Reduce image size
- Use gradient accumulation

**Training too slow?**
- Use GPU if available
- Reduce image resolution
- Use smaller model

**Loss not decreasing?**
- Check learning rate (try 1e-4 or 5e-5)
- Verify data preprocessing
- Check model architecture

## Next Steps

- Read the full guides in `guides/` directory
- Explore advanced techniques in `guides/07-advanced-techniques.md`
- Experiment with different architectures and schedules

Happy training! 🚀


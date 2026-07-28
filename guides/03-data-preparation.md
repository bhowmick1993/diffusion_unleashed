# Step 3: Data Preparation

## Overview

Proper data preparation is crucial for training successful diffusion models. This step covers dataset selection, preprocessing, and creating data loaders.

## Dataset Selection

### Recommended Datasets for Learning

1. **CIFAR-10** (32×32 images)
   - Small, fast to train
   - Good for experimentation
   - 10 classes, 50k training images

2. **CelebA** (178×218 faces)
   - Medium size
   - Good for face generation
   - 200k+ celebrity face images

3. **Custom Dataset**
   - Your own images
   - Any domain you're interested in

### Downloading Datasets

**CIFAR-10:**
```python
from torchvision import datasets
dataset = datasets.CIFAR10(root='./data', download=True)
```

**CelebA:**
- Download from: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- Extract to `data/raw/celeba/`

## Data Preprocessing

### 1. Image Resizing and Normalization

Diffusion models typically work with square images. Common sizes:
- **32×32**: Fast training, good for CIFAR-10
- **64×64**: Balanced speed/quality
- **128×128**: Higher quality, slower
- **256×256+**: High quality, requires more GPU memory

### 2. Data Augmentation (Optional)

Common augmentations:
- Random horizontal flip
- Color jitter (slight)
- **Note**: Be careful with augmentations - they can affect the learned distribution

### 3. Normalization

Images should be normalized to [-1, 1] range:
```python
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Maps [0,1] to [-1,1]
])
```

## Creating a Dataset Class

Here's a template for a custom dataset:

```python
from torch.utils.data import Dataset
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, f) 
            for f in os.listdir(root_dir) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image
```

## Data Loader Setup

```python
from torch.utils.data import DataLoader

# Create dataset
dataset = ImageDataset('data/processed', transform=transform)

# Create data loader
dataloader = DataLoader(
    dataset,
    batch_size=32,  # Adjust based on GPU memory
    shuffle=True,
    num_workers=4,  # Parallel data loading
    pin_memory=True  # Faster GPU transfer
)
```

## Data Validation

Before training, verify your data:

```python
import matplotlib.pyplot as plt

# Get a batch
batch = next(iter(dataloader))
print(f"Batch shape: {batch.shape}")
print(f"Min value: {batch.min()}, Max value: {batch.max()}")

# Visualize
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    img = batch[i].permute(1, 2, 0)  # CHW -> HWC
    img = (img + 1) / 2  # Denormalize [-1,1] -> [0,1]
    ax.imshow(img.clamp(0, 1))
    ax.axis('off')
plt.tight_layout()
plt.savefig('data_check.png')
```

## Preprocessing Script

Use the provided `scripts/preprocess_data.py`:

```bash
python scripts/preprocess_data.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --size 64 \
    --num_workers 4
```

## Data Statistics

It's helpful to compute dataset statistics:

```python
def compute_stats(dataloader):
    mean = 0.0
    std = 0.0
    total = 0
    
    for images in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total += batch_samples
    
    mean /= total
    std /= total
    return mean, std
```

## Tips

1. **Start Small**: Begin with a small subset (1000-5000 images) for faster iteration
2. **Consistent Format**: Ensure all images are RGB and same size
3. **Quality Check**: Remove corrupted or low-quality images
4. **Storage**: Consider using a fast SSD for data loading speed

## Next Steps

With your data prepared, move to [Step 4: Model Architecture](./04-model-architecture.md) to build the diffusion model.


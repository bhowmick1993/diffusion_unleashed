# 🎮 Side Quest: Transform Diffusion-101 into a Portfolio Piece

A step-by-step guide to upgrading this learning project into something impressive for your portfolio.

---

## 📋 Progress Tracker

| # | Task | Status | Priority | Effort |
|---|------|--------|----------|--------|
| 1 | [Add Gradio Demo](#1-add-gradio-demo) | ⬜ | 🔴 High | 2-3 hrs |
| 2 | [Deploy to Hugging Face Spaces](#2-deploy-to-hugging-face-spaces) | ⬜ | 🔴 High | 30 min |
| 3 | [Create Results Gallery](#3-create-results-gallery) | ⬜ | 🔴 High | 1-2 hrs |
| 4 | [Upload Pretrained Checkpoint](#4-upload-pretrained-checkpoint) | ⬜ | 🟡 Medium | 30 min |
| 5 | [Add Class-Conditional Generation](#5-add-class-conditional-generation) | ⬜ | 🔴 High | 3-4 hrs |
| 6 | [Add Classifier-Free Guidance (CFG)](#6-add-classifier-free-guidance-cfg) | ⬜ | 🟡 Medium | 2-3 hrs |
| 7 | [Create Benchmark Comparison](#7-create-benchmark-comparison) | ⬜ | 🟡 Medium | 2-3 hrs |
| 8 | [Write Technical Blog Post](#8-write-technical-blog-post) | ⬜ | 🟢 Low | 3-4 hrs |

**Legend:** ⬜ Not Started | 🔄 In Progress | ✅ Complete

---

## 1. Add Gradio Demo

**Why:** Lets anyone interact with your model without running code. Instant "wow" factor.

### Step 1.1: Install Gradio

```bash
pip install gradio
echo "gradio>=4.0.0" >> requirements.txt
```

### Step 1.2: Create `app.py`

Create a new file `app.py` in the project root:

```python
"""
Gradio demo for Diffusion-101.
Run with: python app.py
"""
import torch
import gradio as gr
import numpy as np
from PIL import Image

from src.models.unet import UNet
from src.models.diffusion import DiffusionModel

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 32
CHECKPOINT_PATH = "outputs/checkpoints/best_model.pt"  # Update this path

# Load model
def load_model(checkpoint_path):
    unet = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=128,
        channel_mult=(1, 2, 4, 8),
        time_emb_dim=128
    )
    
    diffusion = DiffusionModel(
        model=unet,
        timesteps=1000,
        schedule_type='linear',
        device=DEVICE
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    diffusion.model.load_state_dict(checkpoint['model_state_dict'])
    diffusion.model.eval()
    
    return diffusion

# Try to load model, handle missing checkpoint gracefully
try:
    model = load_model(CHECKPOINT_PATH)
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False
    print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}")


def generate_images(
    sampler: str,
    num_steps: int,
    num_images: int,
    seed: int
):
    """Generate images using the diffusion model."""
    if not MODEL_LOADED:
        # Return placeholder images if model isn't loaded
        return [Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='gray') for _ in range(num_images)]
    
    torch.manual_seed(seed)
    
    with torch.no_grad():
        if sampler == "DDPM":
            samples = model.sample(
                image_size=IMAGE_SIZE,
                batch_size=num_images,
                channels=3
            )
        # Add DDIM/DPM-Solver options here when implemented
        else:
            samples = model.sample(
                image_size=IMAGE_SIZE,
                batch_size=num_images,
                channels=3
            )
    
    # Convert to PIL images
    images = []
    for i in range(samples.shape[0]):
        img = samples[i].cpu().permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        images.append(Image.fromarray(img))
    
    return images


# Build Gradio interface
with gr.Blocks(title="Diffusion-101 Demo") as demo:
    gr.Markdown("""
    # 🎨 Diffusion-101: Interactive Image Generator
    
    Generate images using a diffusion model trained from scratch on CIFAR-10.
    
    **How it works:** The model starts with random noise and gradually denoises it 
    into a realistic image over multiple steps.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            sampler = gr.Dropdown(
                choices=["DDPM", "DDIM", "DPM-Solver"],
                value="DDPM",
                label="Sampler"
            )
            num_steps = gr.Slider(
                minimum=10,
                maximum=1000,
                value=100,
                step=10,
                label="Number of Steps"
            )
            num_images = gr.Slider(
                minimum=1,
                maximum=16,
                value=4,
                step=1,
                label="Number of Images"
            )
            seed = gr.Slider(
                minimum=0,
                maximum=10000,
                value=42,
                step=1,
                label="Random Seed"
            )
            generate_btn = gr.Button("Generate", variant="primary")
        
        with gr.Column(scale=2):
            gallery = gr.Gallery(
                label="Generated Images",
                columns=4,
                height="auto"
            )
    
    generate_btn.click(
        fn=generate_images,
        inputs=[sampler, num_steps, num_images, seed],
        outputs=gallery
    )
    
    gr.Markdown("""
    ---
    **About this project:** This is an educational implementation of diffusion models 
    built from scratch. Check out the [GitHub repository](https://github.com/yourusername/diffusion-101) 
    for the full codebase and tutorials.
    """)

if __name__ == "__main__":
    demo.launch(share=True)
```

### Step 1.3: Test Locally

```bash
python app.py
```

---

## 2. Deploy to Hugging Face Spaces

**Why:** Free hosting with GPU support. Shareable link for your resume/portfolio.

### Step 2.1: Create Hugging Face Account

1. Go to [huggingface.co](https://huggingface.co)
2. Create an account
3. Create a new Space: `huggingface.co/new-space`
   - Name: `diffusion-101`
   - SDK: Gradio
   - Hardware: CPU Basic (free) or GPU if available

### Step 2.2: Prepare Files

Create `requirements.txt` for the Space:

```
torch
torchvision
gradio>=4.0.0
numpy
Pillow
tqdm
```

### Step 2.3: Upload Files

Either use the web interface or:

```bash
# Install huggingface_hub
pip install huggingface_hub

# Login
huggingface-cli login

# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/diffusion-101
cd diffusion-101

# Copy your files
cp -r ../src .
cp ../app.py .
cp ../outputs/checkpoints/best_model.pt .

# Push
git add .
git commit -m "Initial deployment"
git push
```

---

## 3. Create Results Gallery

**Why:** Visual proof that your model works. Worth more than words.

### Step 3.1: Create Directory Structure

```bash
mkdir -p results/{training_progress,sampler_comparison,samples}
```

### Step 3.2: Generate Training Progress GIF

Add this to a notebook or script:

```python
import imageio
import os
from glob import glob

def create_training_gif(sample_dir, output_path, duration=0.5):
    """Create a GIF showing training progression."""
    images = []
    sample_files = sorted(glob(f"{sample_dir}/epoch_*.png"))
    
    for filepath in sample_files:
        images.append(imageio.imread(filepath))
    
    imageio.mimsave(output_path, images, duration=duration, loop=0)
    print(f"Saved training progress GIF to {output_path}")

# Usage (run after training with periodic sample saves)
create_training_gif("outputs/samples", "results/training_progress/progress.gif")
```

### Step 3.3: Generate Sampler Comparison Grid

```python
import matplotlib.pyplot as plt
import torch

def compare_samplers(model, seed=42):
    """Generate comparison grid of different samplers."""
    torch.manual_seed(seed)
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    
    samplers = ["DDPM (1000 steps)", "DDIM (100 steps)", "DPM-Solver (50 steps)"]
    
    for row, (sampler_name, samples) in enumerate([
        ("DDPM (1000 steps)", model.sample(32, 4, 3)),
        # ("DDIM (100 steps)", model.ddim_sample(32, 4, 3, steps=100)),
        # ("DPM-Solver (50 steps)", model.dpm_sample(32, 4, 3, steps=50)),
    ]):
        for col in range(4):
            img = samples[col].cpu().permute(1, 2, 0).numpy()
            img = (img * 255).astype('uint8')
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(sampler_name, fontsize=12)
    
    plt.suptitle("Sampler Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/sampler_comparison/comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
```

### Step 3.4: Update README with Results

Add to your main `README.md`:

```markdown
## Results

### Training Progress
![Training Progress](results/training_progress/progress.gif)

### Sampler Comparison
![Sampler Comparison](results/sampler_comparison/comparison.png)

### Sample Gallery
| DDPM | DDIM | DPM-Solver |
|------|------|------------|
| ![](results/samples/ddpm_1.png) | ![](results/samples/ddim_1.png) | ![](results/samples/dpm_1.png) |
```

---

## 4. Upload Pretrained Checkpoint

**Why:** Lets others use your model instantly without training.

### Step 4.1: Create Model Card

Create `model_card.md`:

```markdown
---
license: mit
tags:
  - diffusion
  - image-generation
  - pytorch
  - cifar10
datasets:
  - cifar10
---

# Diffusion-101 CIFAR-10

A U-Net based diffusion model trained on CIFAR-10 for educational purposes.

## Model Details

- **Architecture:** U-Net with attention
- **Parameters:** ~8M
- **Training:** 100 epochs on CIFAR-10
- **Resolution:** 32x32
- **Samplers:** DDPM, DDIM, DPM-Solver

## Usage

```python
from huggingface_hub import hf_hub_download
import torch

checkpoint = hf_hub_download("YOUR_USERNAME/diffusion-101-cifar10", "model.pt")
# Load and use...
```

## Training

Trained using the Diffusion-101 tutorial: [GitHub Link]
```

### Step 4.2: Upload to Hugging Face

```bash
# Create model repo
huggingface-cli repo create diffusion-101-cifar10 --type model

# Upload checkpoint
huggingface-cli upload YOUR_USERNAME/diffusion-101-cifar10 \
    outputs/checkpoints/best_model.pt model.pt

# Upload model card
huggingface-cli upload YOUR_USERNAME/diffusion-101-cifar10 \
    model_card.md README.md
```

---

## 5. Add Class-Conditional Generation

**Why:** Major feature upgrade. Lets users generate specific classes (cars, planes, cats, etc.)

### Step 5.1: Create `src/models/conditional_unet.py`

```python
"""
Class-conditional U-Net for diffusion models.
"""
import torch
import torch.nn as nn
from .unet import UNet, SinusoidalPositionEmbeddings


class ConditionalUNet(UNet):
    """
    U-Net with class conditioning via embedding addition.
    
    Args:
        num_classes: Number of classes (10 for CIFAR-10)
        **kwargs: Arguments passed to parent UNet
    """
    
    def __init__(self, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        
        time_emb_dim = kwargs.get('time_emb_dim', 128)
        
        # Class embedding
        self.num_classes = num_classes
        self.class_embed = nn.Embedding(num_classes + 1, time_emb_dim)  # +1 for null class
        self.null_class = num_classes  # Index for unconditional
    
    def forward(self, x, timestep, class_label=None):
        """
        Forward pass with optional class conditioning.
        
        Args:
            x: Input tensor (B, C, H, W)
            timestep: Timestep tensor (B,)
            class_label: Optional class labels (B,). None = unconditional.
        
        Returns:
            Predicted noise (B, C, H, W)
        """
        # Time embedding
        t_emb = self.time_embed(timestep)
        
        # Add class embedding
        if class_label is not None:
            c_emb = self.class_embed(class_label)
            t_emb = t_emb + c_emb
        else:
            # Use null class embedding for unconditional
            null_labels = torch.full(
                (x.shape[0],), self.null_class, 
                device=x.device, dtype=torch.long
            )
            c_emb = self.class_embed(null_labels)
            t_emb = t_emb + c_emb
        
        # Rest is same as parent UNet
        h = self.conv_in(x)
        
        skip_connections = []
        for down_block, down_sample in zip(self.down_blocks, self.down_samples):
            h = down_block(h, t_emb)
            skip_connections.append(h)
            h = down_sample(h)
        
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        for up_sample, up_block in zip(self.up_samples, self.up_blocks):
            h = up_sample(h)
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1)
            h = up_block(h, t_emb)
        
        h = self.norm_out(h)
        h = nn.SiLU()(h)
        h = self.conv_out(h)
        
        return h
```

### Step 5.2: Update Training to Use Labels

Modify `src/training/trainer.py` to pass class labels:

```python
# In train_epoch method, change:
for batch_idx, images in enumerate(pbar):
# To:
for batch_idx, (images, labels) in enumerate(pbar):
    images = images.to(self.device)
    labels = labels.to(self.device)
    
    # Random dropout of labels for classifier-free guidance training
    # 10% of the time, use null class
    mask = torch.rand(labels.shape[0]) < 0.1
    labels[mask] = self.model.model.null_class
    
    # Pass labels to compute_loss
    loss = self.model.compute_loss(images, t, labels=labels)
```

### Step 5.3: Update Gradio Demo

Add class selection dropdown:

```python
class_names = ["airplane", "automobile", "bird", "cat", "deer", 
               "dog", "frog", "horse", "ship", "truck", "random"]

class_dropdown = gr.Dropdown(
    choices=class_names,
    value="random",
    label="Class (CIFAR-10)"
)
```

---

## 6. Add Classifier-Free Guidance (CFG)

**Why:** The technique that makes modern diffusion models (Stable Diffusion, DALL-E) work so well.

### Step 6.1: Add CFG Sampling to `src/models/diffusion.py`

```python
def sample_with_cfg(
    self, 
    image_size, 
    batch_size=16, 
    channels=3,
    class_labels=None,
    guidance_scale=7.5
):
    """
    Generate samples with classifier-free guidance.
    
    Args:
        image_size: Size of generated images
        batch_size: Number of samples
        channels: Number of channels
        class_labels: Class labels for conditional generation
        guidance_scale: CFG scale (higher = more adherence to class)
    
    Returns:
        Generated images
    """
    self.model.eval()
    shape = (batch_size, channels, image_size, image_size)
    
    # Start from pure noise
    img = torch.randn(shape, device=self.device)
    
    # Null labels for unconditional prediction
    null_labels = torch.full(
        (batch_size,), self.model.null_class,
        device=self.device, dtype=torch.long
    )
    
    from tqdm import tqdm
    for i in tqdm(reversed(range(0, self.timesteps)), desc='CFG Sampling'):
        t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
        
        # Predict conditional and unconditional noise
        with torch.no_grad():
            noise_cond = self.model(img, t, class_labels)
            noise_uncond = self.model(img, t, null_labels)
        
        # Classifier-free guidance
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        
        # Denoising step (simplified)
        img = self._denoise_step(img, t, i, noise_pred)
    
    img = (img + 1) / 2
    img = torch.clamp(img, 0, 1)
    
    return img
```

### Step 6.2: Add Guidance Scale Slider to Demo

```python
guidance_scale = gr.Slider(
    minimum=1.0,
    maximum=15.0,
    value=7.5,
    step=0.5,
    label="Guidance Scale (CFG)"
)
```

---

## 7. Create Benchmark Comparison

**Why:** Quantitative evidence that your implementation works correctly.

### Step 7.1: Create `benchmarks/evaluate.py`

```python
"""
Benchmark evaluation for diffusion models.
"""
import torch
import numpy as np
from tqdm import tqdm
import time

# Install: pip install pytorch-fid
try:
    from pytorch_fid import fid_score
    HAS_FID = True
except ImportError:
    HAS_FID = False
    print("Install pytorch-fid for FID evaluation: pip install pytorch-fid")


def compute_fid(model, real_images_path, num_samples=10000, batch_size=64):
    """Compute FID score between generated and real images."""
    if not HAS_FID:
        return None
    
    # Generate samples
    generated_path = "benchmarks/generated_samples"
    os.makedirs(generated_path, exist_ok=True)
    
    num_batches = num_samples // batch_size
    for i in tqdm(range(num_batches), desc="Generating samples"):
        samples = model.sample(32, batch_size, 3)
        for j, img in enumerate(samples):
            save_image(img, f"{generated_path}/{i*batch_size + j}.png")
    
    # Compute FID
    fid = fid_score.calculate_fid_given_paths(
        [real_images_path, generated_path],
        batch_size=50,
        device='cuda',
        dims=2048
    )
    
    return fid


def benchmark_samplers(model, num_samples=16):
    """Benchmark different samplers for speed and quality."""
    results = []
    
    samplers = [
        ("DDPM", 1000, lambda: model.sample(32, num_samples, 3)),
        # ("DDIM-100", 100, lambda: model.ddim_sample(32, num_samples, 3, steps=100)),
        # ("DDIM-50", 50, lambda: model.ddim_sample(32, num_samples, 3, steps=50)),
        # ("DPM-Solver-50", 50, lambda: model.dpm_sample(32, num_samples, 3, steps=50)),
    ]
    
    for name, steps, sample_fn in samplers:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        samples = sample_fn()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        
        results.append({
            "sampler": name,
            "steps": steps,
            "time_seconds": elapsed,
            "time_per_image": elapsed / num_samples
        })
        
        print(f"{name}: {elapsed:.2f}s ({elapsed/num_samples:.3f}s per image)")
    
    return results


if __name__ == "__main__":
    # Load model and run benchmarks
    from src.models.unet import UNet
    from src.models.diffusion import DiffusionModel
    
    model = ...  # Load your model
    
    print("\n=== Sampler Benchmark ===")
    benchmark_samplers(model)
    
    print("\n=== FID Evaluation ===")
    fid = compute_fid(model, "data/cifar10_test")
    print(f"FID Score: {fid:.2f}")
```

### Step 7.2: Add Results Table to README

```markdown
## Benchmarks

| Model Config | Params | FID ↓ | Sampler | Steps | Time (16 imgs) |
|--------------|--------|-------|---------|-------|----------------|
| Small U-Net | 2.5M | 48.3 | DDPM | 1000 | 12.4s |
| Medium U-Net | 8M | 31.2 | DDPM | 1000 | 35.1s |
| Medium U-Net | 8M | 33.8 | DDIM | 100 | 3.8s |
| Medium U-Net | 8M | 32.1 | DPM-Solver | 50 | 1.9s |

*Evaluated on CIFAR-10 test set. FID computed with 10,000 samples.*
```

---

## 8. Write Technical Blog Post

**Why:** Shows communication skills. Helps others learn. Great for LinkedIn/portfolio.

### Step 8.1: Create `BLOG.md` Outline

```markdown
# Building a Diffusion Model from Scratch: What I Learned

## Introduction
- Why I built this
- What diffusion models are (1-2 paragraphs)

## The Math (Simplified)
- Forward process: adding noise
- Reverse process: predicting noise
- Key insight: we learn to denoise, not generate directly

## Architecture Decisions
- Why U-Net?
- Attention placement
- Time embedding design

## Training Insights
- Loss function choice (epsilon vs x0 prediction)
- Learning rate scheduling
- Common failure modes I encountered

## Sampler Deep Dive
- DDPM: the baseline
- DDIM: faster but same quality
- DPM-Solver: even faster with math tricks

## Results & Analysis
- FID scores achieved
- Visual quality assessment
- What worked, what didn't

## What I'd Do Differently
- Lessons learned
- Future improvements

## Resources
- Papers that helped
- Other implementations I referenced
```

### Step 8.2: Publish

- **Medium:** Good reach, easy formatting
- **Dev.to:** Developer-focused audience
- **Personal blog:** Full control, SEO benefits
- **LinkedIn article:** Professional visibility

---

## 📁 Final Project Structure

After completing all tasks:

```
diffusion_unleashed/
├── app.py                          # 🆕 Gradio demo
├── README.md                       # Updated with results
├── SideQuestReadme.md              # This file
├── BLOG.md                         # 🆕 Technical write-up
├── requirements.txt                # Updated with gradio
│
├── src/
│   ├── models/
│   │   ├── unet.py
│   │   ├── conditional_unet.py     # 🆕 Class-conditional
│   │   └── diffusion.py            # Updated with CFG
│   ├── training/
│   │   └── trainer.py              # Updated for conditional
│   └── ...
│
├── benchmarks/                     # 🆕
│   ├── evaluate.py
│   └── results.json
│
├── results/                        # 🆕
│   ├── training_progress/
│   │   └── progress.gif
│   ├── sampler_comparison/
│   │   └── comparison.png
│   └── samples/
│       └── *.png
│
├── notebooks/
│   └── ...
│
└── guides/
    └── ...
```

---

## 🎯 Quick Start Order

**Minimum Viable Portfolio (do these first):**

1. ⬜ Train a model and save checkpoint
2. ⬜ Create `app.py` with Gradio demo
3. ⬜ Generate results gallery (GIFs, comparison images)
4. ⬜ Deploy to Hugging Face Spaces
5. ⬜ Update README with live demo link and results

**Level Up (do these next):**

6. ⬜ Add class-conditional generation
7. ⬜ Add classifier-free guidance
8. ⬜ Create benchmark table
9. ⬜ Upload pretrained weights to HF Hub
10. ⬜ Write and publish blog post

---

## ✅ Completion Checklist

When you're done, your project should have:

- [ ] Live demo anyone can try (HF Spaces link)
- [ ] Visual results (GIFs, comparison grids)
- [ ] Pretrained checkpoint available for download
- [ ] Class-conditional generation
- [ ] Classifier-free guidance support
- [ ] Benchmark comparison table
- [ ] Technical blog post
- [ ] Clean, well-documented code

**Good luck on your side quest! 🚀**

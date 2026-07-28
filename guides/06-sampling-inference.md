# Step 6: Sampling & Inference

## Overview

After training, we need to generate images by reversing the diffusion process. This guide covers different sampling methods to generate images from noise.

## DDPM Sampling (Original Method)

### Reverse Diffusion Process

The reverse process iteratively denoises from pure noise:

```python
import torch
import torch.nn.functional as F
from tqdm import tqdm

def p_sample(model, x, t, t_index, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance):
    """
    Sample from p_θ(x_{t-1} | x_t)
    """
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Predict noise
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape, timesteps, betas, device='cuda'):
    """
    Generate samples by iteratively denoising
    """
    # Precompute values
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = betas * (1. - alphas_cumprod) / (1. - alphas_cumprod)
    
    # Start from pure noise
    img = torch.randn(shape, device=device)
    
    # Denoise step by step
    for i in tqdm(reversed(range(0, timesteps)), desc='Sampling'):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i, betas, sqrt_one_minus_alphas_cumprod, 
                      sqrt_recip_alphas, posterior_variance)
    
    return img

def sample(model, image_size, batch_size=16, channels=3, device='cuda', timesteps=1000):
    """
    Generate samples from the model
    """
    model.eval()
    shape = (batch_size, channels, image_size, image_size)
    betas = linear_beta_schedule(timesteps).to(device)
    
    samples = p_sample_loop(model, shape, timesteps, betas, device)
    
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    return samples
```

## DDIM Sampling (Faster Alternative)

DDIM allows faster sampling with fewer steps:

```python
@torch.no_grad()
def ddim_sample(model, shape, num_inference_steps=50, eta=0.0, device='cuda'):
    """
    DDIM sampling - faster with fewer steps
    eta=0: deterministic, eta=1: stochastic
    """
    model.eval()
    
    # Create timestep sequence
    timesteps = torch.linspace(1000, 0, num_inference_steps + 1, dtype=torch.long, device=device)
    
    # Precompute alphas
    betas = linear_beta_schedule(1000).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Start from noise
    img = torch.randn(shape, device=device)
    
    for i in tqdm(range(num_inference_steps), desc='DDIM Sampling'):
        t = timesteps[i]
        next_t = timesteps[i + 1] if i < num_inference_steps - 1 else torch.tensor(0)
        
        # Predict noise
        noise_pred = model(img, t)
        
        # Compute alpha values
        alpha_t = alphas_cumprod[t]
        alpha_next = alphas_cumprod[next_t]
        
        # Predict x_0
        pred_x0 = (img - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        
        # Compute direction
        dir_xt = torch.sqrt(1 - alpha_next - eta ** 2 * (1 - alpha_next)) * noise_pred
        
        # Sample
        if eta == 0:
            img = torch.sqrt(alpha_next) * pred_x0 + dir_xt
        else:
            noise = torch.randn_like(img)
            img = torch.sqrt(alpha_next) * pred_x0 + dir_xt + eta * torch.sqrt(1 - alpha_next) * noise
    
    # Denormalize
    img = (img + 1) / 2
    img = torch.clamp(img, 0, 1)
    
    return img
```

## Visualization

### Save Generated Images

```python
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def save_samples(samples, path, nrow=4):
    """
    Save a grid of generated images
    """
    grid = make_grid(samples, nrow=nrow, normalize=False)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()

# Generate and save
samples = sample(model, image_size=64, batch_size=16)
save_samples(samples, 'outputs/samples/generated.png')
```

### Progressive Generation Visualization

Show the denoising process:

```python
@torch.no_grad()
def sample_progressive(model, shape, timesteps=1000, save_steps=10, device='cuda'):
    """
    Generate samples and save intermediate steps
    """
    model.eval()
    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = betas * (1. - alphas_cumprod) / (1. - alphas_cumprod)
    
    img = torch.randn(shape, device=device)
    images = []
    
    for i in tqdm(reversed(range(0, timesteps)), desc='Progressive Sampling'):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i, betas, sqrt_one_minus_alphas_cumprod,
                      sqrt_recip_alphas, posterior_variance)
        
        if i % (timesteps // save_steps) == 0:
            img_vis = (img + 1) / 2
            img_vis = torch.clamp(img_vis, 0, 1)
            images.append(img_vis[0].cpu())
    
    return images

# Create animation
images = sample_progressive(model, (1, 3, 64, 64))
fig, axes = plt.subplots(1, len(images), figsize=(20, 4))
for ax, img in zip(axes, images):
    ax.imshow(img.permute(1, 2, 0))
    ax.axis('off')
plt.savefig('outputs/samples/progressive.png')
```

## Inference Script

Use the provided `scripts/sample.py`:

```bash
python scripts/sample.py \
    --checkpoint outputs/checkpoints/checkpoint_epoch_100.pt \
    --output_dir outputs/samples \
    --num_samples 16 \
    --image_size 64 \
    --method ddim \
    --steps 50
```

## Sampling Methods Comparison

| Method | Steps | Speed | Quality | Deterministic |
|--------|-------|-------|---------|---------------|
| **DDPM** | 1000 | Slow | High | No |
| **DDIM** | 50 | Fast | High | Yes (eta=0) |
| **DPM-Solver** | 20 | Very Fast | High | Yes |

## Tips for Better Generation

1. **Temperature Scaling**: Adjust noise during sampling
   ```python
   noise = torch.randn_like(x) * temperature
   ```

2. **Classifier-Free Guidance**: Use conditional generation for better control
   ```python
   # Combine conditional and unconditional predictions
   noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
   ```

3. **Seed Control**: Set random seed for reproducibility
   ```python
   torch.manual_seed(42)
   ```

4. **Batch Generation**: Generate multiple samples and pick best

## Evaluation Metrics

### FID Score (Fréchet Inception Distance)

```python
# Install: pip install pytorch-fid
from pytorch_fid import fid_score

fid_value = fid_score.calculate_fid_given_paths(
    ['data/real', 'outputs/samples'],
    batch_size=50,
    device='cuda',
    dims=2048
)
print(f'FID Score: {fid_value:.2f}')
```

### IS Score (Inception Score)

```python
# Measures quality and diversity
# Lower FID and higher IS = better
```

## Next Steps

Explore [Step 7: Advanced Techniques](./07-advanced-techniques.md) for optimizations and improvements.


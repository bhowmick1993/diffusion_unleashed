# Step 7: Advanced Techniques

## Overview

This guide covers advanced techniques to improve your diffusion model's quality, speed, and capabilities.

## 1. Improved Noise Schedules

### Cosine Schedule

Better for high-resolution images:

```python
def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
```

## 2. Classifier-Free Guidance

Enable conditional generation with better control:

```python
class ConditionalUNet(UNet):
    def __init__(self, num_classes=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.class_embed = nn.Embedding(num_classes, kwargs['time_emb_dim'])
    
    def forward(self, x, timestep, class_labels=None):
        t_emb = self.time_embed(timestep)
        
        if class_labels is not None:
            class_emb = self.class_embed(class_labels)
            t_emb = t_emb + class_emb
        
        # Rest of forward pass...
        return super().forward(x, t_emb)

# During sampling with guidance
def sample_with_guidance(model, class_label, guidance_scale=7.5):
    # Unconditional prediction
    uncond_pred = model(x_t, t, class_labels=None)
    
    # Conditional prediction
    cond_pred = model(x_t, t, class_labels=class_label)
    
    # Combine with guidance
    noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
    return noise_pred
```

## 3. Latent Diffusion

Work in a compressed latent space for efficiency:

```python
from torch import nn
import torch.nn.functional as F

class VAEEncoder(nn.Module):
    """Encode images to latent space"""
    def __init__(self):
        super().__init__()
        # Encoder architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(512, 8, 4, 2, 1)  # Latent channels
        )
    
    def forward(self, x):
        return self.encoder(x)

class VAEDecoder(nn.Module):
    """Decode latents back to images"""
    def __init__(self):
        super().__init__()
        # Decoder architecture
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 512, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.decoder(z)

# Training in latent space
def train_latent_diffusion(vae, diffusion_model, dataloader):
    vae.eval()  # Freeze VAE
    with torch.no_grad():
        latents = vae.encoder(images)
    
    # Train diffusion on latents instead of images
    # (much faster, less memory)
```

## 4. Efficient Sampling Methods

### DPM-Solver

Very fast sampling (20 steps):

```python
def dpm_solver_sample(model, shape, num_steps=20, device='cuda'):
    """
    DPM-Solver: Fast ODE-based sampling
    """
    # Implementation of DPM-Solver algorithm
    # See: https://arxiv.org/abs/2206.00927
    pass
```

### PLMS (Pseudo Linear Multi-Step)

```python
def plms_sample(model, shape, num_steps=50, device='cuda'):
    """
    PLMS: Improved DDIM with multi-step prediction
    """
    # Uses history of previous predictions
    # More stable than DDIM
    pass
```

## 5. Attention Optimizations

### Flash Attention

For faster attention computation:

```python
try:
    from flash_attn import flash_attn_func
    
    def flash_attention(q, k, v):
        return flash_attn_func(q, k, v)
except ImportError:
    # Fallback to standard attention
    def flash_attention(q, k, v):
        return standard_attention(q, k, v)
```

### Sparse Attention

For larger images:

```python
def sparse_attention(q, k, v, block_size=32):
    """
    Process attention in blocks for memory efficiency
    """
    # Divide into blocks and process separately
    pass
```

## 6. Training Optimizations

### Gradient Accumulation

Train with larger effective batch size:

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### EMA (Exponential Moving Average)

Smoother model weights:

```python
from copy import deepcopy

class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.ema_model = deepcopy(model)
        self.decay = decay
    
    def update(self):
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(), 
                self.model.parameters()
            ):
                ema_param.data.mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )
    
    def state_dict(self):
        return self.ema_model.state_dict()

# Usage
ema = EMA(model)
# After each training step:
ema.update()
# Use ema.state_dict() for inference
```

## 7. Multi-Resolution Training

Train on multiple resolutions:

```python
def multi_resolution_training(model, dataloader_32, dataloader_64, dataloader_128):
    """
    Train on different resolutions to improve generalization
    """
    for batch_32, batch_64, batch_128 in zip(
        dataloader_32, dataloader_64, dataloader_128
    ):
        # Train on each resolution
        loss_32 = train_step(model, batch_32, image_size=32)
        loss_64 = train_step(model, batch_64, image_size=64)
        loss_128 = train_step(model, batch_128, image_size=128)
        
        total_loss = loss_32 + loss_64 + loss_128
        total_loss.backward()
```

## 8. Text-to-Image (CLIP Guidance)

Add text conditioning:

```python
import clip

class TextConditionedDiffusion(nn.Module):
    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda")
        self.text_proj = nn.Linear(512, diffusion_model.time_emb_dim)
    
    def forward(self, x, t, text_prompt):
        # Encode text
        text_tokens = clip.tokenize([text_prompt]).to(x.device)
        text_features = self.clip_model.encode_text(text_tokens)
        text_emb = self.text_proj(text_features)
        
        # Combine with time embedding
        t_emb = self.diffusion_model.time_embed(t)
        combined_emb = t_emb + text_emb
        
        # Forward through diffusion model
        return self.diffusion_model.forward_with_embedding(x, combined_emb)
```

## 9. Evaluation and Monitoring

### FID Score Tracking

```python
def compute_fid_periodically(model, real_data_path, epoch):
    if epoch % 10 == 0:
        # Generate samples
        samples = generate_samples(model, num_samples=1000)
        save_samples(samples, 'temp_samples')
        
        # Compute FID
        fid = calculate_fid(real_data_path, 'temp_samples')
        print(f'Epoch {epoch} FID: {fid:.2f}')
```

### Training Metrics Dashboard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/diffusion_training')

# Log metrics
writer.add_scalar('Loss/Train', loss, step)
writer.add_scalar('Loss/Validation', val_loss, step)
writer.add_scalar('Metrics/FID', fid_score, step)
writer.add_image('Samples/Generated', sample_grid, step)
```

## 10. Deployment Considerations

### Model Quantization

Reduce model size:

```python
import torch.quantization as quantization

# Quantize model
quantized_model = quantization.quantize_dynamic(
    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
)
```

### ONNX Export

For cross-platform deployment:

```python
torch.onnx.export(
    model,
    (dummy_input, dummy_timestep),
    "diffusion_model.onnx",
    input_names=['image', 'timestep'],
    output_names=['noise_pred']
)
```

## Best Practices Summary

1. **Start Simple**: Begin with basic DDPM, then add improvements
2. **Monitor Training**: Use TensorBoard, track FID/IS scores
3. **Save Regularly**: Checkpoint frequently, keep best model
4. **Experiment**: Try different schedules, architectures, guidance scales
5. **Optimize Gradually**: Add optimizations one at a time

## Resources for Further Learning

- **Stable Diffusion**: https://github.com/StableDiffusion
- **DALL-E 2 Paper**: Understanding large-scale diffusion models
- **Latent Diffusion**: https://arxiv.org/abs/2112.10752
- **DPM-Solver**: https://arxiv.org/abs/2206.00927

## Conclusion

You now have a complete understanding of diffusion models! Continue experimenting, and don't hesitate to explore the latest research papers for cutting-edge techniques.

Happy training! 🚀


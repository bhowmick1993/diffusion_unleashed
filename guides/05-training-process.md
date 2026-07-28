# Step 5: Training Process

## Overview

This guide covers implementing the complete training loop for a diffusion model, including the forward diffusion process, loss computation, and optimization.

## Forward Diffusion Process

### Noise Schedule

First, define how noise is added over time:

```python
import torch
import torch.nn.functional as F

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """Linear noise schedule"""
    return torch.linspace(start, end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine noise schedule (often better quality)"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

# Precompute values for efficiency
def extract(a, t, x_shape):
    """Extract values from a at timestep t"""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
```

### Forward Diffusion Function

```python
def q_sample(x_start, t, noise=None):
    """
    Sample from q(x_t | x_0)
    Adds noise to x_0 according to timestep t
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
```

## Training Loop

### Complete Training Function

```python
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

def train_diffusion(
    model,
    dataloader,
    num_epochs=100,
    lr=1e-4,
    timesteps=1000,
    device='cuda'
):
    # Setup
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Precompute noise schedule
    betas = linear_beta_schedule(timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)
            batch_size = images.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            
            # Sample noise
            noise = torch.randn_like(images)
            
            # Add noise to images
            noisy_images = q_sample(images, t, noise)
            
            # Predict noise
            noise_pred = model(noisy_images, t)
            
            # Compute loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Logging
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'outputs/checkpoints/checkpoint_epoch_{epoch+1}.pt')
    
    return model
```

## Loss Function Details

### Simple MSE Loss

The standard loss is mean squared error between predicted and actual noise:

```python
loss = F.mse_loss(noise_pred, noise)
```

### Weighted Loss (Optional)

You can weight different timesteps differently:

```python
def weighted_mse_loss(pred, target, weights):
    """Weighted MSE loss"""
    return (weights * (pred - target) ** 2).mean()
```

## Training Tips

### 1. Learning Rate Scheduling

```python
# Cosine annealing (recommended)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# Or warmup + cosine
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )
    return LambdaLR(optimizer, lr_lambda)
```

### 2. Gradient Clipping

Prevent exploding gradients:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3. Mixed Precision Training

Speed up training with less memory:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    noise_pred = model(noisy_images, t)
    loss = F.mse_loss(noise_pred, noise)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4. Monitoring Training

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('outputs/logs')

# In training loop:
writer.add_scalar('Loss/Train', loss.item(), global_step)
writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)

# Visualize samples during training
if batch_idx % 500 == 0:
    with torch.no_grad():
        sample = sample_image(model, device, timesteps)
        writer.add_image('Samples', sample, global_step)
```

## Training Script

Use the provided `scripts/train.py`:

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data_dir data/processed \
    --output_dir outputs \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 1e-4
```

## Expected Training Behavior

1. **Early epochs**: Loss decreases rapidly
2. **Middle epochs**: Gradual improvement, loss stabilizes
3. **Later epochs**: Fine-tuning, small improvements

Typical loss values:
- Initial: ~0.5-1.0
- After 50 epochs: ~0.1-0.3
- After 100 epochs: ~0.05-0.15

## Troubleshooting

### Loss not decreasing
- Check learning rate (try 1e-4 or 5e-5)
- Verify data preprocessing
- Check model architecture

### Out of memory
- Reduce batch size
- Reduce image resolution
- Use gradient accumulation

### Training too slow
- Use mixed precision
- Increase num_workers in DataLoader
- Use smaller model initially

## Next Steps

Once training is complete, learn to generate images in [Step 6: Sampling & Inference](./06-sampling-inference.md).


# Step 4: Model Architecture

## Overview

The core of a diffusion model is a U-Net architecture that predicts noise at each timestep. This guide walks through building this architecture.

## U-Net Architecture

### Why U-Net?

U-Net is ideal for diffusion because:
- **Encoder-Decoder**: Captures both high-level and fine-grained features
- **Skip Connections**: Preserves spatial details during upsampling
- **Proven**: Successfully used in many image generation tasks

### Architecture Components

```
Input (x_t, t)
    ↓
Time Embedding (t)
    ↓
Encoder (Downsampling)
    ↓
Bottleneck
    ↓
Decoder (Upsampling) + Skip Connections
    ↓
Output (predicted noise ε)
```

## Building Blocks

### 1. Sinusoidal Positional Embeddings

Time step t needs to be embedded for the model:

```python
import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
```

### 2. Residual Block with Time Conditioning

Each block processes the image and time information:

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.res_conv = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)
```

### 3. Attention Block (Optional but Recommended)

Self-attention helps capture long-range dependencies:

```python
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.view(B, C, H * W).transpose(1, 2)
        k = k.view(B, C, H * W)
        v = v.view(B, C, H * W).transpose(1, 2)
        
        # Attention
        attn = (q @ k) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        h = (attn @ v).transpose(1, 2).view(B, C, H, W)
        
        return x + self.proj(h)
```

### 4. Complete U-Net

Putting it all together:

```python
class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        time_emb_dim=128,
        model_channels=128,
        channel_mult=(1, 2, 4, 8),
        attention_resolutions=(16,),
        dropout=0.1
    ):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch = model_channels
        
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            self.down_blocks.append(
                ResidualBlock(ch, out_ch, time_emb_dim, dropout)
            )
            if i < len(channel_mult) - 1:
                self.down_samples.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            else:
                self.down_samples.append(nn.Identity())
            ch = out_ch
        
        # Middle block
        self.mid_block1 = ResidualBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResidualBlock(ch, ch, time_emb_dim, dropout)
        
        # Decoder (upsampling)
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            self.up_samples.append(
                nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1) 
                if i > 0 else nn.Identity()
            )
            self.up_blocks.append(
                ResidualBlock(ch + out_ch, out_ch, time_emb_dim, dropout)
            )
            ch = out_ch
        
        # Output
        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)
    
    def forward(self, x, timestep):
        # Time embedding
        t_emb = self.time_embed(timestep)
        
        # Initial conv
        h = self.conv_in(x)
        
        # Encoder
        skip_connections = []
        for down_block, down_sample in zip(self.down_blocks, self.down_samples):
            h = down_block(h, t_emb)
            skip_connections.append(h)
            h = down_sample(h)
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Decoder
        for up_sample, up_block in zip(self.up_samples, self.up_blocks):
            h = up_sample(h)
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1)
            h = up_block(h, t_emb)
        
        # Output
        h = self.norm_out(h)
        h = nn.SiLU()(h)
        h = self.conv_out(h)
        
        return h
```

## Model Initialization

```python
# Create model
model = UNet(
    in_channels=3,
    out_channels=3,
    time_emb_dim=128,
    model_channels=128,
    channel_mult=(1, 2, 4, 8)
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Test forward pass
x = torch.randn(1, 3, 64, 64)
t = torch.randint(0, 1000, (1,))
with torch.no_grad():
    output = model(x, t)
print(f"Input shape: {x.shape}, Output shape: {output.shape}")
```

## Model Variants

### Smaller Model (Faster Training)
```python
model = UNet(
    model_channels=64,
    channel_mult=(1, 2, 4)
)
```

### Larger Model (Better Quality)
```python
model = UNet(
    model_channels=256,
    channel_mult=(1, 1, 2, 2, 4, 4),
    attention_resolutions=(32, 16, 8)
)
```

## Next Steps

With the model architecture ready, proceed to [Step 5: Training Process](./05-training-process.md) to implement the training loop.


"""
U-Net architecture for diffusion models.
"""
import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for timesteps."""
    
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


class ResidualBlock(nn.Module):
    """Residual block with time conditioning."""
    
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


class AttentionBlock(nn.Module):
    """Self-attention block for capturing long-range dependencies."""
    
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


class UNet(nn.Module):
    """
    U-Net architecture for diffusion models.
    
    Args:
        in_channels: Number of input channels (typically 3 for RGB)
        out_channels: Number of output channels (typically 3 for RGB)
        time_emb_dim: Dimension of time embeddings
        model_channels: Base number of channels
        channel_mult: Multipliers for channel numbers at each resolution level
        attention_resolutions: Resolutions to apply attention (as tuple of sizes)
        dropout: Dropout rate
    """
    
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
        input_block_chans = [model_channels]
        
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            self.down_blocks.append(
                ResidualBlock(ch, out_ch, time_emb_dim, dropout)
            )
            input_block_chans.append(out_ch)
            
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
        
        # Skip connections come from encoder outputs (excluding initial conv_in)
        # Reverse to match decoder order (pop from end)
        skip_chans = list(reversed(input_block_chans[1:]))  # Exclude initial conv_in channels
        
        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            skip_ch = skip_chans[i]  # Channel count from corresponding encoder block output
            self.up_samples.append(
                nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1) 
                if i > 0 else nn.Identity()
            )
            # Input channels = upsampled channels + skip connection channels
            self.up_blocks.append(
                ResidualBlock(out_ch + skip_ch, out_ch, time_emb_dim, dropout)
            )
            ch = out_ch
        
        # Output
        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)
    
    def forward(self, x, timestep):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            timestep: Timestep tensor of shape (B,)
        
        Returns:
            Predicted noise tensor of shape (B, C, H, W)
        """
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


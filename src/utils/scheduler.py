"""
Noise scheduling utilities for diffusion models.
"""
import torch
import math


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """
    Linear noise schedule.
    
    Args:
        timesteps: Number of diffusion timesteps
        start: Starting beta value
        end: Ending beta value
    
    Returns:
        Tensor of beta values
    """
    return torch.linspace(start, end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine noise schedule (often better quality than linear).
    
    Args:
        timesteps: Number of diffusion timesteps
        s: Small offset to prevent beta from being too small
    
    Returns:
        Tensor of beta values
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def extract(a, t, x_shape):
    """
    Extract values from tensor a at timestep t.
    
    Args:
        a: Tensor to extract from
        t: Timestep indices
        x_shape: Shape of the data tensor (for broadcasting)
    
    Returns:
        Extracted values with proper shape for broadcasting
    """
    batch_size = t.shape[0]
    # Ensure both tensors are on the same device
    # If a is on a different device, move t to match a's device
    if a.device != t.device:
        t = t.to(a.device)
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def get_noise_schedule(timesteps, schedule_type='linear', **kwargs):
    """
    Get noise schedule by type.
    
    Args:
        timesteps: Number of timesteps
        schedule_type: 'linear' or 'cosine'
        **kwargs: Additional arguments for the schedule
    
    Returns:
        Tuple of (betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, 
                 sqrt_one_minus_alphas_cumprod)
    """
    if schedule_type == 'linear':
        betas = linear_beta_schedule(timesteps, **kwargs)
    elif schedule_type == 'cosine':
        betas = cosine_beta_schedule(timesteps, **kwargs)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    return betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod


"""
Diffusion model wrapper with forward and reverse processes.
"""
import torch
import torch.nn.functional as F
from .unet import UNet
from ..utils.scheduler import get_noise_schedule, extract


class DiffusionModel:
    """
    Wrapper class for diffusion model training and sampling.
    
    Args:
        model: U-Net model
        timesteps: Number of diffusion timesteps
        schedule_type: 'linear' or 'cosine'
        device: Device to run on
    """
    
    def __init__(self, model, timesteps=1000, schedule_type='linear', device='cuda'):
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device
        
        # Precompute noise schedule
        betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = \
            get_noise_schedule(timesteps, schedule_type)
        
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
        
        # For sampling
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Sample from q(x_t | x_0) - forward diffusion.
        
        Args:
            x_start: Original images
            t: Timesteps
            noise: Optional noise tensor
        
        Returns:
            Noisy images at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, x, t, t_index):
        """
        Sample from p_θ(x_{t-1} | x_t) - reverse diffusion step.
        
        Args:
            x: Noisy image at timestep t
            t: Current timestep
            t_index: Current timestep index
        
        Returns:
            Denoised image at timestep t-1
        """
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Predict noise
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def sample(self, image_size, batch_size=16, channels=3):
        """
        Generate samples by reversing the diffusion process.
        
        Args:
            image_size: Size of generated images
            batch_size: Number of samples to generate
            channels: Number of channels
        
        Returns:
            Generated images (denormalized to [0, 1])
        """
        self.model.eval()
        shape = (batch_size, channels, image_size, image_size)
        
        # Start from pure noise
        img = torch.randn(shape, device=self.device)
        
        # Denoise step by step
        from tqdm import tqdm
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling'):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t, i)
        
        # Denormalize from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)
        
        return img
    
    def compute_loss(self, x_start, t, noise=None):
        """
        Compute training loss.
        
        Args:
            x_start: Original images
            t: Random timesteps
            noise: Optional noise tensor
        
        Returns:
            MSE loss between predicted and actual noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Add noise
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        noise_pred = self.model(x_noisy, t)
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss


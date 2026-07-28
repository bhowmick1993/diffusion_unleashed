"""
Visualization utilities for diffusion models.
"""
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np


def save_samples(samples, path, nrow=4, normalize=True):
    """
    Save a grid of generated images.
    
    Args:
        samples: Tensor of shape (N, C, H, W) with values in [0, 1]
        path: Path to save the image
        nrow: Number of images per row in the grid
        normalize: Whether to normalize the grid
    """
    # Ensure values are in [0, 1]
    samples = torch.clamp(samples, 0, 1)
    
    grid = make_grid(samples, nrow=nrow, normalize=normalize, padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_training_curves(losses, save_path=None):
    """
    Plot training loss curve.
    
    Args:
        losses: List or array of loss values
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()
    plt.close()


def visualize_noise_schedule(betas, save_path=None):
    """
    Visualize the noise schedule.
    
    Args:
        betas: Beta values
        save_path: Optional path to save the plot
    """
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(betas.cpu().numpy())
    axes[0].set_title('Beta Schedule')
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('Beta')
    axes[0].grid(True)
    
    axes[1].plot(alphas.cpu().numpy())
    axes[1].set_title('Alpha Schedule')
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Alpha')
    axes[1].grid(True)
    
    axes[2].plot(alphas_cumprod.cpu().numpy())
    axes[2].set_title('Alpha Cumulative Product')
    axes[2].set_xlabel('Timestep')
    axes[2].set_ylabel('Alpha Cumprod')
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()
    plt.close()


def show_progressive_generation(images_list, save_path=None):
    """
    Show progressive generation at different timesteps.
    
    Args:
        images_list: List of tensors showing generation at different steps
        save_path: Optional path to save the plot
    """
    n_steps = len(images_list)
    fig, axes = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4))
    
    if n_steps == 1:
        axes = [axes]
    
    for i, img in enumerate(images_list):
        if isinstance(img, torch.Tensor):
            img = img.cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f'Step {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()
    plt.close()


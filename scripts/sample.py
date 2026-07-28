"""
Sampling script for generating images from trained diffusion models.
"""
import argparse
import torch
import yaml
from torchvision.utils import save_image

from src.models.unet import UNet
from src.models.diffusion import DiffusionModel
from src.utils.visualization import save_samples


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Generate samples from diffusion model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='outputs/samples',
                       help='Directory to save generated images')
    parser.add_argument('--num_samples', type=int, default=16,
                       help='Number of samples to generate')
    parser.add_argument('--image_size', type=int, default=64,
                       help='Size of generated images')
    parser.add_argument('--method', type=str, default='ddpm',
                       choices=['ddpm', 'ddim'],
                       help='Sampling method')
    parser.add_argument('--steps', type=int, default=None,
                       help='Number of sampling steps (for DDIM)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Create model
    print("Loading model...")
    unet = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        time_emb_dim=config['model']['time_emb_dim'],
        model_channels=config['model']['model_channels'],
        channel_mult=tuple(config['model']['channel_mult']),
        attention_resolutions=tuple(config['model']['attention_resolutions']),
        dropout=config['model']['dropout']
    )
    
    # Create diffusion model
    timesteps = args.steps if args.steps else config['diffusion']['timesteps']
    diffusion_model = DiffusionModel(
        model=unet,
        timesteps=timesteps,
        schedule_type=config['diffusion']['schedule_type'],
        device=device
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    diffusion_model.model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    samples = diffusion_model.sample(
        image_size=args.image_size,
        batch_size=args.num_samples,
        channels=config['model']['out_channels']
    )
    
    # Save samples
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save as grid
    save_samples(samples, 
                os.path.join(args.output_dir, 'samples_grid.png'),
                nrow=4)
    
    # Save individual images
    for i, sample in enumerate(samples):
        save_image(sample, 
                  os.path.join(args.output_dir, f'sample_{i:03d}.png'))
    
    print(f"Samples saved to {args.output_dir}")

if __name__ == '__main__':
    main()


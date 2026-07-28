"""
Training script for diffusion models.
"""
import os
import sys
sys.path.append(r"C:\AB_Personal\diffusion_101\Diffusion-101")
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.models.diffusion import DiffusionModel
from src.data.dataset import ImageDataset
from src.training.trainer import Trainer


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train a diffusion model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Override data directory from config')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Override output directory from config')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size from config')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Override number of epochs from config')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate from config')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.data_dir:
        config['data']['root_dir'] = args.data_dir
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Setup device
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Create dataset and dataloader
    print(f"Loading dataset from {config['data']['root_dir']}...")
    dataset = ImageDataset(
        os.path.join(config['data']['root_dir'],'train'),
        image_size=config['data']['image_size']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Create model
    print("Creating model...")
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
    diffusion_model = DiffusionModel(
        model=unet,
        timesteps=config['diffusion']['timesteps'],
        schedule_type=config['diffusion']['schedule_type'],
        device=device
    )
    
    # Create trainer
    trainer = Trainer(
        model=diffusion_model,
        dataloader=dataloader,
        num_epochs=config['training']['num_epochs'],
        lr=config['training']['learning_rate'],
        device=device,
        output_dir=config['training']['output_dir']
    )
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()


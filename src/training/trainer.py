"""
Training utilities for diffusion models.
"""
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os


class Trainer:
    """
    Trainer class for diffusion models.
    
    Args:
        model: Diffusion model
        dataloader: Data loader
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        output_dir: Directory to save checkpoints and logs
    """
    
    def __init__(
        self,
        model,
        dataloader,
        num_epochs=100,
        lr=1e-4,
        device='cuda',
        output_dir='outputs'
    ):
        self.model = model
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.device = device
        self.output_dir = output_dir
        
        # Setup optimizer
        self.optimizer = Adam(model.model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        # Setup logging
        os.makedirs(output_dir, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(output_dir, 'logs'))
        
        self.global_step = 0
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.model.train()
        total_loss = 0
        
        pbar = tqdm(self.dataloader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
        
        for batch_idx, images in enumerate(pbar):
            images = images.to(self.device)
            batch_size = images.shape[0]
            
            # Sample random timesteps
            t = torch.randint(
                0, self.model.timesteps, (batch_size,), device=self.device
            ).long()
            
            # Compute loss
            loss = self.model.compute_loss(images, t)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Logging
            self.writer.add_scalar('Loss/Train', loss.item(), self.global_step)
            self.writer.add_scalar('Learning_Rate', 
                                 self.optimizer.param_groups[0]['lr'], 
                                 self.global_step)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.dataloader)
        return avg_loss
    
    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
        
        for epoch in range(self.num_epochs):
            avg_loss = self.train_epoch(epoch)
            self.scheduler.step()
            
            print(f'Epoch {epoch+1}/{self.num_epochs} completed. Average Loss: {avg_loss:.4f}')
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.save_checkpoint(epoch, avg_loss)
        
        self.writer.close()
        print("Training completed!")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint.get('epoch', 0)


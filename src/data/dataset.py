"""
Dataset classes for diffusion model training.
"""
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    """
    Custom dataset for image files.
    
    Args:
        root_dir: Root directory containing images
        transform: Optional transform to apply to images
        image_size: Target image size (will be resized to square)
    """
    
    def __init__(self, root_dir, transform=None, image_size=64):
        self.root_dir = root_dir
        self.image_size = image_size
        self.extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        self.image_paths = []
        folders_in_root_dir = os.listdir(root_dir)
        print(folders_in_root_dir)
        for folder in folders_in_root_dir:
            self.image_paths.extend(
                [os.path.join(root_dir, folder, f) 
                 for f in os.listdir(os.path.join(root_dir, folder)) 
                 if any(f.lower().endswith(ext) for ext in self.extensions)]
            )
        print(len(self.image_paths))
        """
        # Get all image paths
        self.image_paths = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            self.image_paths.extend(
                [os.path.join(root_dir, f) 
                 for f in os.listdir(root_dir) 
                 if f.lower().endswith(ext)]
            )
        """
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # If image is corrupted, return a random image
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (self.image_size, self.image_size), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image


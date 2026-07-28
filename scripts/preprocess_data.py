"""
Data preprocessing script.
"""
import argparse
import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


def preprocess_images(input_dir, output_dir, size=64, num_workers=1):
    """
    Preprocess images: resize and normalize.
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for processed images
        size: Target image size (square)
        num_workers: Number of worker processes
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend([
            f for f in os.listdir(input_dir)
            if f.lower().endswith(ext)
        ])
    
    print(f"Found {len(image_files)} images to process")
    
    # Transform: resize and convert to RGB
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Lambda(lambda img: img.convert('RGB'))
    ])
    
    # Process images
    for img_file in tqdm(image_files, desc='Processing images'):
        try:
            img_path = os.path.join(input_dir, img_file)
            img = Image.open(img_path)
            img = transform(img)
            
            # Save processed image
            output_path = os.path.join(output_dir, img_file)
            img.save(output_path, quality=95)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    print(f"Processed images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess images for training')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing raw images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed images')
    parser.add_argument('--size', type=int, default=64,
                       help='Target image size (default: 64)')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of worker processes (default: 1)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    preprocess_images(
        args.input_dir,
        args.output_dir,
        args.size,
        args.num_workers
    )


if __name__ == '__main__':
    main()


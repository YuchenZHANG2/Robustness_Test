"""
Generate preview images for all corruption types using preview.jpg
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch_corruptions import TorchCorruptions, get_corruption_names

# Fix numpy deprecations
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128

# Patch skimage.filters.gaussian for compatibility
import skimage.filters
_orig_gaussian = skimage.filters.gaussian

def _patched_gaussian(*args, **kwargs):
    if "multichannel" in kwargs:
        channel = kwargs.pop("multichannel")
        kwargs.setdefault("channel_axis", -1 if channel else None)
    return _orig_gaussian(*args, **kwargs)

skimage.filters.gaussian = _patched_gaussian


def generate_all_corruption_previews(image_path='preview.jpg', output_dir='static/previews', severity=3):
    """
    Generate preview images for all corruption types organized by category.
    
    Args:
        image_path: Path to the source image
        output_dir: Directory to save preview images
        severity: Corruption severity level (1-5)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if preview image exists
    if not Path(image_path).exists():
        print(f"Error: {image_path} not found!")
        return
    
    # Load image
    image = np.array(Image.open(image_path).convert('RGB'))
    print(f"Loaded image: {image_path} - Shape: {image.shape}")
    
    # Initialize PyTorch corruptions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    corruptor = TorchCorruptions(device=device)
    
    # Define corruption categories
    corruption_categories = {
        'Noise': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
        'Blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
        'Weather': ['snow', 'frost', 'fog', 'brightness'],
        'Digital': ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    }
    
    print(f"\nGenerating previews with severity={severity}")
    print("=" * 60)
    
    # Generate a comprehensive preview for each category
    for category, corruptions in corruption_categories.items():
        print(f"\nProcessing {category} corruptions...")
        
        # Create figure
        cols = min(4, len(corruptions) + 1)
        rows = (len(corruptions) + 1 + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
        
        # Flatten axes
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original', fontsize=13, fontweight='bold', pad=10)
        axes[0].axis('off')
        
        # Corrupted versions
        for idx, corruption_name in enumerate(corruptions, start=1):
            try:
                corrupted = corruptor.corrupt(image, corruption_name=corruption_name, severity=severity)
                axes[idx].imshow(corrupted)
                title = corruption_name.replace('_', ' ').title()
                axes[idx].set_title(title, fontsize=12, pad=10)
                axes[idx].axis('off')
                print(f"  ✓ {corruption_name}")
            except Exception as e:
                axes[idx].text(0.5, 0.5, f'Error:\n{corruption_name}\n{str(e)[:30]}', 
                             ha='center', va='center', transform=axes[idx].transAxes,
                             fontsize=10, color='red')
                axes[idx].axis('off')
                print(f"  ✗ {corruption_name}: {e}")
        
        # Hide unused subplots
        for idx in range(len(corruptions) + 1, len(axes)):
            axes[idx].axis('off')
        
        # Add title
        fig.suptitle(f'{category} Corruptions (Severity {severity})', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.5, w_pad=1.5)
        
        # Save
        output_filename = f'{category.lower()}_corruptions.png'
        output_path = os.path.join(output_dir, output_filename)
        fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
        plt.close(fig)
        
        print(f"  Saved: {output_path}")
    
    # Generate one comprehensive preview with all corruptions
    print(f"\nGenerating comprehensive preview with all corruptions...")
    all_corruptions = []
    for corruptions in corruption_categories.values():
        all_corruptions.extend(corruptions)
    
    # Create large grid
    cols = 4
    rows = (len(all_corruptions) + 1 + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(image)
    axes[0].set_title('Original', fontsize=13, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    # All corruptions
    for idx, corruption_name in enumerate(all_corruptions, start=1):
        try:
            corrupted = corruptor.corrupt(image, corruption_name=corruption_name, severity=severity)
            axes[idx].imshow(corrupted)
            title = corruption_name.replace('_', ' ').title()
            axes[idx].set_title(title, fontsize=11, pad=8)
            axes[idx].axis('off')
        except Exception as e:
            axes[idx].axis('off')
    
    # Hide unused
    for idx in range(len(all_corruptions) + 1, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'All Corruptions (Severity {severity})', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.5, w_pad=1.5)
    
    output_path = os.path.join(output_dir, 'all_corruptions.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    
    print(f"  Saved: {output_path}")
    
    print("\n" + "=" * 60)
    print("✓ All corruption previews generated successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate corruption preview images')
    parser.add_argument('--image', default='preview.jpg', help='Source image path')
    parser.add_argument('--output-dir', default='static/previews', help='Output directory')
    parser.add_argument('--severity', type=int, default=3, choices=[1,2,3,4,5], 
                       help='Corruption severity (1-5)')
    
    args = parser.parse_args()
    
    generate_all_corruption_previews(
        image_path=args.image,
        output_dir=args.output_dir,
        severity=args.severity
    )

"""
Corruption preview generator for the web interface.
Generates a preview image showing selected corruptions applied to the showcase image.
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

import torch
from torch_corruptions import TorchCorruptions


def generate_preview(corruption_names, severity=3, output_dir='static/previews'):
    """
    Generate a preview image showing the selected corruptions.
    
    Args:
        corruption_names: List of corruption names to preview
        severity: Corruption severity level (1-5), default is 3
        output_dir: Directory to save the preview image
    
    Returns:
        Path to the generated preview image (relative to static folder)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize PyTorch corruptions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    corruptor = TorchCorruptions(device=device)
    
    # Load preview image
    preview_path = Path('preview.jpg')
    if not preview_path.exists():
        # Fallback to showcase.png or create placeholder
        preview_path = Path('showcase.png')
        if not preview_path.exists():
            create_placeholder_image(preview_path)
    
    image = np.array(Image.open(preview_path).convert('RGB'))
    
    # Calculate grid dimensions
    num_corruptions = len(corruption_names)
    if num_corruptions == 0:
        return None
    
    # Create figure with clean image + corrupted versions
    cols = min(4, num_corruptions + 1)  # Max 4 columns
    rows = (num_corruptions + 1 + cols - 1) // cols  # Ceiling division
    
    # Adjust figure size for better spacing
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
    
    # Flatten axes for easier indexing
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # First cell: clean image
    axes[0].imshow(image)
    axes[0].set_title('Original', fontsize=12, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    # Remaining cells: corrupted images
    for idx, corruption_name in enumerate(corruption_names, start=1):
        if idx < len(axes):
            try:
                corrupted = corruptor.corrupt(image, corruption_name=corruption_name, severity=severity)
                axes[idx].imshow(corrupted)
                # Format title: replace underscores and capitalize
                title = corruption_name.replace('_', ' ').title()
                axes[idx].set_title(title, fontsize=11, pad=10)
                axes[idx].axis('off')
            except Exception as e:
                # If corruption fails, show error message
                axes[idx].text(0.5, 0.5, f'Error:\n{corruption_name}', 
                             ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_corruptions + 1, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'Corruption Preview (Severity {severity})', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.5, w_pad=1.5)
    
    # Save the preview
    output_path = os.path.join(output_dir, 'corruption_preview.png')
    fig.savefig(output_path, dpi=120, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    
    # Return relative path from static folder
    return 'previews/corruption_preview.png'


def create_placeholder_image(path, size=(640, 480)):
    """Create a simple placeholder image if showcase.png doesn't exist."""
    from PIL import Image, ImageDraw
    
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple gradient pattern
    for i in range(0, size[0], 40):
        for j in range(0, size[1], 40):
            color = ((i * 255) // size[0], (j * 255) // size[1], 128)
            draw.rectangle([i, j, i+39, j+39], fill=color)
    
    # Add text
    text = "Showcase Image"
    bbox = draw.textbbox((0, 0), text)
    position = ((size[0] - (bbox[2] - bbox[0])) // 2, (size[1] - (bbox[3] - bbox[1])) // 2)
    draw.text((position[0]+2, position[1]+2), text, fill='black')  # Shadow
    draw.text(position, text, fill='white')
    
    img.save(path)
    print(f"Created placeholder image at {path}")


if __name__ == '__main__':
    # Test the preview generator
    test_corruptions = ['gaussian_noise', 'motion_blur', 'snow', 'contrast']
    preview_path = generate_preview(test_corruptions)
    print(f"Preview generated at: {preview_path}")

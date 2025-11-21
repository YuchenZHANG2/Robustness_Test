"""
Test script to verify PyTorch corruptions work correctly and benchmark performance.
"""
import time
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from torch_corruptions import TorchCorruptions, get_corruption_names

# Try importing the original for comparison
try:
    from imagecorruptions import corrupt as numpy_corrupt
    HAS_NUMPY_VERSION = True
except ImportError:
    HAS_NUMPY_VERSION = False
    print("imagecorruptions not available for comparison")


def test_corruptions():
    """Test that all corruptions work correctly."""
    print("=" * 60)
    print("Testing PyTorch Corruptions")
    print("=" * 60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create test image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize corruptor
    corruptor = TorchCorruptions(device=device)
    
    # Test all corruptions
    all_corruptions = get_corruption_names('all')
    print(f"\nTesting {len(all_corruptions)} corruptions...")
    
    failed = []
    success = []
    
    for corruption in all_corruptions:
        try:
            result = corruptor.corrupt(test_img, corruption, severity=3)
            
            # Verify output
            assert result.shape == test_img.shape, f"Shape mismatch: {result.shape} vs {test_img.shape}"
            assert result.dtype == np.uint8, f"Type mismatch: {result.dtype}"
            assert 0 <= result.min() <= result.max() <= 255, f"Value range error: {result.min()}-{result.max()}"
            
            success.append(corruption)
            print(f"✓ {corruption}")
            
        except Exception as e:
            failed.append((corruption, str(e)))
            print(f"✗ {corruption}: {e}")
    
    print(f"\n{len(success)}/{len(all_corruptions)} corruptions working")
    
    if failed:
        print("\nFailed corruptions:")
        for corr, err in failed:
            print(f"  - {corr}: {err}")
    
    return len(failed) == 0


def benchmark_speed():
    """Benchmark PyTorch corruptions performance."""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load real image
    test_img_path = Path('/home/yuchen/YuchenZ/Datasets/coco/val2017')
    img_files = list(test_img_path.glob('*.jpg'))[:50]  # Use 50 images
    
    if not img_files:
        print("No test images found, creating synthetic images")
        images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(50)]
    else:
        images = [np.array(Image.open(f).convert('RGB')) for f in img_files[:50]]
        print(f"Loaded {len(images)} images from COCO")
    
    # Test corruptions
    test_corruptions = ['gaussian_noise', 'gaussian_blur', 'snow', 'contrast', 'brightness', 'pixelate']
    
    print(f"\nBenchmarking on {len(images)} images with {len(test_corruptions)} corruptions")
    
    # Initialize PyTorch corruptor
    torch_corruptor = TorchCorruptions(device=device)
    
    # Benchmark PyTorch version (single)
    print(f"\nPyTorch ({device}) single processing:")
    torch_single_times = []
    
    for corruption in test_corruptions:
        start = time.time()
        for img in images:
            _ = torch_corruptor.corrupt(img, corruption, severity=3)
        elapsed = time.time() - start
        torch_single_times.append(elapsed)
        print(f"  {corruption}: {elapsed:.3f}s ({len(images)/elapsed:.2f} img/s)")
    
    torch_single_total = sum(torch_single_times)
    print(f"  Total: {torch_single_total:.3f}s")
    
    # Benchmark PyTorch version (batch)
    if device == 'cuda':
        print(f"\nPyTorch (GPU) batch processing:")
        torch_batch_times = []
        
        # Resize all images to same size for batching
        target_size = (480, 640)
        resized_images = []
        for img in images:
            if img.shape[0] != target_size[0] or img.shape[1] != target_size[1]:
                img_pil = Image.fromarray(img)
                img_pil = img_pil.resize((target_size[1], target_size[0]))
                img = np.array(img_pil)
            resized_images.append(img)
        
        # Stack images into batch
        batch_images = np.stack(resized_images)
        
        for corruption in test_corruptions:
            start = time.time()
            _ = torch_corruptor.corrupt(batch_images, corruption, severity=3)
            elapsed = time.time() - start
            torch_batch_times.append(elapsed)
            print(f"  {corruption}: {elapsed:.3f}s ({len(images)/elapsed:.2f} img/s)")
        
        torch_batch_total = sum(torch_batch_times)
        print(f"  Total: {torch_batch_total:.3f}s")
        
        # Speedup comparison
        print("\n" + "-" * 60)
        print("Batch vs Single processing speedup:")
        print(f"  Overall: {torch_single_total/torch_batch_total:.2f}x faster")
        print(f"  Single: {torch_single_total:.3f}s total ({len(images)*len(test_corruptions)/torch_single_total:.2f} corruptions/s)")
        print(f"  Batch: {torch_batch_total:.3f}s total ({len(images)*len(test_corruptions)/torch_batch_total:.2f} corruptions/s)")
    else:
        print("\nGPU not available - skipping batch benchmark")


def visual_comparison():
    """Generate visual comparison of corruptions."""
    print("\n" + "=" * 60)
    print("Visual Comparison")
    print("=" * 60)
    
    # Load test image
    test_img_path = Path('/home/yuchen/YuchenZ/Datasets/coco/val2017')
    img_files = list(test_img_path.glob('*.jpg'))
    
    if not img_files:
        print("No test images found")
        return
    
    img = np.array(Image.open(img_files[0]).convert('RGB'))
    print(f"Using image: {img_files[0].name} ({img.shape})")
    
    # Initialize corruptor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    corruptor = TorchCorruptions(device=device)
    
    # Test corruptions to visualize
    test_corruptions = ['gaussian_noise', 'defocus_blur', 'snow', 'contrast', 'brightness', 'pixelate']
    
    # Create figure
    rows = 2
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original', fontweight='bold')
    axes[0].axis('off')
    
    # Corrupted images
    for idx, corruption in enumerate(test_corruptions, start=1):
        if idx < len(axes):
            try:
                corrupted = corruptor.corrupt(img, corruption, severity=3)
                axes[idx].imshow(corrupted)
                title = corruption.replace('_', ' ').title()
                axes[idx].set_title(title, fontweight='bold')
                axes[idx].axis('off')
            except Exception as e:
                axes[idx].text(0.5, 0.5, f'Error:\n{corruption}', 
                             ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].axis('off')
    
    # Hide unused
    for idx in range(len(test_corruptions) + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('PyTorch Corruptions (Severity 3)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = 'torch_corruptions_test.png'
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f"Saved visual comparison to: {output_path}")
    plt.close()


if __name__ == '__main__':
    # Run tests
    success = test_corruptions()
    
    # Run benchmark if tests passed
    if success:
        benchmark_speed()
        visual_comparison()
    else:
        print("\nSkipping benchmarks due to test failures")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

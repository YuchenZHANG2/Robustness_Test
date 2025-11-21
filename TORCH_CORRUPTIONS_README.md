# PyTorch-Accelerated Image Corruptions

This directory contains GPU-accelerated implementations of image corruptions using PyTorch, designed to significantly speed up robustness testing.

## Overview

The original `imagecorruptions` library uses NumPy, SciPy, and OpenCV, which run on CPU and process images sequentially. Our PyTorch implementation leverages GPU parallelization to process batches of images simultaneously, resulting in substantial speedup.

## Key Features

- **GPU Acceleration**: All corruptions implemented in PyTorch for CUDA execution
- **Batch Processing**: Process multiple images simultaneously on GPU
- **Backward Compatible**: Drop-in replacement for `imagecorruptions.corrupt()`
- **Automatic Fallback**: Uses CPU if CUDA not available
- **All Corruption Types**: Supports all standard corruption categories

## Files

### Core Implementation

- **`torch_corruptions.py`**: Main PyTorch corruption implementation
  - `TorchCorruptions` class: GPU-accelerated corruption generator
  - `corrupt()` function: Backward-compatible API
  - `get_corruption_names()`: List available corruptions

### Integration

- **`testing_pipeline.py`**: Standard testing pipeline (updated to use PyTorch)
- **`batch_testing_pipeline.py`**: Optimized batch processing pipeline
- **`corruption_preview.py`**: Preview generator (updated to use PyTorch)

### Testing

- **`test_torch_corruptions.py`**: Comprehensive test and benchmark script

## Corruption Categories

### Noise
- `gaussian_noise`: Additive Gaussian noise
- `shot_noise`: Poisson (shot) noise
- `impulse_noise`: Salt and pepper noise
- `speckle_noise`: Multiplicative noise

### Blur
- `gaussian_blur`: Gaussian blur filter
- `defocus_blur`: Defocus/bokeh blur
- `motion_blur`: Motion blur (directional)
- `zoom_blur`: Radial zoom blur
- `glass_blur`: Glass distortion blur

### Weather
- `snow`: Snow effect with motion blur
- `frost`: Frost texture overlay
- `fog`: Fog/haze effect
- `spatter`: Water/mud spatter

### Digital
- `contrast`: Contrast reduction
- `brightness`: Brightness adjustment
- `saturate`: Saturation adjustment
- `pixelate`: Pixelation effect
- `jpeg_compression`: JPEG artifacts (approximation)
- `elastic_transform`: Elastic deformation

## Usage

### Basic Usage

```python
from torch_corruptions import corrupt

# Load image
image = np.array(Image.open('photo.jpg'))

# Apply corruption (automatically uses GPU if available)
corrupted = corrupt(image, 'gaussian_noise', severity=3)
```

### Advanced Usage with TorchCorruptions Class

```python
from torch_corruptions import TorchCorruptions

# Initialize (specify device)
corruptor = TorchCorruptions(device='cuda')

# Single image
corrupted = corruptor.corrupt(image, 'snow', severity=5)

# Batch processing (much faster on GPU)
batch_images = np.stack([img1, img2, img3, img4])
corrupted_batch = corruptor.corrupt(batch_images, 'gaussian_blur', severity=2)
```

### Batch Processing Pipeline

```python
from batch_testing_pipeline import BatchRobustnessTest

# Initialize with GPU and batch size
test = BatchRobustnessTest(
    model_loader=model_loader,
    evaluator=evaluator,
    batch_size=8,  # Process 8 images simultaneously
    device='cuda'
)

# Run tests (automatically batches corruptions)
results = test.run_full_test(
    model_keys=['frcnn_v2', 'detr'],
    corruption_names=['gaussian_noise', 'snow', 'contrast'],
    image_ids=coco_image_ids,
    severities=[1, 2, 3, 4, 5]
)
```

## Performance

Expected speedups compared to NumPy implementation (on NVIDIA GPU):

| Processing Mode | Speedup |
|----------------|---------|
| Single image (GPU) | 2-5x |
| Batch processing (GPU) | 10-30x |

*Actual speedup depends on GPU model, image size, and corruption type*

### Benchmark Results

Run the benchmark script to test on your hardware:

```bash
python test_torch_corruptions.py
```

Example output:
```
NumPy (CPU) version:
  gaussian_noise: 1.234s (8.10 img/s)
  gaussian_blur: 2.456s (4.07 img/s)
  snow: 3.789s (2.64 img/s)
  Total: 7.479s

PyTorch (GPU) batch processing:
  gaussian_noise: 0.089s (112.36 img/s)
  gaussian_blur: 0.156s (64.10 img/s)
  snow: 0.234s (42.74 img/s)
  Total: 0.479s

Speedup vs NumPy (CPU):
  Batch processing: 15.61x
```

## Implementation Notes

### Differences from Original

1. **Frost Corruption**: Requires frost texture images in `frost/` directory
   - Falls back to fog effect if textures not available
   - Textures from original library can be copied

2. **JPEG Compression**: Simplified using quantization
   - True JPEG compression requires codec, which is not available in pure PyTorch
   - Approximation provides similar visual artifacts

3. **Elastic Transform**: Currently simplified
   - Full elastic deformation requires complex grid sampling
   - Can be enhanced for better accuracy if needed

4. **Glass Blur**: Simplified pixel shuffling
   - Full local shuffling is hard to parallelize on GPU
   - Uses Gaussian blur as approximation

### Memory Considerations

- Batch processing uses more GPU memory
- Adjust `batch_size` parameter based on:
  - GPU memory available
  - Image resolution
  - Corruption type complexity

Typical memory usage:
- 640x480 images, batch_size=8: ~500 MB GPU memory
- 1920x1080 images, batch_size=8: ~2 GB GPU memory

## Integration with Flask App

The Flask application automatically detects GPU availability and switches between:

1. **GPU available**: Uses `BatchRobustnessTest` with batch_size=8
2. **CPU only**: Uses standard `RobustnessTest` with PyTorch on CPU

See `app.py`:
```python
if torch.cuda.is_available():
    test = BatchRobustnessTest(model_loader, evaluator, batch_size=8, device='cuda')
else:
    test = RobustnessTest(model_loader, evaluator)
```

## Testing

Run the test suite:

```bash
# Test all corruptions
python test_torch_corruptions.py

# This will:
# 1. Test all corruption types for correctness
# 2. Benchmark speed vs NumPy implementation
# 3. Generate visual comparison image
```

## Requirements

```
torch >= 2.0.0
numpy >= 1.20.0
pillow >= 9.0.0
```

GPU acceleration requires:
```
CUDA-capable GPU
CUDA Toolkit >= 11.0
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```python
test = BatchRobustnessTest(..., batch_size=4)  # Try smaller batch
```

### Corruptions Look Different

Some corruptions (frost, elastic, glass_blur) are simplified. For exact reproduction of original library, you may need to:
1. Copy frost textures to `frost/` directory
2. Enhance specific corruption implementations

### CPU Slower Than NumPy

PyTorch on CPU may have overhead. For CPU-only systems:
- Use original `imagecorruptions` library, or
- Use larger batch sizes to amortize overhead

## Future Enhancements

Potential improvements:
- [ ] Full elastic deformation implementation
- [ ] Exact JPEG compression using external codec
- [ ] Multi-GPU support for very large batches
- [ ] Mixed precision (FP16) for even faster processing
- [ ] Cached frost textures as tensors
- [ ] Progressive batch sizing based on GPU memory

## License

Compatible with original `imagecorruptions` library license.

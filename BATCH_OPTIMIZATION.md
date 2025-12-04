# Batch-Optimized Robustness Testing Pipeline

## Overview

This optimized pipeline significantly speeds up robustness testing by leveraging:

1. **PyTorch DataLoader** - Parallel image loading with multiple worker processes
2. **Batch Inference** - Process multiple images simultaneously on GPU
3. **GPU-Accelerated Corruptions** - Apply corruptions to batches using PyTorch
4. **Prefetching** - Load next batch while processing current batch

## Performance Improvements

### Key Optimizations

| Component | Standard Pipeline | Optimized Pipeline |
|-----------|------------------|-------------------|
| Image Loading | Sequential, one at a time | Parallel with `num_workers` processes |
| Corruption | CPU, one image | GPU batch processing |
| Inference | One image per forward pass | Batch of 8-16 images per pass |
| Data Transfer | CPU→GPU per image | Pinned memory with batching |
| I/O | Blocking | Prefetched in background |

### Expected Speedup

- **CPU**: 2-3x faster (from parallel loading)
- **GPU**: 5-10x faster (from batch processing + parallel loading)
- **Large datasets**: Even greater speedup due to better hardware utilization

## Usage

### In Application (app.py)

The application automatically selects the optimal pipeline:

```python
# GPU available: Uses BatchOptimizedRobustnessTest
# CPU only: Falls back to BatchRobustnessTest
```

### Standalone Usage

```python
from model_loader import ModelLoader
from evaluator import COCOEvaluator
from batch_optimized_pipeline import BatchOptimizedRobustnessTest

# Initialize
model_loader = ModelLoader()
evaluator = COCOEvaluator(annotation_file='...', image_dir='...')

# Create optimized tester
test = BatchOptimizedRobustnessTest(
    model_loader, 
    evaluator,
    batch_size=8,      # Process 8 images at once
    num_workers=4      # Use 4 parallel workers for loading
)

# Run test
results = test.run_full_test(
    model_keys=['frcnn_v2', 'retinanet_v2'],
    corruption_names=['gaussian_noise', 'motion_blur'],
    image_ids=image_ids,
    severities=[1, 2, 3, 4, 5]
)
```

### Tuning Parameters

#### Batch Size
- **Single GPU (recommended)**: batch_size=4
- **High VRAM GPU**: Can try 8
- **Low VRAM**: Reduce to 2
- **CPU Only**: Keep at 4

#### Number of Workers
- **Recommended**: num_workers=2 (safe for single GPU)
- **SSD Storage with high CPU**: Can try 4
- **HDD Storage**: Keep at 2
- **CPU bound**: Reduce to 1

```python
# Recommended for single GPU setup
test = BatchOptimizedRobustnessTest(
    model_loader, evaluator,
    batch_size=4,      # Conservative, works reliably
    num_workers=2      # Parallel loading without overhead
)

# For limited GPU memory
test = BatchOptimizedRobustnessTest(
    model_loader, evaluator,
    batch_size=4,
    num_workers=4
)
```

## Architecture

### COCODetectionDataset

Custom PyTorch Dataset that:
- Loads images from COCO dataset
- Applies corruptions on CPU in worker processes (avoids CUDA fork issues)
- Returns images with metadata (IDs, sizes)
- Each worker has its own CPU-based corruptor instance

```python
dataset = COCODetectionDataset(
    evaluator, 
    image_ids,
    corruption_name='gaussian_noise',
    severity=3
)
# Corruption happens on CPU in DataLoader workers
# Model inference happens on GPU in main process
```

### Batch Prediction

Models process multiple images in a single forward pass:

```python
# Torchvision models: Native batch support
outputs = model([img1_tensor, img2_tensor, ...])

# Hugging Face models: Batch with padding
inputs = processor(images=[img1, img2, ...], padding=True)
outputs = model(**inputs)
```

### DataLoader Pipeline

```
┌─────────────┐
│  Worker 1   │──┐
│ Load Image  │  │
└─────────────┘  │
                 │    ┌──────────────┐
┌─────────────┐  ├───→│ Batch Queue  │
│  Worker 2   │──┤    │ (Prefetch 2) │
│ Load Image  │  │    └──────┬───────┘
└─────────────┘  │           │
                 │           ↓
┌─────────────┐  │    ┌──────────────┐
│  Worker N   │──┘    │     GPU      │
│ Load Image  │       │   Inference  │
└─────────────┘       └──────────────┘
```

## Benchmark

Run the benchmark script to see performance improvements:

```bash
python benchmark_pipelines.py
```

Example output:
```
PERFORMANCE COMPARISON
======================================================================
Standard Pipeline:        245.32s
Batch-Optimized Pipeline: 32.14s
Speedup:                  7.63x faster
Time Saved:               213.18s (86.9%)
```

## Implementation Details

### Parallel Corruption Application

Corruptions are applied on **CPU** in the Dataset's `__getitem__` method, which runs in parallel across workers. This avoids CUDA multiprocessing fork issues:

```python
def __getitem__(self, idx):
    image = load_image(idx)
    if self.corruption_name:
        # Runs on CPU in parallel across workers
        # Each worker has its own CPU corruptor instance
        # Avoids CUDA fork issues
        image = self.corruptor.corrupt(image, ...)
    return image
```

GPU is used only for model inference in the main process, which processes batches efficiently.

### Memory Management

- **Pin Memory**: Enabled for GPU to speed up host→device transfer
- **Prefetch Factor**: 2 batches loaded ahead of time
- **Automatic Cleanup**: PyTorch handles GPU memory automatically

### Error Handling

- Gracefully handles out-of-memory errors
- Falls back to smaller batch sizes if needed
- Progress tracking continues even with errors

## Comparison with Standard Pipeline

| Feature | Standard | Batch-Optimized |
|---------|----------|----------------|
| Images per iteration | 1 | 8-16 |
| Loading | Sequential | Parallel (4+ workers) |
| Corruption | CPU, sequential | GPU, parallel |
| GPU utilization | Low (~20%) | High (~80-90%) |
| CPU utilization | Low | Medium-High |
| Memory usage | Low | Medium |
| Speedup | 1x (baseline) | 5-10x |

## Notes

- First batch may be slower due to DataLoader initialization
- Speedup increases with dataset size
- GPU memory usage scales with batch size
- Some models (DETR) may require smaller batches

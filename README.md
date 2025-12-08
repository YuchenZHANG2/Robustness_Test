---
title: Object Detector Robustness Testing
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Object Detector Robustness Testing & Image Matcher

A Flask-based web application suite with two main applications:
1. **Detector Robustness Tester** - Test object detectors against image corruptions
2. **Image Matcher** - Match and align image pairs from folders

## Features

### Detector Robustness Tester (app.py)

- **3-Step Workflow**: Intuitive multi-page interface guiding users through the testing process
- **Detector Selection**: Choose from pre-trained models (Faster R-CNN, RT-DETR, DETR, RetinaNet, FCOS, etc.)
- **Interactive Corruption Preview**: Visualize corruptions with adjustable severity sliders (0-5)
- **Corruption Testing**: Apply various corruption types organized in 4 categories:
  - **Noise**: Gaussian, Shot, Impulse
  - **Blur**: Defocus, Glass, Motion, Zoom
  - **Weather**: Snow, Frost, Fog, Brightness, Dust
  - **Digital**: Contrast, Elastic Transform, Pixelate, JPEG Compression
- **Batch Processing**: GPU-accelerated batch inference with parallel data loading (4x faster)
- **Comprehensive Metrics**: COCO evaluation metrics (mAP, mAP@50, mAP@75, etc.)

### Image Matcher (image_matcher_app.py)

- **Folder-Level Matching**: Match images with same names across two folders
- **Feature Detection**: SIFT-based image alignment and matching
- **Overlay Visualization**: Toggle between side-by-side and overlay views
- **Adjustable Opacity**: Control transparency for better comparison
- **Batch Saving**: Save aligned cropped regions for all matched pairs
- **Browser-Based**: Access remotely via web browser

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv DetectorTest
source DetectorTest/bin/activate  # On Windows: DetectorTest\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Detector Robustness Tester

1. Start the application:
```bash
python app.py
```

2. Open browser and navigate to: `http://localhost:5001`

3. Follow the 3-step process:
   - **Step 1/3**: Select detectors to test
   - **Step 2/3**: Choose corruption types and preview them
   - **Step 3/3**: Configure and start testing

### Image Matcher

1. Start the application:
```bash
python image_matcher_app.py
```

2. Open browser and navigate to: `http://localhost:5001`

3. Enter two folder paths containing images with matching names

4. Navigate through pairs, toggle overlay, and save cropped regions

## Project Structure

```
Detector_test/
├── app.py                          # Main detector testing Flask app
├── image_matcher_app.py            # Image matcher Flask app
├── batch_optimized_pipeline.py     # Optimized batch testing pipeline
├── model_loader.py                 # Object detection model loader
├── evaluator.py                    # COCO evaluation metrics
├── visualization.py                # Detection visualization
├── torch_corruptions.py            # GPU-accelerated image corruptions
├── corruption_preview.py           # Corruption preview generator
├── requirements.txt                # Python dependencies
├── templates/                      # HTML templates
│   ├── base.html                  # Base template
│   ├── step1.html                 # Detector selection
│   ├── step2.html                 # Corruption selection
│   ├── step3.html                 # Test configuration
│   ├── run_testing.html           # Test execution
│   ├── show_results.html          # Results display
│   ├── validate_model.html        # Model validation
│   └── image_matcher.html         # Image matcher UI
├── static/                         # Static assets
│   ├── style.css
│   ├── test_results.json          # Saved test results
│   └── previews/                  # Corruption previews
└── matched_crops/                  # Matched image outputs
    ├── cropped_img1/
    └── cropped_img2/
```

## Performance

### Batch Optimization
- **GPU Batch Processing**: 4 images processed simultaneously
- **Parallel Loading**: 2 worker processes for concurrent image loading
- **CPU Corruptions**: Applied in parallel workers (avoids CUDA fork issues)
- **GPU Inference**: Model runs on GPU for maximum speed
- **Speedup**: 3-4x faster than sequential processing

### Configuration
```python
# Single GPU setup (default in app.py)
batch_size=4      # Process 4 images at once
num_workers=2     # 2 parallel workers for loading
```

## Configuration

- **Port**: Applications run on port 5001 by default (configurable)
- **Upload Directory**: Custom files stored in `uploads/` folder
- **Output Directory**: Matched crops saved to `matched_crops/` folder
- **Max File Size**: 16MB limit for uploaded files

## Requirements

- Python 3.8+
- PyTorch (with CUDA support recommended)
- OpenCV
- Flask
- torchvision
- Pillow
- numpy
- pycocotools

## Future Enhancements

- Real-time progress tracking improvements
- More corruption types
- Additional object detection models
- Export results in various formats
- Batch image upload
- Configuration presets

## License

MIT

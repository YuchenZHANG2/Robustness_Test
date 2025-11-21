# Testing Backend Setup Guide

## Overview

The testing backend supports:
- Loading models from torchvision and Hugging Face
- Model validation with sample predictions before testing
- Robustness evaluation across corruptions and severity levels (1-5)
- COCO mAP evaluation using pycocotools
- Progress tracking during model loading and testing

## Architecture

### Core Modules

1. **model_loader.py** - Model loading and inference
   - Supports torchvision detection models (Faster R-CNN, RetinaNet, FCOS, SSD)
   - Supports Hugging Face models (DETR, RT-DETR, Deformable DETR, etc.)
   - Unified prediction interface
   - GPU/CPU automatic device selection

2. **evaluator.py** - COCO evaluation
   - Loads COCO annotations
   - Converts predictions to COCO format
   - Computes mAP metrics using pycocotools

3. **visualization.py** - Prediction visualization
   - Draws bounding boxes with class labels
   - Converts matplotlib figures to base64 for web display

4. **testing_pipeline.py** - Robustness testing
   - Tests models on clean and corrupted images
   - Evaluates across all severity levels
   - Generates comprehensive results

## Workflow

### 1. Model Selection (Step 1/3)
User selects which models to test from predefined list or uploads custom model.

### 2. Corruption Selection (Step 2/3)
User selects corruption types and previews them on showcase image.

### 3. Report Options (Step 3/3)
User chooses report format (visual only or PDF with analysis).

### 4. Model Validation
**Before testing begins, each model is validated:**
- Model is loaded with progress tracking
- Prediction is made on one sample image
- User reviews the visualization to verify:
  - Classes are correctly mapped
  - Bounding boxes look reasonable
- User approves or rejects each model

### 5. Testing Execution
Once all models are approved:
- 50 random images are selected from COCO val2017
- Each model is tested on:
  - Clean images (baseline mAP)
  - Each corruption at severities 1-5
- Progress is tracked and displayed
- Results are saved as JSON

### 6. Results Display
- Summary table with mAP comparison
- Detailed results by corruption and severity
- Degradation and robustness scores
- Downloadable JSON results

## Installation

Install dependencies (in your conda environment):

```bash
pip install flask torch torchvision transformers pycocotools tqdm
pip install imagecorruptions scikit-image matplotlib pillow opencv-python
```

## Configuration

Update paths in `app.py` if needed:

```python
evaluator = COCOEvaluator(
    annotation_file='/path/to/instances_val2017.json',
    image_dir='/path/to/val2017'
)
```

## Model Configurations

Available models are defined in `model_loader.py` under `MODEL_CONFIGS`:

**Torchvision models:**
- frcnn_v1, frcnn_v2, frcnn_mobilenet_large, frcnn_mobilenet_320
- retinanet_v1, retinanet_v2
- fcos_v1
- ssd300, ssdlite_320

**Hugging Face models:**
- detr (DETR ResNet50)
- deformable_detr
- conditional_detr
- dab_detr
- rt_detr

## Label Mapping

Both torchvision and Hugging Face models use COCO category IDs (1-91 with gaps).
The mapping is handled automatically in the evaluator.

## Testing Parameters

- **Number of images**: 50 (random sample from val2017)
- **Severity levels**: 1, 2, 3, 4, 5
- **Score threshold**: 0.05 (for evaluation), 0.3 (for visualization)
- **Label offset**: 0 (both model types use COCO IDs directly)

## Results Format

Results are saved as nested JSON:

```json
{
  "model_key": {
    "name": "Model Name",
    "clean": {
      "mAP": 0.xxx,
      "mAP_50": 0.xxx,
      "mAP_75": 0.xxx
    },
    "corrupted": {
      "corruption_name": {
        "1": {"mAP": 0.xxx, ...},
        "2": {"mAP": 0.xxx, ...},
        ...
      }
    }
  }
}
```

## Performance Notes

- First model load downloads weights (may take time)
- GPU highly recommended for testing
- 50 images × 5 severities × N corruptions × M models = many evaluations
- Expect 1-5 minutes per model depending on hardware

## Troubleshooting

**Model loading fails:**
- Check internet connection (for downloading weights)
- Ensure sufficient disk space
- Verify GPU drivers if using CUDA

**Label mapping issues:**
- Both torchvision and HF models should use COCO IDs
- Validation step helps catch mapping errors
- Check visualizations carefully

**Memory issues:**
- Reduce number of images
- Use smaller models
- Enable CPU mode if GPU OOM

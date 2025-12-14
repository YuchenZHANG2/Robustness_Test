# Object Detector Robustness Testing

Test object detection models against image corruptions and compare performance across different datasets.

## Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Datasets

- **COCO**: Download from [COCO website](https://cocodataset.org/#download) the Coco Val 2017 images and annotations and update paths in `DATASET_CONFIG` in `app.py` (lines 36-40)
- **DustyConstruction**: Download from [Hugging Face](https://huggingface.co/datasets/YuchenGua/Dusty_Construction) and update paths in `DATASET_CONFIG` in `app.py` (lines 42-46)

### 3. Run the Application
```bash
python app.py
```
Then navigate to `http://localhost:5001`

## What This Does

- Tests object detectors (Faster R-CNN, RT-DETR, YOLO, etc.) against 15+ corruption types
- Evaluates robustness with COCO metrics (mAP, mAP@50, mAP@75)
- GPU-accelerated batch processing
- Interactive web interface for configuration and results

## Additional Tools

**Image Matcher** (optional): Match and align image pairs
```bash
python image_matcher_app.py
```
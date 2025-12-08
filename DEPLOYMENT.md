# Hugging Face Spaces Deployment Guide

## Prerequisites
- Hugging Face account (create at https://huggingface.co)
- Git installed locally
- Your datasets ready

## Step-by-Step Deployment

### 1. Create a New Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in:
   - **Space name**: `detector-robustness-test` (or your preferred name)
   - **License**: Choose appropriate license
   - **SDK**: Select **Docker**
   - **Hardware**: Select **CPU basic** (free) or **GPU T4** (upgrade later)
4. Click "Create Space"

### 2. Clone Your Space Repository

```bash
# Clone the empty space
git clone https://huggingface.co/spaces/YOUR_USERNAME/detector-robustness-test
cd detector-robustness-test
```

### 3. Copy Your Application Files

Copy these files from your project to the space directory:

```bash
# Essential files
cp /path/to/Detector_test/app.py .
cp /path/to/Detector_test/model_loader.py .
cp /path/to/Detector_test/evaluator.py .
cp /path/to/Detector_test/visualization.py .
cp /path/to/Detector_test/batch_optimized_pipeline.py .
cp /path/to/Detector_test/torch_corruptions.py .
cp /path/to/Detector_test/requirements.txt .
cp /path/to/Detector_test/Dockerfile .
cp /path/to/Detector_test/.dockerignore .

# Copy directories
cp -r /path/to/Detector_test/templates .
cp -r /path/to/Detector_test/static .

# Create empty directories (will be populated at runtime)
mkdir -p uploads/temp/folder1 uploads/temp/folder2
mkdir -p static/validation static/previews
mkdir -p matched_crops
```

### 4. Handle Large Dataset Files

**Option A: Use Hugging Face Datasets Hub**
```bash
# Upload your dataset to a separate dataset repository
# Then download in your app using datasets library
```

**Option B: Git LFS for Medium-Sized Files**
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pt"
git lfs track "DustyConstruction.v2i.coco/**"

# Copy your dataset
cp -r /path/to/Detector_test/DustyConstruction.v2i.coco .
```

**Option C: External Storage (Recommended for COCO)**
- Upload COCO dataset to Hugging Face Datasets
- Or use cloud storage (AWS S3, Google Cloud Storage)
- Download on first run

### 5. Update Dataset Paths in app.py

Edit `app.py` to use relative paths or download datasets on startup:

```python
# Option 1: Relative paths
DATASET_CONFIG = {
    'COCO': {
        'annotation_file': './coco/annotations/instances_val2017.json',
        'image_dir': './coco/val2017',
        # ...
    },
    'Construction': {
        'annotation_file': './DustyConstruction.v2i.coco/_annotations.coco.json',
        'image_dir': './DustyConstruction.v2i.coco/train',
        # ...
    }
}
```

### 6. Push to Hugging Face

```bash
# Add all files
git add .

# Commit
git commit -m "Initial deployment of detector robustness testing app"

# Push to Hugging Face
git push
```

### 7. Enable GPU (Optional but Recommended)

1. Go to your Space settings on Hugging Face
2. Click "Settings" tab
3. Under "Hardware", select:
   - **T4 small** (Free tier, limited hours)
   - **T4 medium** (~$0.60/hour)
   - **A10G small** (~$1.30/hour) - Recommended for production
4. Click "Save"

### 8. Monitor Deployment

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/detector-robustness-test`
2. Watch the build logs
3. Wait for "Running" status
4. Your app will be live!

## Post-Deployment

### Access Your App
```
https://huggingface.co/spaces/YOUR_USERNAME/detector-robustness-test
```

### Update Your App
```bash
# Make changes locally
git add .
git commit -m "Description of changes"
git push
```

### Share Your App
- Share the Space URL with anyone
- Optionally make it private in Settings (requires paid account)

## Troubleshooting

### Build Fails
- Check logs in the Space page
- Verify Dockerfile syntax
- Ensure all dependencies in requirements.txt

### Out of Memory
- Reduce batch size in `batch_optimized_pipeline.py`
- Upgrade to larger GPU instance
- Optimize model loading

### Slow Performance
- Enable GPU in Space settings
- Check if CUDA is being used properly
- Monitor resource usage in logs

### Dataset Not Found
- Verify dataset paths are correct
- Check if datasets are included in repo or downloaded at startup
- Review .dockerignore to ensure datasets aren't excluded

## Tips for Free Tier

1. **GPU Hours**: T4 free tier has limited monthly hours
2. **Sleep Mode**: Space sleeps after inactivity (restarts on next visit)
3. **Optimize**: Reduce model count, use smaller datasets for demos
4. **Cache**: Cache model downloads to avoid re-downloading

## Support

- Hugging Face Docs: https://huggingface.co/docs/hub/spaces
- Community Forum: https://discuss.huggingface.co/

# Object Detector Robustness Testing

Test object detection models against image corruptions and compare performance across different datasets.

## Quick Setup

### 1. Create Virtual Environment and Install Dependencies
```bash
python -m venv DetectorTest
source DetectorTest/bin/activate  # On Linux/Mac
# DetectorTest\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 2. Download Datasets

Run the automated download script to get all datasets:
```bash
python download_datasets.py
```


### 3. Run the Application
```bash
python app.py
```
Then navigate to `http://localhost:7860`

## Additional Tools

**Image Matcher** (optional): Match and align image pairs
```bash
python image_matcher_app.py
```
# Object Detector Robustness Tester

A Flask-based web application for testing the robustness of pre-trained object detectors against various image corruptions.

## Features

- **3-Step Workflow**: Intuitive multi-page interface guiding users through the testing process
- **Detector Selection**: Choose from pre-trained models (Faster R-CNN, RT-DETR, DETR, etc.) or upload custom models
- **Corruption Testing**: Apply various corruption types organized in 4 categories:
  - **Noise**: Gaussian, Shot, Impulse
  - **Blur**: Defocus, Glass, Motion, Zoom
  - **Weather**: Snow, Frost, Fog, Brightness
  - **Digital**: Contrast, Elastic Transform, Pixelate, JPEG Compression
- **Live Preview**: Visualize selected corruptions before running full tests
- **Flexible Reporting**: Choose between visual-only reports or comprehensive PDF reports with AI analysis

## Installation

1. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv DetectorTest
source DetectorTest/bin/activate  # On Windows: DetectorTest\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5001
```

3. Follow the 3-step process:
   - **Step 1/3**: Select detectors to test
   - **Step 2/3**: Choose corruption types and preview them
   - **Step 3/3**: Select report format and start testing

## Project Structure

```
Detector_test/
├── app.py                    # Main Flask application
├── corruption_preview.py     # Corruption preview generator
├── test_corruption.py        # Original corruption testing script
├── showcase.png             # Sample image for previews
├── requirements.txt         # Python dependencies
├── templates/               # HTML templates
│   ├── base.html
│   ├── step1.html
│   ├── step2.html
│   ├── step3.html
│   └── results.html
└── static/                  # Static assets
    ├── style.css
    └── previews/           # Generated preview images
```

## Configuration

- **Port**: The application runs on port 5001 by default (configurable in `app.py`)
- **Upload Directory**: Custom models are stored in the `uploads/` folder
- **Max File Size**: 16MB limit for uploaded models

## Future Enhancements

- Integration with actual object detection models
- Real-time testing progress tracking
- Detailed metrics and performance analysis
- PDF report generation with vision-language model interpretations
- Batch processing for multiple images
- Export results in various formats

## License

MIT

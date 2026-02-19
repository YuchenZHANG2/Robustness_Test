# PDF Report Module

This module generates professional PDF reports for robustness evaluation results of object detection models.

## Structure

```
pdf_report/
├── __init__.py                     # Main RobustnessReportGenerator class
├── constants.py                    # Shared constants (colors, thresholds, etc.)
├── styles.py                       # All paragraph styles and color definitions
├── components.py                   # Page backgrounds components
├── utils.py                        # Helper functions (fonts, calculations, formatting)
├── table_utils.py                  # Table styling utilities
├── visualization_utils.py          # Plot and chart generation utilities
├── title_page.py                   # Title page generation
├── table_of_contents_page.py       # Table of contents generation
├── map_comparison_page.py          # mAP comparison table and radar chart
├── corruption_detail_page.py       # Individual corruption analysis pages
├── qualitative_examples.py         # Qualitative visualization grids
├── test_generator.py               # Test script
└── README.md                       # This file
```

## Module Organization

### Core Modules
- **`__init__.py`**: Main `RobustnessReportGenerator` class orchestrating report creation
- **`constants.py`**: Centralized constants (MODEL_COLORS, severity levels, thresholds)
- **`styles.py`**: All paragraph and text styles used throughout the report
- **`components.py`**: Page background functions (blue for title, white for content)

### Utility Modules
- **`utils.py`**: Core utilities
  - Font registration
  - Metrics calculation
  - Corruption name formatting
  - Qualitative image selection/management
  
- **`table_utils.py`**: Table styling utilities
  - Three-line table style (standard academic format)
  - Severity table style (dual-row per model)
  - Notes section formatting
  
- **`visualization_utils.py`**: Visualization generation
  - Severity line plots
  - Spider/radar charts
  - Grid image composition
  - PIL to ReportLab conversion
  - Column headers flowable

### Page Modules
- **`title_page.py`**: Title page with project information
- **`table_of_contents_page.py`**: Clickable table of contents
- **`map_comparison_page.py`**: Overall mAP comparison with radar chart
- **`corruption_detail_page.py`**: Per-corruption analysis with plots and tables
- **`qualitative_examples.py`**: Qualitative visualization grids

## Usage

### Basic Usage

```python
from pdf_report import RobustnessReportGenerator

# Initialize generator
generator = RobustnessReportGenerator(output_dir='static')

# Generate basic report (quantitative only)
pdf_path = generator.generate_report(
    detectors=['YOLO11', 'Faster R-CNN V2'],
    corruptions=['gaussian_noise', 'motion_blur'],
    results=test_results_dict,
    dataset_name='COCO Val2017'
)
```

### With Qualitative Examples

To include qualitative visualizations showing detector performance on sample images:

```python
from pdf_report import RobustnessReportGenerator
from model_loader import ModelLoader
from evaluator import COCOEvaluator, format_coco_label_mapping
from torch_corruptions import TorchCorruptions

# Load required objects
model_loader = ModelLoader()
model_loader.load_model('yolov11')
model_loader.load_model('detr')

evaluator = COCOEvaluator(annotation_file, image_dir)
corruptor = TorchCorruptions(device='cuda')
category_names = format_coco_label_mapping()

# Generate report with qualitative examples
generator = RobustnessReportGenerator(output_dir='static')
pdf_path = generator.generate_report(
    detectors=['YOLO11', 'DETR'],
    corruptions=['gaussian_noise', 'motion_blur'],
    results=test_results_dict,
    dataset_name='COCO Val2017',
    # Additional parameters for qualitative examples
    model_loader=model_loader,
    evaluator=evaluator,
    corruptor=corruptor,
    category_names=category_names,
    include_qualitative=True,
    num_qualitative_images=3  # Number of random images to visualize
)
```

**Note**: Generating qualitative examples requires running inference on-the-fly during PDF generation, which may take several minutes depending on the number of detectors and corruptions.

### Qualitative Examples Format

For each corruption type, the report shows:
- **N grids** (one per randomly selected image)
- Each grid layout: **detectors (rows) × severity levels (columns)**
- **Color-coded bounding boxes** by detector (no labels for clarity)
- **Legend** mapping colors to detector names

## Report Structure

1. **Title Page** (blue background)
   - Project title
   - List of tested models
   - Corruption types
   - Dataset information

2. **Table of Contents** (white background)
   - Clickable navigation links
   - Section numbering

3. **mAP Comparison Page**
   - Results table with models ranked by clean mAP
   - Metrics: Clean mAP, Avg Corrupted mAP, Degradation, Robustness Score
   - Spider/radar chart for visual comparison
   - Calculation notes

4. **Individual Corruption Pages** (one per corruption)
   - Section title and number
   - Line plot: mAP vs. severity levels (0-5)
   - Dual-row table: absolute mAP values + relative degradation %
   - Qualitative examples (if enabled):
     - Multiple image grids showing detector performance
     - Color-coded detection boxes
     - Notes explaining image selection and thresholds

## Design Principles

- **Modularity**: Each page and utility function is independent and reusable
- **Separation of Concerns**: Constants, styles, utilities, and page logic are separate
- **DRY Principle**: Common patterns (table styling, plotting) are centralized
- **Clarity**: Functions are focused and well-documented
- **Maintainability**: Easy to add, remove, or modify pages and features

## Color Scheme

All colors are defined in `styles.py` and `constants.py`:
- **Primary Blue**: `#114584` (headers, titles, primary elements)
- **White**: `#ffffff`
- **Gray shades**: For text and backgrounds
- **MODEL_COLORS**: Consistent 10-color palette for charts and visualizations

## Testing

Run the test script to verify PDF generation:

```bash
cd /path/to/Detector_test
python pdf_report/test_generator.py
```

This will:
1. Load test results from `static/test_results.json`
2. Initialize YOLO11 and DETR models
3. Set up evaluator and corruptor
4. Generate a complete PDF with qualitative examples
5. Print the report structure

## Dependencies

- **ReportLab**: PDF generation library
- **Matplotlib**: For plots and charts
- **PIL/Pillow**: Image processing
- **DejaVu Serif fonts**: With automatic fallback to Helvetica
- **PyTorch**: For corruption generation (qualitative examples only)
- **COCO API**: For dataset evaluation (qualitative examples only)

## Adding New Features

### Adding a New Page Type

1. Create a new module (e.g., `my_new_page.py`)
2. Define a function that returns a list of flowable elements:
   ```python
   def create_my_new_page(data, styles):
       """Create my new page."""
       elements = []
       # Add your content...
       return elements
   ```
3. Import in `__init__.py`:
   ```python
   from .my_new_page import create_my_new_page
   ```
4. Add to `generate_report()` method in `RobustnessReportGenerator`

### Adding New Visualizations

Add visualization functions to `visualization_utils.py`:
```python
def create_my_plot(data):
    """Create a custom plot."""
    # Matplotlib plotting code
    # ...
    return RLImage(img_buffer, width=6*inch, height=4*inch)
```

### Adding New Table Styles

Add style functions to `table_utils.py`:
```python
def create_my_table_style(num_rows):
    """Create a custom table style."""
    return TableStyle([...])
```

## Troubleshooting

### Fonts Not Found
If DejaVu Serif fonts are not available, the system automatically falls back to Helvetica. To use custom fonts, install them system-wide or modify `utils.register_fonts()`.

### Qualitative Examples Failing
Ensure all required parameters are provided:
- `model_loader`: with models already loaded
- `evaluator`: properly initialized with dataset
- `corruptor`: TorchCorruptions instance
- `category_names`: dictionary mapping IDs to names

### Memory Issues with Large PDFs
If generating PDFs with many qualitative examples causes memory issues:
- Reduce `num_qualitative_images` parameter
- Process fewer corruptions at once
- Use smaller images or reduce visualization resolution in `visualization_utils.py`

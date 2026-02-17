# PDF Report Module

This module generates professional PDF reports for robustness evaluation results.

## Structure

```
pdf_report/
├── __init__.py                  # Main generator class and document setup
├── styles.py                    # All paragraph styles and color definitions
├── components.py                # Reusable components (backgrounds, boxes)
├── utils.py                     # Helper functions (fonts, calculations)
├── title_page.py                # Title page generation
├── map_comparison_page.py       # mAP comparison table page
├── test_generator.py            # Test script
└── README.md                    # This file
```

## Usage

```python
from pdf_report import RobustnessReportGenerator

# Initialize generator
generator = RobustnessReportGenerator(output_dir='static')

# Generate report
pdf_path = generator.generate_report(
    detectors=['YOLO11', 'Faster R-CNN V2'],
    corruptions=['gaussian_noise', 'motion_blur'],
    results=test_results_dict,
    dataset_name='COCO Val2017'
)
```

## Adding New Pages

To add a new page to the report:

1. **Create a new page module** (e.g., `corruption_breakdown_page.py`):
   ```python
   def create_corruption_breakdown_page(results, styles):
       """
       Create a page showing detailed corruption analysis
       
       Args:
           results: Test results dictionary
           styles: StyleSheet object
           
       Returns:
           list: Flowable elements for the page
       """
       elements = []
       # Add your page content...
       return elements
   ```

2. **Import in `__init__.py`**:
   ```python
   from .corruption_breakdown_page import create_corruption_breakdown_page
   ```

3. **Add to the report generation** in `__init__.py` `generate_report()` method:
   ```python
   # 3. Corruption breakdown page
   if results:
       story.extend(create_corruption_breakdown_page(
           results=results,
           styles=self.styles
       ))
   ```

## Design Principles

- **Modularity**: Each page is independent and can be tested separately
- **Reusability**: Common components and styles are centralized
- **Clarity**: Functions have clear purposes and documentation
- **Maintainability**: Easy to add, remove, or modify pages

## Color Scheme

All colors are defined in `styles.py`:
- Primary Blue: `#114584` (headers, titles)
- White: `#ffffff`
- Gray shades for text and backgrounds

## Testing

Run the test script to verify PDF generation:
```bash
python pdf_report/test_generator.py
```

## Dependencies

- ReportLab: PDF generation library
- DejaVu Serif fonts (with automatic fallback to Helvetica)

#!/usr/bin/env python3
"""
Test script for PDF report generator

This script tests the PDF generation functionality with actual test results,
loading models and generating both quantitative tables and qualitative visualizations.
"""
import json
import sys
import torch
from pathlib import Path

# Add parent directory to path to import pdf_report
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_report import RobustnessReportGenerator
from model_loader import ModelLoader
from evaluator import COCOEvaluator, format_coco_label_mapping
from torch_corruptions import TorchCorruptions


# Configuration
RESULTS_FILE = 'static/test_results.json'
ANNOTATIONS_FILE = 'DustyConstruction.v2i.coco/_annotations.coco.json'
IMAGE_DIR = 'DustyConstruction.v2i.coco/train'
FILTER_CLASSES = [3]  # Only person class
TEST_CORRUPTIONS = ['gaussian_noise', 'pixelate']
NUM_QUALITATIVE_IMAGES = 3


def load_test_results(results_path):
    """
    Load test results from JSON file.
    
    Args:
        results_path: Path to test results JSON file
        
    Returns:
        dict: Test results, or None if file not found
    """
    if not Path(results_path).exists():
        print(f"❌ Test results file not found: {results_path}")
        print("   Please run the robustness tests first to generate test_results.json")
        return None
    
    with open(results_path, 'r') as f:
        return json.load(f)


def initialize_models(model_loader):
    """
    Load all required models for testing.
    
    Args:
        model_loader: ModelLoader instance
    """
    print("  Loading YOLO11...")
    model_loader.load_model('yolov11')
    print("  Loading DETR ResNet50...")
    model_loader.load_model('detr')


def initialize_evaluator_and_corruptor():
    """
    Initialize the evaluator and corruptor instances.
    
    Returns:
        tuple: (evaluator, corruptor, category_names)
    """
    print("  Initializing evaluator...")
    evaluator = COCOEvaluator(
        annotation_file=ANNOTATIONS_FILE,
        image_dir=IMAGE_DIR,
        filter_classes=FILTER_CLASSES
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Initializing corruptor (device: {device})...")
    corruptor = TorchCorruptions(device=device)
    
    category_names = format_coco_label_mapping()
    
    return evaluator, corruptor, category_names


def print_report_structure(pdf_path, results, corruptions):
    """
    Print the structure of the generated PDF report.
    
    Args:
        pdf_path: Path to the generated PDF
        results: Test results dictionary
        corruptions: List of corruption types
    """
    print(f"\n✅ PDF generated successfully: {pdf_path}")
    print("\n📄 Report contents:")
    print("   ├─ Page 1: Title page (blue background)")
    print("   │  ├─ Project title")
    print("   │  ├─ Tested models list")
    print("   │  ├─ Corruption types")
    print("   │  └─ Dataset information")
    print("   │")
    print("   ├─ Page 2: Table of Contents (white background)")
    print("   │  ├─ Clickable links to all sections")
    print("   │  ├─ Overall mAP Comparison")
    print("   │  └─ Individual corruption analyses")
    print("   │")
    print("   ├─ Page 3: mAP Comparison (white background)")
    print("   │  ├─ Results table with #114584 headers")
    print("   │  ├─ Three-line table style")
    print("   │  ├─ Alternating row colors")
    print("   │  ├─ Models ranked by clean mAP")
    print("   │  ├─ Spider/radar chart")
    print("   │  └─ Calculation notes")
    print("   │")
    
    # Show corruption pages
    all_corruptions = set()
    for model_data in results.values():
        if 'corrupted' in model_data:
            all_corruptions.update(model_data['corrupted'].keys())
    
    sorted_corruptions = sorted(all_corruptions)
    for idx, corruption in enumerate(sorted_corruptions, start=4):
        corruption_name = corruption.replace('_', ' ').title()
        is_last = (idx == len(sorted_corruptions) + 3)
        prefix = "   └─" if is_last else "   ├─"
        
        print(f"{prefix} Page {idx}: {corruption_name} Analysis")
        print(f"   │  ├─ Line plot (mAP vs severity)")
        print(f"   │  ├─ Dual-row table (mAP + degradation %)")
        print(f"   │  ├─ Qualitative Examples ({NUM_QUALITATIVE_IMAGES} images)")
        print(f"   │  │  ├─ Grids: detectors × severity levels")
        print(f"   │  │  ├─ Color-coded bounding boxes")
        print(f"   │  │  └─ Detector legend")


def test_pdf_generation():
    """
    Test the PDF generator with actual test results and qualitative examples.
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Load test results
    results_path = Path(__file__).parent.parent / RESULTS_FILE
    results = load_test_results(results_path)
    
    if results is None:
        return False
    
    # Get detector names from results
    detector_names = [model_data['name'] for model_data in results.values()]
    
    print("🔄 Initializing models and dataset for qualitative examples...")
    
    # Initialize ModelLoader and load models
    model_loader = ModelLoader()
    initialize_models(model_loader)
    
    # Initialize evaluator and corruptor
    evaluator, corruptor, category_names = initialize_evaluator_and_corruptor()
    
    # Generate PDF with qualitative examples
    print("\n📝 Generating PDF report with qualitative examples...")
    print("   This will take several minutes as it runs inference on-the-fly")
    
    generator = RobustnessReportGenerator()
    pdf_path = generator.generate_report(
        detectors=detector_names,
        corruptions=TEST_CORRUPTIONS,
        results=results,
        dataset_name='DustyConstruction.v2i.coco',
        model_loader=model_loader,
        evaluator=evaluator,
        corruptor=corruptor,
        category_names=category_names,
        include_qualitative=True,
        num_qualitative_images=NUM_QUALITATIVE_IMAGES
    )
    
    # Print report structure
    print_report_structure(pdf_path, results, TEST_CORRUPTIONS)
    
    return True


if __name__ == "__main__":
    success = test_pdf_generation()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Test script for PDF report generator
"""
import json
import sys
from pathlib import Path
import torch

# Add parent directory to path to import pdf_report
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_report import RobustnessReportGenerator
from model_loader import ModelLoader
from evaluator import COCOEvaluator, format_coco_label_mapping
from torch_corruptions import TorchCorruptions


def test_pdf_generation():
    """Test the PDF generator with actual test results and qualitative examples"""
    
    # Load test results
    results_path = Path(__file__).parent.parent / 'static' / 'test_results.json'
    
    if not results_path.exists():
        print(f"❌ Test results file not found: {results_path}")
        print("   Please run the robustness tests first to generate test_results.json")
        return False
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Get detector names from results
    detector_names = [model_data['name'] for model_data in results.values()]
    
    # Corruptions to test
    corruptions = ['gaussian_noise', 'pixelate']
    
    print("🔄 Initializing models and dataset for qualitative examples...")
    
    # Initialize ModelLoader and load YOLO11 and DETR models
    model_loader = ModelLoader()
    print("  Loading YOLO11...")
    model_loader.load_model('yolov11')
    print("  Loading DETR ResNet50...")
    model_loader.load_model('detr')
    
    # Initialize COCOEvaluator for Construction dataset
    print("  Initializing evaluator...")
    evaluator = COCOEvaluator(
        annotation_file='DustyConstruction.v2i.coco/_annotations.coco.json',
        image_dir='DustyConstruction.v2i.coco/train',
        filter_classes=[3]  # Only person class
    )
    
    # Initialize TorchCorruptions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Initializing corruptor (device: {device})...")
    corruptor = TorchCorruptions(device=device)
    
    # Get category names mapping
    category_names = format_coco_label_mapping()
    
    # Generate PDF with qualitative examples
    print("\nGenerating PDF report with qualitative examples...")
    print("  This will take several minutes as it runs inference on-the-fly")
    generator = RobustnessReportGenerator()
    pdf_path = generator.generate_report(
        detectors=detector_names,
        corruptions=corruptions,
        results=results,
        dataset_name='DustyConstruction.v2i.coco',
        model_loader=model_loader,
        evaluator=evaluator,
        corruptor=corruptor,
        category_names=category_names,
        include_qualitative=True,
        num_qualitative_images=3
    )
    
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
    print("   │  └─ Calculation notes")
    print("   │")
    
    # Show corruption pages
    all_corruptions = set()
    for model_data in results.values():
        if 'corrupted' in model_data:
            all_corruptions.update(model_data['corrupted'].keys())
    
    for idx, corruption in enumerate(sorted(all_corruptions), start=4):
        corruption_name = corruption.replace('_', ' ').title()
        if idx < len(all_corruptions) + 3:
            print(f"   ├─ Page {idx}: {corruption_name} Analysis")
        else:
            print(f"   └─ Page {idx}: {corruption_name} Analysis")
        print(f"   │  ├─ Line plot (mAP vs severity)")
        print(f"   │  ├─ Dual-row table (mAP + degradation %)")
        print(f"   │  ├─ Qualitative Examples (3 images)")
        print(f"   │  │  ├─ Grids: detectors × severity levels")
        print(f"   │  │  ├─ Color-coded bounding boxes")
        print(f"   │  │  └─ Detector legend")
    
    return True


if __name__ == "__main__":
    success = test_pdf_generation()
    sys.exit(0 if success else 1)

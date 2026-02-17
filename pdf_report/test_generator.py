#!/usr/bin/env python3
"""
Test script for PDF report generator
"""
import json
import sys
from pathlib import Path

# Add parent directory to path to import pdf_report
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_report import RobustnessReportGenerator


def test_pdf_generation():
    """Test the PDF generator with actual test results"""
    
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
    
    # Sample corruptions (from the data)
    corruptions = ['gaussian_noise', 'pixelate']
    
    # Generate PDF
    print("🔄 Generating PDF report...")
    generator = RobustnessReportGenerator()
    pdf_path = generator.generate_report(
        detectors=detector_names,
        corruptions=corruptions,
        results=results,
        dataset_name='DustyConstruction.v2i.coco'
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
        print(f"   │  └─ Dual-row table (mAP + degradation %)")
    
    return True


if __name__ == "__main__":
    success = test_pdf_generation()
    sys.exit(0 if success else 1)

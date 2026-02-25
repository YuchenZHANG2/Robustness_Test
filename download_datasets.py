"""
Download and prepare all datasets for Object Detector Robustness Testing.

This script downloads:
1. COCO 2017 validation dataset (subset) using FiftyOne
2. DustyConstruction dataset from Hugging Face
3. OpenImages OOD dataset using FiftyOne

All datasets are placed in the correct directories expected by app.py.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import fiftyone as fo
import fiftyone.zoo as foz
import shutil

# Base paths
PROJECT_ROOT = Path(__file__).parent
COCO_BASE_DIR = PROJECT_ROOT / "Coco"
CONSTRUCTION_DIR = PROJECT_ROOT / "DustyConstruction.v2i.coco"
OOD_BASE_DIR = PROJECT_ROOT / "OOD_dataset" / "OpenImage"


def fix_fiftyone_coco_categories(annotations_file):
    """
    Fix FiftyOne's alphabetically-ordered category IDs to match standard COCO IDs.
    
    FiftyOne exports categories alphabetically with contiguous IDs (1, 2, 3...),
    but COCO uses specific IDs with gaps. This function remaps to standard COCO IDs.
    
    Args:
        annotations_file: Path to the COCO annotations JSON file
    """
    # Standard COCO category names to IDs (80 classes)
    standard_coco_categories = {
        'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5,
        'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10,
        'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14, 'bench': 15,
        'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21,
        'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27,
        'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 'frisbee': 34,
        'skis': 35, 'snowboard': 36, 'sports ball': 37, 'kite': 38, 'baseball bat': 39,
        'baseball glove': 40, 'skateboard': 41, 'surfboard': 42, 'tennis racket': 43,
        'bottle': 44, 'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50,
        'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54, 'orange': 55,
        'broccoli': 56, 'carrot': 57, 'hot dog': 58, 'pizza': 59, 'donut': 60,
        'cake': 61, 'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65,
        'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73, 'mouse': 74,
        'remote': 75, 'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79,
        'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84, 'clock': 85,
        'vase': 86, 'scissors': 87, 'teddy bear': 88, 'hair drier': 89, 'toothbrush': 90
    }
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # Create mapping from old IDs to new IDs based on category names
    old_to_new_id = {}
    new_categories = []
    
    for cat in data['categories']:
        old_id = cat['id']
        cat_name = cat['name']
        
        if cat_name in standard_coco_categories:
            new_id = standard_coco_categories[cat_name]
            old_to_new_id[old_id] = new_id
            
            new_categories.append({
                'id': new_id,
                'name': cat_name,
                'supercategory': cat.get('supercategory', 'object')
            })
    
    # Update categories
    data['categories'] = new_categories
    
    # Update category_id in all annotations
    for ann in data['annotations']:
        old_cat_id = ann['category_id']
        if old_cat_id in old_to_new_id:
            ann['category_id'] = old_to_new_id[old_cat_id]
    
    # Save fixed annotations
    with open(annotations_file, 'w') as f:
        json.dump(data, f, indent=2)


def download_coco(max_samples=100):
    """
    Download COCO 2017 validation dataset using FiftyOne.
    
    Args:
        max_samples: Number of images to download (default: 100 for quick testing)
    """
    print("\n" + "="*60)
    print("DOWNLOADING COCO 2017 VALIDATION DATASET")
    print("="*60)
    
    try:
        # Create directories
        COCO_BASE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download dataset using FiftyOne
        print(f"Downloading {max_samples} COCO validation images...")
        print("Note: This will download the full validation set (~1.9GB) but it will be cleaned up after export.")
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="validation",
            label_types=["detections"],
            max_samples=max_samples,
        )
        
        # Export to COCO format with original category IDs preserved
        export_dir = COCO_BASE_DIR
        print(f"Exporting dataset to {export_dir}...")
        
        dataset.export(
            export_dir=str(export_dir),
            dataset_type=fo.types.COCODetectionDataset,
            label_field="ground_truth",
            use_polylines=False,
            tolerance=2,
        )
        
        # Check and rename if needed
        annotations_dir = export_dir / "annotations"
        images_dir = export_dir / "data"
        target_images_dir = export_dir / "val2017"
        
        if images_dir.exists() and not target_images_dir.exists():
            shutil.move(str(images_dir), str(target_images_dir))
        
        # Rename labels.json to instances_val2017.json if needed
        labels_file = export_dir / "labels.json"
        target_labels_file = annotations_dir / "instances_val2017.json"
        
        if labels_file.exists():
            annotations_dir.mkdir(exist_ok=True)
            shutil.move(str(labels_file), str(target_labels_file))
            fix_fiftyone_coco_categories(target_labels_file)
        
        print(f"✓ COCO dataset downloaded successfully!")
        print(f"  Images: {target_images_dir}")
        print(f"  Annotations: {target_labels_file}")
        
        # Cleanup FiftyOne cache to save disk space
        try:
            fo.delete_dataset("coco-2017-validation")
            print("Cleaned up full dataset cache (~1.9GB freed)")
        except:
            pass
        
    except Exception as e:
        print(f"Error downloading COCO dataset: {e}")
        print("You can manually download from: https://cocodataset.org/#download")
        return False
    
    return True


def download_dusty_construction():
    """
    Download DustyConstruction dataset from Hugging Face.
    """
    print("\n" + "="*60)
    print("DOWNLOADING DUSTY CONSTRUCTION DATASET")
    print("="*60)
    
    try:
        from huggingface_hub import snapshot_download
        
        print("Downloading from Hugging Face...")
        snapshot_download(
            repo_id="YuchenGua/Dusty_Construction",
            repo_type="dataset",
            local_dir=str(CONSTRUCTION_DIR),
            local_dir_use_symlinks=False,
        )
        
        print(f" DustyConstruction dataset downloaded successfully!")
        print(f"  Location: {CONSTRUCTION_DIR}")
        
    except ImportError:
        print(" huggingface_hub not installed.")
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
        print("\nPlease run this script again after installation.")
        return False
    except Exception as e:
        print(f" Error downloading DustyConstruction dataset: {e}")
        print("\nManual download instructions:")
        print("1. Visit: https://huggingface.co/datasets/YuchenGua/Dusty_Construction")
        print("2. Download the dataset")
        print(f"3. Extract to: {CONSTRUCTION_DIR}")
        return False
    
    return True


def download_ood_dataset():
    """
    Download and prepare OpenImages OOD dataset by running existing scripts.
    Note: Requires COCO dataset to be downloaded first for category ID mapping.
    """
    print("\n" + "="*60)
    print("DOWNLOADING OOD DATASET (OpenImages)")
    print("="*60)
    
    try:
        # Check if COCO is available (required for category mapping)
        coco_annotations = COCO_BASE_DIR / "annotations" / "instances_val2017.json"
        if not coco_annotations.exists():
            print(f"⚠ Warning: COCO annotations not found at {coco_annotations}")
            print("  The OOD dataset requires COCO for category ID mapping.")
            print("  Continuing anyway, but b_coco_openimage_id.py may fail...")
        
        # Create directories
        OOD_BASE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Path to existing scripts
        script_a = OOD_BASE_DIR / "a_download_dataset.py"
        script_b = OOD_BASE_DIR / "b_coco_openimage_id.py"
        
        # Check if scripts exist
        if not script_a.exists():
            print(f" Script not found: {script_a}")
            return False
        if not script_b.exists():
            print(f" Script not found: {script_b}")
            return False
        
        print("Step 1: Downloading and filtering OpenImages dataset...")
        
        # Run a_download_dataset.py
        import subprocess
        result = subprocess.run(
            [sys.executable, str(script_a)],
            cwd=str(OOD_BASE_DIR),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f" Error running {script_a.name}:")
            print(result.stderr)
            return False
        
        print(result.stdout)
        
        print("\nStep 2: Remapping category IDs to align with COCO...")
        
        # Run b_coco_openimage_id.py
        result = subprocess.run(
            [sys.executable, str(script_b)],
            cwd=str(OOD_BASE_DIR),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f" Error running {script_b.name}:")
            print(result.stderr)
            return False
        
        print(result.stdout)
        
        # Verify output
        final_dir = OOD_BASE_DIR / "Dataset_final"
        target_labels_file = final_dir / "labels_new.json"
        data_dir = final_dir / "data"
        
        if target_labels_file.exists() and data_dir.exists():
            print(f"\nOOD dataset downloaded successfully!")
            print(f"  Images: {data_dir}")
            print(f"  Annotations: {target_labels_file}")
            return True
        else:
            print("Output files not found after processing")
            return False
        
    except Exception as e:
        print(f" Error downloading OOD dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to download all datasets."""
    print("="*60)
    print("OBJECT DETECTOR ROBUSTNESS TESTING - DATASET DOWNLOADER")
    print("="*60)
    print("\nThis script will download all required datasets:")
    print("1. COCO 2017 Validation (subset)")
    print("2. DustyConstruction")
    print("3. OpenImages OOD Dataset")

    
    # Download datasets
    results = {}
    
    # COCO
    results['coco'] = download_coco(max_samples=200)
    
    # DustyConstruction
    results['construction'] = download_dusty_construction()
    
    # OOD
    results['ood'] = download_ood_dataset()
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"COCO:              {' Success' if results['coco'] else ' Failed'}")
    print(f"DustyConstruction: {' Success' if results['construction'] else ' Failed'}")
    print(f"OOD Dataset:       {' Success' if results['ood'] else ' Failed'}")
    
    if all(results.values()):
        print("\n All datasets downloaded successfully!")
        print("\nYou can now run: python app.py")
    else:
        print("\n⚠ Some datasets failed to download.")
        print("Please check the error messages above and download manually if needed.")


if __name__ == "__main__":
    main()

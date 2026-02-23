"""
Remap OpenImages categories to align with COCO dataset.
Creates labels_new.json with:
- IN categories: Use COCO IDs (1-90 with gaps)
- OUT categories: New IDs starting from 91
- IGNORE categories: Removed entirely
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

# Paths
SCRIPT_DIR = Path(__file__).parent
OPENIMAGES_LABELS = SCRIPT_DIR / "Dataset_final" / "labels.json"
COCO_ANNOTATIONS = Path("/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json")
MAPPING_CSV = SCRIPT_DIR / "openimages_coco_mapping.csv"
OUTPUT_FILE = SCRIPT_DIR / "Dataset_final" / "labels_new.json"

def load_coco_categories():
    """Load COCO category list with exact IDs."""
    print("Loading COCO categories...")
    with open(COCO_ANNOTATIONS, 'r') as f:
        coco_data = json.load(f)
    
    # Create mapping: coco_name (lowercase) -> coco_id
    coco_name_to_id = {}
    coco_categories = []
    
    for cat in coco_data['categories']:
        coco_name_to_id[cat['name'].lower()] = cat['id']
        coco_categories.append({
            'id': cat['id'],
            'name': cat['name'],
            'supercategory': cat.get('supercategory', None)
        })
    
    print(f"  Found {len(coco_categories)} COCO categories (IDs 1-90 with gaps)")
    return coco_name_to_id, coco_categories

def load_openimages_mapping():
    """Load OpenImages to COCO mapping from CSV."""
    print("Loading OpenImages-COCO mapping...")
    mapping = {}
    
    with open(MAPPING_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            openimages_name = row['OpenImages'].strip()
            status = row['Status'].strip()
            coco_class = row['COCO_Class'].strip()
            
            mapping[openimages_name.lower()] = {
                'status': status,
                'coco_class': coco_class.lower() if coco_class else None,
                'original_name': openimages_name
            }
    
    print(f"  Loaded {len(mapping)} mappings")
    print(f"    IN: {sum(1 for m in mapping.values() if m['status'] == 'IN')}")
    print(f"    OUT: {sum(1 for m in mapping.values() if m['status'] == 'OUT')}")
    print(f"    IGNORE: {sum(1 for m in mapping.values() if m['status'] == 'IGNORE')}")
    
    return mapping

def load_openimages_data():
    """Load current OpenImages labels.json."""
    print("Loading OpenImages dataset...")
    with open(OPENIMAGES_LABELS, 'r') as f:
        data = json.load(f)
    
    print(f"  Categories: {len(data['categories'])}")
    print(f"  Images: {len(data['images'])}")
    print(f"  Annotations: {len(data['annotations'])}")
    
    return data

def create_category_mapping(openimages_data, oi_mapping, coco_name_to_id):
    """
    Create mapping from old OpenImages category IDs to new IDs.
    Returns: (old_id -> new_id mapping, list of OUT categories)
    """
    print("\nCreating category ID mapping...")
    
    old_id_to_new_id = {}
    out_categories = []
    
    # Collect all OUT categories first
    for cat in openimages_data['categories']:
        cat_name_lower = cat['name'].lower()
        
        if cat_name_lower in oi_mapping:
            mapping_info = oi_mapping[cat_name_lower]
            status = mapping_info['status']
            
            if status == 'IN':
                # Map to COCO ID
                coco_class = mapping_info['coco_class']
                if coco_class and coco_class in coco_name_to_id:
                    new_id = coco_name_to_id[coco_class]
                    old_id_to_new_id[cat['id']] = new_id
                else:
                    print(f"  Warning: IN category '{cat['name']}' has no COCO mapping")
            
            elif status == 'OUT':
                # Will assign new ID starting from 91
                out_categories.append({
                    'old_id': cat['id'],
                    'name': cat['name']  # Keep original OpenImages name
                })
            
            elif status == 'IGNORE':
                # Mark for deletion (no mapping)
                old_id_to_new_id[cat['id']] = None
        else:
            print(f"  Warning: Category '{cat['name']}' not found in mapping CSV")
    
    # Sort OUT categories alphabetically and assign IDs starting from 91
    out_categories.sort(key=lambda x: x['name'].lower())
    next_id = 91
    
    for cat in out_categories:
        old_id_to_new_id[cat['old_id']] = next_id
        cat['new_id'] = next_id
        next_id += 1
    
    print(f"  Mapped {sum(1 for v in old_id_to_new_id.values() if v and v <= 90)} categories to COCO IDs")
    print(f"  Mapped {len(out_categories)} OUT categories to IDs 91-{next_id-1}")
    print(f"  Marked {sum(1 for v in old_id_to_new_id.values() if v is None)} categories for removal (IGNORE)")
    
    return old_id_to_new_id, out_categories

def remap_annotations(annotations, old_id_to_new_id):
    """Remap annotations to new category IDs, removing IGNORE categories."""
    print("\nRemapping annotations...")
    
    new_annotations = []
    removed_count = 0
    
    for ann in annotations:
        old_cat_id = ann['category_id']
        
        if old_cat_id in old_id_to_new_id:
            new_cat_id = old_id_to_new_id[old_cat_id]
            
            if new_cat_id is None:
                # IGNORE category - skip this annotation
                removed_count += 1
                continue
            
            # Create new annotation with only essential fields
            new_ann = {
                'id': ann['id'],
                'image_id': ann['image_id'],
                'category_id': new_cat_id,
                'bbox': ann['bbox'],
                'area': ann['area'],
                'iscrowd': ann.get('iscrowd', 0)
            }
            new_annotations.append(new_ann)
        else:
            print(f"  Warning: Annotation with unknown category_id {old_cat_id}")
    
    print(f"  Kept: {len(new_annotations)} annotations")
    print(f"  Removed: {removed_count} annotations (IGNORE categories)")
    
    return new_annotations

def create_new_labels(openimages_data, coco_categories, out_categories, new_annotations):
    """Create the new labels.json structure."""
    print("\nCreating new labels structure...")
    
    # Build final category list: COCO (80 categories) + OUT categories
    final_categories = coco_categories.copy()
    
    for cat in out_categories:
        final_categories.append({
            'id': cat['new_id'],
            'name': cat['name'],
            'supercategory': None
        })
    
    # Sort by ID for cleaner output
    final_categories.sort(key=lambda x: x['id'])
    
    new_labels = {
        'info': openimages_data.get('info', {}),
        'licenses': openimages_data.get('licenses', []),
        'categories': final_categories,
        'images': openimages_data['images'],
        'annotations': new_annotations
    }
    
    print(f"  Total categories: {len(final_categories)}")
    print(f"    COCO (1-90): 80 categories")
    print(f"    OUT (91+): {len(out_categories)} categories")
    
    return new_labels

def main():
    print("=" * 80)
    print("Remapping OpenImages Categories to COCO Format")
    print("=" * 80)
    
    # Load all data
    coco_name_to_id, coco_categories = load_coco_categories()
    oi_mapping = load_openimages_mapping()
    openimages_data = load_openimages_data()
    
    # Create mapping from old to new category IDs
    old_id_to_new_id, out_categories = create_category_mapping(
        openimages_data, oi_mapping, coco_name_to_id
    )
    
    # Remap annotations
    new_annotations = remap_annotations(
        openimages_data['annotations'],
        old_id_to_new_id
    )
    
    # Create new labels file
    new_labels = create_new_labels(
        openimages_data,
        coco_categories,
        out_categories,
        new_annotations
    )
    
    # Save output
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(new_labels, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✓ Successfully created labels_new.json")
    print("=" * 80)
    
    # Summary statistics
    print("\nSummary:")
    print(f"  Input annotations:  {len(openimages_data['annotations'])}")
    print(f"  Output annotations: {len(new_annotations)}")
    print(f"  Difference:         {len(openimages_data['annotations']) - len(new_annotations)} removed")
    print(f"\n  Input categories:   {len(openimages_data['categories'])}")
    print(f"  Output categories:  {len(new_labels['categories'])}")
    print(f"    - COCO categories: 80")
    print(f"    - OUT categories:  {len(out_categories)}")
    print()

if __name__ == "__main__":
    main()

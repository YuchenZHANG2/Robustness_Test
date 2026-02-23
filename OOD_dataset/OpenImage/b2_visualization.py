"""
Visualize random images from Dataset_final with annotations from both labels.json and labels_new.json.
Shows side-by-side comparison. Saves annotated images locally (for remote machine use).
"""

import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import colorsys

# Paths
SCRIPT_DIR = Path(__file__).parent
LABELS_FILE_OLD = SCRIPT_DIR / "Dataset_final" / "labels.json"
LABELS_FILE_NEW = SCRIPT_DIR / "Dataset_final" / "labels_new.json"
DATA_DIR = SCRIPT_DIR / "Dataset_final" / "data"
OUTPUT_DIR = SCRIPT_DIR / "visualizations"

# Visualization settings
NUM_IMAGES = 5
BOX_WIDTH = 3
FONT_SIZE = 16
TITLE_FONT_SIZE = 24


def generate_colors(n):
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.8
        value = 0.95
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(x * 255) for x in rgb))
    return colors


def load_labels(labels_file):
    """Load a labels JSON file."""
    print(f"Loading annotations from {labels_file.name}...")
    with open(labels_file, 'r') as f:
        data = json.load(f)
    
    # Create category lookup
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Create image lookup
    image_id_to_info = {img['id']: img for img in data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    print(f"  Categories: {len(cat_id_to_name)}")
    print(f"  Images: {len(image_id_to_info)}")
    print(f"  Annotations: {len(data['annotations'])}")
    
    return cat_id_to_name, image_id_to_info, annotations_by_image


def draw_annotations(image, annotations, cat_id_to_name, category_colors):
    """Draw bounding boxes and labels on image."""
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", FONT_SIZE)
        except:
            font = ImageFont.load_default()
    
    for ann in annotations:
        # Get category info
        cat_id = ann['category_id']
        cat_name = cat_id_to_name.get(cat_id, f"ID_{cat_id}")
        color = category_colors.get(cat_id, (255, 0, 0))
        
        # Get bounding box (COCO format: [x, y, width, height])
        x, y, w, h = ann['bbox']
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=BOX_WIDTH)
        
        # Draw label background
        label_text = cat_name
        
        # Get text size using textbbox
        bbox = draw.textbbox((x1, y1), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw label background
        label_y = max(0, y1 - text_height - 4)
        draw.rectangle(
            [x1, label_y, x1 + text_width + 4, label_y + text_height + 4],
            fill=color
        )
        
        # Draw label text
        draw.text((x1 + 2, label_y + 2), label_text, fill=(255, 255, 255), font=font)
    
    return image


def create_side_by_side_comparison(img_path, annotations_old, annotations_new, 
                                   cat_id_to_name_old, cat_id_to_name_new, 
                                   category_colors, title_left, title_right):
    """Create a side-by-side comparison image with annotations from both label files."""
    # Load image twice (one for each side)
    image_old = Image.open(img_path).convert('RGB')
    image_new = Image.open(img_path).convert('RGB')
    
    # Draw annotations on each
    image_old = draw_annotations(image_old, annotations_old, cat_id_to_name_old, category_colors)
    image_new = draw_annotations(image_new, annotations_new, cat_id_to_name_new, category_colors)
    
    # Add title space
    title_height = TITLE_FONT_SIZE + 20
    width, height = image_old.size
    
    # Create combined image with titles
    combined = Image.new('RGB', (width * 2, height + title_height), color=(255, 255, 255))
    
    # Paste images
    combined.paste(image_old, (0, title_height))
    combined.paste(image_new, (width, title_height))
    
    # Draw titles
    draw = ImageDraw.Draw(combined)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", TITLE_FONT_SIZE)
    except:
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", TITLE_FONT_SIZE)
        except:
            title_font = ImageFont.load_default()
    
    # Draw left title
    bbox_left = draw.textbbox((0, 0), title_left, font=title_font)
    text_width_left = bbox_left[2] - bbox_left[0]
    draw.text((width // 2 - text_width_left // 2, 10), title_left, fill=(0, 0, 0), font=title_font)
    
    # Draw right title
    bbox_right = draw.textbbox((0, 0), title_right, font=title_font)
    text_width_right = bbox_right[2] - bbox_right[0]
    draw.text((width + width // 2 - text_width_right // 2, 10), title_right, fill=(0, 0, 0), font=title_font)
    
    # Draw dividing line
    draw.line([(width, 0), (width, height + title_height)], fill=(0, 0, 0), width=3)
    
    return combined


def visualize_random_images():
    """Randomly select and visualize images with annotations from both label files."""
    print("=" * 80)
    print("Visualizing Random Images - Comparing labels.json vs labels_new.json")
    print("=" * 80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Load data from both label files
    print("\n--- Loading labels.json (OLD) ---")
    cat_id_to_name_old, image_id_to_info_old, annotations_by_image_old = load_labels(LABELS_FILE_OLD)
    
    print("\n--- Loading labels_new.json (NEW) ---")
    cat_id_to_name_new, image_id_to_info_new, annotations_by_image_new = load_labels(LABELS_FILE_NEW)
    
    # Merge all categories for consistent coloring
    all_cat_names = set(cat_id_to_name_old.values()) | set(cat_id_to_name_new.values())
    sorted_cat_names = sorted(all_cat_names)
    colors = generate_colors(len(sorted_cat_names))
    
    # Create color mapping by category ID for both label sets
    category_colors_old = {}
    for cat_id, cat_name in cat_id_to_name_old.items():
        color_idx = sorted_cat_names.index(cat_name)
        category_colors_old[cat_id] = colors[color_idx]
    
    category_colors_new = {}
    for cat_id, cat_name in cat_id_to_name_new.items():
        color_idx = sorted_cat_names.index(cat_name)
        category_colors_new[cat_id] = colors[color_idx]
    
    # Get image IDs that exist in both datasets (use file_name for matching)
    filename_to_id_old = {img['file_name']: img_id for img_id, img in image_id_to_info_old.items()}
    filename_to_id_new = {img['file_name']: img_id for img_id, img in image_id_to_info_new.items()}
    
    common_filenames = set(filename_to_id_old.keys()) & set(filename_to_id_new.keys())
    print(f"\n--- Common images in both label files: {len(common_filenames)} ---")
    
    # Get filenames that have annotations in at least one of the files
    filenames_with_annotations = []
    for filename in common_filenames:
        img_id_old = filename_to_id_old[filename]
        img_id_new = filename_to_id_new[filename]
        if img_id_old in annotations_by_image_old or img_id_new in annotations_by_image_new:
            filenames_with_annotations.append(filename)
    
    print(f"Images with annotations (in at least one file): {len(filenames_with_annotations)}")
    
    # Randomly select images
    if len(filenames_with_annotations) < NUM_IMAGES:
        selected_filenames = filenames_with_annotations
        print(f"Warning: Only {len(selected_filenames)} images available")
    else:
        selected_filenames = random.sample(filenames_with_annotations, NUM_IMAGES)
    
    print(f"Selected {len(selected_filenames)} random images for visualization\n")
    print("=" * 80)
    
    # Process each selected image
    for idx, filename in enumerate(selected_filenames, 1):
        img_path = DATA_DIR / filename
        
        print(f"\n[{idx}/{len(selected_filenames)}] Processing: {filename}")
        
        if not img_path.exists():
            print(f"  Error: Image file not found: {img_path}")
            continue
        
        # Load image to get size
        test_img = Image.open(img_path).convert('RGB')
        print(f"  Image size: {test_img.size}")
        
        # Get annotations from both files
        img_id_old = filename_to_id_old[filename]
        img_id_new = filename_to_id_new[filename]
        
        annotations_old = annotations_by_image_old.get(img_id_old, [])
        annotations_new = annotations_by_image_new.get(img_id_new, [])
        
        print(f"  Annotations in labels.json (OLD): {len(annotations_old)}")
        print(f"  Annotations in labels_new.json (NEW): {len(annotations_new)}")
        
        # Count categories for both
        cat_counts_old = {}
        for ann in annotations_old:
            cat_id = ann['category_id']
            cat_name = cat_id_to_name_old.get(cat_id, f"ID_{cat_id}")
            cat_counts_old[cat_name] = cat_counts_old.get(cat_name, 0) + 1
        
        cat_counts_new = {}
        for ann in annotations_new:
            cat_id = ann['category_id']
            cat_name = cat_id_to_name_new.get(cat_id, f"ID_{cat_id}")
            cat_counts_new[cat_name] = cat_counts_new.get(cat_name, 0) + 1
        
        if cat_counts_old:
            print(f"  OLD Categories: {', '.join(f'{name}({count})' for name, count in sorted(cat_counts_old.items()))}")
        if cat_counts_new:
            print(f"  NEW Categories: {', '.join(f'{name}({count})' for name, count in sorted(cat_counts_new.items()))}")
        
        # Create side-by-side comparison
        comparison_image = create_side_by_side_comparison(
            img_path, 
            annotations_old, 
            annotations_new,
            cat_id_to_name_old,
            cat_id_to_name_new,
            {**category_colors_old, **category_colors_new},
            f"labels.json ({len(annotations_old)} boxes)",
            f"labels_new.json ({len(annotations_new)} boxes)"
        )
        
        # Save
        output_filename = f"comparison_{idx}_{filename}"
        output_path = OUTPUT_DIR / output_filename
        comparison_image.save(output_path)
        print(f"  ✓ Saved: {output_path}")
    
    print("\n" + "=" * 80)
    print(f"✓ Visualization complete! Saved {len(selected_filenames)} comparison images to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    visualize_random_images()

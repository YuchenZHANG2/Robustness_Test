import json
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Load COCO annotations
with open('DustyConstruction.v2i.coco/_annotations.coco.json', 'r') as f:
    coco_data = json.load(f)

# Create mappings
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
images_by_filename = {img['file_name']: img for img in coco_data['images']}

# Get image lists from train and test folders
train_images = os.listdir('DustyConstruction.v2i.coco/train')
test_images = os.listdir('DustyConstruction.v2i.coco/test')

# Group images by their first 7 digits
train_by_prefix = {}
for img in train_images:
    prefix = img[:7]
    train_by_prefix[prefix] = img

test_by_prefix = {}
for img in test_images:
    prefix = img[:7]
    test_by_prefix[prefix] = img

# Find paired images (same first 7 digits)
common_prefixes = set(train_by_prefix.keys()).intersection(set(test_by_prefix.keys()))
print(f"Total paired images: {len(common_prefixes)}")
print(f"Train only prefixes: {len(set(train_by_prefix.keys()) - set(test_by_prefix.keys()))}")
print(f"Test only prefixes: {len(set(test_by_prefix.keys()) - set(train_by_prefix.keys()))}")

# Select 10 random pairs
sample_prefixes = random.sample(list(common_prefixes), min(10, len(common_prefixes)))

# Create visualization directory
os.makedirs('static/dataset_visualization', exist_ok=True)

# Visualize each pair
for idx, prefix in enumerate(sample_prefixes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Get the actual filenames for this prefix
    train_img_name = train_by_prefix[prefix]
    test_img_name = test_by_prefix[prefix]
    
    # Load images
    train_path = f'DustyConstruction.v2i.coco/train/{train_img_name}'
    test_path = f'DustyConstruction.v2i.coco/test/{test_img_name}'
    
    train_img = Image.open(train_path)
    test_img = Image.open(test_path)
    
    ax1.imshow(train_img)
    ax2.imshow(test_img)
    
    # Get annotations for images with this prefix (first 7 digits)
    # Find all images with this prefix
    matching_images = [img for img in coco_data['images'] if img['file_name'][:7] == prefix]
    matching_image_ids = [img['id'] for img in matching_images]
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in matching_image_ids]
    
    # Draw annotations (excluding "dummy" classes)
    for ann in annotations:
        cat_name = categories[ann['category_id']]
        
        # Skip dummy classes
        if 'dummy' in cat_name.lower():
            continue
        
        bbox = ann['bbox']  # [x, y, width, height]
        
        # Draw on train image
        rect1 = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax1.add_patch(rect1)
        ax1.text(bbox[0], bbox[1]-5, cat_name, color='red', 
                fontsize=10, backgroundcolor='white')
        
        # Draw on test image
        rect2 = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax2.add_patch(rect2)
        ax2.text(bbox[0], bbox[1]-5, cat_name, color='red',
                fontsize=10, backgroundcolor='white')
    
    ax1.set_title(f'Train: {train_img_name}', fontsize=12)
    ax2.set_title(f'Test: {test_img_name}', fontsize=12)
    ax1.axis('off')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'static/dataset_visualization/pair_{idx+1}_{prefix}.png', 
                dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization {idx+1}/10: {prefix} (train: {train_img_name}, test: {test_img_name})")

print(f"\nVisualizations saved in 'static/dataset_visualization/' directory")
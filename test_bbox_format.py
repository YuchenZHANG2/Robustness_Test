"""
Quick test to check bbox format from different models
"""
import torch
from PIL import Image
from model_loader import ModelLoader
from evaluator import COCOEvaluator
import numpy as np

# Load a test image
evaluator = COCOEvaluator(
    "/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json",
    "/home/yuchen/YuchenZ/Datasets/coco/val2017"
)

# Get a valid image ID
image_ids = evaluator.get_random_images(n=1)
image_id = image_ids[0]
image_path = evaluator.get_image_path(image_id)
image = Image.open(image_path).convert('RGB')

print(f"Image size: {image.size}")
print(f"Image ID: {image_id}")

# Test one DETR model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loader = ModelLoader(device=device)

print("\nLoading DETR model...")
loader.load_model('detr')

print("\nRunning prediction...")
preds = loader.predict('detr', image, score_threshold=0.3)

print(f"\nNumber of predictions: {len(preds['boxes'])}")

if len(preds['boxes']) > 0:
    print("\nFirst 3 predictions:")
    for i in range(min(3, len(preds['boxes']))):
        box = preds['boxes'][i]
        label = preds['labels'][i]
        score = preds['scores'][i]
        print(f"  Box {i}: {box}")
        print(f"    Label: {label}, Score: {score:.3f}")
        print(f"    Format check - x1={box[0]:.1f}, y1={box[1]:.1f}, x2={box[2]:.1f}, y2={box[3]:.1f}")
        print(f"    Width: {box[2]-box[0]:.1f}, Height: {box[3]-box[1]:.1f}")
        
    print("\n\nConverting to COCO format...")
    coco_preds = evaluator.convert_predictions_to_coco_format(preds, image_id, label_offset=0)
    
    print("\nFirst 3 COCO format predictions:")
    for i in range(min(3, len(coco_preds))):
        print(f"  COCO pred {i}: {coco_preds[i]}")
        bbox = coco_preds[i]['bbox']
        print(f"    bbox format: [x={bbox[0]:.1f}, y={bbox[1]:.1f}, w={bbox[2]:.1f}, h={bbox[3]:.1f}]")

print("\n\nChecking if category IDs are valid in COCO dataset...")
valid_cat_ids = evaluator.coco_gt.getCatIds()
print(f"Valid COCO category IDs: {sorted(valid_cat_ids)[:20]}...")  # Show first 20

unique_labels = np.unique(preds['labels'])
print(f"\nPredicted labels: {unique_labels[:20]}")  # Show first 20

invalid_labels = [l for l in unique_labels if int(l) not in valid_cat_ids]
if invalid_labels:
    print(f"WARNING: Invalid category IDs found: {invalid_labels}")
else:
    print("✓ All predicted category IDs are valid!")

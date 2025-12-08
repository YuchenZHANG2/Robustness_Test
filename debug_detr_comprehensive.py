"""
Comprehensive DETR debug - compare simple prediction vs batch pipeline
"""
import torch
from PIL import Image
from model_loader import ModelLoader
from batch_optimized_pipeline import BatchOptimizedRobustnessTest
from evaluator import COCOEvaluator
import numpy as np

# Initialize
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")

# Load evaluator
evaluator = COCOEvaluator(
    "/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json",
    "/home/yuchen/YuchenZ/Datasets/coco/val2017"
)

# Get 50 random images (same as in the actual test)
image_ids = evaluator.get_random_images(n=50)
print(f"Testing on {len(image_ids)} images\n")

# Initialize model loader
loader = ModelLoader(device=device)
loader.load_model('detr')

# Test 1: Simple prediction (like test_eval.py)
print("="*70)
print("TEST 1: Simple Prediction (working method)")
print("="*70)

all_predictions_simple = []
for img_id in image_ids:
    image_path = evaluator.get_image_path(img_id)
    image = Image.open(image_path).convert('RGB')
    
    preds = loader.predict('detr', image, score_threshold=0.05)
    
    # Convert to COCO format with label_offset=0
    coco_preds = evaluator.convert_predictions_to_coco_format(
        preds, img_id, label_offset=0
    )
    all_predictions_simple.extend(coco_preds)

print(f"Total predictions: {len(all_predictions_simple)}")
if len(all_predictions_simple) > 0:
    print(f"Sample prediction: {all_predictions_simple[0]}")
    print(f"Category IDs: {sorted(set(p['category_id'] for p in all_predictions_simple))[:20]}")
    
    # Evaluate
    results_simple = evaluator.evaluate_predictions(all_predictions_simple, image_ids)
    print(f"\nSimple Method Results:")
    print(f"  mAP: {results_simple['mAP']:.4f}")
    print(f"  mAP@50: {results_simple['mAP_50']:.4f}")
    print(f"  mAP@75: {results_simple['mAP_75']:.4f}")

# Test 2: Batch pipeline (like the actual test)
print("\n" + "="*70)
print("TEST 2: Batch Pipeline (the method used in testing)")
print("="*70)

batch_test = BatchOptimizedRobustnessTest(
    model_loader=loader,
    evaluator=evaluator,
    batch_size=4
)

# Use the batch pipeline method to run predictions
print("Running batch predictions...")
all_predictions_batch = []

# The actual test calls test_model_on_clean which eventually calls predict_batch
# Let's simulate that by processing images in batches
from torch.utils.data import DataLoader
from batch_optimized_pipeline import COCODetectionDataset, collate_fn

dataset = COCODetectionDataset(evaluator, image_ids)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

for batch_data in dataloader:
    batch_tensors = batch_data['images']  # Shape: (B, C, H, W)
    batch_image_ids = batch_data['image_ids']
    batch_transforms = batch_data['transformations']
    
    # Call the batch prediction method
    batch_results = batch_test.predict_batch_huggingface_tensor(
        'detr',
        batch_tensors,
        batch_transforms,
        score_threshold=0.05
    )
    
    # Convert to COCO format
    for img_id, preds in zip(batch_image_ids, batch_results):
        coco_preds = evaluator.convert_predictions_to_coco_format(
            preds, int(img_id), label_offset=0
        )
        all_predictions_batch.extend(coco_preds)

print(f"Total predictions: {len(all_predictions_batch)}")
if len(all_predictions_batch) > 0:
    print(f"Sample prediction: {all_predictions_batch[0]}")
    print(f"Category IDs: {sorted(set(p['category_id'] for p in all_predictions_batch))[:20]}")
    
    # Evaluate
    results_batch = evaluator.evaluate_predictions(all_predictions_batch, image_ids)
    print(f"\nBatch Pipeline Results:")
    print(f"  mAP: {results_batch['mAP']:.4f}")
    print(f"  mAP@50: {results_batch['mAP_50']:.4f}")
    print(f"  mAP@75: {results_batch['mAP_75']:.4f}")

# Compare
print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"Simple method mAP:   {results_simple['mAP']:.4f}")
print(f"Batch method mAP:    {results_batch['mAP']:.4f}")
print(f"Difference:          {abs(results_simple['mAP'] - results_batch['mAP']):.4f}")

if abs(results_simple['mAP'] - results_batch['mAP']) > 0.01:
    print("\n⚠️  SIGNIFICANT DIFFERENCE DETECTED!")
    print("\nDEBUG INFO:")
    print(f"Number of predictions (simple): {len(all_predictions_simple)}")
    print(f"Number of predictions (batch):  {len(all_predictions_batch)}")
    
    # Check a few predictions in detail
    print("\nFirst 5 predictions from simple method:")
    for p in all_predictions_simple[:5]:
        print(f"  {p}")
    
    print("\nFirst 5 predictions from batch method:")
    for p in all_predictions_batch[:5]:
        print(f"  {p}")
else:
    print("\n✓ Both methods produce similar results!")

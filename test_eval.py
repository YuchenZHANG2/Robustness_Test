"""
Test evaluation with actual COCO evaluation to see where the issue is
"""
import torch
from PIL import Image
from model_loader import ModelLoader
from evaluator import COCOEvaluator
import numpy as np

# Load evaluator  
evaluator = COCOEvaluator(
    "/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json",
    "/home/yuchen/YuchenZ/Datasets/coco/val2017"
)

# Get 10 random images
image_ids = evaluator.get_random_images(n=10)
print(f"Testing on {len(image_ids)} images: {image_ids}")

# Test one DETR model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loader = ModelLoader(device=device)

print("\nLoading DETR model...")
loader.load_model('detr')

print("\nRunning predictions...")
all_predictions = []

for img_id in image_ids:
    image_path = evaluator.get_image_path(img_id)
    image = Image.open(image_path).convert('RGB')
    
    preds = loader.predict('detr', image, score_threshold=0.05)
    
    # Convert to COCO format
    coco_preds = evaluator.convert_predictions_to_coco_format(
        preds, img_id, label_offset=0
    )
    
    all_predictions.extend(coco_preds)
    print(f"  Image {img_id}: {len(coco_preds)} predictions")

print(f"\nTotal predictions: {len(all_predictions)}")

if len(all_predictions) > 0:
    print("\nSample predictions:")
    for i in range(min(5, len(all_predictions))):
        print(f"  {all_predictions[i]}")
    
    print("\n" + "="*70)
    print("Running COCO evaluation...")
    print("="*70)
    
    try:
        results = evaluator.evaluate_predictions(all_predictions, image_ids)
        
        print("\nEvaluation Results:")
        print(f"  mAP: {results['mAP']:.4f}")
        print(f"  mAP@50: {results['mAP_50']:.4f}")
        print(f"  mAP@75: {results['mAP_75']:.4f}")
        
        if results['mAP'] < 0.1:
            print("\n⚠️  WARNING: mAP is very low! This indicates a problem.")
            print("\nDEBUG INFO:")
            print(f"  Number of predictions submitted: {len(all_predictions)}")
            print(f"  Number of images: {len(image_ids)}")
            print(f"  Unique category IDs in predictions: {sorted(set(p['category_id'] for p in all_predictions))[:20]}")
            
            # Check ground truth
            valid_cats = evaluator.coco_gt.getCatIds()
            print(f"  Valid category IDs in dataset: {sorted(valid_cats)[:20]}...")
            
    except Exception as e:
        print(f"\nERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n⚠️  No predictions generated!")

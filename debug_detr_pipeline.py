"""
Test the exact way run_full_test processes DETR predictions
"""
import torch
import json
from model_loader import ModelLoader
from batch_optimized_pipeline import BatchOptimizedRobustnessTest
from evaluator import COCOEvaluator

# Initialize
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")

# Load evaluator
evaluator = COCOEvaluator(
    "/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json",
    "/home/yuchen/YuchenZ/Datasets/coco/val2017"
)

# Get 50 random images
image_ids = evaluator.get_random_images(n=50)
print(f"Testing on {len(image_ids)} images\n")

# Initialize model loader and load DETR
loader = ModelLoader(device=device)
loader.load_model('detr')

# Create batch test
batch_test = BatchOptimizedRobustnessTest(
    model_loader=loader,
    evaluator=evaluator,
    batch_size=4,
    num_workers=0  # No workers to avoid multiprocessing issues
)

# Run the full test for DETR on clean images only
print("="*70)
print("Running DETR test via run_full_test (EXACT pipeline method)")
print("="*70)

results = batch_test.run_full_test(
    model_keys=['detr'],
    corruption_names=[],  # No corruptions, just clean
    image_ids=image_ids,
    severities=[],
    progress_callback=None
)

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(json.dumps(results, indent=2))

detr_map = results['detr']['clean']['mAP']
print(f"\n{'='*70}")
print(f"DETR mAP: {detr_map:.4f}")
print(f"{'='*70}")

if detr_map < 0.10:
    print("\n⚠️  ERROR: mAP is very low! This is the same problem as in the test results.")
    print("Let me investigate further...")
    
    # Check the predictions
    print(f"\nNumber of predictions made: {len(results.get('_predictions', {}).get('clean', []))}")
else:
    print("\n✓ mAP looks normal!")

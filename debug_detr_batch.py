"""
Debug script to check DETR predictions in the batch pipeline
"""
import torch
from PIL import Image
from model_loader import ModelLoader
from batch_optimized_pipeline import BatchOptimizedRobustnessTest
from evaluator import COCOEvaluator
import numpy as np

# Initialize
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load one test image
evaluator = COCOEvaluator(
    "/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json",
    "/home/yuchen/YuchenZ/Datasets/coco/val2017"
)

image_ids = evaluator.get_random_images(n=5)
print(f"Testing on {len(image_ids)} images: {image_ids}\n")

# Test 1: Standard ModelLoader prediction
print("="*70)
print("TEST 1: Standard ModelLoader.predict()")
print("="*70)

loader = ModelLoader(device=device)
loader.load_model('detr')

for img_id in image_ids[:2]:  # Test on first 2 images
    image_path = evaluator.get_image_path(img_id)
    image = Image.open(image_path).convert('RGB')
    
    preds = loader.predict('detr', image, score_threshold=0.3)
    
    print(f"\nImage {img_id}:")
    print(f"  Boxes shape: {preds['boxes'].shape}")
    print(f"  Labels range: {preds['labels'].min():.0f} - {preds['labels'].max():.0f}")
    print(f"  Labels (first 10): {preds['labels'][:10]}")
    print(f"  Scores (first 10): {preds['scores'][:10]}")
    
    # Convert to COCO format
    coco_preds = evaluator.convert_predictions_to_coco_format(
        preds, img_id, label_offset=0
    )
    print(f"  COCO predictions: {len(coco_preds)}")
    if len(coco_preds) > 0:
        print(f"  Sample COCO pred: {coco_preds[0]}")

# Test 2: Batch pipeline prediction
print("\n" + "="*70)
print("TEST 2: Batch Pipeline predict_batch_huggingface_tensor()")
print("="*70)

# Create batch pipeline
batch_test = BatchOptimizedRobustnessTest(
    model_loader=loader,
    evaluator=evaluator,
    batch_size=2
)

# Prepare a batch
images_batch = []
for img_id in image_ids[:2]:
    image_path = evaluator.get_image_path(img_id)
    image = Image.open(image_path).convert('RGB')
    images_batch.append(image)

# Get processor and model
from model_loader import MODEL_CONFIGS
config = MODEL_CONFIGS['detr']
processor = batch_test.model_loader.processors[config['hf_model_id']]
model = batch_test.model_loader.models['detr']

# Process batch
inputs = processor(images=images_batch, return_tensors="pt", padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

print(f"\nInput tensor shape: {inputs['pixel_values'].shape}")

with torch.no_grad():
    outputs = model(**inputs)

# Post-process
target_sizes = torch.tensor([img.size[::-1] for img in images_batch]).to(device)
print(f"Target sizes: {target_sizes}")

results = processor.post_process_object_detection(
    outputs, 
    target_sizes=target_sizes,
    threshold=0.3
)

print(f"\nBatch predictions:")
for i, result in enumerate(results):
    print(f"\nImage {i} (ID {image_ids[i]}):")
    print(f"  Boxes shape: {result['boxes'].shape}")
    print(f"  Labels shape: {result['labels'].shape}")
    print(f"  Labels (raw): {result['labels'][:10]}")
    print(f"  Labels range: {result['labels'].min():.0f} - {result['labels'].max():.0f}")
    print(f"  Scores (first 10): {result['scores'][:10]}")
    
    # Check if labels need mapping
    labels_np = result['labels'].cpu().numpy()
    print(f"  Unique labels: {sorted(np.unique(labels_np).astype(int))[:20]}")

print("\n" + "="*70)
print("LABEL MAPPING CHECK")
print("="*70)

# Check what labels DETR actually outputs
print("\nDETR should output COCO category IDs (1-90 with gaps)")
print("RT-DETR outputs contiguous 0-79 which needs mapping")

from evaluator import format_coco_label_mapping, get_rtdetr_to_coco_mapping

coco_labels = format_coco_label_mapping()
print(f"\nValid COCO IDs: {sorted(coco_labels.keys())[:20]}...")

# Check if DETR is outputting 0-79 or 1-90
sample_labels = result['labels'].cpu().numpy()
has_label_91_or_higher = (sample_labels >= 91).any()
has_label_0 = (sample_labels == 0).any()
max_label = sample_labels.max()

print(f"\nDETR output analysis:")
print(f"  Has label 0: {has_label_0}")
print(f"  Max label: {max_label}")
print(f"  Has label >= 91: {has_label_91_or_higher}")

if max_label <= 90 and not has_label_0:
    print("  ✓ Labels appear to be COCO IDs (1-90)")
elif max_label < 80:
    print("  ⚠ Labels might be contiguous 0-79 (needs RT-DETR mapping)")
else:
    print("  ? Unclear label format")

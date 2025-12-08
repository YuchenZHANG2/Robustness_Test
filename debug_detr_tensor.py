"""
Debug the batch prediction tensors to see what's wrong
"""
import torch
from PIL import Image
from model_loader import ModelLoader, MODEL_CONFIGS
from batch_optimized_pipeline import BatchOptimizedRobustnessTest, COCODetectionDataset, collate_fn
from evaluator import COCOEvaluator
from torch.utils.data import DataLoader

# Initialize
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load evaluator
evaluator = COCOEvaluator(
    "/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json",
    "/home/yuchen/YuchenZ/Datasets/coco/val2017"
)

# Get 4 images (one batch)
image_ids = evaluator.get_random_images(n=4)
print(f"Testing on {len(image_ids)} images: {image_ids}\n")

# Initialize model loader
loader = ModelLoader(device=device)
loader.load_model('detr')

# Create batch test
batch_test = BatchOptimizedRobustnessTest(
    model_loader=loader,
    evaluator=evaluator,
    batch_size=4
)

# Create dataset and dataloader
dataset = COCODetectionDataset(evaluator, image_ids)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Get one batch
batch = next(iter(dataloader))
batch_tensor = batch['images']  # (B, C, H, W)
batch_image_ids = batch['image_ids']
transformations = batch['transformations']

print("="*70)
print("BATCH INFO")
print("="*70)
print(f"Batch tensor shape: {batch_tensor.shape}")
print(f"Image IDs: {batch_image_ids}")
print(f"Number of transformations: {len(transformations)}")
print("\nTransformations:")
for i, t in enumerate(transformations):
    print(f"  Image {i}: {t}")

# Get model and processor
config = MODEL_CONFIGS['detr']
model = loader.models['detr']
processor = loader.processors[config['hf_model_id']]

print("\n" + "="*70)
print("METHOD 1: Standard predict() - one image at a time")
print("="*70)

for i, img_id in enumerate(image_ids):
    image_path = evaluator.get_image_path(img_id)
    image = Image.open(image_path).convert('RGB')
    
    preds = loader.predict('detr', image, score_threshold=0.05)
    print(f"\nImage {img_id} (standard method):")
    print(f"  Image size: {image.size}")
    print(f"  Predictions: {len(preds['boxes'])}")
    if len(preds['boxes']) > 0:
        print(f"  Boxes (first 3): {preds['boxes'][:3]}")
        print(f"  Labels (first 10): {preds['labels'][:10]}")
        print(f"  Scores (first 10): {preds['scores'][:10]}")

print("\n" + "="*70)
print("METHOD 2: Batch pipeline predict_batch_huggingface_tensor()")
print("="*70)

# Call the batch prediction
batch_results = batch_test.predict_batch_huggingface_tensor(
    'detr',
    batch_tensor.to(device),
    transformations,
    score_threshold=0.05
)

for i, (preds, img_id) in enumerate(zip(batch_results, batch_image_ids)):
    print(f"\nImage {img_id} (batch method):")
    print(f"  Transformation: {transformations[i]}")
    print(f"  Predictions: {len(preds['boxes'])}")
    if len(preds['boxes']) > 0:
        print(f"  Boxes (first 3): {preds['boxes'][:3]}")
        print(f"  Labels (first 10): {preds['labels'][:10]}")
        print(f"  Scores (first 10): {preds['scores'][:10]}")
    else:
        print("  ⚠️  NO PREDICTIONS!")

print("\n" + "="*70)
print("DETAILED INSPECTION OF BATCH PROCESSING")
print("="*70)

# Manually process the batch to see what's happening
import torchvision.transforms.functional as F

# Convert batch tensor to PIL
pil_images = []
for i in range(batch_tensor.shape[0]):
    img_pil = F.to_pil_image(batch_tensor[i])
    pil_images.append(img_pil)
    print(f"\nConverted PIL image {i}: size={img_pil.size}, mode={img_pil.mode}")

# Process with HF processor
print("\nProcessing with HuggingFace processor...")
inputs = processor(images=pil_images, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

print(f"Processor input tensor shape: {inputs['pixel_values'].shape}")
print(f"Processor pixel_mask shape: {inputs.get('pixel_mask', 'N/A')}")

# Run model
with torch.no_grad():
    outputs = model(**inputs)

print(f"\nModel outputs:")
if isinstance(outputs, dict):
    print(f"  logits shape: {outputs['logits'].shape}")
    print(f"  pred_boxes shape: {outputs['pred_boxes'].shape}")
else:
    print(f"  logits shape: {outputs.logits.shape}")
    print(f"  pred_boxes shape: {outputs.pred_boxes.shape}")

# Post-process each image
print("\nPost-processing each image...")
from types import SimpleNamespace

for i, transform in enumerate(transformations):
    resized_h, resized_w = transform['resized_size']
    target_size = torch.tensor([[resized_h, resized_w]]).to(device)
    
    print(f"\nImage {i}:")
    print(f"  Resized size: ({resized_h}, {resized_w})")
    print(f"  Target size tensor: {target_size}")
    
    if isinstance(outputs, dict):
        single_output = SimpleNamespace(
            logits=outputs['logits'][i:i+1],
            pred_boxes=outputs['pred_boxes'][i:i+1]
        )
    else:
        single_output = SimpleNamespace(
            logits=outputs.logits[i:i+1],
            pred_boxes=outputs.pred_boxes[i:i+1]
        )
    
    processed = processor.post_process_object_detection(
        single_output,
        target_sizes=target_size,
        threshold=0.05
    )[0]
    
    print(f"  Post-processed detections: {len(processed['boxes'])}")
    print(f"  Boxes shape: {processed['boxes'].shape}")
    if len(processed['boxes']) > 0:
        print(f"  First 3 boxes: {processed['boxes'][:3]}")
        print(f"  First 10 labels: {processed['labels'][:10]}")
        print(f"  First 10 scores: {processed['scores'][:10]}")
        
        # Check box coordinates
        boxes_np = processed['boxes'].cpu().numpy()
        print(f"  Box x-range: [{boxes_np[:, 0].min():.1f}, {boxes_np[:, 2].max():.1f}]")
        print(f"  Box y-range: [{boxes_np[:, 1].min():.1f}, {boxes_np[:, 3].max():.1f}]")
        print(f"  Resized image size: ({resized_w}, {resized_h})")
        
        # Apply scaling
        scale = transform['scale']
        boxes_scaled = boxes_np / scale
        print(f"  After scaling by 1/{scale:.4f}:")
        print(f"    Box x-range: [{boxes_scaled[:, 0].min():.1f}, {boxes_scaled[:, 2].max():.1f}]")
        print(f"    Box y-range: [{boxes_scaled[:, 1].min():.1f}, {boxes_scaled[:, 3].max():.1f}]")
        print(f"    Original image size: {transform['original_size']}")

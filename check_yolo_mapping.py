"""
Check if YOLO class mapping is correct
"""
from model_loader import ModelLoader
from evaluator import COCOEvaluator, format_coco_label_mapping, get_rtdetr_to_coco_mapping
from PIL import Image

evaluator = COCOEvaluator(
    '/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json',
    '/home/yuchen/YuchenZ/Datasets/coco/val2017'
)

# Get COCO label mapping
coco_labels = format_coco_label_mapping()
print("COCO label IDs (should have gaps):")
print(sorted(coco_labels.keys())[:30])
print(f"Total: {len(coco_labels)} classes")
print()

# Get RT-DETR mapping (this should be the SAME as what YOLO needs)
rtdetr_to_coco = get_rtdetr_to_coco_mapping()
print("RT-DETR mapping (0-79 to COCO IDs):")
print("First 10:", {i: rtdetr_to_coco[i] for i in range(10)})
print(f"Total: {len(rtdetr_to_coco)} classes")
print()

# Test YOLO predictions
loader = ModelLoader(device='cuda')
loader.load_model('yolov11')

image_ids = evaluator.get_random_images(n=5)
for img_id in image_ids:
    image_path = evaluator.get_image_path(img_id)
    image = Image.open(image_path).convert('RGB')
    
    preds = loader.predict('yolov11', image, score_threshold=0.25)
    
    if len(preds['labels']) > 0:
        print(f"\nImage {img_id}:")
        print(f"  Raw YOLO classes (before +1): {preds['labels'] - 1}")  # Reverse the +1
        print(f"  After +1 (current method): {preds['labels']}")
        print(f"  What they SHOULD be (using RT-DETR mapping):")
        raw_labels = preds['labels'] - 1  # Get back to 0-79
        correct_labels = [rtdetr_to_coco.get(int(l), int(l)) for l in raw_labels]
        print(f"    {correct_labels}")
        print(f"  Difference: {preds['labels'].tolist() != correct_labels}")
        break

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("YOLO uses 0-79 contiguous labels, just like RT-DETR!")
print("We're currently doing: label + 1")
print("We SHOULD do: rtdetr_to_coco[label] mapping")
print("\nExample:")
print(f"  YOLO class 0 (person) -> Currently maps to {0+1} -> Should map to {rtdetr_to_coco[0]}")
print(f"  YOLO class 10 (fire hydrant) -> Currently maps to {10+1} -> Should map to {rtdetr_to_coco[10]}")

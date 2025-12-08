"""
Check YOLO11 raw outputs before filtering
"""
from ultralytics import YOLO
from PIL import Image
from evaluator import COCOEvaluator

evaluator = COCOEvaluator(
    '/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json',
    '/home/yuchen/YuchenZ/Datasets/coco/val2017'
)

# Load model directly
model = YOLO('yolo11n.pt')

# Test on one image
image_ids = evaluator.get_random_images(n=1)
image_path = evaluator.get_image_path(image_ids[0])
image = Image.open(image_path).convert('RGB')

print(f'Image {image_ids[0]}: size={image.size}\n')

# Run inference with different settings
print('Default settings:')
results = model(image, verbose=False)[0]
boxes = results.boxes
print(f'  Total boxes before filtering: {len(boxes)}')
print(f'  Confidence scores: {boxes.conf.cpu().numpy()}')
print(f'  Classes: {boxes.cls.cpu().numpy()}')

print('\nWith lower conf threshold:')
results = model(image, conf=0.01, verbose=False)[0]
boxes = results.boxes
print(f'  Total boxes: {len(boxes)}')
print(f'  Confidence scores: {boxes.conf.cpu().numpy()}')

print('\nWith max_det increased:')
results = model(image, conf=0.01, max_det=300, verbose=False)[0]
boxes = results.boxes
print(f'  Total boxes: {len(boxes)}')
print(f'  Confidence scores: {boxes.conf.cpu().numpy()}')

print('\nWith lower IOU threshold for NMS:')
results = model(image, conf=0.01, iou=0.3, max_det=300, verbose=False)[0]
boxes = results.boxes
print(f'  Total boxes: {len(boxes)}')
print(f'  Confidence scores: {boxes.conf.cpu().numpy()}')

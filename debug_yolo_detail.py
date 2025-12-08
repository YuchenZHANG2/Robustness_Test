"""
Debug YOLO11 predictions in detail
"""
from model_loader import ModelLoader
from evaluator import COCOEvaluator
from PIL import Image
import numpy as np

evaluator = COCOEvaluator(
    '/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json',
    '/home/yuchen/YuchenZ/Datasets/coco/val2017'
)

# Test on a few images
image_ids = evaluator.get_random_images(n=5)
print(f'Testing YOLO11 on {len(image_ids)} images\n')

loader = ModelLoader(device='cuda')
loader.load_model('yolov11')

for img_id in image_ids:
    image_path = evaluator.get_image_path(img_id)
    image = Image.open(image_path).convert('RGB')
    
    print(f'Image {img_id}: size={image.size}')
    
    # Try different thresholds
    for threshold in [0.01, 0.05, 0.1, 0.25, 0.5]:
        preds = loader.predict('yolov11', image, score_threshold=threshold)
        print(f'  Threshold {threshold:.2f}: {len(preds["boxes"])} predictions')
        if len(preds["boxes"]) > 0 and threshold == 0.01:
            print(f'    Score range: [{preds["scores"].min():.3f}, {preds["scores"].max():.3f}]')
            print(f'    Scores: {preds["scores"][:10]}')
            print(f'    Labels: {preds["labels"][:10]}')
            print(f'    Boxes (first 3): {preds["boxes"][:3]}')
    print()

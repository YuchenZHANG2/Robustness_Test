"""
Check what PIL image sizes are created from batch tensor
"""
import torch
from PIL import Image
from evaluator import COCOEvaluator
from batch_optimized_pipeline import COCODetectionDataset, collate_fn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

# Load evaluator
evaluator = COCOEvaluator(
    "/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json",
    "/home/yuchen/YuchenZ/Datasets/coco/val2017"
)

image_ids = evaluator.get_random_images(n=4)
print(f"Testing on {len(image_ids)} images\n")

dataset = COCODetectionDataset(evaluator, image_ids)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)

for batch_data in dataloader:
    batch_tensor = batch_data['images']  # Shape: (B, C, H, W)
    transforms = batch_data['transformations']
    
    print(f"Batch tensor shape: {batch_tensor.shape}")
    print()
    
    for i in range(batch_tensor.shape[0]):
        # Convert to PIL
        img_pil = F.to_pil_image(batch_tensor[i].cpu())
        
        transform = transforms[i]
        
        print(f"Image {i}:")
        print(f"  Original size: {transform['original_size']}")
        print(f"  Resized size: {transform['resized_size']}")
        print(f"  Padding (L,R,T,B): {transform['padding']}")
        print(f"  Scale factor: {transform['scale']:.4f}")
        print(f"  PIL image size (W, H): {img_pil.size}")
        print(f"  Expected padded size: ({transform['resized_size'][0] + transform['padding'][1]}, {transform['resized_size'][1] + transform['padding'][3]})")
        print()
    
    break

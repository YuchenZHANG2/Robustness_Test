"""
Final verification: Compare simple vs batch pipeline for DETR
"""
from model_loader import ModelLoader
from batch_optimized_pipeline import BatchOptimizedRobustnessTest, COCODetectionDataset, collate_fn
from evaluator import COCOEvaluator
from torch.utils.data import DataLoader
from PIL import Image

evaluator = COCOEvaluator(
    '/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json',
    '/home/yuchen/YuchenZ/Datasets/coco/val2017'
)

# Use same set of images for both tests
image_ids = evaluator.get_random_images(n=50)
print(f'Testing DETR on {len(image_ids)} images\n')

loader = ModelLoader(device='cuda')
loader.load_model('detr')

# Method 1: Simple prediction
print('='*70)
print('METHOD 1: Simple Prediction')
print('='*70)
all_preds_simple = []
for img_id in image_ids:
    image_path = evaluator.get_image_path(img_id)
    image = Image.open(image_path).convert('RGB')
    preds = loader.predict('detr', image, score_threshold=0.05)
    coco_preds = evaluator.convert_predictions_to_coco_format(preds, img_id, label_offset=0)
    all_preds_simple.extend(coco_preds)

print(f'Total predictions: {len(all_preds_simple)}')
results_simple = evaluator.evaluate_predictions(all_preds_simple, image_ids)
print(f'mAP: {results_simple["mAP"]:.4f}')

# Method 2: Batch pipeline
print('\n' + '='*70)
print('METHOD 2: Batch Pipeline (FIXED)')
print('='*70)
batch_test = BatchOptimizedRobustnessTest(model_loader=loader, evaluator=evaluator, batch_size=4)
dataset = COCODetectionDataset(evaluator, image_ids)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)

all_preds_batch = []
for batch_data in dataloader:
    batch_results = batch_test.predict_batch_huggingface_tensor(
        'detr', batch_data['images'], batch_data['transformations'], score_threshold=0.05
    )
    for img_id, preds in zip(batch_data['image_ids'], batch_results):
        coco_preds = evaluator.convert_predictions_to_coco_format(preds, int(img_id), label_offset=0)
        all_preds_batch.extend(coco_preds)

print(f'Total predictions: {len(all_preds_batch)}')
results_batch = evaluator.evaluate_predictions(all_preds_batch, image_ids)
print(f'mAP: {results_batch["mAP"]:.4f}')

# Comparison
print('\n' + '='*70)
print('COMPARISON')
print('='*70)
print(f'Simple method:  mAP = {results_simple["mAP"]:.4f}')
print(f'Batch method:   mAP = {results_batch["mAP"]:.4f}')
print(f'Difference:           {abs(results_simple["mAP"] - results_batch["mAP"]):.4f}')

if abs(results_simple['mAP'] - results_batch['mAP']) < 0.05:
    print('\n✓ SUCCESS: Both methods produce similar results!')
else:
    print('\n⚠ WARNING: Methods still differ significantly')

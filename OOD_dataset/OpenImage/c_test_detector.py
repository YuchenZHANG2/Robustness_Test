"""
Test RT-DETR detector on OpenImage Dataset_final images.
Processes images in batches and reports statistics.
Evaluates mAP on COCO classes (category id <= 90).
Evaluates OOD detection on non-COCO classes (category id > 90).
"""
import sys
import os
import json
import time
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Add parent directory to path to import model_loader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model_loader import ModelLoader

# Configuration
DATASET_DIR = Path(__file__).parent / "Dataset_final"
DATA_DIR = DATASET_DIR / "data"
LABELS_FILE = DATASET_DIR / "labels_new.json"  # Use labels_new.json for COCO classes
OUTPUT_FILE = Path(__file__).parent / "detection_results.json"
OOD_EVAL_FILE = Path(__file__).parent / "ood_evaluation.json"
MODEL_KEY = "rt_detr"
CONFIDENCE_THRESHOLD = 0.1
BATCH_SIZE = 8  # Adjust based on your GPU VRAM
MAX_COCO_CATEGORY_ID = 90  # Only evaluate on COCO classes (id <= 90)
TOP_N_OOD_CLASSES = 10  # Number of top OOD classes to analyze (configurable)


def load_dataset_info():
    """Load dataset labels to understand the data."""
    if LABELS_FILE.exists():
        with open(LABELS_FILE, 'r') as f:
            return json.load(f)
    return None


def get_image_paths():
    """Get all image paths from the data directory."""
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_paths = [
        p for p in sorted(DATA_DIR.glob('*'))
        if p.suffix.lower() in image_extensions
    ]
    return image_paths


def process_images_batch(model_loader, model_key, image_paths_batch, filename_to_img_id):
    """
    Process a batch of images.
    For HuggingFace models, we still process one at a time but efficiently.
    """
    batch_results = []
    
    for img_path in image_paths_batch:
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Run inference
            predictions = model_loader.predict(
                model_key=model_key,
                image=image,
                score_threshold=CONFIDENCE_THRESHOLD
            )
            
            # Get the COCO image_id from filename
            img_filename = img_path.name
            coco_image_id = filename_to_img_id.get(img_filename, None)
            
            # Store results
            result = {
                'image_id': coco_image_id if coco_image_id is not None else img_path.stem,
                'image_filename': img_filename,
                'image_path': str(img_path),
                'num_detections': len(predictions['scores']),
                'detections': {
                    'boxes': predictions['boxes'].tolist(),
                    'labels': predictions['labels'].tolist(),
                    'scores': predictions['scores'].tolist()
                }
            }
            batch_results.append(result)
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            batch_results.append({
                'image_id': None,
                'image_filename': img_path.name,
                'image_path': str(img_path),
                'error': str(e)
            })
    
    return batch_results


def convert_to_coco_format(all_results):
    """
    Convert detection results to COCO format for evaluation.
    Filter to only COCO classes (category_id <= 90).
    """
    coco_detections = []
    detection_id = 1
    
    for result in all_results:
        if 'error' in result or result['image_id'] is None:
            continue
        
        image_id = result['image_id']
        boxes = result['detections']['boxes']
        labels = result['detections']['labels']
        scores = result['detections']['scores']
        
        for box, label, score in zip(boxes, labels, scores):
            # Filter to only COCO classes (id <= 90)
            if label <= MAX_COCO_CATEGORY_ID:
                # Convert box format from [x1, y1, x2, y2] to COCO format [x, y, width, height]
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                coco_detections.append({
                    'id': detection_id,
                    'image_id': image_id,
                    'category_id': int(label),
                    'bbox': [float(x1), float(y1), float(width), float(height)],
                    'score': float(score),
                    'area': float(width * height)
                })
                detection_id += 1
    
    return coco_detections


def evaluate_coco_map(gt_file, detection_results):
    """
    Evaluate mAP using COCO evaluation tools.
    Only evaluate on COCO classes (category_id <= 90).
    """

    print(f"\n{'=' * 80}")
    print("COCO EVALUATION (mAP on COCO classes, category_id <= 90)")
    print(f"{'=' * 80}\n")
    
    # Convert detections to COCO format
    coco_detections = convert_to_coco_format(detection_results)
    
    if len(coco_detections) == 0:
        print("No detections found for COCO classes. Cannot evaluate.")
        return None
    
    print(f"Total detections (COCO classes only): {len(coco_detections)}")
    
    # Load ground truth
    coco_gt = COCO(str(gt_file))
    
    # Filter ground truth to only COCO classes
    coco_cat_ids = [cat_id for cat_id in coco_gt.getCatIds() if cat_id <= MAX_COCO_CATEGORY_ID]
    print(f"COCO categories being evaluated: {len(coco_cat_ids)}")
    
    # Load detections
    coco_dt = coco_gt.loadRes(coco_detections)
    
    # Create evaluator
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Set category IDs to evaluate (only COCO classes)
    coco_eval.params.catIds = coco_cat_ids
    
    # Run evaluation
    print("\nRunning COCO evaluation...")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract metrics
    metrics = {
        'mAP': coco_eval.stats[0],
        'mAP_50': coco_eval.stats[1],
        'mAP_75': coco_eval.stats[2],
        'mAP_small': coco_eval.stats[3],
        'mAP_medium': coco_eval.stats[4],
        'mAP_large': coco_eval.stats[5],
        'mAR_1': coco_eval.stats[6],
        'mAR_10': coco_eval.stats[7],
        'mAR_100': coco_eval.stats[8],
        'mAR_small': coco_eval.stats[9],
        'mAR_medium': coco_eval.stats[10],
        'mAR_large': coco_eval.stats[11]
    }
    
    return metrics


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes.
    Boxes are in format [x1, y1, x2, y2].
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def evaluate_ood_detection(gt_file, detection_results, top_n=10, iou_threshold=0.5):
    """
    Evaluate OOD (Out-of-Distribution) detection performance.
    OOD classes are those with category_id > MAX_COCO_CATEGORY_ID.
    
    Returns:
    - General OOD recall
    - Per-class recall for top N most frequent OOD classes
    - Confusion analysis showing which categories the model predicted for OOD objects
    """
    print(f"\n{'=' * 80}")
    print(f"OOD EVALUATION (category_id > {MAX_COCO_CATEGORY_ID})")
    print(f"{'=' * 80}\n")
    
    # Load ground truth
    coco_gt = COCO(str(gt_file))
    
    # Get all OOD categories (id > MAX_COCO_CATEGORY_ID)
    all_cat_ids = coco_gt.getCatIds()
    ood_cat_ids = [cat_id for cat_id in all_cat_ids if cat_id > MAX_COCO_CATEGORY_ID]
    
    print(f"Total OOD categories in dataset: {len(ood_cat_ids)}")
    
    # Get all OOD annotations
    ood_annotations = []
    for cat_id in ood_cat_ids:
        anns = coco_gt.loadAnns(coco_gt.getAnnIds(catIds=[cat_id]))
        ood_annotations.extend(anns)
    
    print(f"Total OOD ground truth annotations: {len(ood_annotations)}")
    
    if len(ood_annotations) == 0:
        print("No OOD annotations found. Skipping OOD evaluation.")
        return None
    
    # Count frequency of each OOD category
    ood_cat_frequency = {}
    for ann in ood_annotations:
        cat_id = ann['category_id']
        ood_cat_frequency[cat_id] = ood_cat_frequency.get(cat_id, 0) + 1
    
    # Rank OOD categories by frequency
    sorted_ood_cats = sorted(ood_cat_frequency.items(), key=lambda x: x[1], reverse=True)
    top_n_ood_cats = sorted_ood_cats[:top_n]
    
    print(f"\nTop {top_n} most frequent OOD categories:")
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_gt.loadCats(coco_gt.getCatIds())}
    for rank, (cat_id, count) in enumerate(top_n_ood_cats, 1):
        cat_name = cat_id_to_name.get(cat_id, f"Unknown_{cat_id}")
        print(f"  {rank}. {cat_name} (ID:{cat_id}): {count} annotations")
    
    # Create mapping from image_id to detections
    img_id_to_detections = {}
    for result in detection_results:
        if 'error' in result or result['image_id'] is None:
            continue
        img_id = result['image_id']
        boxes = result['detections']['boxes']
        labels = result['detections']['labels']
        scores = result['detections']['scores']
        
        detections = []
        for box, label, score in zip(boxes, labels, scores):
            detections.append({
                'box': box,
                'category_id': label,
                'score': score
            })
        img_id_to_detections[img_id] = detections
    
    # Evaluate OOD detection
    total_ood_detected = 0
    ood_gt_count = len(ood_annotations)
    
    # Per-class statistics
    class_detected = {}  # {cat_id: detected_count}
    class_total = {}  # {cat_id: total_count}
    class_confusion = {}  # {cat_id: {predicted_cat_id: count}}
    
    for cat_id in ood_cat_ids:
        class_detected[cat_id] = 0
        class_total[cat_id] = 0
        class_confusion[cat_id] = {}
    
    # Check each OOD ground truth annotation
    for gt_ann in ood_annotations:
        gt_cat_id = gt_ann['category_id']
        gt_img_id = gt_ann['image_id']
        
        # Convert COCO bbox [x, y, width, height] to [x1, y1, x2, y2]
        x, y, w, h = gt_ann['bbox']
        gt_box = [x, y, x + w, y + h]
        
        class_total[gt_cat_id] += 1
        
        # Get detections for this image
        detections = img_id_to_detections.get(gt_img_id, [])
        
        # Find overlapping detections
        overlapping_detections = []
        for det in detections:
            det_box = det['box']
            iou = calculate_iou(gt_box, det_box)
            if iou > iou_threshold:
                overlapping_detections.append({
                    'category_id': det['category_id'],
                    'score': det['score'],
                    'iou': iou
                })
        
        # If any detection overlaps, mark as detected
        if overlapping_detections:
            total_ood_detected += 1
            class_detected[gt_cat_id] += 1
            
            # Use the detection with highest confidence score
            best_detection = max(overlapping_detections, key=lambda x: x['score'])
            predicted_cat_id = best_detection['category_id']
            
            # Record confusion
            if predicted_cat_id not in class_confusion[gt_cat_id]:
                class_confusion[gt_cat_id][predicted_cat_id] = 0
            class_confusion[gt_cat_id][predicted_cat_id] += 1
    
    # Calculate overall OOD recall
    general_ood_recall = total_ood_detected / ood_gt_count if ood_gt_count > 0 else 0.0
    
    print(f"\n{'=' * 80}")
    print("OOD DETECTION RESULTS")
    print(f"{'=' * 80}")
    print(f"\nGeneral OOD Recall: {general_ood_recall:.4f} ({total_ood_detected}/{ood_gt_count})")
    
    # Calculate per-class recall for top N
    top_n_recalls = {}
    print(f"\nTop {top_n} OOD Class Recalls:")
    for rank, (cat_id, total_count) in enumerate(top_n_ood_cats, 1):
        detected_count = class_detected[cat_id]
        recall = detected_count / class_total[cat_id] if class_total[cat_id] > 0 else 0.0
        cat_name = cat_id_to_name.get(cat_id, f"Unknown_{cat_id}")
        top_n_recalls[cat_id] = {
            'category_name': cat_name,
            'recall': recall,
            'detected': detected_count,
            'total': class_total[cat_id]
        }
        print(f"  {rank}. {cat_name} (ID:{cat_id}): {recall:.4f} ({detected_count}/{class_total[cat_id]})")
    
    # Confusion analysis for top N
    top_n_confusion = {}
    print(f"\nTop {top_n} OOD Class Confusion (Predicted Categories):")
    for rank, (cat_id, _) in enumerate(top_n_ood_cats, 1):
        cat_name = cat_id_to_name.get(cat_id, f"Unknown_{cat_id}")
        confusion_dict = class_confusion[cat_id]
        
        # Sort by frequency
        sorted_confusion = sorted(confusion_dict.items(), key=lambda x: x[1], reverse=True)
        
        top_n_confusion[cat_id] = {
            'category_name': cat_name,
            'predicted_categories': {}
        }
        
        print(f"  {rank}. {cat_name} (ID:{cat_id}):")
        if sorted_confusion:
            for pred_cat_id, count in sorted_confusion[:5]:  # Show top 5 predictions
                pred_cat_name = cat_id_to_name.get(pred_cat_id, f"Unknown_{pred_cat_id}")
                percentage = count / class_detected[cat_id] * 100 if class_detected[cat_id] > 0 else 0
                top_n_confusion[cat_id]['predicted_categories'][pred_cat_id] = {
                    'category_name': pred_cat_name,
                    'count': count,
                    'percentage': percentage
                }
                print(f"     -> {pred_cat_name} (ID:{pred_cat_id}): {count} ({percentage:.1f}%)")
            
            # Store all confusion data
            for pred_cat_id, count in sorted_confusion:
                if pred_cat_id not in top_n_confusion[cat_id]['predicted_categories']:
                    pred_cat_name = cat_id_to_name.get(pred_cat_id, f"Unknown_{pred_cat_id}")
                    percentage = count / class_detected[cat_id] * 100 if class_detected[cat_id] > 0 else 0
                    top_n_confusion[cat_id]['predicted_categories'][pred_cat_id] = {
                        'category_name': pred_cat_name,
                        'count': count,
                        'percentage': percentage
                    }
        else:
            print(f"     -> No detections overlapped with this OOD class")
    
    # Prepare results
    ood_results = {
        'iou_threshold': iou_threshold,
        'top_n': top_n,
        'general_ood_recall': general_ood_recall,
        'total_ood_detected': total_ood_detected,
        'total_ood_gt': ood_gt_count,
        'top_n_class_recalls': top_n_recalls,
        'top_n_class_confusion': top_n_confusion
    }
    
    return ood_results


def main():
    print("=" * 80)
    print("RT-DETR Object Detection on OpenImage Dataset")
    print("=" * 80)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load dataset info
    print(f"\nDataset directory: {DATASET_DIR}")
    dataset_info = load_dataset_info()
    
    # Create filename to image_id mapping for COCO evaluation
    filename_to_img_id = {}
    if dataset_info:
        print(f"  Categories: {len(dataset_info.get('categories', []))}")
        print(f"  Images in labels: {len(dataset_info.get('images', []))}")
        
        # Count COCO categories
        coco_cats = [c for c in dataset_info.get('categories', []) if c['id'] <= MAX_COCO_CATEGORY_ID]
        print(f"  COCO categories (id <= {MAX_COCO_CATEGORY_ID}): {len(coco_cats)}")
        
        # Build filename to image_id mapping
        for img in dataset_info.get('images', []):
            filename_to_img_id[img['file_name']] = img['id']
    
    # Get image paths
    image_paths = get_image_paths()
    print(f"\nFound {len(image_paths)} images in {DATA_DIR}")
    
    if len(image_paths) == 0:
        print("No images found! Exiting.")
        return
    
    # Initialize model loader
    print(f"\n{'─' * 80}")
    print("Loading RT-DETR model...")
    print(f"{'─' * 80}")
    
    model_loader = ModelLoader(device=device)
    
    # Load the model with progress
    start_load = time.time()
    model = model_loader.load_model(MODEL_KEY)
    load_time = time.time() - start_load
    
    print(f"Model loaded in {load_time:.2f} seconds")
    print(f"Model: {model_loader.get_model_name(MODEL_KEY)}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Process images
    print(f"\n{'─' * 80}")
    print("Running inference on images...")
    print(f"{'─' * 80}\n")
    
    all_results = []
    total_detections = 0
    start_inference = time.time()
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Processing batches"):
        batch = image_paths[i:i + BATCH_SIZE]
        batch_results = process_images_batch(model_loader, MODEL_KEY, batch, filename_to_img_id)
        
        # Accumulate statistics
        for result in batch_results:
            if 'error' not in result:
                total_detections += result['num_detections']
        
        all_results.extend(batch_results)
    
    inference_time = time.time() - start_inference
    
    # Calculate statistics
    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 80}")
    
    num_successful = sum(1 for r in all_results if 'error' not in r)
    num_errors = len(all_results) - num_successful
    
    print(f"\nProcessing Statistics:")
    print(f"  Successfully processed: {num_successful}/{len(image_paths)} images")
    if num_errors > 0:
        print(f"  Errors: {num_errors}")
    print(f"  Total inference time: {inference_time:.2f} seconds")
    print(f"  Average time per image: {inference_time/len(image_paths):.3f} seconds")
    print(f"  Images per second: {len(image_paths)/inference_time:.2f}")
    
    print(f"\nDetection Statistics:")
    print(f"  Total detections: {total_detections}")
    print(f"  Average detections per image: {total_detections/num_successful:.2f}")
    
    # Detection distribution
    detection_counts = [r['num_detections'] for r in all_results if 'error' not in r]
    if detection_counts:
        print(f"  Min detections in image: {min(detection_counts)}")
        print(f"  Max detections in image: {max(detection_counts)}")
        print(f"  Median detections: {np.median(detection_counts):.1f}")
    
    # Count COCO class detections
    coco_class_detections = sum(
        sum(1 for label in r['detections']['labels'] if label <= MAX_COCO_CATEGORY_ID)
        for r in all_results if 'error' not in r
    )
    print(f"  COCO class detections (id <= {MAX_COCO_CATEGORY_ID}): {coco_class_detections}")
    
    # Show examples
    print(f"\nExample detections (first 5 images):")
    for i, result in enumerate(all_results[:5]):
        if 'error' not in result:
            img_id = result.get('image_filename', result['image_id'])
            print(f"  {i+1}. {img_id}: {result['num_detections']} objects")
            if result['num_detections'] > 0:
                # Show top detection
                scores = result['detections']['scores']
                labels = result['detections']['labels']
                if scores:
                    max_idx = scores.index(max(scores))
                    print(f"     -> Top detection: label={labels[max_idx]}, score={scores[max_idx]:.3f}")
    
    # Save results
    print(f"\nSaving detection results to {OUTPUT_FILE}...")
    output_data = {
        'model': model_loader.get_model_name(MODEL_KEY),
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'num_images': num_successful,
        'total_detections': total_detections,
        'coco_class_detections': coco_class_detections,
        'inference_time': inference_time,
        'results': all_results
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved successfully.")
    
    # Evaluate mAP using COCO tools
    if dataset_info and num_successful > 0:
        metrics = evaluate_coco_map(LABELS_FILE, all_results)
        
        if metrics:
            print(f"\n{'=' * 80}")
            print("COCO METRICS SUMMARY")
            print(f"{'=' * 80}")
            print(f"mAP (COCO classes, id <= {MAX_COCO_CATEGORY_ID}):")
            print(f"  mAP @ IoU=0.50:0.95: {metrics['mAP']:.4f}")
            print(f"  mAP @ IoU=0.50:      {metrics['mAP_50']:.4f}")
            print(f"  mAP @ IoU=0.75:      {metrics['mAP_75']:.4f}")
            print(f"  mAP (small):         {metrics['mAP_small']:.4f}")
            print(f"  mAP (medium):        {metrics['mAP_medium']:.4f}")
            print(f"  mAP (large):         {metrics['mAP_large']:.4f}")
        
        # Evaluate OOD detection
        ood_results = evaluate_ood_detection(LABELS_FILE, all_results, top_n=TOP_N_OOD_CLASSES)
        
        if ood_results:
            # Save OOD evaluation results
            print(f"\nSaving OOD evaluation results to {OOD_EVAL_FILE}...")
            with open(OOD_EVAL_FILE, 'w') as f:
                json.dump(ood_results, f, indent=2)
            print(f"OOD evaluation results saved successfully.")
    
    print(f"\n{'=' * 80}")
    print("Processing complete!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()

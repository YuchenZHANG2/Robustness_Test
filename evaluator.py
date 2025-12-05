"""
Evaluation utilities for object detection models.
Handles COCO dataset evaluation using pycocotools.
"""
import json
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import random


class COCOEvaluator:
    """Evaluate object detection models on COCO dataset."""
    
    def __init__(self, annotation_file, image_dir, filter_classes=None, class_mapping=None):
        """
        Initialize COCO evaluator.
        
        Args:
            annotation_file: Path to COCO annotation JSON file
            image_dir: Directory containing COCO images
            filter_classes: Optional list of class IDs to filter ground truth annotations
                          (e.g., [3] to only evaluate on "person" class in Construction dataset)
            class_mapping: Optional dict to map dataset class IDs to COCO IDs
                         (e.g., {3: 1} to map Construction's person (3) to COCO's person (1))
        """
        self.annotation_file = annotation_file
        self.image_dir = Path(image_dir)
        self.coco_gt = COCO(annotation_file)
        self.filter_classes = filter_classes
        self.class_mapping = class_mapping or {}
        
    def get_random_images(self, n=50):
        """
        Get random image IDs from the dataset.
        
        Args:
            n: Number of images to sample
        
        Returns:
            List of image IDs
        """
        all_img_ids = self.coco_gt.getImgIds()
        return random.sample(all_img_ids, min(n, len(all_img_ids)))
    
    def get_image_path(self, image_id):
        """Get the file path for an image ID."""
        img_info = self.coco_gt.loadImgs([image_id])[0]
        return self.image_dir / img_info['file_name']
    
    def evaluate_predictions(self, predictions, image_ids=None):
        """
        Evaluate predictions using COCO metrics.
        
        Args:
            predictions: List of detection dictionaries with format:
                         {'image_id': int, 'category_id': int, 'bbox': [x, y, w, h], 'score': float}
            image_ids: Optional list of image IDs to evaluate on
        
        Returns:
            Dictionary with mAP and other COCO metrics
        """
        if not predictions:
            return {'mAP': 0.0, 'mAP_50': 0.0, 'mAP_75': 0.0}
        
        # Filter predictions by class if specified
        if self.filter_classes:
            predictions = [p for p in predictions if p['category_id'] in self.filter_classes]
        
        if not predictions:
            return {'mAP': 0.0, 'mAP_50': 0.0, 'mAP_75': 0.0}
        
        # Create results in COCO format
        coco_dt = self.coco_gt.loadRes(predictions)
        
        # Run COCO evaluation
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        
        if image_ids:
            coco_eval.params.imgIds = image_ids
        
        # Filter evaluation by specific classes if specified
        if self.filter_classes:
            coco_eval.params.catIds = self.filter_classes
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract key metrics
        return {
            'mAP': coco_eval.stats[0],  # mAP @ IoU=0.50:0.95
            'mAP_50': coco_eval.stats[1],  # mAP @ IoU=0.50
            'mAP_75': coco_eval.stats[2],  # mAP @ IoU=0.75
            'mAP_small': coco_eval.stats[3],  # mAP for small objects
            'mAP_medium': coco_eval.stats[4],  # mAP for medium objects
            'mAP_large': coco_eval.stats[5],  # mAP for large objects
        }
    
    def convert_predictions_to_coco_format(self, model_predictions, image_id, 
                                          label_offset=0):
        """
        Convert model predictions to COCO format.
        
        Args:
            model_predictions: Dict with 'boxes', 'labels', 'scores' from model
            image_id: COCO image ID
            label_offset: Offset to add to labels (e.g., +1 for torchvision models)
        
        Returns:
            List of COCO-format predictions
        """
        coco_results = []
        
        # Apply inverse class mapping if needed (map COCO IDs back to dataset IDs)
        inverse_mapping = {v: k for k, v in self.class_mapping.items()} if self.class_mapping else {}
        
        boxes = model_predictions['boxes']
        labels = model_predictions['labels']
        scores = model_predictions['scores']
        
        for box, label, score in zip(boxes, labels, scores):
            # Convert from [x1, y1, x2, y2] to [x, y, width, height]
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Apply label offset (e.g., for torchvision models)
            predicted_label = int(label) + label_offset
            
            # Map COCO class ID back to dataset class ID if mapping exists
            # (e.g., model predicts COCO person=1, map back to Construction person=3)
            if inverse_mapping and predicted_label in inverse_mapping:
                category_id = inverse_mapping[predicted_label]
            else:
                category_id = predicted_label
            
            coco_results.append({
                'image_id': int(image_id),
                'category_id': category_id,
                'bbox': [float(x1), float(y1), float(width), float(height)],
                'score': float(score)
            })
        
        return coco_results
    
    def get_category_names(self):
        """Get mapping from category IDs to names."""
        cats = self.coco_gt.loadCats(self.coco_gt.getCatIds())
        return {cat['id']: cat['name'] for cat in cats}


def format_coco_label_mapping():
    """
    Return the COCO category ID to name mapping.
    Torchvision models use labels 1-91 (with gaps), HF models use 1-91.
    """
    # Standard COCO 80 classes with their actual IDs
    coco_names = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
        22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
        28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
        35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
        40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
        44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
        51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
        56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
        61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
        67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
        75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
        80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
        86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
    }
    return coco_names

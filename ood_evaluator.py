"""
OOD (Out-of-Distribution) Evaluator for Object Detection
Evaluates detector performance on OOD classes (category_id > 90)
"""
import json
from pathlib import Path
from pycocotools.coco import COCO
import numpy as np


class OODEvaluator:
    """Evaluates OOD detection performance on OpenImages dataset"""
    
    MAX_COCO_CATEGORY_ID = 90  # COCO classes have id <= 90
    
    def __init__(self, annotation_file, image_dir, top_n_classes=5, iou_threshold=0.5):
        """
        Initialize OOD evaluator
        
        Args:
            annotation_file: Path to labels_new.json (COCO format)
            image_dir: Path to image directory
            top_n_classes: Number of top frequent OOD classes to analyze
            iou_threshold: IoU threshold for matching predictions to ground truth
        """
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.top_n_classes = top_n_classes
        self.iou_threshold = iou_threshold
        
        # Load COCO ground truth
        self.coco_gt = COCO(annotation_file)
        
        # Get OOD categories (id > MAX_COCO_CATEGORY_ID)
        all_cat_ids = self.coco_gt.getCatIds()
        self.ood_cat_ids = [cat_id for cat_id in all_cat_ids if cat_id > self.MAX_COCO_CATEGORY_ID]
        
        # Category name mapping
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.coco_gt.loadCats(self.coco_gt.getCatIds())}
        
        print(f"OOD Evaluator initialized: {len(self.ood_cat_ids)} OOD categories")
    
    def get_image_path(self, image_id):
        """Get absolute path to an image by its ID"""
        image_info = self.coco_gt.loadImgs(image_id)[0]
        return str(Path(self.image_dir) / image_info['file_name'])
    
    def get_all_ood_images(self):
        """Get all image IDs that contain OOD annotations"""
        ood_img_ids = set()
        for cat_id in self.ood_cat_ids:
            ann_ids = self.coco_gt.getAnnIds(catIds=[cat_id])
            anns = self.coco_gt.loadAnns(ann_ids)
            for ann in anns:
                ood_img_ids.add(ann['image_id'])
        return list(ood_img_ids)
    
    @staticmethod
    def calculate_iou(box1, box2):
        """
        Calculate IoU between two boxes
        Boxes are in format [x1, y1, x2, y2]
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
    
    def evaluate_ood(self, model_predictions):
        """
        Evaluate OOD detection performance
        
        Args:
            model_predictions: Dict of {image_id: {'boxes': [...], 'labels': [...], 'scores': [...]}}
        
        Returns:
            Dictionary containing:
            - general_ood_recall: Overall recall on all OOD classes
            - top_n_classes: List of top N most frequent OOD classes with their stats
            - class_recalls: Dict of {class_id: recall} for top N classes
            - class_confusion: Dict of {class_id: {predicted_class_id: count}} for top N classes
        """
        # Get all OOD annotations
        ood_annotations = []
        for cat_id in self.ood_cat_ids:
            anns = self.coco_gt.loadAnns(self.coco_gt.getAnnIds(catIds=[cat_id]))
            ood_annotations.extend(anns)
        
        if len(ood_annotations) == 0:
            print("No OOD annotations found")
            return None
        
        # Count frequency of each OOD category
        ood_cat_frequency = {}
        for ann in ood_annotations:
            cat_id = ann['category_id']
            ood_cat_frequency[cat_id] = ood_cat_frequency.get(cat_id, 0) + 1
        
        # Get top N most frequent OOD categories
        sorted_ood_cats = sorted(ood_cat_frequency.items(), key=lambda x: x[1], reverse=True)
        top_n_ood_cats = sorted_ood_cats[:self.top_n_classes]
        
        # Initialize statistics
        class_detected = {cat_id: 0 for cat_id in self.ood_cat_ids}
        class_total = {cat_id: 0 for cat_id in self.ood_cat_ids}
        class_confusion = {cat_id: {} for cat_id in self.ood_cat_ids}
        
        total_ood_detected = 0
        ood_gt_count = len(ood_annotations)
        
        # Evaluate each ground truth annotation
        for gt_ann in ood_annotations:
            gt_cat_id = gt_ann['category_id']
            gt_img_id = gt_ann['image_id']
            
            # Convert COCO bbox [x, y, width, height] to [x1, y1, x2, y2]
            x, y, w, h = gt_ann['bbox']
            gt_box = [x, y, x + w, y + h]
            
            class_total[gt_cat_id] += 1
            
            # Get predictions for this image
            if gt_img_id not in model_predictions:
                continue
            
            preds = model_predictions[gt_img_id]
            boxes = preds['boxes']
            labels = preds['labels']
            scores = preds['scores']
            
            # Find overlapping detections
            overlapping_detections = []
            for box, label, score in zip(boxes, labels, scores):
                iou = self.calculate_iou(gt_box, box)
                if iou > self.iou_threshold:
                    overlapping_detections.append({
                        'category_id': label,
                        'score': score,
                        'iou': iou
                    })
            
            # If any detection overlaps, mark as detected
            if overlapping_detections:
                total_ood_detected += 1
                class_detected[gt_cat_id] += 1
                
                # Use detection with highest confidence score
                best_detection = max(overlapping_detections, key=lambda x: x['score'])
                predicted_cat_id = best_detection['category_id']
                
                # Record confusion
                if predicted_cat_id not in class_confusion[gt_cat_id]:
                    class_confusion[gt_cat_id][predicted_cat_id] = 0
                class_confusion[gt_cat_id][predicted_cat_id] += 1
        
        # Calculate overall OOD recall
        general_ood_recall = total_ood_detected / ood_gt_count if ood_gt_count > 0 else 0.0
        
        # Build results for top N classes
        top_n_results = []
        class_recalls = {}
        class_confusion_cleaned = {}
        
        for cat_id, total_count in top_n_ood_cats:
            detected_count = class_detected[cat_id]
            recall = detected_count / class_total[cat_id] if class_total[cat_id] > 0 else 0.0
            cat_name = self.cat_id_to_name.get(cat_id, f"Unknown_{cat_id}")
            
            class_recalls[cat_id] = recall
            
            # Get confusion data
            confusion_dict = class_confusion[cat_id]
            sorted_confusion = sorted(confusion_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Build confusion with category names
            confusion_with_names = {}
            for pred_cat_id, count in sorted_confusion:
                pred_cat_name = self.cat_id_to_name.get(pred_cat_id, f"Unknown_{pred_cat_id}")
                percentage = count / detected_count * 100 if detected_count > 0 else 0
                confusion_with_names[int(pred_cat_id)] = {
                    'category_name': pred_cat_name,
                    'count': int(count),
                    'percentage': float(percentage)
                }
            
            class_confusion_cleaned[cat_id] = confusion_with_names
            
            top_n_results.append({
                'category_id': int(cat_id),
                'category_name': cat_name,
                'total_annotations': int(total_count),
                'detected': int(detected_count),
                'recall': float(recall),
                'confusion': confusion_with_names
            })
        
        return {
            'iou_threshold': self.iou_threshold,
            'top_n': self.top_n_classes,
            'general_ood_recall': float(general_ood_recall),
            'total_ood_detected': int(total_ood_detected),
            'total_ood_gt': int(ood_gt_count),
            'top_n_classes': top_n_results,
            'class_recalls': {int(k): float(v) for k, v in class_recalls.items()},
            'class_confusion': {int(k): v for k, v in class_confusion_cleaned.items()}
        }

"""
Optimized batch testing pipeline using GPU-accelerated corruptions.
Processes multiple images in parallel for significant speedup.
"""
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm

# Fix numpy deprecations
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128

from torch_corruptions import TorchCorruptions


class BatchRobustnessTest:
    """
    GPU-accelerated robustness testing with batch processing.
    """
    
    def __init__(self, model_loader, evaluator, batch_size=8, device='cuda'):
        """
        Initialize batch testing pipeline.
        
        Args:
            model_loader: ModelLoader instance
            evaluator: COCOEvaluator instance
            batch_size: Number of images to process simultaneously
            device: Device to use for corruptions ('cuda' or 'cpu')
        """
        self.model_loader = model_loader
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.results = {}
        
        # Initialize GPU-accelerated corruptions
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = 'cpu'
        
        self.device = device
        self.corruptor = TorchCorruptions(device=device)
        print(f"BatchRobustnessTest: Using {device} for corruptions with batch_size={batch_size}")
    
    def load_images_batch(self, image_ids):
        """
        Load a batch of images.
        
        Args:
            image_ids: List of COCO image IDs
        
        Returns:
            List of PIL Images and their paths
        """
        images = []
        paths = []
        
        for image_id in image_ids:
            img_path = self.evaluator.get_image_path(image_id)
            image = Image.open(img_path).convert('RGB')
            images.append(image)
            paths.append(img_path)
        
        return images, paths
    
    def test_model_on_clean_images_batch(self, model_key, image_ids, 
                                         progress_callback=None):
        """
        Test a model on clean images with batch processing.
        
        Args:
            model_key: Model configuration key
            image_ids: List of COCO image IDs to test
            progress_callback: Optional function(current, total, message)
        
        Returns:
            Dictionary with evaluation metrics
        """
        all_predictions = []
        num_batches = (len(image_ids) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(image_ids))
            batch_ids = image_ids[start_idx:end_idx]
            
            if progress_callback:
                progress_callback(end_idx, len(image_ids),
                                f"Processing batch {batch_idx + 1}/{num_batches}")
            
            # Load batch of images
            images, _ = self.load_images_batch(batch_ids)
            
            # Get predictions for each image in batch
            for image, image_id in zip(images, batch_ids):
                preds = self.model_loader.predict(model_key, image, score_threshold=0.05)
                
                # Convert to COCO format
                label_offset = 0
                coco_preds = self.evaluator.convert_predictions_to_coco_format(
                    preds, image_id, label_offset=label_offset
                )
                all_predictions.extend(coco_preds)
        
        # Evaluate
        metrics = self.evaluator.evaluate_predictions(all_predictions, image_ids)
        
        return {
            'predictions': all_predictions,
            'metrics': metrics
        }
    
    def test_model_with_corruption_batch(self, model_key, image_ids, corruption_name,
                                         severity, progress_callback=None):
        """
        Test a model on corrupted images with GPU-accelerated batch corruption.
        
        Args:
            model_key: Model configuration key
            image_ids: List of COCO image IDs to test
            corruption_name: Name of corruption to apply
            severity: Corruption severity (1-5)
            progress_callback: Optional function(current, total, message)
        
        Returns:
            Dictionary with evaluation metrics
        """
        all_predictions = []
        num_batches = (len(image_ids) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(image_ids))
            batch_ids = image_ids[start_idx:end_idx]
            
            if progress_callback:
                progress_callback(end_idx, len(image_ids),
                                f"Processing {corruption_name} (severity {severity}) - "
                                f"batch {batch_idx + 1}/{num_batches}")
            
            # Load batch of images
            images, _ = self.load_images_batch(batch_ids)
            
            # Convert to numpy arrays for batch corruption
            images_np = [np.array(img) for img in images]
            
            # Find max dimensions for padding
            max_h = max(img.shape[0] for img in images_np)
            max_w = max(img.shape[1] for img in images_np)
            
            # Pad images to same size for batch processing
            padded_images = []
            original_sizes = []
            
            for img in images_np:
                h, w = img.shape[:2]
                original_sizes.append((h, w))
                
                if h < max_h or w < max_w:
                    # Pad to max size
                    padded = np.zeros((max_h, max_w, 3), dtype=img.dtype)
                    padded[:h, :w] = img
                    padded_images.append(padded)
                else:
                    padded_images.append(img)
            
            # Stack into batch and apply corruption on GPU
            batch_array = np.stack(padded_images)
            
            # Apply corruption to entire batch at once (GPU accelerated)
            corrupted_batch = self.corruptor.corrupt(batch_array, 
                                                     corruption_name=corruption_name,
                                                     severity=severity)
            
            # Get predictions for each corrupted image
            for idx, (image_id, (orig_h, orig_w)) in enumerate(zip(batch_ids, original_sizes)):
                # Crop back to original size
                corrupted_img = corrupted_batch[idx, :orig_h, :orig_w]
                
                # Get predictions
                preds = self.model_loader.predict(model_key, corrupted_img, 
                                                 score_threshold=0.05)
                
                # Convert to COCO format
                label_offset = 0
                coco_preds = self.evaluator.convert_predictions_to_coco_format(
                    preds, image_id, label_offset=label_offset
                )
                all_predictions.extend(coco_preds)
        
        # Evaluate
        metrics = self.evaluator.evaluate_predictions(all_predictions, image_ids)
        
        return {
            'predictions': all_predictions,
            'metrics': metrics
        }
    
    def run_full_test(self, model_keys, corruption_names, image_ids,
                     severities=[1, 2, 3, 4, 5], progress_callback=None):
        """
        Run comprehensive robustness test with batch processing.
        
        Args:
            model_keys: List of model configuration keys
            corruption_names: List of corruption names to test
            image_ids: List of COCO image IDs
            severities: List of severity levels to test
            progress_callback: Optional function(current, total, message)
        
        Returns:
            Nested dictionary with all results
        """
        results = {}
        
        total_tests = len(model_keys) * (1 + len(corruption_names) * len(severities))
        current_test = 0
        
        for model_key in model_keys:
            model_name = self.model_loader.get_model_name(model_key)
            results[model_key] = {
                'name': model_name,
                'clean': {},
                'corrupted': {}
            }
            
            # Test on clean images
            current_test += 1
            if progress_callback:
                progress_callback(current_test, total_tests,
                                f"Testing {model_name} on clean images...")
            
            clean_results = self.test_model_on_clean_images_batch(model_key, image_ids)
            results[model_key]['clean'] = clean_results['metrics']
            
            # Test on corrupted images
            for corruption_name in corruption_names:
                if corruption_name not in results[model_key]['corrupted']:
                    results[model_key]['corrupted'][corruption_name] = {}
                
                for severity in severities:
                    current_test += 1
                    
                    if progress_callback:
                        progress_callback(current_test, total_tests,
                                        f"Testing {model_name} with {corruption_name} "
                                        f"(severity {severity})...")
                    
                    print(f"\n=== Batch processing: {model_name} - {corruption_name} - severity {severity} ===")
                    
                    # Use batch corruption processing
                    corrupt_results = self.test_model_with_corruption_batch(
                        model_key, image_ids, corruption_name, severity,
                        progress_callback=None
                    )
                    
                    print(f"=== Finished batch: {model_name} - {corruption_name} - severity {severity} ===\n")
                    
                    results[model_key]['corrupted'][corruption_name][severity] = \
                        corrupt_results['metrics']
        
        self.results = results
        return results
    
    def save_results(self, output_path):
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def get_summary_table(self):
        """
        Generate a summary table of mAP values.
        
        Returns:
            Dictionary suitable for display
        """
        summary = {}
        
        for model_key, model_results in self.results.items():
            model_name = model_results['name']
            summary[model_name] = {
                'Clean': model_results['clean'].get('mAP', 0.0)
            }
            
            # Average mAP across all corruptions and severities
            all_corrupt_maps = []
            for corruption, severity_results in model_results['corrupted'].items():
                for severity, metrics in severity_results.items():
                    all_corrupt_maps.append(metrics.get('mAP', 0.0))
            
            if all_corrupt_maps:
                summary[model_name]['Corrupted (avg)'] = np.mean(all_corrupt_maps)
                summary[model_name]['Degradation'] = \
                    summary[model_name]['Clean'] - summary[model_name]['Corrupted (avg)']
        
        return summary


if __name__ == '__main__':
    print("Batch testing pipeline module loaded successfully")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

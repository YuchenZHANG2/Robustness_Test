"""
Testing pipeline for evaluating detector robustness across corruptions.
"""
import numpy as np
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm

# Fix numpy deprecations
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128

# Patch skimage for compatibility
import skimage.filters
_orig_gaussian = skimage.filters.gaussian

def _patched_gaussian(*args, **kwargs):
    if "multichannel" in kwargs:
        channel = kwargs.pop("multichannel")
        kwargs.setdefault("channel_axis", -1 if channel else None)
    return _orig_gaussian(*args, **kwargs)

skimage.filters.gaussian = _patched_gaussian

import torch
from torch_corruptions import TorchCorruptions


class RobustnessTest:
    """
    Manage robustness testing across multiple models, corruptions, and severity levels.
    """
    
    def __init__(self, model_loader, evaluator):
        """
        Initialize testing pipeline.
        
        Args:
            model_loader: ModelLoader instance
            evaluator: COCOEvaluator instance
        """
        self.model_loader = model_loader
        self.evaluator = evaluator
        self.results = {}
        
        # Initialize PyTorch corruptions for GPU acceleration
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.corruptor = TorchCorruptions(device=device)
        print(f"RobustnessTest: Using {device} for corruptions")
        
    def test_model_on_clean_images(self, model_key, image_ids, 
                                   progress_callback=None):
        """
        Test a model on clean (uncorrupted) images.
        
        Args:
            model_key: Model configuration key
            image_ids: List of COCO image IDs to test
            progress_callback: Optional function(current, total, message)
        
        Returns:
            Dictionary with evaluation metrics
        """
        all_predictions = []
        
        for idx, image_id in enumerate(image_ids):
            if progress_callback:
                progress_callback(idx + 1, len(image_ids), 
                                f"Processing image {idx + 1}/{len(image_ids)}")
            
            # Load image
            img_path = self.evaluator.get_image_path(image_id)
            image = Image.open(img_path).convert('RGB')
            
            # Get predictions
            preds = self.model_loader.predict(model_key, image, score_threshold=0.05)
            
            # Determine label offset (torchvision models need no offset, they use COCO IDs)
            label_offset = 0
            
            # Convert to COCO format
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
    
    def test_model_with_corruption(self, model_key, image_ids, corruption_name, 
                                   severity, progress_callback=None):
        """
        Test a model on corrupted images.
        
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
        
        for idx, image_id in enumerate(image_ids):
            if progress_callback:
                progress_callback(idx + 1, len(image_ids),
                                f"Processing {corruption_name} (severity {severity}) - "
                                f"image {idx + 1}/{len(image_ids)}")
            
            # Load and corrupt image
            img_path = self.evaluator.get_image_path(image_id)
            image = np.array(Image.open(img_path).convert('RGB'))
            corrupted_image = self.corruptor.corrupt(image, corruption_name=corruption_name, 
                                                     severity=severity)
            
            # Get predictions
            preds = self.model_loader.predict(model_key, corrupted_image, 
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
        Run comprehensive robustness test.
        
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
            
            clean_results = self.test_model_on_clean_images(model_key, image_ids)
            results[model_key]['clean'] = clean_results['metrics']
            
            # Test on corrupted images
            for corruption_name in corruption_names:
                if corruption_name not in results[model_key]['corrupted']:
                    results[model_key]['corrupted'][corruption_name] = {}
                
                for severity in severities:
                    current_test += 1
                    
                    # Show starting message
                    if progress_callback:
                        progress_callback(current_test, total_tests,
                                        f"Testing {model_name} with {corruption_name} "
                                        f"(severity {severity})...")
                    
                    print(f"\n=== Starting test: {model_name} - {corruption_name} - severity {severity} ===")
                    
                    # Don't pass progress_callback to the inner function
                    corrupt_results = self.test_model_with_corruption(
                        model_key, image_ids, corruption_name, severity,
                        progress_callback=None  # Don't use inner callback
                    )
                    
                    print(f"=== Finished test: {model_name} - {corruption_name} - severity {severity} ===\n")
                    
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

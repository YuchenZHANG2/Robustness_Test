"""
Optimized batch-based testing pipeline with parallel corruption and inference.
Uses PyTorch DataLoader and batch processing for maximum performance.
"""
import numpy as np
from PIL import Image
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Fix numpy deprecations
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128

from torch_corruptions import TorchCorruptions


class COCODetectionDataset(Dataset):
    """Dataset for loading COCO images (corruption applied later in batch)."""
    
    def __init__(self, evaluator, image_ids):
        """
        Initialize dataset.
        
        Args:
            evaluator: COCOEvaluator instance
            image_ids: List of COCO image IDs
        """
        self.evaluator = evaluator
        self.image_ids = image_ids
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image only - corruption happens in batch on GPU
        img_path = self.evaluator.get_image_path(image_id)
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        return {
            'image': image,
            'image_id': image_id,
            'original_size': original_size
        }
def collate_fn(batch):
    """Custom collate function that keeps images as PIL for model input."""
    return {
        'images': [item['image'] for item in batch],
        'image_ids': [item['image_id'] for item in batch],
        'original_sizes': [item['original_size'] for item in batch]
    }


class BatchOptimizedRobustnessTest:
    """
    Optimized robustness testing with batch processing and parallel execution.
    Uses CPU for corruptions in workers, GPU only for model inference.
    """
    
    def __init__(self, model_loader, evaluator, batch_size=4, num_workers=2):
        """
        Initialize optimized testing pipeline.
        
        Args:
            model_loader: ModelLoader instance
            evaluator: COCOEvaluator instance
            batch_size: Number of images to process in parallel (default: 4 for single GPU)
            num_workers: Number of DataLoader worker processes (default: 2)
        """
        self.model_loader = model_loader
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.results = {}
        
        # GPU is used for both corruption and model inference
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.corruptor = TorchCorruptions(device=self.device)
        
        print(f"BatchOptimizedRobustnessTest: Using {self.device}")
        print(f"Batch size: {batch_size}, Workers: {num_workers}")
        print(f"GPU batch processing: Corruptions + Model inference")
    
    def apply_batch_corruption(self, images: List[Image.Image], 
                               corruption_name: str, severity: int) -> List[Image.Image]:
        """
        Apply corruption to a batch of images on GPU.
        
        Args:
            images: List of PIL Images
            corruption_name: Name of corruption to apply
            severity: Severity level (1-5)
        
        Returns:
            List of corrupted PIL Images
        """
        from torchvision import transforms
        
        # Convert batch to tensor (B, C, H, W)
        tensors = torch.stack([transforms.ToTensor()(img) for img in images])
        tensors = tensors.to(self.device)
        
        # Apply corruption to entire batch at once
        corrupted_tensors = self.corruptor.corrupt(
            tensors,
            corruption_name=corruption_name,
            severity=severity,
            return_tensor=True
        )
        
        # Convert back to PIL images
        corrupted_images = []
        for i in range(corrupted_tensors.shape[0]):
            img = self.corruptor.to_pil(corrupted_tensors[i])
            corrupted_images.append(img)
        
        return corrupted_images
    
    def predict_batch_torchvision(self, model, images: List[Image.Image], 
                                   score_threshold: float = 0.05) -> List[Dict]:
        """
        Batch prediction for torchvision models.
        
        Args:
            model: Torchvision detection model
            images: List of PIL Images
            score_threshold: Minimum confidence score
        
        Returns:
            List of predictions (one dict per image)
        """
        model.eval()
        
        # Convert images to tensors
        tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            tensor = F.to_tensor(img)
            tensors.append(tensor)
        
        # Move to device
        tensors = [t.to(self.device) for t in tensors]
        
        # Batch inference
        with torch.no_grad():
            outputs = model(tensors)
        
        # Filter by score threshold
        results = []
        for output in outputs:
            mask = output['scores'] > score_threshold
            results.append({
                'boxes': output['boxes'][mask].cpu().numpy(),
                'labels': output['labels'][mask].cpu().numpy(),
                'scores': output['scores'][mask].cpu().numpy()
            })
        
        return results
    
    def predict_batch_huggingface(self, model_key, images: List[Image.Image],
                                   score_threshold: float = 0.05) -> List[Dict]:
        """
        Batch prediction for Hugging Face models.
        
        Args:
            model_key: Model configuration key
            images: List of PIL Images
            score_threshold: Minimum confidence score
        
        Returns:
            List of predictions (one dict per image)
        """
        from model_loader import MODEL_CONFIGS
        
        config = MODEL_CONFIGS[model_key]
        model = self.model_loader.models[model_key]
        processor = self.model_loader.processors[config['hf_model_id']]
        
        # Convert numpy arrays to PIL if needed
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            pil_images.append(img)
        
        # Batch process images
        inputs = processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process each image separately
        results = []
        for i, img in enumerate(pil_images):
            target_size = torch.tensor([img.size[::-1]]).to(self.device)
            
            # Extract outputs for this image
            single_output = {
                'logits': outputs.logits[i:i+1],
                'pred_boxes': outputs.pred_boxes[i:i+1]
            }
            
            processed = processor.post_process_object_detection(
                single_output,
                target_sizes=target_size,
                threshold=score_threshold
            )[0]
            
            results.append({
                'boxes': processed['boxes'].cpu().numpy(),
                'labels': processed['labels'].cpu().numpy(),
                'scores': processed['scores'].cpu().numpy()
            })
        
        return results
    
    def predict_batch(self, model_key, images: List[Image.Image],
                      score_threshold: float = 0.05) -> List[Dict]:
        """
        Universal batch prediction method.
        
        Args:
            model_key: Model configuration key
            images: List of PIL Images
            score_threshold: Minimum confidence score
        
        Returns:
            List of predictions (one dict per image)
        """
        from model_loader import MODEL_CONFIGS
        
        config = MODEL_CONFIGS[model_key]
        model = self.model_loader.models[model_key]
        
        if config['type'] == 'torchvision':
            return self.predict_batch_torchvision(model, images, score_threshold)
        elif config['type'] == 'huggingface':
            return self.predict_batch_huggingface(model_key, images, score_threshold)
        else:
            raise ValueError(f"Unknown model type: {config['type']}")
    
    def test_model_batch(self, model_key, image_ids, corruption_name=None,
                        severity=None, progress_callback=None) -> Dict:
        """
        Test model with batch processing and optional corruption.
        
        Args:
            model_key: Model configuration key
            image_ids: List of COCO image IDs
            corruption_name: Optional corruption to apply
            severity: Optional corruption severity (1-5)
            progress_callback: Optional callback(current, total, message)
        
        Returns:
            Dictionary with predictions and metrics
        """
        # Create dataset (just loads images)
        dataset = COCODetectionDataset(self.evaluator, image_ids)
        
        # Create DataLoader with multiple workers for parallel loading
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True if self.device == 'cuda' else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        all_predictions = []
        total_batches = len(dataloader)
        
        # Process in batches
        for batch_idx, batch in enumerate(dataloader):
            if progress_callback:
                current = (batch_idx + 1) * self.batch_size
                total = len(image_ids)
                msg_suffix = f" ({corruption_name} sev{severity})" if corruption_name else " (clean)"
                progress_callback(
                    min(current, total), 
                    total,
                    f"Processing batch {batch_idx + 1}/{total_batches}{msg_suffix}"
                )
            
            # Apply corruption to entire batch on GPU if needed
            images = batch['images']
            if corruption_name:
                images = self.apply_batch_corruption(images, corruption_name, severity)
            
            # Batch prediction
            batch_preds = self.predict_batch(
                model_key, 
                images,
                score_threshold=0.05
            )
            
            # Convert to COCO format
            for preds, image_id in zip(batch_preds, batch['image_ids']):
                coco_preds = self.evaluator.convert_predictions_to_coco_format(
                    preds, image_id, label_offset=0
                )
                all_predictions.extend(coco_preds)
        
        # Evaluate
        metrics = self.evaluator.evaluate_predictions(all_predictions, image_ids)
        
        return {
            'predictions': all_predictions,
            'metrics': metrics
        }
    
    def run_full_test(self, model_keys, corruption_names, image_ids,
                     severities=[1, 2, 3, 4, 5], progress_callback=None) -> Dict:
        """
        Run comprehensive robustness test with batch processing.
        
        Args:
            model_keys: List of model configuration keys
            corruption_names: List of corruption names
            image_ids: List of COCO image IDs
            severities: List of severity levels
            progress_callback: Optional callback(current, total, message)
        
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
            
            print(f"\n{'='*60}")
            print(f"Testing {model_name}")
            print(f"{'='*60}")
            
            # Test on clean images
            current_test += 1
            if progress_callback:
                progress_callback(current_test, total_tests,
                                f"Testing {model_name} on clean images...")
            
            print(f"\nProcessing clean images (batch mode)...")
            clean_results = self.test_model_batch(
                model_key, 
                image_ids,
                corruption_name=None,
                severity=None
            )
            results[model_key]['clean'] = clean_results['metrics']
            print(f"Clean mAP: {clean_results['metrics']['mAP']:.4f}")
            
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
                    
                    print(f"\nProcessing {corruption_name} (severity {severity}) - batch mode...")
                    
                    corrupt_results = self.test_model_batch(
                        model_key,
                        image_ids,
                        corruption_name=corruption_name,
                        severity=severity
                    )
                    
                    results[model_key]['corrupted'][corruption_name][severity] = \
                        corrupt_results['metrics']
                    
                    print(f"{corruption_name} (sev {severity}) mAP: "
                          f"{corrupt_results['metrics']['mAP']:.4f}")
        
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

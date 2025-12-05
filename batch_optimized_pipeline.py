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
    """Dataset for loading COCO images with annotations."""
    
    def __init__(self, evaluator, image_ids):
        """
        Initialize dataset.
        
        Args:
            evaluator: COCOEvaluator instance
            image_ids: List of COCO image IDs
        """
        self.image_dir = evaluator.image_dir
        self.image_ids = image_ids
        
        # Cache image info to avoid COCO object access in workers
        self.image_info = {}
        for img_id in image_ids:
            img_data = evaluator.coco_gt.loadImgs([img_id])[0]
            self.image_info[img_id] = img_data['file_name']
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image using cached info
        img_path = self.image_dir / self.image_info[image_id]
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        return {
            'image': image,
            'image_id': image_id,
            'original_size': original_size
        }


def collate_fn(batch):
    """
    Custom collate function that resizes to largest image and pads.
    Returns tensors ready for batch corruption and tracking of transformations.
    """
    from torchvision import transforms
    
    images = [item['image'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    
    # Find largest dimensions in batch
    max_width = max(img.size[0] for img in images)
    max_height = max(img.size[1] for img in images)
    
    # Convert to tensors and track transformations
    tensors = []
    transformations = []
    
    for img, orig_size in zip(images, original_sizes):
        orig_w, orig_h = orig_size
        
        # Calculate scaling (maintain aspect ratio to fit in max dimensions)
        scale = min(max_width / orig_w, max_height / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Resize image
        resized_img = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Convert to tensor
        tensor = transforms.ToTensor()(resized_img)  # (C, H, W)
        
        # Pad to max dimensions (pad right and bottom)
        pad_right = max_width - new_w
        pad_bottom = max_height - new_h
        
        # Pad: (left, right, top, bottom)
        padded_tensor = transforms.Pad((0, 0, pad_right, pad_bottom), fill=0)(tensor)
        
        tensors.append(padded_tensor)
        transformations.append({
            'original_size': orig_size,
            'scale': scale,
            'resized_size': (new_w, new_h),
            'padding': (0, pad_right, 0, pad_bottom)  # left, right, top, bottom
        })
    
    # Stack into batch tensor (B, C, H, W)
    batch_tensor = torch.stack(tensors)
    
    return {
        'images': batch_tensor,  # (B, C, H, W) - padded and ready for corruption
        'image_ids': image_ids,
        'original_sizes': original_sizes,
        'transformations': transformations
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
        print(f"Batch-first approach: Corrupt once, test all models")
    
    def apply_batch_corruption(self, batch_tensor: torch.Tensor, 
                               corruption_name: str, severity: int) -> torch.Tensor:
        """
        Apply corruption to a batch tensor on GPU.
        
        Args:
            batch_tensor: Tensor of shape (B, C, H, W)
            corruption_name: Name of corruption to apply
            severity: Severity level (1-5)
        
        Returns:
            Corrupted tensor of same shape
        """
        batch_tensor = batch_tensor.to(self.device)
        
        # Apply corruption to entire batch at once
        corrupted_tensor = self.corruptor.corrupt(
            batch_tensor,
            corruption_name=corruption_name,
            severity=severity,
            return_tensor=True
        )
        
        return corrupted_tensor
    
    def unpad_and_resize_batch(self, batch_tensor: torch.Tensor, 
                                transformations: List[Dict]) -> List[Image.Image]:
        """
        Convert batch tensor back to original-sized PIL images.
        
        Args:
            batch_tensor: Tensor of shape (B, C, H, W)
            transformations: List of transformation dicts from collate_fn
        
        Returns:
            List of PIL Images at original sizes
        """
        images = []
        
        for i, transform in enumerate(transformations):
            # Get single image tensor
            img_tensor = batch_tensor[i]  # (C, H, W)
            
            # Remove padding
            resized_w, resized_h = transform['resized_size']
            img_tensor = img_tensor[:, :resized_h, :resized_w]
            
            # Resize back to original size
            orig_w, orig_h = transform['original_size']
            img_pil = self.corruptor.to_pil(img_tensor)
            img_pil = img_pil.resize((orig_w, orig_h), Image.BILINEAR)
            
            images.append(img_pil)
        
        return images
    
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
    
    def predict_batch_torchvision_tensor(self, model, batch_tensor: torch.Tensor,
                                          transformations: List[Dict],
                                          score_threshold: float = 0.05) -> List[Dict]:
        """
        Batch prediction for torchvision models from tensor.
        Process padded batch directly and transform predictions back to original coordinates.
        
        Args:
            model: Torchvision detection model
            batch_tensor: Tensor of shape (B, C, H, W) - padded batch
            transformations: List of transformation dicts
            score_threshold: Minimum confidence score
        
        Returns:
            List of predictions (one dict per image) in original image coordinates
        """
        model.eval()
        
        # Ensure tensor is on correct device
        batch_tensor = batch_tensor.to(self.device)
        
        # Split into list for torchvision (expects List[Tensor])
        # Each tensor stays on GPU
        image_list = [batch_tensor[i] for i in range(batch_tensor.shape[0])]
        
        # Batch inference on padded/resized images
        with torch.no_grad():
            outputs = model(image_list)
        
        # Transform predictions back to original image coordinates
        results = []
        for i, (output, transform) in enumerate(zip(outputs, transformations)):
            mask = output['scores'] > score_threshold
            boxes = output['boxes'][mask].cpu().numpy()
            
            # Transform boxes back to original coordinates
            # Boxes are in resized+padded space, need to map to original space
            scale = transform['scale']
            
            # Undo scaling
            boxes = boxes / scale
            
            results.append({
                'boxes': boxes,
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
        inputs = processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process each image separately
        results = []
        for i, img in enumerate(pil_images):
            target_size = torch.tensor([img.size[::-1]]).to(self.device)
            
            # Extract outputs for this image - handle both dict and object outputs
            if isinstance(outputs, dict):
                single_output = {
                    'logits': outputs['logits'][i:i+1],
                    'pred_boxes': outputs['pred_boxes'][i:i+1]
                }
            else:
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
    
    def predict_batch_huggingface_tensor(self, model_key, batch_tensor: torch.Tensor,
                                          transformations: List[Dict],
                                          score_threshold: float = 0.05) -> List[Dict]:
        """
        Batch prediction for Hugging Face models from tensor.
        Process padded batch and transform predictions back to original coordinates.
        
        Args:
            model_key: Model configuration key
            batch_tensor: Tensor of shape (B, C, H, W) - padded batch
            transformations: List of transformation dicts
            score_threshold: Minimum confidence score
        
        Returns:
            List of predictions (one dict per image) in original image coordinates
        """
        from model_loader import MODEL_CONFIGS
        
        config = MODEL_CONFIGS[model_key]
        model = self.model_loader.models[model_key]
        processor = self.model_loader.processors[config['hf_model_id']]
        
        # Convert batch tensor to PIL (HuggingFace processor requires PIL)
        pil_images = []
        for i in range(batch_tensor.shape[0]):
            img_pil = F.to_pil_image(batch_tensor[i].cpu())
            pil_images.append(img_pil)
        
        # Batch process images (don't use padding parameter - processor handles it automatically)
        inputs = processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process each image and transform to original coordinates
        results = []
        for i, transform in enumerate(transformations):
            # HF models expect target size of the resized (not padded) image
            resized_h, resized_w = transform['resized_size']
            target_size = torch.tensor([[resized_h, resized_w]]).to(self.device)
            
            # Extract outputs for this image - handle both dict and object outputs
            if isinstance(outputs, dict):
                single_output = {
                    'logits': outputs['logits'][i:i+1],
                    'pred_boxes': outputs['pred_boxes'][i:i+1]
                }
            else:
                single_output = {
                    'logits': outputs.logits[i:i+1],
                    'pred_boxes': outputs.pred_boxes[i:i+1]
                }
            
            processed = processor.post_process_object_detection(
                single_output,
                target_sizes=target_size,
                threshold=score_threshold
            )[0]
            
            boxes = processed['boxes'].cpu().numpy()
            
            # Transform boxes back to original coordinates
            # Boxes are in resized space, scale back to original
            scale = transform['scale']
            boxes = boxes / scale
            
            results.append({
                'boxes': boxes,
                'labels': processed['labels'].cpu().numpy(),
                'scores': processed['scores'].cpu().numpy()
            })
        
        return results
    
    def predict_batch_tensor(self, model_key, batch_tensor: torch.Tensor,
                             transformations: List[Dict],
                             score_threshold: float = 0.05) -> List[Dict]:
        """
        Universal batch prediction method from tensor.
        
        Args:
            model_key: Model configuration key
            batch_tensor: Tensor of shape (B, C, H, W)
            transformations: List of transformation dicts
            score_threshold: Minimum confidence score
        
        Returns:
            List of predictions (one dict per image)
        """
        from model_loader import MODEL_CONFIGS
        
        config = MODEL_CONFIGS[model_key]
        model = self.model_loader.models[model_key]
        
        if config['type'] == 'torchvision':
            return self.predict_batch_torchvision_tensor(
                model, batch_tensor, transformations, score_threshold
            )
        elif config['type'] == 'huggingface':
            return self.predict_batch_huggingface_tensor(
                model_key, batch_tensor, transformations, score_threshold
            )
        else:
            raise ValueError(f"Unknown model type: {config['type']}")
    
    def run_full_test(self, model_keys, corruption_names, image_ids,
                     severities=[1, 2, 3, 4, 5], progress_callback=None) -> Dict:
        """
        Run comprehensive robustness test with batch-first processing.
        Corrupt each batch once, then test all models on it - much more efficient!
        
        Args:
            model_keys: List of model configuration keys
            corruption_names: List of corruption names
            image_ids: List of COCO image IDs
            severities: List of severity levels
            progress_callback: Optional callback(current, total, message)
        
        Returns:
            Nested dictionary with all results
        """
        # Initialize results structure
        results = {}
        for model_key in model_keys:
            model_name = self.model_loader.get_model_name(model_key)
            results[model_key] = {
                'name': model_name,
                'clean': {},
                'corrupted': {}
            }
            # Initialize predictions storage for each model
            results[model_key]['_predictions'] = {
                'clean': [],
                'corrupted': {c: {s: [] for s in severities} for c in corruption_names}
            }
        
        # Create dataset and dataloader
        dataset = COCODetectionDataset(self.evaluator, image_ids)
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
        
        total_batches = len(dataloader)
        total_tests_per_batch = 1 + len(corruption_names) * len(severities)  # clean + corruptions
        total_tests = total_batches * total_tests_per_batch
        current_test = 0
        
        print(f"\n{'='*70}")
        print(f"BATCH-FIRST TESTING: Corrupt once, test all {len(model_keys)} models")
        print(f"Total batches: {total_batches}, Batch size: {self.batch_size}")
        print(f"{'='*70}\n")
        
        # Process batch by batch
        for batch_idx, batch in enumerate(dataloader):
            batch_tensor = batch['images'].to(self.device)  # (B, C, H, W)
            batch_image_ids = batch['image_ids']
            transformations = batch['transformations']
            
            print(f"\n--- Batch {batch_idx + 1}/{total_batches} ---")
            
            # 1. TEST CLEAN IMAGES (all models on same clean batch)
            current_test += 1
            if progress_callback:
                progress_callback(current_test, total_tests,
                                f"Batch {batch_idx+1}/{total_batches}: Testing clean images")
            
            print(f"  Clean images...")
            
            for model_key in model_keys:
                batch_preds = self.predict_batch_tensor(
                    model_key, batch_tensor, transformations, score_threshold=0.05
                )
                
                # Store predictions
                for preds, image_id in zip(batch_preds, batch_image_ids):
                    coco_preds = self.evaluator.convert_predictions_to_coco_format(
                        preds, image_id, label_offset=0
                    )
                    results[model_key]['_predictions']['clean'].extend(coco_preds)
            
            # 2. TEST CORRUPTED IMAGES (corrupt once per corruption/severity, test all models)
            for corruption_name in corruption_names:
                for severity in severities:
                    current_test += 1
                    if progress_callback:
                        progress_callback(current_test, total_tests,
                                        f"Batch {batch_idx+1}/{total_batches}: "
                                        f"{corruption_name} sev{severity}")
                    
                    print(f"  {corruption_name} (severity {severity})...")
                    
                    # Apply corruption to batch ONCE
                    corrupted_tensor = self.apply_batch_corruption(
                        batch_tensor, corruption_name, severity
                    )
                    
                    # Test ALL models on this corrupted batch (stay in tensor form!)
                    for model_key in model_keys:
                        batch_preds = self.predict_batch_tensor(
                            model_key, corrupted_tensor, transformations, score_threshold=0.05
                        )
                        
                        # Store predictions
                        for preds, image_id in zip(batch_preds, batch_image_ids):
                            coco_preds = self.evaluator.convert_predictions_to_coco_format(
                                preds, image_id, label_offset=0
                            )
                            results[model_key]['_predictions']['corrupted'][corruption_name][severity].extend(coco_preds)
        
        # Evaluate all collected predictions
        print(f"\n{'='*70}")
        print("Evaluating all predictions...")
        print(f"{'='*70}\n")
        
        for model_key in model_keys:
            model_name = results[model_key]['name']
            print(f"\n{model_name}:")
            
            # Evaluate clean
            clean_preds = results[model_key]['_predictions']['clean']
            results[model_key]['clean'] = self.evaluator.evaluate_predictions(
                clean_preds, image_ids
            )
            print(f"  Clean mAP: {results[model_key]['clean']['mAP']:.4f}")
            
            # Evaluate corruptions
            for corruption_name in corruption_names:
                if corruption_name not in results[model_key]['corrupted']:
                    results[model_key]['corrupted'][corruption_name] = {}
                
                for severity in severities:
                    corrupt_preds = results[model_key]['_predictions']['corrupted'][corruption_name][severity]
                    results[model_key]['corrupted'][corruption_name][severity] = \
                        self.evaluator.evaluate_predictions(corrupt_preds, image_ids)
                    
                    mAP = results[model_key]['corrupted'][corruption_name][severity]['mAP']
                    print(f"  {corruption_name} sev{severity} mAP: {mAP:.4f}")
            
            # Clean up temporary predictions storage
            del results[model_key]['_predictions']
        
        self.results = results
        return results
    
    def save_results(self, output_path):
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

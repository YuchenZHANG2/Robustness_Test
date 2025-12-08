"""
Model loading and management for object detector robustness testing.
Supports both torchvision and Hugging Face models.
"""
import torch
import torchvision
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import numpy as np


MODEL_CONFIGS = {
    "frcnn_v1": {
        "name": "Faster R-CNN ResNet50-FPN V1",
        "type": "torchvision",
        "constructor": torchvision.models.detection.fasterrcnn_resnet50_fpn,
        "weights": torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    },
    "frcnn_v2": {
        "name": "Faster R-CNN ResNet50-FPN V2",
        "type": "torchvision",
        "constructor": torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
        "weights": torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    },
    "frcnn_mobilenet_large": {
        "name": "Faster R-CNN MobileNetV3-Large-FPN",
        "type": "torchvision",
        "constructor": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
        "weights": torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    },
    "frcnn_mobilenet_320": {
        "name": "Faster R-CNN MobileNetV3-Large 320px",
        "type": "torchvision",
        "constructor": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
        "weights": torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    },
    "retinanet_v1": {
        "name": "RetinaNet ResNet50-FPN V1",
        "type": "torchvision",
        "constructor": torchvision.models.detection.retinanet_resnet50_fpn,
        "weights": torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.COCO_V1
    },
    "retinanet_v2": {
        "name": "RetinaNet ResNet50-FPN V2",
        "type": "torchvision",
        "constructor": torchvision.models.detection.retinanet_resnet50_fpn_v2,
        "weights": torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
    },
    "fcos_v1": {
        "name": "FCOS ResNet50-FPN V1",
        "type": "torchvision",
        "constructor": torchvision.models.detection.fcos_resnet50_fpn,
        "weights": torchvision.models.detection.FCOS_ResNet50_FPN_Weights.COCO_V1
    },
    "ssd300": {
        "name": "SSD300 VGG16",
        "type": "torchvision",
        "constructor": torchvision.models.detection.ssd300_vgg16,
        "weights": torchvision.models.detection.SSD300_VGG16_Weights.COCO_V1
    },
    "ssdlite_320": {
        "name": "SSDLite320 MobileNetV3-Large",
        "type": "torchvision",
        "constructor": torchvision.models.detection.ssdlite320_mobilenet_v3_large,
        "weights": torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    },
    # Hugging Face DETR Models
    "detr": {
        "name": "DETR ResNet50",
        "type": "huggingface",
        "hf_model_id": "facebook/detr-resnet-50"
    },
    "deformable_detr": {
        "name": "Deformable DETR",
        "type": "huggingface",
        "hf_model_id": "SenseTime/deformable-detr"
    },
    "conditional_detr": {
        "name": "Conditional DETR",
        "type": "huggingface",
        "hf_model_id": "microsoft/conditional-detr-resnet-50"
    },
    "dab_detr": {
        "name": "DAB-DETR",
        "type": "huggingface",
        "hf_model_id": "IDEA-Research/dab-detr-resnet-50"
    },
    "rt_detr": {
        "name": "RT-DETR",
        "type": "huggingface",
        "hf_model_id": "PekingU/rtdetr_r50vd"
    },
    "yolov11": {
        "name": "YOLO11",
        "type": "ultralytics",
        "model_name": "yolo11m.pt"  # medium version for better accuracy
    },
}


class ModelLoader:
    """Handles loading and managing object detection models."""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.processors = {}
        
    def load_model(self, model_key, progress_callback=None):
        """
        Load a model by its configuration key.
        
        Args:
            model_key: Key from MODEL_CONFIGS
            progress_callback: Optional function(step, total, message) for progress updates
        
        Returns:
            Loaded model in eval mode
        """
        if model_key in self.models:
            return self.models[model_key]
        
        config = MODEL_CONFIGS.get(model_key)
        if not config:
            raise ValueError(f"Unknown model key: {model_key}")
        
        if progress_callback:
            progress_callback(0, 3, f"Loading {config['name']}...")
        
        if config['type'] == 'torchvision':
            model = self._load_torchvision_model(config, progress_callback)
        elif config['type'] == 'huggingface':
            model = self._load_huggingface_model(config, progress_callback)
        elif config['type'] == 'ultralytics':
            model = self._load_ultralytics_model(config, progress_callback)
        else:
            raise ValueError(f"Unknown model type: {config['type']}")
        
        if progress_callback:
            progress_callback(3, 3, f"Loaded {config['name']} successfully!")
        
        self.models[model_key] = model
        return model
    
    def _load_torchvision_model(self, config, progress_callback=None):
        """Load a torchvision model."""
        if progress_callback:
            progress_callback(1, 3, "Downloading weights...")
        
        model = config['constructor'](weights=config['weights'])
        
        if progress_callback:
            progress_callback(2, 3, "Moving to device...")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _load_huggingface_model(self, config, progress_callback=None):
        """Load a Hugging Face model."""
        if progress_callback:
            progress_callback(1, 3, "Loading image processor...")
        
        processor = AutoImageProcessor.from_pretrained(config['hf_model_id'])
        self.processors[config['hf_model_id']] = processor
        
        if progress_callback:
            progress_callback(2, 3, "Loading model weights...")
        
        model = AutoModelForObjectDetection.from_pretrained(config['hf_model_id'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _load_ultralytics_model(self, config, progress_callback=None):
        """Load an Ultralytics YOLO model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package is required for YOLO models. "
                "Install it with: pip install ultralytics"
            )
        
        if progress_callback:
            progress_callback(1, 3, "Loading YOLO model...")
        
        # YOLO models are loaded directly from model name
        model = YOLO(config['model_name'])
        
        if progress_callback:
            progress_callback(2, 3, "Model ready...")
        
        return model
    
    def predict_torchvision(self, model, image, score_threshold=0.5):
        """
        Run inference with a torchvision model.
        
        Args:
            model: Torchvision detection model
            image: PIL Image or numpy array
            score_threshold: Minimum confidence score
        
        Returns:
            List of detections with boxes, labels, scores
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = model(image_tensor)[0]
        
        # Filter by score threshold
        keep = predictions['scores'] > score_threshold
        
        return {
            'boxes': predictions['boxes'][keep].cpu().numpy(),
            'labels': predictions['labels'][keep].cpu().numpy(),
            'scores': predictions['scores'][keep].cpu().numpy()
        }
    
    def predict_huggingface(self, model_key, image, score_threshold=0.5):
        """
        Run inference with a Hugging Face model.
        
        Args:
            model_key: Key to get the model and processor
            image: PIL Image or numpy array
            score_threshold: Minimum confidence score
        
        Returns:
            List of detections with boxes, labels, scores
        """
        config = MODEL_CONFIGS[model_key]
        model = self.models[model_key]
        processor = self.processors[config['hf_model_id']]
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process predictions
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes,
            threshold=score_threshold
        )[0]
        
        # Only RT-DETR uses contiguous 0-79 labels, other DETR models use standard COCO 1-90
        labels = results['labels'].cpu().numpy()
        if 'rtdetr' in config['hf_model_id'].lower():
            from evaluator import get_rtdetr_to_coco_mapping
            rtdetr_to_coco = get_rtdetr_to_coco_mapping()
            labels = np.array([rtdetr_to_coco.get(int(l), int(l)) for l in labels])
        
        return {
            'boxes': results['boxes'].cpu().numpy(),
            'labels': labels,
            'scores': results['scores'].cpu().numpy()
        }
    
    def predict_ultralytics(self, model, image, score_threshold=0.5):
        """
        Run inference with an Ultralytics YOLO model.
        
        Args:
            model: Ultralytics YOLO model
            image: PIL Image or numpy array
            score_threshold: Minimum confidence score
        
        Returns:
            Dictionary with boxes, labels, scores
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Run inference - pass conf threshold to YOLO to get all detections
        results = model(image, conf=score_threshold, verbose=False)[0]
        
        # Extract predictions
        boxes = results.boxes
        
        # YOLO already filtered by score_threshold, no need to filter again
        # YOLO uses 0-79 contiguous labels, need to map to COCO IDs (same as RT-DETR)
        from evaluator import get_rtdetr_to_coco_mapping
        yolo_to_coco = get_rtdetr_to_coco_mapping()
        raw_labels = boxes.cls.cpu().numpy().astype(int)
        labels = np.array([yolo_to_coco.get(int(l), int(l)) for l in raw_labels])
        
        return {
            'boxes': boxes.xyxy.cpu().numpy(),
            'labels': labels,
            'scores': boxes.conf.cpu().numpy()
        }
    
    def predict(self, model_key, image, score_threshold=0.5):
        """
        Universal prediction method that works for both model types.
        
        Args:
            model_key: Model configuration key
            image: PIL Image or numpy array
            score_threshold: Minimum confidence score
        
        Returns:
            Dictionary with boxes, labels, scores
        """
        config = MODEL_CONFIGS[model_key]
        model = self.models[model_key]
        
        if config['type'] == 'torchvision':
            return self.predict_torchvision(model, image, score_threshold)
        elif config['type'] == 'huggingface':
            return self.predict_huggingface(model_key, image, score_threshold)
        elif config['type'] == 'ultralytics':
            return self.predict_ultralytics(model, image, score_threshold)
        else:
            raise ValueError(f"Unknown model type: {config['type']}")
    
    def get_model_name(self, model_key):
        """Get the human-readable name of a model."""
        return MODEL_CONFIGS.get(model_key, {}).get('name', model_key)

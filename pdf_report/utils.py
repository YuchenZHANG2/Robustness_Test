"""
Utility functions for PDF generation
"""
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import random


# Global variable to store selected image IDs for qualitative examples
# This ensures the same images are used across all corruption types
_SELECTED_IMAGE_IDS = None


def register_fonts():
    """
    Register DejaVu Serif font family for use in the PDF.
    Falls back gracefully if fonts are not found.
    """
    try:
        pdfmetrics.registerFont(
            TTFont('DejaVuSerif', '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf')
        )
        pdfmetrics.registerFont(
            TTFont('DejaVuSerif-Bold', '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf')
        )
        pdfmetrics.registerFont(
            TTFont('DejaVuSerif-Italic', '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf')
        )
        pdfmetrics.registerFont(
            TTFont('DejaVuSerif-BoldItalic', '/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf')
        )
    except Exception as e:
        print(f"Warning: DejaVu Serif fonts not found, using default fonts. Error: {e}")


def select_qualitative_images(evaluator, num_images=3, seed=None):
    """
    Select random image IDs for qualitative examples.
    This should be called once at the start of report generation.
    
    Args:
        evaluator: COCOEvaluator instance
        num_images: Number of images to select (default: 3)
        seed: Random seed for reproducibility (default: None = truly random)
        
    Returns:
        List of selected image IDs
    """
    global _SELECTED_IMAGE_IDS
    
    if _SELECTED_IMAGE_IDS is None:
        if seed is not None:
            random.seed(seed)
        all_img_ids = evaluator.get_all_images()
        _SELECTED_IMAGE_IDS = random.sample(all_img_ids, min(num_images, len(all_img_ids)))
    
    return _SELECTED_IMAGE_IDS


def get_qualitative_images():
    """
    Get the selected qualitative image IDs.
    
    Returns:
        List of image IDs or None if not yet selected
    """
    return _SELECTED_IMAGE_IDS


def reset_qualitative_images():
    """Reset the selected qualitative images (useful for testing and new reports)."""
    global _SELECTED_IMAGE_IDS
    _SELECTED_IMAGE_IDS = None


def calculate_metrics(model_data):
    """
    Calculate robustness metrics for a model.
    
    Args:
        model_data: Dictionary containing model test results with structure:
                   {'clean': {'mAP': float}, 'corrupted': {corruption: {severity: {'mAP': float}}}}
        
    Returns:
        dict: Contains clean_map, avg_corrupt, degradation, robustness_score, or None if no data
    """
    clean_map = model_data['clean']['mAP']
    
    # Calculate average corrupted mAP across all corruptions and severities
    corrupt_maps = []
    for corruption, severities in model_data['corrupted'].items():
        for severity, metrics in severities.items():
            corrupt_maps.append(metrics['mAP'])
    
    if not corrupt_maps:
        return None
    
    avg_corrupt = sum(corrupt_maps) / len(corrupt_maps)
    degradation = clean_map - avg_corrupt
    robustness_score = (avg_corrupt / clean_map * 100) if clean_map > 0 else 0
    
    return {
        'clean_map': clean_map,
        'avg_corrupt': avg_corrupt,
        'degradation': degradation,
        'robustness_score': robustness_score
    }


def format_corruption_name(corruption_name):
    """
    Format corruption name for display (e.g., 'gaussian_noise' -> 'Gaussian Noise').
    
    Args:
        corruption_name: Raw corruption name string with underscores
        
    Returns:
        str: Formatted name with title case and spaces
    """
    return corruption_name.replace('_', ' ').title()

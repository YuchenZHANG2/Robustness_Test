"""
Utility functions for PDF generation
"""
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def register_fonts():
    """
    Register DejaVu Serif font family for use in the PDF
    
    Falls back gracefully if fonts are not found
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


def calculate_metrics(model_data):
    """
    Calculate robustness metrics for a model
    
    Args:
        model_data: Dictionary containing model test results
        
    Returns:
        dict: Contains clean_map, avg_corrupt, degradation, robustness_score
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
    Format corruption name for display (e.g., 'gaussian_noise' -> 'Gaussian Noise')
    
    Args:
        corruption_name: Raw corruption name string
        
    Returns:
        str: Formatted name
    """
    return corruption_name.replace('_', ' ').title()

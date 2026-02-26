"""
Qualitative Examples Generation for PDF Reports

Handles the generation of qualitative visualization grids showing
detector performance across severity levels.
"""
from pathlib import Path
import sys
import torch
import torchvision.transforms.functional as TF
from PIL import Image as PILImage
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch

from .constants import MODEL_COLORS, DEFAULT_SCORE_THRESHOLD, SEVERITY_LEVELS
from .styles import COLORS
from .utils import get_qualitative_images, format_corruption_name
from .visualization_utils import create_grid_image, pil_image_to_reportlab, ColumnHeadersFlowable
from .table_utils import add_notes_section

# Import visualization function from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from visualization import visualize_predictions_pdf


def create_qualitative_grids(corruption_name, model_loader, evaluator, corruptor, 
                             category_names, styles):
    """
    Create qualitative visualization grids for selected images.
    Shows all detectors' performance across severity levels.
    
    Args:
        corruption_name: Name of the corruption
        model_loader: ModelLoader instance
        evaluator: COCOEvaluator instance
        corruptor: TorchCorruptions instance
        category_names: Dict mapping category IDs to names
        styles: StyleSheet object
        
    Returns:
        List of flowable elements (title, grids, legend)
    """
    elements = []
    
    # Get selected image IDs
    selected_img_ids = get_qualitative_images()
    if not selected_img_ids:
        return elements
    
    # Get model information
    model_keys = list(model_loader.models.keys())
    model_names = [model_loader.get_model_name(key) for key in model_keys]
    
    # Add section title with detector list
    _add_qualitative_section_title(elements, corruption_name, model_names, styles)
    
    # Add column headers
    elements.append(ColumnHeadersFlowable())
    elements.append(Spacer(1, 0.1*inch))
    
    # Generate visualizations for each selected image
    print(f"Generating qualitative examples for {corruption_name}...")
    
    for img_idx, image_id in enumerate(selected_img_ids):
        print(f"  Processing image {img_idx + 1}/{len(selected_img_ids)} (ID: {image_id})...")
        
        # Generate grid for this image
        grid_image = _generate_image_grid(
            image_id, corruption_name, model_keys, model_loader, 
            evaluator, corruptor, category_names
        )
        
        # Convert to ReportLab image and add to elements (fit within page margins)
        reportlab_img = pil_image_to_reportlab(grid_image, target_width=6*inch, max_height=8.5*inch)
        elements.append(reportlab_img)
        elements.append(Spacer(1, 0.3*inch))
    
    # Add notes section
    _add_qualitative_notes(elements, selected_img_ids, styles)
    
    return elements


def _add_qualitative_section_title(elements, corruption_name, model_names, styles):
    """
    Add the section title with detector list to the elements.
    
    Args:
        elements: List of flowable elements to append to
        corruption_name: Name of the corruption
        model_names: List of model names
        styles: StyleSheet object
    """
    formatted_name = format_corruption_name(corruption_name)
    
    # Create title with detector list using a table for proper alignment
    title_data = []
    for idx, name in enumerate(model_names):
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        if idx == 0:
            # First row: corruption name + "tested on:" + first detector
            title_data.append([
                Paragraph(
                    f'<font color="{COLORS["primary_blue"]}" size="14"><b>{formatted_name} tested on:</b></font>', 
                    styles['Normal']
                ),
                Paragraph(
                    f'<font color="{color}" size="14">■</font> <font color="{COLORS["primary_blue"]}" size="14"><b>{name}</b></font>', 
                    styles['Normal']
                )
            ])
        else:
            # Additional rows: empty first column, detector in second column
            title_data.append([
                Paragraph('', styles['Normal']),
                Paragraph(
                    f'<font color="{color}" size="14">■</font> <font color="{COLORS["primary_blue"]}" size="14"><b>{name}</b></font>', 
                    styles['Normal']
                )
            ])
    
    title_table = Table(title_data)
    title_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    
    elements.append(Spacer(1, 0.3*inch))
    elements.append(title_table)
    elements.append(Spacer(1, 0.5*inch))


def _generate_image_grid(image_id, corruption_name, model_keys, model_loader, 
                        evaluator, corruptor, category_names):
    """
    Generate a visualization grid for a single image across all models and severity levels.
    
    Args:
        image_id: ID of the image to process
        corruption_name: Name of the corruption
        model_keys: List of model keys
        model_loader: ModelLoader instance
        evaluator: COCOEvaluator instance
        corruptor: TorchCorruptions instance
        category_names: Dict mapping category IDs to names
        
    Returns:
        PIL Image of the grid
    """
    # Load original image
    img_path = evaluator.get_image_path(image_id)
    original_image = PILImage.open(img_path).convert('RGB')
    
    # Storage for visualizations: [detector_idx][severity] = PIL Image
    vis_grid = []
    
    # For each detector
    for model_idx, model_key in enumerate(model_keys):
        detector_color = MODEL_COLORS[model_idx % len(MODEL_COLORS)]
        severity_images = []
        
        # For each severity level (0-5)
        for severity in SEVERITY_LEVELS:
            # Apply corruption
            corrupted_image = _apply_corruption(
                original_image, corruption_name, severity, corruptor, img_path
            )
            
            # Run detection
            predictions = model_loader.predict(
                model_key, corrupted_image, score_threshold=DEFAULT_SCORE_THRESHOLD
            )
            
            # Visualize with detector-specific color
            vis_image = visualize_predictions_pdf(
                corrupted_image, 
                predictions, 
                category_names,
                detector_color=detector_color,
                score_threshold=DEFAULT_SCORE_THRESHOLD,
                show_labels=False  # No labels to keep images clean
            )
            
            severity_images.append(vis_image)
        
        vis_grid.append(severity_images)
    
    # Create and return grid image
    return create_grid_image(vis_grid)


def _apply_dust_corruption_single(clean_tensor, severity, device, clean_image_path=None):
    """
    Apply dust corruption to a single image tensor.
    Matches the logic from batch_optimized_pipeline.py.
    
    Args:
        clean_tensor: Clean image tensor (1, C, H, W) or (C, H, W)
        severity: Severity level (1-5)
        device: Device for computation
        clean_image_path: Path to clean image (for finding matching dusty image)
        
    Returns:
        Dusted tensor (same shape as input)
    """
    import cv2
    
    # Ensure tensor is 4D (1, C, H, W)
    if clean_tensor.dim() == 3:
        clean_tensor = clean_tensor.unsqueeze(0)
    
    # Severity determines blend weight: alpha * dust + (1-alpha) * clean
    alpha_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    alpha = alpha_values[severity - 1]
    beta = 1.0 - alpha
    gamma = 0.0
    
    B, C, H, W = clean_tensor.shape
    clean_tensor = clean_tensor.to(device)
    
    # Try to load real dusty image if path is provided
    dust_tensor = None
    if clean_image_path:
        clean_path = Path(clean_image_path)
        # Dust directory is in test folder (sibling of val2017 or data folder)
        dust_dir = clean_path.parent.parent / 'test'
        
        if dust_dir.exists():
            # Extract first 7 digits from clean image filename
            clean_filename = clean_path.name
            prefix = clean_filename[:7]
            
            # Find matching dusty image
            matching_files = list(dust_dir.glob(f"{prefix}*"))
            
            if matching_files:
                # Load dusty image
                dust_path = matching_files[0]
                dust_img = cv2.imread(str(dust_path))
                dust_img = cv2.cvtColor(dust_img, cv2.COLOR_BGR2RGB)
                dust_pil = PILImage.fromarray(dust_img)
                
                # Resize to match clean image size
                dust_pil = dust_pil.resize((W, H), PILImage.BILINEAR)
                
                # Convert to tensor
                dust_tensor = TF.to_tensor(dust_pil).to(device)  # (C, H, W)
    
    # Fallback: Generate procedural dust if no real dust image found
    if dust_tensor is None:
        dust_tensor = torch.ones((C, H, W), device=device)
        
        # Add noise/texture
        noise = torch.randn((1, H, W), device=device) * 0.1 + 0.7
        noise = torch.clamp(noise, 0, 1)
        
        # Make it yellowish-brown (dust color)
        dust_tensor[0] = noise.squeeze() * 0.9  # R
        dust_tensor[1] = noise.squeeze() * 0.8  # G
        dust_tensor[2] = noise.squeeze() * 0.6  # B
    
    # Blend
    dusted = alpha * dust_tensor + beta * clean_tensor.squeeze(0) + gamma
    dusted = torch.clamp(dusted, 0.0, 1.0)
    
    return dusted.unsqueeze(0)


def _apply_corruption(original_image, corruption_name, severity, corruptor, image_path=None):
    """
    Apply corruption to an image at a specific severity level.
    
    Args:
        original_image: PIL Image to corrupt
        corruption_name: Name of the corruption
        severity: Severity level (0 for clean)
        corruptor: TorchCorruptions instance
        image_path: Path to the original image (needed for dust corruption)
        
    Returns:
        PIL Image (corrupted or original if severity is 0)
    """
    if severity == 0:
        return original_image
    
    # Convert to tensor
    img_tensor = TF.to_tensor(original_image).unsqueeze(0)
    img_tensor = img_tensor.to(corruptor.device)
    
    # Apply corruption
    if corruption_name == 'dust':
        # Apply dust blending (same as batch pipeline)
        corrupted_tensor = _apply_dust_corruption_single(
            img_tensor, severity, corruptor.device, image_path
        )
    else:
        corrupted_tensor = corruptor.corrupt(
            img_tensor, 
            corruption_name=corruption_name, 
            severity=severity
        )
    
    # Convert back to PIL
    return TF.to_pil_image(corrupted_tensor.squeeze(0).cpu())


def _add_qualitative_notes(elements, selected_img_ids, styles):
    """
    Add notes section explaining the qualitative examples.
    
    Args:
        elements: List of flowable elements to append to
        selected_img_ids: List of selected image IDs
        styles: StyleSheet object
    """
    elements.append(Spacer(1, 0.2*inch))
    
    # Build note text with image IDs
    image_ids_str = ', '.join(str(img_id) for img_id in selected_img_ids)
    notes = [
        f'Images are chosen at random. For this report, images {image_ids_str} were selected.',
        f'Detection boxes are shown for predictions with confidence scores above {DEFAULT_SCORE_THRESHOLD}.'
    ]
    
    add_notes_section(elements, notes, styles)

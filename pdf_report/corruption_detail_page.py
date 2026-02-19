"""
Corruption Detail Page - Individual corruption analysis with plots and tables
"""
from reportlab.platypus import Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PDF generation
import matplotlib.pyplot as plt
import io
from pathlib import Path
from .styles import COLORS
from reportlab.platypus import Image as RLImage
from .utils import format_corruption_name, get_qualitative_images


# Color palette for different models in plots
MODEL_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
]


def create_corruption_detail_page(corruption_name, results, styles, section_number=None,
                                 model_loader=None, evaluator=None, corruptor=None, 
                                 category_names=None):
    """
    Create a detailed analysis page for a specific corruption type
    
    Args:
        corruption_name: Name of the corruption (e.g., 'gaussian_noise')
        results: Dictionary of test results for all models
        styles: StyleSheet object with all styles
        section_number: Section number for this corruption (e.g., "2.1")
        model_loader: ModelLoader instance (optional, for qualitative examples)
        evaluator: COCOEvaluator instance (optional, for qualitative examples)
        corruptor: TorchCorruptions instance (optional, for qualitative examples)
        category_names: Dict mapping category IDs to names (optional, for qualitative examples)
        
    Returns:
        list: List of flowable elements for the corruption detail page
    """
    elements = []
    
    # Page title with anchor for TOC linking and section number
    formatted_name = format_corruption_name(corruption_name)
    anchor_name = f'corruption_{corruption_name}'
    
    if section_number:
        title = Paragraph(f'<a name="{anchor_name}"/>{section_number} {formatted_name}', styles['ContentPageTitle'])
    else:
        title = Paragraph(f'<a name="{anchor_name}"/>{formatted_name}', styles['ContentPageTitle'])
    
    elements.append(title)
    
    # Subtitle
    subtitle = Paragraph(
        f"Performance degradation analysis across severity levels (0-5)",
        styles['ContentPageSubtitle']
    )
    elements.append(subtitle)
    
    # Generate and add the line plot
    plot_image = _create_severity_plot(corruption_name, results)
    if plot_image:
        elements.append(plot_image)
        elements.append(Spacer(1, 0.2*inch))
    
    # Create and add the severity table
    elements.append(_create_severity_table(corruption_name, results, styles))
    
    # Add page break after table, before qualitative examples
    elements.append(PageBreak())
    
    # Add qualitative examples if all required objects are provided
    if all([model_loader, evaluator, corruptor, category_names]):
        qualitative_elements = create_qualitative_grids(
            corruption_name, model_loader, evaluator, corruptor, 
            category_names, styles
        )
        elements.extend(qualitative_elements)
    
    # Add page break for next corruption
    elements.append(PageBreak())
    
    return elements


def _create_severity_plot(corruption_name, results):
    """
    Create a line plot showing mAP across severity levels
    
    Args:
        corruption_name: Name of the corruption
        results: Dictionary of test results
        
    Returns:
        Image: ReportLab Image object with the plot
    """
    # Set up the plot with taller height
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    severities = [0, 1, 2, 3, 4, 5]
    
    # Plot each model
    for idx, (model_key, model_data) in enumerate(results.items()):
        model_name = model_data.get('name', model_key)
        
        # Check if this model has data for this corruption
        if corruption_name not in model_data.get('corrupted', {}):
            continue
        
        mAP_values = []
        
        # Severity 0 (clean)
        mAP_values.append(model_data['clean']['mAP'])
        
        # Severities 1-5
        corruption_data = model_data['corrupted'][corruption_name]
        for severity in range(1, 6):
            severity_key = str(severity)
            if severity_key in corruption_data:
                mAP_values.append(corruption_data[severity_key]['mAP'])
            else:
                mAP_values.append(None)
        
        # Plot the line
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        ax.plot(severities, mAP_values, marker='o', linewidth=2, 
                label=model_name, color=color, markersize=6)
    
    # Styling
    ax.set_xlabel('Severity Level', fontsize=10, fontweight='bold')
    ax.set_ylabel('mAP', fontsize=10, fontweight='bold')
    ax.set_xticks(severities)
    ax.set_xticklabels(['Clean', '1', '2', '3', '4', '5'])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9, framealpha=0.9, frameon=False)
    
    plt.tight_layout()
    
    # Convert plot to image
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close(fig)
    
    # Create ReportLab Image (width ~6.5 inches, taller height)
    img = Image(img_buffer, width=6.5*inch, height=3.2*inch)
    
    return img


def _create_severity_table(corruption_name, results, styles):
    """
    Create a table showing mAP values and degradation across severities
    Each model has two rows: absolute mAP and relative degradation
    Model name column spans both rows, with a Metric column showing mAP/Deg%
    
    Args:
        corruption_name: Name of the corruption
        results: Dictionary of test results
        styles: StyleSheet object
        
    Returns:
        Table: Formatted severity table
    """
    table_data = []
    
    # Header row
    headers = [
        Paragraph('<b>Model</b>', styles['TableHeader']),
        Paragraph('<b>Metric</b>', styles['TableHeader'])
    ]
    for severity in range(6):
        headers.append(Paragraph(f'<b>Sev {severity}</b>', styles['TableHeader']))
    table_data.append(headers)
    
    # Sort models by clean mAP (descending)
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]['clean']['mAP'],
        reverse=True
    )
    
    # Data rows - each model gets two rows
    for model_key, model_data in sorted_models:
        # Skip if this model doesn't have data for this corruption
        if corruption_name not in model_data.get('corrupted', {}):
            continue
        
        model_name = model_data.get('name', model_key)
        clean_map = model_data['clean']['mAP']
        corruption_data = model_data['corrupted'][corruption_name]
        
        # Row 1: Absolute mAP values
        map_row = [
            Paragraph(f'<b>{model_name}</b>', styles['TableCell']),
            Paragraph('mAP', styles['TableCell'])
        ]
        
        # Severity 0 (clean)
        map_row.append(Paragraph(f'{clean_map:.3f}', styles['TableCell']))
        
        # Severities 1-5
        map_values = []
        for severity in range(1, 6):
            severity_key = str(severity)
            if severity_key in corruption_data:
                map_val = corruption_data[severity_key]['mAP']
                map_values.append(map_val)
                map_row.append(Paragraph(f'{map_val:.3f}', styles['TableCell']))
            else:
                map_values.append(None)
                map_row.append(Paragraph('N/A', styles['TableCell']))
        
        table_data.append(map_row)
        
        # Row 2: Relative degradation (%)
        deg_row = [
            '',  # Empty for model name (will be spanned from above)
            Paragraph('Deg %', styles['TableCell'])
        ]
        
        # Severity 0 (clean) - 0% degradation
        deg_row.append(Paragraph('0.0%', styles['TableCell']))
        
        # Severities 1-5
        for map_val in map_values:
            if map_val is not None:
                degradation_pct = ((map_val - clean_map) / clean_map * 100)
                deg_row.append(Paragraph(f'{degradation_pct:+.1f}%', styles['TableCell']))
            else:
                deg_row.append(Paragraph('N/A', styles['TableCell']))
        
        table_data.append(deg_row)
    
    # Create table
    col_widths = [1.4*inch, 0.7*inch] + [0.7*inch] * 6
    severity_table = Table(table_data, colWidths=col_widths)
    
    # Apply styling
    severity_table.setStyle(_get_severity_table_style(len(table_data)))
    
    return severity_table


def _get_severity_table_style(num_rows):
    """
    Get the table style configuration for severity table
    
    Args:
        num_rows: Total number of rows in the table
        
    Returns:
        TableStyle: Configured table style
    """
    table_style = [
        # Header row styling
        ('BACKGROUND', (0, 0), (-1, 0), HexColor(COLORS['primary_blue'])),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor(COLORS['white'])),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSerif-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        
        # Data rows alignment
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),  # Model name left-aligned
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),  # All other columns centered
        ('FONTNAME', (0, 1), (-1, -1), 'DejaVuSerif'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        
        # Three-line style
        ('LINEABOVE', (0, 0), (-1, 0), 1.5, HexColor(COLORS['dark_gray'])),
        ('LINEBELOW', (0, 0), (-1, 0), 1.0, HexColor(COLORS['dark_gray'])),
        ('LINEBELOW', (0, -1), (-1, -1), 1.5, HexColor(COLORS['dark_gray'])),
        
        # Padding
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]
    
    # Span model names across their two rows
    model_count = 0
    for i in range(1, num_rows, 2):  # Process pairs of rows
        if i + 1 < num_rows:  # Make sure we have both rows
            # Span the model name cell vertically
            table_style.append(('SPAN', (0, i), (0, i+1)))
            
            # Alternating background for model pairs
            if model_count % 2 == 0:
                # Light background for both rows of this model
                table_style.append(('BACKGROUND', (0, i), (-1, i+1), HexColor(COLORS['very_light_gray'])))
            else:
                # White background for both rows of this model
                table_style.append(('BACKGROUND', (0, i), (-1, i+1), HexColor(COLORS['white'])))
            model_count += 1
    
    return TableStyle(table_style)


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
    from PIL import Image as PILImage, ImageDraw, ImageFont
    import torch
    import torchvision.transforms.functional as TF
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from visualization import visualize_predictions_pdf
    
    elements = []
    
    # Get selected image IDs
    selected_img_ids = get_qualitative_images()
    if not selected_img_ids:
        return elements
    
    # Define colors for each detector
    detector_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Get model keys and names
    model_keys = list(model_loader.models.keys())
    model_names = [model_loader.get_model_name(key) for key in model_keys]
    
    # Section title with corruption name and detector list (all same size/font/color)
    formatted_name = format_corruption_name(corruption_name)
    
    # Create title with detector list using a table for proper alignment
    title_data = []
    for idx, name in enumerate(model_names):
        color = detector_colors[idx % len(detector_colors)]
        if idx == 0:
            # First row: corruption name + "tested on:" + first detector
            title_data.append([
                Paragraph(f'<font color="{COLORS["primary_blue"]}" size="14"><b>{formatted_name} tested on:</b></font>', styles['Normal']),
                Paragraph(f'<font color="{color}" size="14">■</font> <font color="{COLORS["primary_blue"]}" size="14"><b>{name}</b></font>', styles['Normal'])
            ])
        else:
            # Additional rows: empty first column, detector in second column
            title_data.append([
                Paragraph('', styles['Normal']),
                Paragraph(f'<font color="{color}" size="14">■</font> <font color="{COLORS["primary_blue"]}" size="14"><b>{name}</b></font>', styles['Normal'])
            ])
    
    title_table = Table(title_data )
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
    
    # Add column headers with "Clean" and "severity" arrow
    column_headers = _create_column_headers(styles)
    elements.append(column_headers)
    elements.append(Spacer(1, 0.1*inch))
    
    # Generate visualizations for each selected image
    print(f"Generating qualitative examples for {corruption_name}...")
    
    for img_idx, image_id in enumerate(selected_img_ids):
        print(f"Processing image {img_idx + 1}/{len(selected_img_ids)} (ID: {image_id})...")
        
        # Load original image
        img_path = evaluator.get_image_path(image_id)
        original_image = PILImage.open(img_path).convert('RGB')
        
        # Storage for visualizations: [detector_idx][severity] = PIL Image
        vis_grid = []
        
        # For each detector
        for model_idx, model_key in enumerate(model_keys):
            print(f"      Testing {model_names[model_idx]}...")
            detector_color = detector_colors[model_idx % len(detector_colors)]
            severity_images = []
            
            # For each severity level (0-5)
            for severity in range(6):
                # Apply corruption (severity 0 = clean)
                if severity == 0:
                    corrupted_image = original_image
                else:
                    # Convert to tensor
                    img_tensor = TF.to_tensor(original_image).unsqueeze(0)
                    img_tensor = img_tensor.to(corruptor.device)
                    
                    # Apply corruption
                    if corruption_name == 'dust':
                        # Special handling for dust - needs image paths
                        # For now, skip dust or use a simpler approach
                        corrupted_tensor = img_tensor
                    else:
                        corrupted_tensor = corruptor.corrupt(
                            img_tensor, 
                            corruption_name=corruption_name, 
                            severity=severity
                        )
                    
                    # Convert back to PIL
                    corrupted_image = TF.to_pil_image(corrupted_tensor.squeeze(0).cpu())
                
                # Run detection
                predictions = model_loader.predict(model_key, corrupted_image, score_threshold=0.3)
                
                # Visualize with detector-specific color
                vis_image = visualize_predictions_pdf(
                    corrupted_image, 
                    predictions, 
                    category_names,
                    detector_color=detector_color,
                    score_threshold=0.3,
                    show_labels=False  # No labels to keep images clean
                )
                
                severity_images.append(vis_image)
            
            vis_grid.append(severity_images)
        
        # Create grid image: rows=detectors, cols=severities (without labels)
        grid_image = _create_grid_image(vis_grid)
        
        # Convert to ReportLab Image
        img_buffer = io.BytesIO()
        grid_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Add to PDF (width to fit page)
        img_width, img_height = grid_image.size

        # Target width
        target_width = 7 * inch

        # Preserve aspect ratio
        aspect_ratio = img_height / img_width
        target_height = target_width * aspect_ratio
        reportlab_img = RLImage(img_buffer, width=target_width, height=target_height)

        elements.append(reportlab_img)
        elements.append(Spacer(1, 0.3*inch))  # Space between image groups
    
    # Add notes section after all images
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph('<b>Note:</b>', styles['NotesTitle']))
    
    # Build note text with image IDs
    image_ids_str = ', '.join(str(img_id) for img_id in selected_img_ids)
    notes = [
        f'Images are chosen at random. For this report, images {image_ids_str} were selected.',
        'Detection boxes are shown for predictions with confidence scores above 0.3.'
    ]
    
    for note in notes:
        bullet = Paragraph(f'• {note}', styles['NotesText'])
        elements.append(bullet)
        elements.append(Spacer(1, 0.08*inch))
    
    return elements


def _create_grid_image(vis_grid):
    """
    Create a grid image from visualization results (without labels).
    
    Args:
        vis_grid: List[List[PIL Image]] - [detector_idx][severity_idx]
        
    Returns:
        PIL Image of the complete grid
    """
    from PIL import Image as PILImage
    
    num_detectors = len(vis_grid)
    num_severities = len(vis_grid[0]) if vis_grid else 0
    
    if num_detectors == 0 or num_severities == 0:
        return PILImage.new('RGB', (800, 600), 'white')
    
    # Get size of individual images
    sample_img = vis_grid[0][0]
    img_width, img_height = sample_img.size
    
    # Add padding between images
    padding = 5
    
    # Calculate grid dimensions (no labels)
    grid_width = num_severities * (img_width + padding) - padding
    grid_height = num_detectors * (img_height + padding) - padding
    
    # Create blank canvas
    grid_img = PILImage.new('RGB', (grid_width, grid_height), 'white')
    
    # Paste images
    for det_idx in range(num_detectors):
        for sev_idx in range(num_severities):
            x = sev_idx * (img_width + padding)
            y = det_idx * (img_height + padding)
            grid_img.paste(vis_grid[det_idx][sev_idx], (x, y))
    
    return grid_img


def _create_column_headers(styles):
    """
    Create column headers with "Clean" and arrow with "severity" using ReportLab.
    
    Args:
        styles: StyleSheet object
        
    Returns:
        Custom flowable with headers
    """
    from reportlab.platypus.flowables import Flowable
    from reportlab.lib.colors import black
    
    class ColumnHeaders(Flowable):
        def __init__(self, width=7*inch, height=0.3*inch):
            Flowable.__init__(self)
            self.width = width
            self.height = height
        
        def draw(self):
            canvas = self.canv
            
            # Calculate positions based on 6 columns (1 clean + 5 severity)
            col_width = self.width / 6
            
            # Draw "Clean" centered on first column
            canvas.setFont('Helvetica-Bold', 11)
            canvas.setFillColor(black)
            clean_x = col_width / 2 - 40  # Center "Clean" text
            canvas.drawString(clean_x, self.height / 2, "Clean")
            
            # Draw arrow spanning columns 2-6
            arrow_start_x = col_width 
            arrow_end_x = self.width - 40
            arrow_y = self.height / 2
            
            # Arrow line
            canvas.setLineWidth(2)
            canvas.line(arrow_start_x, arrow_y, arrow_end_x, arrow_y)
            
            # Arrow head
            arrow_head_size = 6
            canvas.setLineWidth(1)
            path = canvas.beginPath()
            path.moveTo(arrow_end_x, arrow_y)
            path.lineTo(arrow_end_x - arrow_head_size, arrow_y - arrow_head_size/2)
            path.lineTo(arrow_end_x - arrow_head_size, arrow_y + arrow_head_size/2)
            path.close()
            canvas.drawPath(path, fill=1, stroke=0)
            
            # Draw "severity" text above arrow center
            severity_x = (arrow_start_x + arrow_end_x) / 2 - 20
            canvas.drawString(severity_x, arrow_y + 8, "severity")
    
    return ColumnHeaders()

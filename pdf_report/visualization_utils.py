"""
Utility functions for creating visualizations (plots, grids, charts) in PDF reports
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image as PILImage
from reportlab.platypus import Image as RLImage
from reportlab.lib.units import inch
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import black

from .constants import MODEL_COLORS, SEVERITY_LEVELS, SEVERITY_LABELS


def create_severity_line_plot(corruption_name, results):
    """
    Create a line plot showing mAP across severity levels for all models.
    
    Args:
        corruption_name: Name of the corruption
        results: Dictionary of test results
        
    Returns:
        RLImage: ReportLab Image object with the plot, or None if no data
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    has_data = False
    for idx, (model_key, model_data) in enumerate(results.items()):
        model_name = model_data.get('name', model_key)
        
        # Check if this model has data for this corruption
        if corruption_name not in model_data.get('corrupted', {}):
            continue
        
        has_data = True
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
        ax.plot(SEVERITY_LEVELS, mAP_values, marker='o', linewidth=2, 
                label=model_name, color=color, markersize=6)
    
    if not has_data:
        plt.close(fig)
        return None
    
    # Styling
    ax.set_xlabel('Severity Level', fontsize=10, fontweight='bold')
    ax.set_ylabel('mAP', fontsize=10, fontweight='bold')
    ax.set_xticks(SEVERITY_LEVELS)
    ax.set_xticklabels(SEVERITY_LABELS)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9, framealpha=0.9, frameon=False)
    
    plt.tight_layout()
    
    # Convert plot to image
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close(fig)
    
    return RLImage(img_buffer, width=6.5*inch, height=3.2*inch)


def create_spider_chart(results):
    """
    Create a spider/radar chart comparing models across key metrics.
    
    Args:
        results: Dictionary of test results
        
    Returns:
        RLImage: ReportLab Image object with the spider chart, or None if no data
    """
    from .utils import calculate_metrics
    
    # Collect metrics for all models
    model_metrics = []
    model_names = []
    
    for model_key, model_data in results.items():
        metrics = calculate_metrics(model_data)
        if metrics:
            model_metrics.append(metrics)
            model_names.append(model_data.get('name', model_key))
    
    if not model_metrics:
        return None
    
    # Extract values for each axis across all models
    clean_maps = [m['clean_map'] for m in model_metrics]
    corrupted_maps = [m['avg_corrupt'] for m in model_metrics]
    robustness_scores = [m['robustness_score'] for m in model_metrics]
    
    # Find max for each axis to normalize
    max_clean = max(clean_maps) if clean_maps else 1
    max_corrupted = max(corrupted_maps) if corrupted_maps else 1
    max_robustness = max(robustness_scores) if robustness_scores else 1
    
    # Prepare data for spider chart (3 axes)
    categories = ['Clean', 'Corrupted', 'Robustness %']
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Create the plot with square aspect
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    
    # Collect all normalized values to find minimum for chart scaling
    all_normalized_values = []
    
    # Plot each model
    for idx, (metrics, model_name) in enumerate(zip(model_metrics, model_names)):
        # Normalize each value by its axis maximum, then scale to 100
        values = [
            (metrics['clean_map'] / max_clean) * 100,
            (metrics['avg_corrupt'] / max_corrupted) * 100,
            (metrics['robustness_score'] / max_robustness) * 100
        ]
        
        # Collect for min calculation
        all_normalized_values.extend(values)
        values += values[:1]  # Complete the circle
        
        # Plot with transparency
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        ax.plot(angles, values, 'o-', linewidth=2.5, label=model_name, 
                color=color, markersize=8)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    # Calculate dynamic lower bound (round down to nearest 10)
    min_value = min(all_normalized_values)
    lower_bound = int(min_value / 10) * 10
    
    # Customize the chart with dynamic scale
    ax.tick_params(axis='x', pad=12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=15)
    ax.set_ylim(lower_bound, 100)
    
    # Generate appropriate yticks based on range
    tick_step = (100 - lower_bound) / 4
    yticks = [lower_bound + i * tick_step for i in range(5)]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{int(t)}' for t in yticks], fontsize=10, color='gray')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize=12, frameon=False)
    
    # Ensure circular shape
    ax.set_aspect('equal')
    
    plt.tight_layout(pad=1.5)
    
    # Convert to image
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150)
    img_buffer.seek(0)
    plt.close(fig)
    
    return RLImage(img_buffer, width=4.5*inch, height=4.5*inch)


def create_grid_image(vis_grid, padding=5):
    """
    Create a grid image from visualization results.
    
    Args:
        vis_grid: List[List[PIL Image]] - [detector_idx][severity_idx]
        padding: Padding between images in pixels
        
    Returns:
        PIL Image of the complete grid
    """
    num_detectors = len(vis_grid)
    num_severities = len(vis_grid[0]) if vis_grid else 0
    
    if num_detectors == 0 or num_severities == 0:
        return PILImage.new('RGB', (800, 600), 'white')
    
    # Get size of individual images
    sample_img = vis_grid[0][0]
    img_width, img_height = sample_img.size
    
    # Calculate grid dimensions
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


def pil_image_to_reportlab(pil_image, target_width=7*inch):
    """
    Convert a PIL Image to a ReportLab Image with specified width, preserving aspect ratio.
    
    Args:
        pil_image: PIL Image object
        target_width: Target width in ReportLab units (default: 7 inches)
        
    Returns:
        RLImage object
    """
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # Calculate height preserving aspect ratio
    img_width, img_height = pil_image.size
    aspect_ratio = img_height / img_width
    target_height = target_width * aspect_ratio
    
    return RLImage(img_buffer, width=target_width, height=target_height)


class ColumnHeadersFlowable(Flowable):
    """
    Custom flowable for column headers showing "Clean" and an arrow with "severity".
    Used in qualitative visualization grids.
    """
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
        clean_x = col_width / 2 - 40
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

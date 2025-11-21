"""
Visualization utilities for object detection predictions.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import io
import base64


def visualize_predictions(image, predictions, category_names, score_threshold=0.1, 
                         max_boxes=20, title="Predictions"):
    """
    Visualize object detection predictions on an image.
    
    Args:
        image: PIL Image or numpy array
        predictions: Dict with 'boxes', 'labels', 'scores'
        category_names: Dict mapping category IDs to names
        score_threshold: Minimum score to display
        max_boxes: Maximum number of boxes to show
        title: Plot title
    
    Returns:
        Figure object
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']
    
    # Filter by score and limit number
    keep_mask = scores >= score_threshold
    boxes = boxes[keep_mask][:max_boxes]
    labels = labels[keep_mask][:max_boxes]
    scores = scores[keep_mask][:max_boxes]
    
    # Color map for different classes
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Get color for this class
        color = colors[int(label) % len(colors)]
        
        # Draw bounding box
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Get category name
        cat_name = category_names.get(int(label), f'Class {label}')
        
        # Add label with score
        label_text = f'{cat_name}: {score:.2f}'
        ax.text(
            x1, y1 - 5,
            label_text,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
            fontsize=9,
            color='white',
            weight='bold'
        )
    
    ax.set_title(f'{title} ({len(boxes)} detections)', fontsize=14, fontweight='bold', y=1.05)
    ax.axis('off')
    plt.tight_layout()
    
    return fig


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for web display."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def save_prediction_visualization(image, predictions, category_names, 
                                  output_path, score_threshold=0.5):
    """Save visualization to file."""
    fig = visualize_predictions(image, predictions, category_names, 
                                score_threshold=score_threshold)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

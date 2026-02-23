"""
OOD Detail Page - Out-of-Distribution detection analysis for PDF report

This module generates PDF pages for OOD (Out-of-Distribution) detection analysis,
including overall recall summaries, per-class metrics tables, and confusion heatmaps.
"""
from typing import Dict, List, Tuple, Optional, Any
from io import BytesIO

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

from reportlab.platypus import Paragraph, Spacer, Table, PageBreak, Image as RLImage
from reportlab.lib.units import inch
from reportlab.lib import colors

from .table_utils import create_three_line_table_style


# ============================================================================
# Configuration Constants
# ============================================================================

# Visualization parameters for confusion heatmap
HEATMAP_CELL_SIZE = 0.6  # Size of each cell in inches
HEATMAP_PADDING_WIDTH = 2.0  # Horizontal padding for labels (inches)
HEATMAP_PADDING_HEIGHT = 1.5  # Vertical padding for labels (inches)
HEATMAP_DPI = 150  # Resolution for saved images
HEATMAP_FONT_SIZE = 9  # Font size for axis labels
HEATMAP_ANNOTATION_FONT_SIZE = 8  # Font size for cell values
HEATMAP_TITLE_FONT_SIZE = 11  # Font size for title


# ============================================================================
# Main Entry Point
# ============================================================================

def create_ood_detail_page(
    ood_results: Dict[str, Any],
    model_names_map: Dict[str, str],
    category_names: Dict[int, str],
    styles: Any
) -> List[Any]:
    """
    Create OOD analysis page for PDF report.
    
    Args:
        ood_results: Dictionary mapping model keys to OOD evaluation data.
                    Each value should contain 'general_ood_recall', 'total_ood_detected',
                    'total_ood_gt', and 'top_n_classes' keys.
        model_names_map: Dictionary mapping model keys to display names
        category_names: Dictionary mapping category IDs to category names
        styles: StyleSheet object with all text styles
        
    Returns:
        List of ReportLab flowable elements for the OOD analysis page
    """
    elements = []
    
    # Page title with anchor
    elements.append(Paragraph(
        '<a name="ood_analysis"/>3. Out-of-Distribution (OOD) Analysis', 
        styles['ContentPageTitle']
    ))

    
    # Section 3.1: Overall OOD Recall Summary
    elements.append(Paragraph(
        "3.1 Overall OOD Recall by Detector", 
        styles['SectionHeader']
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    summary_table = _create_overall_summary_table(ood_results, model_names_map, styles)
    elements.append(summary_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Section 3.2: Per-class Analysis
    top_n_classes = _get_top_n_classes(ood_results)
    
    if top_n_classes:
        elements.append(Paragraph(
            f"3.2 Top {len(top_n_classes)} Most Frequent OOD Classes", 
            styles['SectionHeader']
        ))
        elements.append(Spacer(1, 0.2*inch))
        
        # Generate detailed analysis for each OOD class
        for idx, class_data in enumerate(top_n_classes, start=1):
            class_elements = _create_class_analysis(
                class_data['category_id'],
                class_data['category_name'],
                ood_results,
                model_names_map,
                category_names,
                styles,
                subsection_number=f"3.2.{idx}"
            )
            elements.extend(class_elements)
            
            # Add page break between classes (except after the last one)
            if idx < len(top_n_classes):
                elements.append(PageBreak())
    
    return elements




# ============================================================================
# Helper Functions - Data Extraction
# ============================================================================

def _get_top_n_classes(ood_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract the list of top N OOD classes from results.
    
    Args:
        ood_results: Dictionary of OOD evaluation results
        
    Returns:
        List of class data dictionaries, or empty list if not available
    """
    if not ood_results:
        return []
    
    # Get top_n_classes from the first model (all models should have the same)
    first_model_data = next(iter(ood_results.values()))
    return first_model_data.get('top_n_classes', [])


def _get_class_data_for_model(
    model_key: str,
    class_id: int,
    ood_results: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Find class data for a specific model and class ID.
    
    Args:
        model_key: Key identifying the model
        class_id: Category ID to search for
        ood_results: Dictionary of OOD evaluation results
        
    Returns:
        Class data dictionary if found, None otherwise
    """
    ood_data = ood_results.get(model_key, {})
    
    for class_data in ood_data.get('top_n_classes', []):
        if class_data['category_id'] == class_id:
            return class_data
    
    return None


def _get_sorted_models_for_class(
    class_id: int,
    ood_results: Dict[str, Any],
    model_names_map: Dict[str, str]
) -> List[Tuple[str, str, float, int, int]]:
    """
    Get models sorted by recall for a specific OOD class.
    
    Args:
        class_id: Category ID to get models for
        ood_results: Dictionary of OOD evaluation results
        model_names_map: Dictionary mapping model keys to display names
        
    Returns:
        List of tuples: (model_key, model_name, recall, detected, total)
        sorted by recall in descending order
    """
    model_recalls = []
    
    for model_key in ood_results.keys():
        class_data = _get_class_data_for_model(model_key, class_id, ood_results)
        
        if class_data:
            model_recalls.append((
                model_key,
                model_names_map.get(model_key, model_key),
                class_data['recall'],
                class_data['detected'],
                class_data['total_annotations']
            ))
    
    # Sort by recall descending (highest recall first)
    model_recalls.sort(key=lambda x: x[2], reverse=True)
    
    return model_recalls


# ============================================================================
# Table Creation Functions
# ============================================================================

def _create_overall_summary_table(
    ood_results: Dict[str, Any],
    model_names_map: Dict[str, str],
    styles: Any
) -> Table:
    """
    Create overall OOD recall summary table, ordered by recall.
    
    Args:
        ood_results: Dictionary of OOD evaluation results
        model_names_map: Dictionary mapping model keys to display names
        styles: StyleSheet object (not used, kept for consistency)
        
    Returns:
        ReportLab Table object with three-line styling
    """
    # Sort models by overall recall
    sorted_models = sorted(
        ood_results.items(),
        key=lambda x: x[1]['general_ood_recall'],
        reverse=True
    )
    
    # Build table data
    data = [['Rank', 'Model', 'OOD Recall', 'Detected / Total']]
    
    for rank, (model_key, ood_data) in enumerate(sorted_models, start=1):
        model_name = model_names_map.get(model_key, model_key)
        recall = ood_data['general_ood_recall']
        detected = ood_data['total_ood_detected']
        total = ood_data['total_ood_gt']
        
        data.append([
            str(rank),
            model_name,
            f"{recall:.3f}",
            f"{detected} / {total}"
        ])
    
    # Create and style table
    table = Table(data, colWidths=[0.8*inch, 3*inch, 1.2*inch, 1.5*inch])
    table.setStyle(create_three_line_table_style(len(data)))
    
    return table




def _create_class_recall_table(
    class_id: int,
    ood_results: Dict[str, Any],
    model_names_map: Dict[str, str],
    styles: Any
) -> Table:
    """
    Create recall table for a specific OOD class.
    
    Args:
        class_id: Category ID for the OOD class
        ood_results: Dictionary of OOD evaluation results
        model_names_map: Dictionary mapping model keys to display names
        styles: StyleSheet object (not used, kept for consistency)
        
    Returns:
        ReportLab Table object with three-line styling
    """
    # Get models sorted by recall for this class
    model_recalls = _get_sorted_models_for_class(class_id, ood_results, model_names_map)
    
    # Build table data
    data = [['Rank', 'Model', 'Recall', 'Detected / Total']]
    
    for rank, (_, model_name, recall, detected, total) in enumerate(model_recalls, start=1):
        data.append([
            str(rank),
            model_name,
            f"{recall:.3f}",
            f"{detected} / {total}"
        ])
    
    # Create and style table
    table = Table(data, colWidths=[0.8*inch, 3*inch, 1.2*inch, 1.5*inch])
    table.setStyle(create_three_line_table_style(len(data)))
    
    return table


# ============================================================================
# Per-Class Analysis
# ============================================================================

def _create_class_analysis(
    class_id: int,
    class_name: str,
    ood_results: Dict[str, Any],
    model_names_map: Dict[str, str],
    category_names: Dict[int, str],
    styles: Any,
    subsection_number: str
) -> List[Any]:
    """
    Create detailed analysis section for one OOD class.
    
    Args:
        class_id: Category ID for the OOD class
        class_name: Display name of the OOD class
        ood_results: Dictionary of OOD evaluation results
        model_names_map: Dictionary mapping model keys to display names
        category_names: Dictionary mapping category IDs to names
        styles: StyleSheet object with all text styles
        subsection_number: Section numbering (e.g., "3.2.1")
        
    Returns:
        List of ReportLab flowable elements for this class analysis
    """
    elements = []
    
    # Subsection title
    elements.append(Paragraph(
        f"<b>{subsection_number} OOD Class: {class_name}</b>",
        styles['SectionHeader']
    ))
    elements.append(Spacer(1, 0.15*inch))
    
    # Recall table showing model performance
    recall_table = _create_class_recall_table(
        class_id, ood_results, model_names_map, styles
    )
    elements.append(recall_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Confusion heatmap showing predicted classes
    heatmap_image = _create_confusion_heatmap(
        class_id, class_name, ood_results, model_names_map, category_names
    )
    
    if heatmap_image:
        elements.append(heatmap_image)
    else:
        # Show message if no confusion data available
        elements.append(Paragraph(
            "<i>No detections overlapped with this OOD class</i>",
            styles['Normal']
        ))
    
    elements.append(Spacer(1, 0.2*inch))
    
    return elements


# ============================================================================
# Confusion Heatmap Generation
# ============================================================================

def _collect_confusion_data(
    class_id: int,
    ood_results: Dict[str, Any]
) -> Tuple[Dict[str, Dict[int, Any]], List[int]]:
    """
    Collect and normalize confusion data for a specific OOD class.
    
    Args:
        class_id: Category ID for the OOD class
        ood_results: Dictionary of OOD evaluation results
        
    Returns:
        Tuple of (confusion_data, sorted_predicted_classes) where:
        - confusion_data: Dict mapping model_key to normalized confusion dict
        - sorted_predicted_classes: Sorted list of predicted class IDs
    """
    all_predicted_classes = set()
    confusion_data = {}
    
    for model_key in ood_results.keys():
        class_data = _get_class_data_for_model(model_key, class_id, ood_results)
        
        if class_data and 'confusion' in class_data:
            # Normalize keys to integers for consistent access
            confusion_dict = class_data['confusion']
            normalized_dict = {int(k): v for k, v in confusion_dict.items()}
            
            confusion_data[model_key] = normalized_dict
            all_predicted_classes.update(normalized_dict.keys())
    
    sorted_predicted_classes = sorted(all_predicted_classes)
    
    return confusion_data, sorted_predicted_classes


def _build_confusion_matrix(
    class_id: int,
    ood_results: Dict[str, Any],
    confusion_data: Dict[str, Dict[int, Any]],
    sorted_predicted_classes: List[int]
) -> Tuple[np.ndarray, List[str]]:
    """
    Build confusion matrix and get ordered model keys.
    
    Args:
        class_id: Category ID for the OOD class
        ood_results: Dictionary of OOD evaluation results
        confusion_data: Dictionary mapping model keys to confusion data
        sorted_predicted_classes: Sorted list of predicted class IDs
        
    Returns:
        Tuple of (matrix, model_keys) where:
        - matrix: NumPy array with confusion counts
        - model_keys: List of model keys in the order they appear in the matrix
    """
    # Sort models by recall for this class
    model_recalls = []
    for model_key in ood_results.keys():
        class_data = _get_class_data_for_model(model_key, class_id, ood_results)
        if class_data:
            model_recalls.append((model_key, class_data['recall']))
    
    model_recalls.sort(key=lambda x: x[1], reverse=True)
    model_keys = [m[0] for m in model_recalls]
    
    # Build matrix
    matrix = []
    for model_key in model_keys:
        row = []
        for pred_class_id in sorted_predicted_classes:
            confusion_dict = confusion_data.get(model_key, {})
            count = confusion_dict.get(pred_class_id, {}).get('count', 0)
            row.append(count)
        matrix.append(row)
    
    return np.array(matrix), model_keys


def _create_confusion_heatmap(
    class_id: int,
    class_name: str,
    ood_results: Dict[str, Any],
    model_names_map: Dict[str, str],
    category_names: Dict[int, str]
) -> Optional[RLImage]:
    """
    Create confusion heatmap showing which COCO classes were predicted.
    
    Args:
        class_id: Category ID for the OOD class
        class_name: Display name of the OOD class
        ood_results: Dictionary of OOD evaluation results
        model_names_map: Dictionary mapping model keys to display names
        category_names: Dictionary mapping category IDs to names
        
    Returns:
        ReportLab Image object, or None if no confusion data available
    """
    # Collect and normalize confusion data
    confusion_data, sorted_predicted_classes = _collect_confusion_data(class_id, ood_results)
    
    if not sorted_predicted_classes:
        return None
    
    # Build confusion matrix
    matrix, model_keys = _build_confusion_matrix(
        class_id, ood_results, confusion_data, sorted_predicted_classes
    )
    
    # Calculate figure dimensions for square cells
    n_cols = len(sorted_predicted_classes)
    n_rows = len(model_keys)
    fig_width = n_cols * HEATMAP_CELL_SIZE + HEATMAP_PADDING_WIDTH
    fig_height = n_rows * HEATMAP_CELL_SIZE + HEATMAP_PADDING_HEIGHT
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create heatmap with square cells
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='equal')
    
    # Configure ticks and labels
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    
    # X-axis: predicted COCO class names
    x_labels = [category_names.get(c, f'ID {c}') for c in sorted_predicted_classes]
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=HEATMAP_FONT_SIZE)
    
    # Y-axis: Model 1, Model 2, etc. (matching table order)
    y_labels = [f'Model {i+1}' for i in range(n_rows)]
    ax.set_yticklabels(y_labels, fontsize=HEATMAP_FONT_SIZE)
    
    # Add count annotations to cells
    for i in range(n_rows):
        for j in range(n_cols):
            value = matrix[i, j]
            if value > 0:
                ax.text(
                    j, i, int(value),
                    ha="center", va="center",
                    color="black", fontsize=HEATMAP_ANNOTATION_FONT_SIZE
                )
    
    # Set axis labels and title
    ax.set_xlabel(
        'Predicted Class',
        fontname='DejaVu Serif',
        fontsize=HEATMAP_FONT_SIZE + 1,
        fontweight='bold'
    )
    ax.set_title(
        f'Confusion Matrix for {class_name}',
        fontname='DejaVu Serif',
        fontsize=HEATMAP_TITLE_FONT_SIZE,
        fontweight='bold',
        pad=10
    )
    
    plt.tight_layout()
    
    # Save to buffer and convert to ReportLab image
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=HEATMAP_DPI, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close(fig)
    
    rl_image = RLImage(
        img_buffer,
        width=fig_width * inch,
        height=fig_height * inch
    )
    
    return rl_image

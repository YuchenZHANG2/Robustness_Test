"""
mAP Comparison Table Page
"""
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import io
from .styles import COLORS
from .utils import calculate_metrics


def create_map_comparison_page(results, styles):
    """
    Create the mAP comparison page with results table
    
    Args:
        results: Dictionary of test results for all models
        styles: StyleSheet object with all styles
        
    Returns:
        list: List of flowable elements for the mAP comparison page
    """
    elements = []
    
    # Page title with anchor for TOC linking and section number
    title = Paragraph('<a name="map_comparison"/>1. mAP Comparison (All Categories)', styles['ContentPageTitle'])
    elements.append(title)
    
    # Subtitle
    subtitle = Paragraph(
        "Overall performance comparison across all tested models",
        styles['ContentPageSubtitle']
    )
    elements.append(subtitle)
    
    # Create and add the results table
    elements.append(_create_results_table(results, styles))
    
    # Add spacing
    elements.append(Spacer(1, 0.3*inch))
    
    # Add calculation notes
    elements.extend(_create_calculation_notes(styles))
    
    # Add spacing before spider chart
    elements.append(Spacer(1, 0.4*inch))
    
    # Add centered spider chart
    spider_chart = _create_spider_chart(results, styles)
    if spider_chart:
        # Center the image using a Table with single cell
        from reportlab.platypus import Table as CenterTable
        centered_chart = CenterTable([[spider_chart]], colWidths=[6.5*inch])
        centered_chart.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(centered_chart)
        
        # Add spacing before radar chart notes
        elements.append(Spacer(1, 0.2*inch))
        
        # Add radar chart notes
        elements.extend(_create_radar_chart_notes(styles))
    
    return elements


# Color palette for different models in spider chart
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


def _create_results_table(results, styles):
    """
    Create the main results table with mAP comparison
    
    Args:
        results: Dictionary of test results
        styles: StyleSheet object
        
    Returns:
        Table: Formatted results table
    """
    table_data = []
    
    # Header row
    headers = [
        Paragraph('<b>Model</b>', styles['TableHeader']),
        Paragraph('<b>Clean mAP</b>', styles['TableHeader']),
        Paragraph('<b>Avg Corrupted<br/>mAP</b>', styles['TableHeader']),
        Paragraph('<b>Degradation</b>', styles['TableHeader']),
        Paragraph('<b>Robustness<br/>Score</b>', styles['TableHeader'])
    ]
    table_data.append(headers)
    
    # Sort models by clean mAP (descending - best first)
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]['clean']['mAP'],
        reverse=True
    )
    
    # Data rows
    for model_key, model_data in sorted_models:
        model_name = model_data.get('name', model_key)
        metrics = calculate_metrics(model_data)
        
        if metrics:
            row = [
                Paragraph(f'<b>{model_name}</b>', styles['TableCell']),
                Paragraph(f"{metrics['clean_map']:.3f}", styles['TableCell']),
                Paragraph(f"{metrics['avg_corrupt']:.3f}", styles['TableCell']),
                Paragraph(f"{metrics['degradation']:.3f}", styles['TableCell']),
                Paragraph(f"{metrics['robustness_score']:.1f}%", styles['TableCell'])
            ]
            table_data.append(row)
    
    # Create table with three-line style
    col_widths = [2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch]
    results_table = Table(table_data, colWidths=col_widths)
    
    # Apply styling
    results_table.setStyle(_get_table_style(len(table_data)))
    
    return results_table


def _create_spider_chart(results, styles):
    """
    Create a spider/radar chart comparing models across key metrics
    
    Args:
        results: Dictionary of test results
        styles: StyleSheet object
        
    Returns:
        Image: ReportLab Image object with the spider chart
    """
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
        # Example: if robustness scores are [75, 60], max=75
        #   Model A: (75/75)*100 = 100
        #   Model B: (60/75)*100 = 80
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
    ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize=12,frameon=False)
    
    # Ensure circular shape
    ax.set_aspect('equal')
    
    plt.tight_layout(pad=1.5)
    
    # Convert to image with consistent dimensions to maintain aspect ratio
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150)
    img_buffer.seek(0)
    plt.close(fig)
    
    # Create ReportLab Image with equal width and height
    img = Image(img_buffer, width=4.5*inch, height=4.5*inch)
    
    return img


def _get_table_style(num_rows):
    """
    Get the table style configuration
    
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
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        
        # Data rows alignment
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),  # Model name left-aligned
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),  # Other columns centered
        ('FONTNAME', (0, 1), (-1, -1), 'DejaVuSerif'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        
        # Three-line style: top, header separator, bottom
        ('LINEABOVE', (0, 0), (-1, 0), 1.5, HexColor(COLORS['dark_gray'])),
        ('LINEBELOW', (0, 0), (-1, 0), 1.0, HexColor(COLORS['dark_gray'])),
        ('LINEBELOW', (0, -1), (-1, -1), 1.5, HexColor(COLORS['dark_gray'])),
        
        # Padding
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]
    
    # Add alternating row colors (skip header row)
    for i in range(1, num_rows):
        if i % 2 == 0:
            table_style.append(
                ('BACKGROUND', (0, i), (-1, i), HexColor(COLORS['very_light_gray']))
            )
        else:
            table_style.append(
                ('BACKGROUND', (0, i), (-1, i), HexColor(COLORS['white']))
            )
    
    return TableStyle(table_style)


def _create_calculation_notes(styles):
    """
    Create the calculation notes section
    
    Args:
        styles: StyleSheet object
        
    Returns:
        list: List of flowable elements for notes
    """
    elements = []
    
    # Notes title
    elements.append(Paragraph('<b>Notes:</b>', styles['NotesTitle']))
    
    # Notes content
    notes = [
        'Models in the table are ordered by Clean mAP (performance on uncorrupted data) in descending order (highest first).',
        '<b>Average Corrupted mAP:</b> Mean of mAP across all corruptions and severity levels (1-5)',
        '<b>Degradation:</b> Clean mAP - Average Corrupted mAP (lower is better)',
        '<b>Robustness Score:</b> (Average Corrupted mAP / Clean mAP) × 100% (higher is better, 100% = no degradation)'
    ]
    
    for note in notes:
        bullet = Paragraph(f'• {note}', styles['NotesText'])
        elements.append(bullet)
        elements.append(Spacer(1, 0.08*inch))
    
    return elements


def _create_radar_chart_notes(styles):
    """
    Create the radar chart notes section
    
    Args:
        styles: StyleSheet object
        
    Returns:
        list: List of flowable elements for radar chart notes
    """
    elements = []
    
    # Notes title
    elements.append(Paragraph('<b>Notes:</b>', styles['NotesTitle']))
    
    # Notes content
    notes = [
        '<b>Axes:</b> The radar chart compares models across three key metrics: <b>Clean</b> (performance on uncorrupted data), <b>Corrupted</b> (average performance across all corruptions and severity levels), and <b>Robustness %</b> (percentage of performance retained under corruption).',
        '<b>Normalization:</b> Each axis is independently normalized to its maximum value across all models, then scaled to 100. The model with the highest value on each axis will reach 100 on that axis, while other models scale proportionally.',
        '<b>Interpretation:</b> Larger polygon area indicates better overall robustness. A model touching 100 on an axis has the best performance for that metric. Polygon shape reveals strengths and weaknesses across different aspects.'
    ]
    
    for note in notes:
        bullet = Paragraph(f'• {note}', styles['NotesText'])
        elements.append(bullet)
        elements.append(Spacer(1, 0.08*inch))
    
    return elements

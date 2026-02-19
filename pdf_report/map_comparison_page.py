"""
mAP Comparison Table Page
"""
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch

from .utils import calculate_metrics
from .table_utils import create_three_line_table_style, add_notes_section
from .visualization_utils import create_spider_chart


def create_map_comparison_page(results, styles):
    """
    Create the mAP comparison page with results table and spider chart.
    
    Args:
        results: Dictionary of test results for all models
        styles: StyleSheet object with all styles
        
    Returns:
        list: List of flowable elements for the mAP comparison page
    """
    elements = []
    
    # Page title with anchor for TOC linking and section number
    title = Paragraph(
        '<a name="map_comparison"/>1. mAP Comparison (All Categories)', 
        styles['ContentPageTitle']
    )
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
    _add_calculation_notes(elements, styles)
    
    # Add spacing before spider chart
    elements.append(Spacer(1, 0.4*inch))
    
    # Add centered spider chart
    spider_chart = create_spider_chart(results)
    if spider_chart:
        # Center the image using a Table with single cell
        centered_chart = Table([[spider_chart]], colWidths=[6.5*inch])
        centered_chart.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(centered_chart)
        
        # Add spacing before radar chart notes
        elements.append(Spacer(1, 0.2*inch))
        
        # Add radar chart notes
        _add_radar_chart_notes(elements, styles)
    
    return elements


def _create_results_table(results, styles):
    """
    Create the main results table with mAP comparison.
    
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
    results_table.setStyle(create_three_line_table_style(len(table_data)))
    
    return results_table


def _add_calculation_notes(elements, styles):
    """Add calculation notes to the elements list."""
    notes = [
        'Models in the table are ordered by Clean mAP (performance on uncorrupted data) in descending order (highest first).',
        '<b>Average Corrupted mAP:</b> Mean of mAP across all corruptions and severity levels (1-5)',
        '<b>Degradation:</b> Clean mAP - Average Corrupted mAP (lower is better)',
        '<b>Robustness Score:</b> (Average Corrupted mAP / Clean mAP) × 100% (higher is better, 100% = no degradation)'
    ]
    add_notes_section(elements, notes, styles)



def _add_radar_chart_notes(elements, styles):
    """Add radar chart notes to the elements list."""
    notes = [
        '<b>Axes:</b> The radar chart compares models across three key metrics: <b>Clean</b> (performance on uncorrupted data), <b>Corrupted</b> (average performance across all corruptions and severity levels), and <b>Robustness %</b> (percentage of performance retained under corruption).',
        '<b>Normalization:</b> Each axis is independently normalized to its maximum value across all models, then scaled to 100. The model with the highest value on each axis will reach 100 on that axis, while other models scale proportionally.',
        '<b>Interpretation:</b> Larger polygon area indicates better overall robustness. A model touching 100 on an axis has the best performance for that metric. Polygon shape reveals strengths and weaknesses across different aspects.'
    ]
    add_notes_section(elements, notes, styles)

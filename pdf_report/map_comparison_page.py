"""
mAP Comparison Table Page
"""
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
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
    
    # Add spacing before notes
    elements.append(Spacer(1, 0.3*inch))
    
    # Add calculation notes
    elements.extend(_create_calculation_notes(styles))
    
    return elements


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

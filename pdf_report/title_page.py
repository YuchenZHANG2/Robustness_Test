"""
Title Page Generation
"""
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, FrameBreak, PageBreak, NextPageTemplate
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from .styles import COLORS
from .utils import format_corruption_name


def create_title_page(detectors, corruptions, dataset_name, styles):
    """
    Create the title page of the report with blue background
    
    Args:
        detectors: List of detector names
        corruptions: List of corruption types
        dataset_name: Name of the dataset used
        styles: StyleSheet object with all styles
        
    Returns:
        list: List of flowable elements for the title page
    """
    elements = []
    
    # Add top space
    elements.append(Spacer(1, 1.2*inch))
    
    # Main title (white on blue background)
    title = Paragraph(
        "Robustness Evaluation of<br/>Object Detection Models",
        styles['MainTitle']
    )
    elements.append(title)
    elements.append(Spacer(1, 0.3*inch))
    
    # Add first separator line
    elements.append(Spacer(1, 0.5*inch))
    elements.append(_create_separator_line())
    elements.append(Spacer(1, 0.5*inch))
    
    # Create information table
    elements.append(_create_info_table(detectors, corruptions, dataset_name, styles))
    
    # Add second separator line
    elements.append(Spacer(1, 0.5*inch))
    elements.append(_create_separator_line())
    elements.append(Spacer(1, 0.3*inch))
    
    # Switch to footnote frame at bottom
    elements.append(FrameBreak())
    
    # Add footnote at true bottom of page
    footnote = Paragraph("MOBIMA 107I", styles['Footnote'])
    elements.append(footnote)
    
    # Switch to white template for next page and break
    elements.append(NextPageTemplate('white_template'))
    elements.append(PageBreak())
    
    return elements


def _create_separator_line():
    """
    Create a white horizontal separator line
    
    Returns:
        Table: Separator line as a table
    """
    line_table = Table([['']], colWidths=[5*inch])
    line_table.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 1.5, HexColor(COLORS['white'])),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    return line_table


def _create_info_table(detectors, corruptions, dataset_name, styles):
    """
    Create the information table with tested models, corruptions, and dataset
    
    Args:
        detectors: List of detector names
        corruptions: List of corruption types
        dataset_name: Name of the dataset
        styles: StyleSheet object
        
    Returns:
        Table: Information table
    """
    table_data = []
    
    # 1. Tested Models
    detectors_bullets = '<br/>'.join([f"• {d}" for d in detectors])
    table_data.append([
        Paragraph('<b>Tested Models:</b>', styles['SectionLabel']),
        Paragraph(detectors_bullets, styles['ContentText'])
    ])
    
    # 2. Corruption Types
    formatted_corruptions = [format_corruption_name(c) for c in corruptions]
    corruptions_bullets = '<br/>'.join([f"• {c}" for c in formatted_corruptions])
    table_data.append([
        Paragraph('<b>Corruption Types:</b>', styles['SectionLabel']),
        Paragraph(corruptions_bullets, styles['ContentText'])
    ])
    
    # 3. Dataset (if available)
    if dataset_name:
        table_data.append([
            Paragraph('<b>Evaluation Dataset:</b>', styles['SectionLabel']),
            Paragraph(f"• {dataset_name}", styles['ContentText'])
        ])
    
    # Create table without borders, centered
    info_table = Table(table_data, colWidths=[2*inch, 3*inch], hAlign='CENTER')
    info_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    return info_table

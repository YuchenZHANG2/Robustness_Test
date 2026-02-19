"""
Utility functions for creating and styling tables in PDF reports
"""
from reportlab.platypus import TableStyle
from reportlab.lib.colors import HexColor
from .styles import COLORS


def create_three_line_table_style(num_rows, has_alternating_rows=True):
    """
    Create a standard three-line table style (top border, header separator, bottom border).
    This is the standard academic table format.
    
    Args:
        num_rows: Total number of rows in the table
        has_alternating_rows: Whether to add alternating row background colors
        
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
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),  # First column left-aligned
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
    if has_alternating_rows:
        for i in range(1, num_rows):
            bg_color = COLORS['very_light_gray'] if i % 2 == 0 else COLORS['white']
            table_style.append(
                ('BACKGROUND', (0, i), (-1, i), HexColor(bg_color))
            )
    
    return TableStyle(table_style)


def create_severity_table_style(num_rows):
    """
    Create a table style for severity tables with dual-row per model format.
    Each model has two rows (mAP and Degradation %), with model name spanning both rows.
    
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
    
    # Span model names across their two rows and apply alternating backgrounds
    model_count = 0
    for i in range(1, num_rows, 2):  # Process pairs of rows
        if i + 1 < num_rows:  # Make sure we have both rows
            # Span the model name cell vertically
            table_style.append(('SPAN', (0, i), (0, i+1)))
            
            # Alternating background for model pairs
            bg_color = COLORS['very_light_gray'] if model_count % 2 == 0 else COLORS['white']
            table_style.append(('BACKGROUND', (0, i), (-1, i+1), HexColor(bg_color)))
            model_count += 1
    
    return TableStyle(table_style)


def add_notes_section(elements, notes, styles):
    """
    Add a formatted notes section to the elements list.
    
    Args:
        elements: List of flowable elements to append to
        notes: List of note strings (can include HTML formatting)
        styles: StyleSheet object
    """
    from reportlab.platypus import Paragraph, Spacer
    from reportlab.lib.units import inch
    
    elements.append(Paragraph('<b>Notes:</b>', styles['NotesTitle']))
    
    for note in notes:
        bullet = Paragraph(f'• {note}', styles['NotesText'])
        elements.append(bullet)
        elements.append(Spacer(1, 0.08*inch))

"""
Table of Contents Page
"""
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from .styles import COLORS
from .utils import format_corruption_name


def create_table_of_contents(corruptions, styles, show_page_numbers=True):
    """
    Create a table of contents page with clickable links to sections
    
    Args:
        corruptions: List of corruption names
        styles: StyleSheet object with all styles
        show_page_numbers: Whether to show page numbers (default: True)
        
    Returns:
        list: List of flowable elements for the TOC page
    """
    elements = []
    
    # Page title
    title = Paragraph("Contents", styles['ContentPageTitle'])
    elements.append(title)
    elements.append(Spacer(1, 0.3*inch))
    
    # Calculate page numbers
    page_nums = {
        'title': 1,
        'toc': 2,
        'map_comparison': 3,
    }
    
    # Corruption pages start after mAP comparison
    current_page = 4
    for corruption in sorted(corruptions):
        page_nums[f'corruption_{corruption}'] = current_page
        current_page += 1
    
    # Build TOC entries (two columns: entry and page number)
    toc_data = []
    
    # 1. Overall mAP Comparison
    entry = f'<a href="#map_comparison" color="{COLORS["primary_blue"]}">1. Overall mAP Comparison</a>'
    if show_page_numbers:
        toc_data.append([
            Paragraph(entry, styles['TOCEntry']),
            Paragraph(f'{page_nums["map_comparison"]}', styles['TOCPage'])
        ])
    else:
        toc_data.append([Paragraph(entry, styles['TOCEntry']), Paragraph('', styles['TOCPage'])])
    
    # 2. Individual Corruption Analysis header
    entry = '<b>2. Individual Corruption Analysis</b>'
    toc_data.append([Paragraph(entry, styles['TOCEntry']), Paragraph('', styles['TOCPage'])])
    
    # 2.x Individual corruption entries (indented with sub-numbering)
    for idx, corruption in enumerate(sorted(corruptions), start=1):
        formatted_name = format_corruption_name(corruption)
        anchor_name = f'corruption_{corruption}'
        
        entry = f'<a href="#{anchor_name}" color="{COLORS["primary_blue"]}">&nbsp;&nbsp;&nbsp;&nbsp;2.{idx} {formatted_name}</a>'
        if show_page_numbers:
            toc_data.append([
                Paragraph(entry, styles['TOCEntry']),
                Paragraph(f'{page_nums[anchor_name]}', styles['TOCPage'])
            ])
        else:
            toc_data.append([Paragraph(entry, styles['TOCEntry']), Paragraph('', styles['TOCPage'])])
    
    # Create two-column table (entry + page number)
    toc_table = Table(toc_data, colWidths=[5.5*inch, 1*inch])
    toc_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),  # Left column left-aligned
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),  # Right column (page numbers) right-aligned
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    elements.append(toc_table)
    elements.append(PageBreak())
    
    return elements

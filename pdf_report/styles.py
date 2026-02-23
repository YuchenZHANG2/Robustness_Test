"""
PDF Report Styles and Color Definitions
"""
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.colors import HexColor


# Color constants
COLORS = {
    'primary_blue': '#114584',
    'white': '#ffffff',
    'light_blue': '#e3f2fd',
    'light_gray': '#f0f0f0',
    'very_light_gray': '#f5f5f5',
    'dark_gray': '#333333',
    'medium_gray': '#666666',
    'text_gray': '#555555',
}


def create_styles():
    """
    Create and return all custom paragraph styles for the PDF report
    
    Returns:
        StyleSheet with all custom styles added
    """
    styles = getSampleStyleSheet()
    
    # ===== TITLE PAGE STYLES (Blue background) =====
    styles.add(ParagraphStyle(
        name='MainTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=HexColor(COLORS['white']),
        spaceAfter=8,
        alignment=TA_CENTER,
        fontName='DejaVuSerif-Bold',
        leading=30
    ))
    
    styles.add(ParagraphStyle(
        name='MainSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=HexColor(COLORS['light_blue']),
        spaceAfter=40,
        alignment=TA_CENTER,
        fontName='DejaVuSerif'
    ))
    
    styles.add(ParagraphStyle(
        name='SectionLabel',
        parent=styles['Normal'],
        fontSize=11,
        textColor=HexColor(COLORS['white']),
        spaceAfter=4,
        alignment=TA_LEFT,
        fontName='DejaVuSerif-Bold'
    ))
    
    styles.add(ParagraphStyle(
        name='ContentText',
        parent=styles['Normal'],
        fontSize=10,
        textColor=HexColor(COLORS['light_gray']),
        spaceAfter=6,
        alignment=TA_LEFT,
        fontName='DejaVuSerif',
        leading=14
    ))
    
    styles.add(ParagraphStyle(
        name='Footnote',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        textColor=HexColor(COLORS['light_blue']),
        alignment=TA_CENTER,
        fontName='DejaVuSerif-Italic'
    ))
    
    # ===== CONTENT PAGE STYLES (White background) =====
    styles.add(ParagraphStyle(
        name='ContentPageTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=HexColor(COLORS['primary_blue']),
        spaceAfter=6,
        spaceBefore=0,
        alignment=TA_LEFT,
        fontName='DejaVuSerif-Bold',
        leading=22
    ))
    
    styles.add(ParagraphStyle(
        name='ContentPageSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=HexColor(COLORS['medium_gray']),
        spaceAfter=20,
        alignment=TA_LEFT,
        fontName='DejaVuSerif-Italic',
        leading=13
    ))
    
    styles.add(ParagraphStyle(
        name='TableHeader',
        parent=styles['Normal'],
        fontSize=10,
        textColor=HexColor(COLORS['white']),
        alignment=TA_CENTER,
        fontName='DejaVuSerif-Bold',
        leading=12
    ))
    
    styles.add(ParagraphStyle(
        name='TableCell',
        parent=styles['Normal'],
        fontSize=9,
        textColor=HexColor(COLORS['dark_gray']),
        alignment=TA_CENTER,
        fontName='DejaVuSerif',
        leading=11
    ))
    
    styles.add(ParagraphStyle(
        name='NotesText',
        parent=styles['Normal'],
        fontSize=9,
        textColor=HexColor(COLORS['text_gray']),
        alignment=TA_LEFT,
        fontName='DejaVuSerif',
        leading=12
    ))
    
    styles.add(ParagraphStyle(
        name='NotesTitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=HexColor(COLORS['dark_gray']),
        alignment=TA_LEFT,
        fontName='DejaVuSerif-Bold',
        leading=12,
        spaceAfter=8
    ))
    
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=HexColor(COLORS['dark_gray']),
        spaceAfter=5,
        spaceBefore=12,
        fontName='DejaVuSerif-Bold'
    ))
    
    # Table of Contents styles
    styles.add(ParagraphStyle(
        name='TOCEntry',
        parent=styles['Normal'],
        fontSize=11,
        textColor=HexColor(COLORS['dark_gray']),
        alignment=TA_LEFT,
        fontName='DejaVuSerif',
        leading=16
    ))
    
    styles.add(ParagraphStyle(
        name='TOCPage',
        parent=styles['Normal'],
        fontSize=11,
        textColor=HexColor(COLORS['dark_gray']),
        alignment=TA_LEFT,
        fontName='DejaVuSerif',
        leading=16
    ))
    
    return styles

"""
PDF Report Generator for Robustness Testing
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph, 
                                 Spacer, PageBreak, Table, TableStyle, KeepTogether, FrameBreak)
from reportlab.platypus.flowables import Flowable, HRFlowable
from reportlab.lib.colors import HexColor, Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
import os


class ColoredBox(Flowable):
    """Custom flowable for drawing colored boxes"""
    def __init__(self, width, height, color):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.color = color
    
    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)


class RobustnessReportGenerator:
    """Generate PDF reports for robustness testing results"""
    
    def __init__(self, output_dir='static'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._register_fonts()
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _register_fonts(self):
        """Register DejaVu Serif font family"""
        try:
            pdfmetrics.registerFont(TTFont('DejaVuSerif', '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'))
            pdfmetrics.registerFont(TTFont('DejaVuSerif-Bold', '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf'))
            pdfmetrics.registerFont(TTFont('DejaVuSerif-Italic', '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf'))
            pdfmetrics.registerFont(TTFont('DejaVuSerif-BoldItalic', '/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf'))
        except:
            # Fallback to Helvetica if DejaVu fonts not found
            print("Warning: DejaVu Serif fonts not found, using Helvetica")
            pass
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Main title style
        self.styles.add(ParagraphStyle(
            name='MainTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=HexColor('#ffffff'),  # White text
            spaceAfter=8,
            alignment=TA_CENTER,
            fontName='DejaVuSerif-Bold',
            leading=30
        ))
        
        # Subtitle style (under main title)
        self.styles.add(ParagraphStyle(
            name='MainSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=HexColor('#e3f2fd'),  # Light blue
            spaceAfter=40,
            alignment=TA_CENTER,
            fontName='DejaVuSerif'
        ))
        
        # Section label
        self.styles.add(ParagraphStyle(
            name='SectionLabel',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=HexColor('#ffffff'),  # White
            spaceAfter=4,
            alignment=TA_LEFT,
            fontName='DejaVuSerif-Bold'
        ))
        
        # Content text
        self.styles.add(ParagraphStyle(
            name='ContentText',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=HexColor('#f0f0f0'),  # Very light gray
            spaceAfter=6,
            alignment=TA_LEFT,
            fontName='DejaVuSerif',
            leading=14
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='DejaVuSerif-Bold'
        ))
        
        # Footnote style
        self.styles.add(ParagraphStyle(
            name='Footnote',
            parent=self.styles['Normal'],
            fontSize=8,
            leading=10,
            textColor=HexColor('#b3d9ff'),
            alignment=TA_CENTER,
            fontName='DejaVuSerif-Italic'
        ))
    
    def generate_report(self, detectors, corruptions, results=None, dataset_name=None):
        """
        Generate a PDF report for robustness testing
        
        Args:
            detectors: List of detector names
            corruptions: List of corruption types
            results: Dictionary of test results (optional for now)
            dataset_name: Name of the dataset used
        
        Returns:
            Path to generated PDF file
        """
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"robustness_report_{timestamp}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        # Page dimensions
        width, height = letter
        
        # Define frames: main content and footnote at bottom
        main_frame = Frame(
            inch,
            1.2 * inch,  # Leave space at bottom for footnote
            width - 2 * inch,
            height - 2.2 * inch,
            id="main"
        )
        
        footnote_frame = Frame(
            inch,
            0.3 * inch,  # Bottom margin
            width - 2 * inch,
            0.5 * inch,
            id="footnote"
        )
        
        # Create page template with both frames
        page_template = PageTemplate(id="template", frames=[main_frame, footnote_frame],
                                    onPage=self._add_page_background)
        
        # Create PDF document with BaseDocTemplate
        doc = BaseDocTemplate(filepath, pagesize=letter)
        doc.addPageTemplates([page_template])
        
        # Build content
        story = []
        
        # Title page
        story.extend(self._create_title_page(detectors, corruptions, dataset_name))
        
        # Build PDF
        doc.build(story)
        
        return filepath
    
    def _add_page_background(self, canvas, doc):
        """Add blue background to entire page"""
        canvas.saveState()
        canvas.setFillColor(HexColor("#114584"))  # RGB(0,101,189)
        canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
        canvas.restoreState()
    
    def _create_title_page(self, detectors, corruptions, dataset_name):
        """Create the title page of the report"""
        elements = []
        
        # Add top space
        elements.append(Spacer(1, 1.2*inch))
        
        # Main title (white on blue background)
        title = Paragraph("Robustness Evaluation of<br/>Object Detection Models", self.styles['MainTitle'])
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))
        

        elements.append(Spacer(1, 0.5*inch))
        # White horizontal line separator
        line_table = Table([['']], colWidths=[5*inch])
        line_table.setStyle(TableStyle([
            ('LINEABOVE', (0, 0), (-1, 0), 1.5, HexColor('#ffffff')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        elements.append(line_table)
        elements.append(Spacer(1, 0.5*inch))
        
        # Create 2-column table for information
        table_data = []
        
        # 1. Tested Models
        detectors_bullets = '<br/>'.join([f"• {d}" for d in detectors])
        table_data.append([
            Paragraph('<b>Tested Models:</b>', self.styles['SectionLabel']),
            Paragraph(detectors_bullets, self.styles['ContentText'])
        ])
        
        # 2. Corruption Types
        formatted_corruptions = [c.replace('_', ' ').title() for c in corruptions]
        corruptions_bullets = '<br/>'.join([f"• {c}" for c in formatted_corruptions])
        table_data.append([
            Paragraph('<b>Corruption Types:</b>', self.styles['SectionLabel']),
            Paragraph(corruptions_bullets, self.styles['ContentText'])
        ])
        
        # 3. Dataset - if available
        if dataset_name:
            table_data.append([
                Paragraph('<b>Evaluation Dataset:</b>', self.styles['SectionLabel']),
                Paragraph(f"• {dataset_name}", self.styles['ContentText'])
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
        elements.append(info_table)

        
        elements.append(Spacer(1, 0.5*inch))
        # White horizontal line separator
        line_table = Table([['']], colWidths=[5*inch])
        line_table.setStyle(TableStyle([
            ('LINEABOVE', (0, 0), (-1, 0), 1.5, HexColor('#ffffff')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        elements.append(line_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Switch to footnote frame at bottom
        elements.append(FrameBreak())
        
        # # Add horizontal line separator for footnote
        # elements.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#ffffff')))
        
        # Add footnote at true bottom of page
        footnote = Paragraph("MOBIMA 107I", self.styles['Footnote'])
        elements.append(footnote)
        
        # Page break after title page
        elements.append(PageBreak())
        
        return elements
    
    def _create_info_section(self, label, content):
        """
        Create a simple information section with label and content
        
        Args:
            label: Section label
            content: List of items to display
        """
        elements = []
        
        # Label
        label_para = Paragraph(f"<b>{label}:</b>", self.styles['SectionLabel'])
        elements.append(label_para)
        elements.append(Spacer(1, 0.1*inch))
        
        # Format content into a clean list
        if isinstance(content, list):
            if len(content) <= 5:
                # Show all items
                content_items = content
            else:
                # Show first 5 and count
                content_items = content[:5]
                content_items.append(f"... and {len(content) - 5} more")
            
            content_text = '<br/>'.join([f"  • {item}" for item in content_items])
        else:
            content_text = f"  {str(content)}"
        
        # Content directly on blue background without box
        content_para = Paragraph(content_text, self.styles['ContentText'])
        elements.append(content_para)
        
        return elements


if __name__ == "__main__":
    # Test the PDF generator
    generator = RobustnessReportGenerator()
    
    test_detectors = ['YOLO11', 'Faster R-CNN V2', 'DETR']
    test_corruptions = ['gaussian_noise', 'motion_blur', 'fog', 'jpeg_compression']
    
    pdf_path = generator.generate_report(
        detectors=test_detectors,
        corruptions=test_corruptions,
        dataset_name='COCO Val2017'
    )
    
    print(f"PDF generated: {pdf_path}")

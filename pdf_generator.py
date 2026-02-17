"""
PDF Report Generator for Robustness Testing
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.colors import HexColor
from datetime import datetime
import os


class RobustnessReportGenerator:
    """Generate PDF reports for robustness testing results"""
    
    def __init__(self, output_dir='static'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Main title style
        self.styles.add(ParagraphStyle(
            name='MainTitle',
            parent=self.styles['Heading1'],
            fontSize=28,
            textColor=HexColor('#1a1a1a'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=HexColor('#4a4a4a'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica'
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=18,
            textColor=HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
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
        
        # Create PDF document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build content
        story = []
        
        # Title page
        story.extend(self._create_title_page(detectors, corruptions, dataset_name))
        
        # Build PDF
        doc.build(story)
        
        return filepath
    
    def _create_title_page(self, detectors, corruptions, dataset_name):
        """Create the title page of the report"""
        elements = []
        
        # Add vertical space
        elements.append(Spacer(1, 1.5*inch))
        
        # Main title
        title = Paragraph("Robustness Test Report", self.styles['MainTitle'])
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))
        
        # Format detectors list
        if len(detectors) == 1:
            detectors_text = detectors[0]
        elif len(detectors) == 2:
            detectors_text = f"{detectors[0]} and {detectors[1]}"
        else:
            detectors_text = f"{', '.join(detectors[:-1])}, and {detectors[-1]}"
        
        # Subtitle: Among detectors
        subtitle1 = Paragraph(
            f"<b>Among:</b> {detectors_text}",
            self.styles['Subtitle']
        )
        elements.append(subtitle1)
        elements.append(Spacer(1, 0.2*inch))
        
        # Format corruptions list
        if len(corruptions) == 1:
            corruptions_text = corruptions[0].replace('_', ' ').title()
        elif len(corruptions) == 2:
            corruptions_text = f"{corruptions[0].replace('_', ' ').title()} and {corruptions[1].replace('_', ' ').title()}"
        else:
            formatted_corruptions = [c.replace('_', ' ').title() for c in corruptions]
            corruptions_text = f"{', '.join(formatted_corruptions[:-1])}, and {formatted_corruptions[-1]}"
        
        # Subtitle: Under corruptions
        subtitle2 = Paragraph(
            f"<b>Under:</b> {corruptions_text}",
            self.styles['Subtitle']
        )
        elements.append(subtitle2)
        
        # Add dataset info if available
        if dataset_name:
            elements.append(Spacer(1, 0.3*inch))
            dataset_info = Paragraph(
                f"<b>Dataset:</b> {dataset_name}",
                self.styles['Subtitle']
            )
            elements.append(dataset_info)
        
        # Add generation date
        elements.append(Spacer(1, 1*inch))
        date_text = Paragraph(
            f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            self.styles['Normal']
        )
        date_text.style.alignment = TA_CENTER
        date_text.style.fontSize = 10
        date_text.style.textColor = HexColor('#7f8c8d')
        elements.append(date_text)
        
        # Page break after title page
        elements.append(PageBreak())
        
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

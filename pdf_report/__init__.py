"""
PDF Report Generator for Robustness Testing

This module provides a clean interface for generating PDF reports
from robustness evaluation results.
"""
from reportlab.lib.pagesizes import letter
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, PageBreak
from reportlab.lib.units import inch
from datetime import datetime
import os

from .styles import create_styles
from .components import add_blue_background, add_white_background
from .utils import register_fonts, select_qualitative_images, reset_qualitative_images
from .title_page import create_title_page
from .table_of_contents_page import create_table_of_contents
from .map_comparison_page import create_map_comparison_page
from .corruption_detail_page import create_corruption_detail_page


class RobustnessReportGenerator:
    """
    Generate PDF reports for robustness testing results
    
    This class handles the creation of comprehensive PDF reports containing:
    - Title page with project information
    - mAP comparison tables
    - Detailed results by corruption type
    - Statistical analysis and visualizations
    """
    
    def __init__(self, output_dir='static'):
        """
        Initialize the report generator
        
        Args:
            output_dir: Directory where PDF files will be saved
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        register_fonts()
        self.styles = create_styles()
    
    def generate_report(self, detectors, corruptions, results=None, dataset_name=None,
                       model_loader=None, evaluator=None, corruptor=None, category_names=None,
                       include_qualitative=True, num_qualitative_images=3):
        """
        Generate a complete PDF report for robustness testing
        
        Args:
            detectors: List of detector names tested
            corruptions: List of corruption types applied
            results: Dictionary of test results (optional)
            dataset_name: Name of the dataset used (optional)
            model_loader: ModelLoader instance (optional, for qualitative examples)
            evaluator: COCOEvaluator instance (optional, for qualitative examples)
            corruptor: TorchCorruptions instance (optional, for qualitative examples)
            category_names: Dict mapping category IDs to names (optional, for qualitative examples)
            include_qualitative: Whether to include qualitative examples (default: True)
            num_qualitative_images: Number of images to show in qualitative section (default: 3)
        
        Returns:
            str: Path to the generated PDF file
        """
        # Reset any previously selected images
        reset_qualitative_images()
        
        # Select random images for qualitative examples if requested
        if include_qualitative and all([model_loader, evaluator, corruptor, category_names]):
            print(f"Selecting {num_qualitative_images} random images for qualitative examples...")
            select_qualitative_images(evaluator, num_images=num_qualitative_images)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"robustness_report_{timestamp}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        # Create document with page templates
        doc = self._create_document(filepath)
        
        # Build content
        story = []
        
        # 1. Title page (blue background)
        story.extend(create_title_page(
            detectors=detectors,
            corruptions=corruptions,
            dataset_name=dataset_name,
            styles=self.styles
        ))
        
        # 2. Table of Contents (white background) - if results provided
        if results:
            # Collect all unique corruptions from results
            all_corruptions = set()
            for model_data in results.values():
                if 'corrupted' in model_data:
                    all_corruptions.update(model_data['corrupted'].keys())
            
            story.extend(create_table_of_contents(
                corruptions=all_corruptions,
                styles=self.styles,
                show_page_numbers=True
            ))
        
        # 3. mAP Comparison page (white background) - if results provided
        if results:
            story.extend(create_map_comparison_page(
                results=results,
                styles=self.styles
            ))
            story.append(PageBreak())
            
            # 4. Corruption detail pages - one page per corruption
            # Generate a page for each corruption with section numbering
            can_generate_qualitative = include_qualitative and all([model_loader, evaluator, corruptor, category_names])
            
            for idx, corruption in enumerate(sorted(all_corruptions), start=1):
                story.extend(create_corruption_detail_page(
                    corruption_name=corruption,
                    results=results,
                    styles=self.styles,
                    section_number=f'2.{idx}',
                    model_loader=model_loader if can_generate_qualitative else None,
                    evaluator=evaluator if can_generate_qualitative else None,
                    corruptor=corruptor if can_generate_qualitative else None,
                    category_names=category_names if can_generate_qualitative else None
                ))
        
        # Build PDF
        doc.build(story)
        
        return filepath
    
    def _create_document(self, filepath):
        """
        Create a BaseDocTemplate with configured page templates
        
        Args:
            filepath: Path where the PDF will be saved
            
        Returns:
            BaseDocTemplate: Configured document
        """
        width, height = letter
        
        # Create document
        doc = BaseDocTemplate(filepath, pagesize=letter)
        
        # === Blue template (for title page) ===
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
        
        blue_template = PageTemplate(
            id="blue_template",
            frames=[main_frame, footnote_frame],
            onPage=add_blue_background
        )
        
        # === White template (for content pages) ===
        white_frame = Frame(
            inch,
            inch,
            width - 2 * inch,
            height - 2 * inch,
            id="white_main"
        )
        
        white_template = PageTemplate(
            id="white_template",
            frames=[white_frame],
            onPage=add_white_background
        )
        
        # Add templates to document
        doc.addPageTemplates([blue_template, white_template])
        
        return doc


# Export main class
__all__ = ['RobustnessReportGenerator']

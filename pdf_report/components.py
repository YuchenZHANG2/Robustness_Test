"""
Reusable components for PDF generation
"""
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from .styles import COLORS


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


def add_blue_background(canvas, doc):
    """
    Add blue background to entire page (for title page)
    
    Args:
        canvas: ReportLab canvas object
        doc: Document object
    """
    canvas.saveState()
    canvas.setFillColor(HexColor(COLORS['primary_blue']))
    canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
    canvas.restoreState()


def add_white_background(canvas, doc):
    """
    Add white background to entire page (for content pages)
    
    Args:
        canvas: ReportLab canvas object
        doc: Document object
    """
    canvas.saveState()
    canvas.setFillColor(HexColor(COLORS['white']))
    canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
    canvas.restoreState()

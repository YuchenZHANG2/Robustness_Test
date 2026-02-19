"""
Corruption Detail Page - Individual corruption analysis with plots and tables
"""
from reportlab.platypus import Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PDF generation
import matplotlib.pyplot as plt
import io
from .styles import COLORS
from .utils import format_corruption_name


# Color palette for different models in plots
MODEL_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
]


def create_corruption_detail_page(corruption_name, results, styles, section_number=None):
    """
    Create a detailed analysis page for a specific corruption type
    
    Args:
        corruption_name: Name of the corruption (e.g., 'gaussian_noise')
        results: Dictionary of test results for all models
        styles: StyleSheet object with all styles
        section_number: Section number for this corruption (e.g., "2.1")
        
    Returns:
        list: List of flowable elements for the corruption detail page
    """
    elements = []
    
    # Page title with anchor for TOC linking and section number
    formatted_name = format_corruption_name(corruption_name)
    anchor_name = f'corruption_{corruption_name}'
    
    if section_number:
        title = Paragraph(f'<a name="{anchor_name}"/>{section_number} {formatted_name}', styles['ContentPageTitle'])
    else:
        title = Paragraph(f'<a name="{anchor_name}"/>{formatted_name}', styles['ContentPageTitle'])
    
    elements.append(title)
    
    # Subtitle
    subtitle = Paragraph(
        f"Performance degradation analysis across severity levels (0-5)",
        styles['ContentPageSubtitle']
    )
    elements.append(subtitle)
    
    # Generate and add the line plot
    plot_image = _create_severity_plot(corruption_name, results)
    if plot_image:
        elements.append(plot_image)
        elements.append(Spacer(1, 0.2*inch))
    
    # Create and add the severity table
    elements.append(_create_severity_table(corruption_name, results, styles))
    
    # Add page break for next corruption
    elements.append(PageBreak())
    
    return elements


def _create_severity_plot(corruption_name, results):
    """
    Create a line plot showing mAP across severity levels
    
    Args:
        corruption_name: Name of the corruption
        results: Dictionary of test results
        
    Returns:
        Image: ReportLab Image object with the plot
    """
    # Set up the plot with taller height
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    severities = [0, 1, 2, 3, 4, 5]
    
    # Plot each model
    for idx, (model_key, model_data) in enumerate(results.items()):
        model_name = model_data.get('name', model_key)
        
        # Check if this model has data for this corruption
        if corruption_name not in model_data.get('corrupted', {}):
            continue
        
        mAP_values = []
        
        # Severity 0 (clean)
        mAP_values.append(model_data['clean']['mAP'])
        
        # Severities 1-5
        corruption_data = model_data['corrupted'][corruption_name]
        for severity in range(1, 6):
            severity_key = str(severity)
            if severity_key in corruption_data:
                mAP_values.append(corruption_data[severity_key]['mAP'])
            else:
                mAP_values.append(None)
        
        # Plot the line
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        ax.plot(severities, mAP_values, marker='o', linewidth=2, 
                label=model_name, color=color, markersize=6)
    
    # Styling
    ax.set_xlabel('Severity Level', fontsize=10, fontweight='bold')
    ax.set_ylabel('mAP', fontsize=10, fontweight='bold')
    ax.set_xticks(severities)
    ax.set_xticklabels(['Clean', '1', '2', '3', '4', '5'])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9, framealpha=0.9, frameon=False)
    
    plt.tight_layout()
    
    # Convert plot to image
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close(fig)
    
    # Create ReportLab Image (width ~6.5 inches, taller height)
    img = Image(img_buffer, width=6.5*inch, height=3.2*inch)
    
    return img


def _create_severity_table(corruption_name, results, styles):
    """
    Create a table showing mAP values and degradation across severities
    Each model has two rows: absolute mAP and relative degradation
    Model name column spans both rows, with a Metric column showing mAP/Deg%
    
    Args:
        corruption_name: Name of the corruption
        results: Dictionary of test results
        styles: StyleSheet object
        
    Returns:
        Table: Formatted severity table
    """
    table_data = []
    
    # Header row
    headers = [
        Paragraph('<b>Model</b>', styles['TableHeader']),
        Paragraph('<b>Metric</b>', styles['TableHeader'])
    ]
    for severity in range(6):
        headers.append(Paragraph(f'<b>Sev {severity}</b>', styles['TableHeader']))
    table_data.append(headers)
    
    # Sort models by clean mAP (descending)
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]['clean']['mAP'],
        reverse=True
    )
    
    # Data rows - each model gets two rows
    for model_key, model_data in sorted_models:
        # Skip if this model doesn't have data for this corruption
        if corruption_name not in model_data.get('corrupted', {}):
            continue
        
        model_name = model_data.get('name', model_key)
        clean_map = model_data['clean']['mAP']
        corruption_data = model_data['corrupted'][corruption_name]
        
        # Row 1: Absolute mAP values
        map_row = [
            Paragraph(f'<b>{model_name}</b>', styles['TableCell']),
            Paragraph('mAP', styles['TableCell'])
        ]
        
        # Severity 0 (clean)
        map_row.append(Paragraph(f'{clean_map:.3f}', styles['TableCell']))
        
        # Severities 1-5
        map_values = []
        for severity in range(1, 6):
            severity_key = str(severity)
            if severity_key in corruption_data:
                map_val = corruption_data[severity_key]['mAP']
                map_values.append(map_val)
                map_row.append(Paragraph(f'{map_val:.3f}', styles['TableCell']))
            else:
                map_values.append(None)
                map_row.append(Paragraph('N/A', styles['TableCell']))
        
        table_data.append(map_row)
        
        # Row 2: Relative degradation (%)
        deg_row = [
            '',  # Empty for model name (will be spanned from above)
            Paragraph('Deg %', styles['TableCell'])
        ]
        
        # Severity 0 (clean) - 0% degradation
        deg_row.append(Paragraph('0.0%', styles['TableCell']))
        
        # Severities 1-5
        for map_val in map_values:
            if map_val is not None:
                degradation_pct = ((map_val - clean_map) / clean_map * 100)
                deg_row.append(Paragraph(f'{degradation_pct:+.1f}%', styles['TableCell']))
            else:
                deg_row.append(Paragraph('N/A', styles['TableCell']))
        
        table_data.append(deg_row)
    
    # Create table
    col_widths = [1.4*inch, 0.7*inch] + [0.7*inch] * 6
    severity_table = Table(table_data, colWidths=col_widths)
    
    # Apply styling
    severity_table.setStyle(_get_severity_table_style(len(table_data)))
    
    return severity_table


def _get_severity_table_style(num_rows):
    """
    Get the table style configuration for severity table
    
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
    
    # Span model names across their two rows
    model_count = 0
    for i in range(1, num_rows, 2):  # Process pairs of rows
        if i + 1 < num_rows:  # Make sure we have both rows
            # Span the model name cell vertically
            table_style.append(('SPAN', (0, i), (0, i+1)))
            
            # Alternating background for model pairs
            if model_count % 2 == 0:
                # Light background for both rows of this model
                table_style.append(('BACKGROUND', (0, i), (-1, i+1), HexColor(COLORS['very_light_gray'])))
            else:
                # White background for both rows of this model
                table_style.append(('BACKGROUND', (0, i), (-1, i+1), HexColor(COLORS['white'])))
            model_count += 1
    
    return TableStyle(table_style)

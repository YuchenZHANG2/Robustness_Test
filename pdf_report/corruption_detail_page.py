"""
Corruption Detail Page - Individual corruption analysis with plots and tables
"""
from reportlab.platypus import Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch

from .utils import format_corruption_name
from .table_utils import create_severity_table_style
from .visualization_utils import create_severity_line_plot
from .qualitative_examples import create_qualitative_grids
from reportlab.platypus import Table


def create_corruption_detail_page(corruption_name, results, styles, section_number=None,
                                 model_loader=None, evaluator=None, corruptor=None, 
                                 category_names=None):
    """
    Create a detailed analysis page for a specific corruption type.
    
    Args:
        corruption_name: Name of the corruption (e.g., 'gaussian_noise')
        results: Diction of test results for all models
        styles: StyleSheet object with all styles
        section_number: Section number for this corruption (e.g., "2.1")
        model_loader: ModelLoader instance (optional, for qualitative examples)
        evaluator: COCOEvaluator instance (optional, for qualitative examples)
        corruptor: TorchCorruptions instance (optional, for qualitative examples)
        category_names: Dict mapping category IDs to names (optional, for qualitative examples)
        
    Returns:
        list: List of flowable elements for the corruption detail page
    """
    elements = []
    
    # Page title with anchor and section number
    formatted_name = format_corruption_name(corruption_name)
    anchor_name = f'corruption_{corruption_name}'
    
    if section_number:
        title = Paragraph(
            f'<a name="{anchor_name}"/>{section_number} {formatted_name}', 
            styles['ContentPageTitle']
        )
    else:
        title = Paragraph(
            f'<a name="{anchor_name}"/>{formatted_name}', 
            styles['ContentPageTitle']
        )
    
    elements.append(title)
    
    # Subtitle
    subtitle = Paragraph(
        f"Performance degradation analysis across severity levels (0-5)",
        styles['ContentPageSubtitle']
    )
    elements.append(subtitle)
    
    # Generate and add the line plot
    plot_image = create_severity_line_plot(corruption_name, results)
    if plot_image:
        elements.append(plot_image)
        elements.append(Spacer(1, 0.2*inch))
    
    # Create and add the severity table
    elements.append(_create_severity_table(corruption_name, results, styles))
    
    # Add page break after table, before qualitative examples
    elements.append(PageBreak())
    
    # Add qualitative examples if all required objects are provided
    if all([model_loader, evaluator, corruptor, category_names]):
        qualitative_elements = create_qualitative_grids(
            corruption_name, model_loader, evaluator, corruptor, 
            category_names, styles
        )
        elements.extend(qualitative_elements)
    
    # Add page break for next corruption
    elements.append(PageBreak())
    
    return elements


def _create_severity_table(corruption_name, results, styles):
    """
    Create a table showing mAP values and degradation across severities.
    Each model has two rows: absolute mAP and relative degradation.
    
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
        
        # Add rows for this model
        _add_model_rows(table_data, model_name, clean_map, corruption_data, styles)
    
    # Create table
    col_widths = [1.4*inch, 0.7*inch] + [0.7*inch] * 6
    severity_table = Table(table_data, colWidths=col_widths)
    
    # Apply styling
    severity_table.setStyle(create_severity_table_style(len(table_data)))
    
    return severity_table


def _add_model_rows(table_data, model_name, clean_map, corruption_data, styles):
    """
    Add mAP and degradation rows for a single model to the table data.
    
    Args:
        table_data: List of table data rows
        model_name: Name of the model
        clean_map: Clean mAP value
        corruption_data: Dictionary of corruption data by severity
        styles: StyleSheet object
    """
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

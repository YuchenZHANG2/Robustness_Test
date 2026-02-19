"""
Constants used across the PDF report module
"""

# Color palette for different models in plots and charts
# Used consistently across spider charts, line plots, and qualitative visualizations
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

# Severity levels for corruption testing
SEVERITY_LEVELS = [0, 1, 2, 3, 4, 5]
SEVERITY_LABELS = ['Clean', '1', '2', '3', '4', '5']

# Default thresholds for detection and visualization
DEFAULT_SCORE_THRESHOLD = 0.3

# Grid visualization settings
GRID_IMAGE_PADDING = 5

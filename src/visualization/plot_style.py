"""
Shared plotting style utilities for report-quality figures.
"""

from pathlib import Path
import matplotlib.pyplot as plt


CLASS_COLORS = {
    "healthy": "#2E8B57",  # green
    "sick": "#B22222",     # red
}

CLASS_LABELS = {
    "healthy": "Healthy",
    "sick": "Sick",
}


def set_report_style():
    """
    Apply consistent Matplotlib styling for the report.
    """
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": False,
    })


def save_figure(fig, output_path):
    """
    Save a figure as PNG with tight bounding box.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output_path
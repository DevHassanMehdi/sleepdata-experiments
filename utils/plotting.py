# =============================================================================
# utils/plotting.py — Publication-ready matplotlib/seaborn style helpers
# =============================================================================
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# set_thesis_style
# ---------------------------------------------------------------------------

def set_thesis_style():
    """Apply a clean, publication-ready style to all subsequent matplotlib figures.

    Settings applied:
    - White background (no grey grid)
    - Sans-serif font family
    - Default figure size 12 × 6 inches
    - Top and right spines hidden via seaborn despine
    """
    sns.set_theme(style="white", font="sans-serif")
    plt.rcParams.update({
        "figure.figsize":    (12, 6),
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "sans-serif",
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "axes.titlesize":    14,
        "axes.labelsize":    12,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
    })


# ---------------------------------------------------------------------------
# save_figure
# ---------------------------------------------------------------------------

def save_figure(fig, output_dir, filename):
    """Save *fig* as an SVG file to *output_dir/filename*, then close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    output_dir : str or Path
        Destination directory.  Created (with parents) if it does not exist.
    filename : str
        Output filename, e.g. ``"01_missing_values_heatmap.svg"``.
        A ``.svg`` extension is appended automatically if not present.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not filename.endswith(".svg"):
        filename = filename + ".svg"

    out_path = output_dir / filename
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Figure saved → {out_path}")

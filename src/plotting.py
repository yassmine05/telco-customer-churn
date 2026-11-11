"""Shared matplotlib style so every figure in the report looks consistent."""
import matplotlib

matplotlib.use("Agg")  # headless backend: figures are saved, never shown
import matplotlib.pyplot as plt

import config

# Palette: one colour per job, never cycled.
BLUE = "#2a78d6"     # primary series / XGBoost
ORANGE = "#eb6834"   # second series / baseline
GREY = "#8a8985"     # reference lines (random, overall rate)
INK = "#0b0b0b"
INK_SOFT = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SOFT,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "text.color": INK,
        "font.size": 10.5,
        "legend.frameon": False,
        "lines.linewidth": 2,
    })


def save(fig, name: str):
    """Save a figure into reports/figures and close it."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(config.ROOT)}")
    return path

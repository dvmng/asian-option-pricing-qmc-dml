from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def set_thesis_style() -> None:
    """Apply common visual defaults for thesis figures.

    Figure-specific line widths, markers, transparency, bins and scales remain
    explicit in each plotting routine.
    """
    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.edgecolor": "black",
        "axes.linewidth": 1,
        "grid.color": "grey",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.4,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig, path: str | Path, *, also_png: bool = True) -> list[Path]:
    """Save a thesis figure as vector PDF and optionally a 300-DPI PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = path.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    saved = [pdf]
    if also_png:
        png = path.with_suffix(".png")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        saved.append(png)
    return saved

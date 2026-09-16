"""Shared matplotlib style for the ResumeBiasBench publication figures.

Static, two-column-paper figures: no seaborn, no interactivity, no dark mode.
Import `apply_style()` once per script before creating any figure, and use
the color/marker constants and helpers below so all eight figures read as
one consistent set.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Palette ──────────────────────────────────────────────────────────────
# Exactly two categorical colors. Colorblind-safe, separates in grayscale.
HUMAN_COLOR = "#2a78d6"
AI_COLOR = "#eb6834"
# Third color only when genuinely unavoidable; always gets a direct label
# because it sits below 3:1 contrast on white.
THIRD_COLOR = "#1baf7a"

HUMAN_MARKER = "o"
AI_MARKER = "^"
THIRD_MARKER = "s"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#dddad4"
AXIS_COLOR = "#b7b4ae"
PARITY_COLOR = "#8a8880"

SINGLE_COL_WIDTH = 3.5
FULL_WIDTH = 7.2

MARKER_SIZE = 6.5  # points; ~8-9px at 300dpi given default marker scaling
MARKER_EDGE_WIDTH = 0.9
LINE_WIDTH = 2.0


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 8,
            "font.family": "DejaVu Sans",
            "text.color": INK_PRIMARY,
            "axes.edgecolor": AXIS_COLOR,
            "axes.labelcolor": INK_PRIMARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_fig(fig: "plt.Figure", out_path_no_ext: str) -> None:
    """Save both a 300dpi PNG and a vector PDF at the same basename."""
    out_path_no_ext = str(out_path_no_ext)
    Path(out_path_no_ext).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_path_no_ext}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_path_no_ext}.pdf", bbox_inches="tight")


def style_grid_x(ax) -> None:
    """One-direction, recessive gridlines drawn behind the marks."""
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def style_grid_y(ax) -> None:
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a simple binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (center - half, center + half)


def ring_kwargs(color: str) -> dict:
    """Marker kwargs giving a thin white ring so overlapping markers read as
    two points rather than a blob."""
    return dict(
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=MARKER_EDGE_WIDTH,
        markersize=MARKER_SIZE,
    )

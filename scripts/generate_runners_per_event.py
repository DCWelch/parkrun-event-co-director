#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate parkrun-style visualizations for per-event counts:
- Runners (runner == 1)
- Participants (participant == 1)  [runners OR volunteers]
- Volunteers (volunteer == 1)

Outputs:
  visualizations/event_counts_series.csv      # event, runners, participants, volunteers
  visualizations/runners_per_event.png
  visualizations/participants_per_event.png
  visualizations/volunteers_per_event.png
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import MaxNLocator
from PIL import Image


# ------------------------- Paths -------------------------

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EVENT_DIR = DATA_DIR / "event_results"
ASSETS_DIR = PROJECT_ROOT / "assets"
VIS_DIR = PROJECT_ROOT / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

SERIES_CSV = VIS_DIR / "event_counts_series.csv"
PLOT_RUNNERS = VIS_DIR / "runners_per_event.png"
PLOT_PARTICIPANTS = VIS_DIR / "participants_per_event.png"
PLOT_VOLUNTEERS = VIS_DIR / "volunteers_per_event.png"
LOGO_PATH = ASSETS_DIR / "parkrun_logo_white.png"


# ------------------------- Colors & Styling -------------------------

PARKRUN_PURPLE = "#4B2E83"  # deep purple
PARKRUN_YELLOW = "#FFA300"  # parkrun yellow
TEAL           = "#00DBC9"  # parkrun teal
NEAR_WHITE     = "#F4F4F6"  # almost white for text/grid

TITLE_SIZE   = 26
LABEL_SIZE   = 18
TICK_SIZE    = 16
AXIS_XY_LW   = 2.8
AXIS_TR_LW   = 1.2
GRID_LW      = 1.6
LINE_LW      = 3.2
LEGEND_FS    = 14
GRID_ALPHA   = 0.55
LOGO_ALPHA   = 0.12


# ------------------------- Load Data -------------------------

def load_all_events() -> pd.DataFrame:
    """Load all event CSVs and concatenate into one DataFrame."""
    rows = []
    for fn in sorted(EVENT_DIR.glob("event_*.csv")):
        m = re.search(r"event_(\d{4})\.csv$", fn.name)
        if not m:
            continue
        evno = int(m.group(1))
        try:
            df = pd.read_csv(fn, dtype={"id": "string"})
        except Exception:
            continue
        df["event"] = evno
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ------------------------- Plot helpers -------------------------

def _add_centered_background_logo(fig: plt.Figure, alpha: float = LOGO_ALPHA):
    if not LOGO_PATH.exists():
        return
    try:
        img = Image.open(LOGO_PATH).convert("RGBA")
    except Exception:
        return
    fig_w, fig_h = fig.get_size_inches()
    fig_h_px = fig_h * fig.dpi
    zoom = fig_h_px / img.height  # make image height ≈ figure height
    oi = OffsetImage(img, zoom=zoom, alpha=alpha)
    ab = AnnotationBbox(oi, (0.5, 0.5), xycoords=fig.transFigure, frameon=False, zorder=0)
    fig.add_artist(ab)


def _apply_axes_style(ax: plt.Axes):
    fig = ax.figure
    fig.patch.set_facecolor(PARKRUN_PURPLE)
    ax.set_facecolor(PARKRUN_PURPLE)

    # grid
    ax.grid(True, linestyle="--", linewidth=GRID_LW, color=NEAR_WHITE, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)

    # spines
    ax.spines["bottom"].set_color(NEAR_WHITE)
    ax.spines["left"].set_color(NEAR_WHITE)
    ax.spines["bottom"].set_linewidth(AXIS_XY_LW)
    ax.spines["left"].set_linewidth(AXIS_XY_LW)
    for sp in ["top", "right"]:
        ax.spines[sp].set_color(NEAR_WHITE)
        ax.spines[sp].set_alpha(0.4)
        ax.spines[sp].set_linewidth(AXIS_TR_LW)

    # ticks
    ax.tick_params(colors=NEAR_WHITE, which="both", width=1.6, length=6, labelsize=TICK_SIZE)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")


# ------------------------- Compute Series -------------------------

def compute_event_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return DataFrame with columns: event, runners, participants, volunteers
    runners      : runner == 1
    participants : participant == 1 (ran OR volunteered)
    volunteers   : volunteer == 1
    """
    if df.empty:
        return pd.DataFrame(columns=["event", "runners", "participants", "volunteers"])

    # Ensure numeric flags
    for col in ["runner", "participant", "volunteer"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Group counts
    runners = df[df["runner"] == 1].groupby("event", as_index=False).size().rename(columns={"size": "runners"})
    participants = (
        df[df["participant"] == 1].groupby("event", as_index=False).size().rename(columns={"size": "participants"})
    )
    volunteers = df[df["volunteer"] == 1].groupby("event", as_index=False).size().rename(columns={"size": "volunteers"})

    # Merge and fill missing events
    all_events = pd.DataFrame({"event": sorted(df["event"].unique())})
    full = (
        all_events
        .merge(runners, on="event", how="left")
        .merge(participants, on="event", how="left")
        .merge(volunteers, on="event", how="left")
        .fillna(0)
    )
    full[["runners", "participants", "volunteers"]] = full[["runners", "participants", "volunteers"]].astype(int)
    return full


# ------------------------- Plot -------------------------

def plot_single_series(full: pd.DataFrame, col: str, title: str, color: str, outpath: Path):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    _add_centered_background_logo(fig, alpha=LOGO_ALPHA)

    ax.plot(full["event"], full[col], color=color, linewidth=LINE_LW, label=title)

    # y limits with padding
    ymin, ymax = 0, int(full[col].max())
    pad = max(1, int(round(0.06 * (ymax - ymin if ymax > ymin else 10))))
    ax.set_ylim(0, ymax + pad)

    ax.set_xlabel("Event number", color=NEAR_WHITE, fontweight="bold",
                  fontsize=LABEL_SIZE, labelpad=10)
    ax.set_ylabel("Count", color=NEAR_WHITE, fontweight="bold",
                  fontsize=LABEL_SIZE, labelpad=10)
    ax.set_title(title, color=NEAR_WHITE, fontweight="bold",
                 fontsize=TITLE_SIZE, pad=22)

    _apply_axes_style(ax)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Legend
    leg = ax.legend(facecolor=PARKRUN_PURPLE, edgecolor=NEAR_WHITE,
                    fontsize=LEGEND_FS)
    for txt in leg.get_texts():
        txt.set_color(NEAR_WHITE)
        txt.set_fontweight("bold")
    leg.get_frame().set_alpha(0.0)

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    fig.savefig(outpath, dpi=160, facecolor=PARKRUN_PURPLE)
    plt.close(fig)


# ------------------------- Main -------------------------

def main():
    df = load_all_events()
    if df.empty:
        print(f"No event CSVs found in: {EVENT_DIR}")
        return

    series = compute_event_counts(df)
    series.to_csv(SERIES_CSV, index=False)
    print(f"Wrote event counts CSV -> {SERIES_CSV}")

    # Use NEAR_WHITE for all three series so they’re not gender-coded
    plot_single_series(series, "runners", "Runners per Event", NEAR_WHITE, PLOT_RUNNERS)
    print(f"Wrote runners plot -> {PLOT_RUNNERS}")

    plot_single_series(series, "participants", "Participants per Event", NEAR_WHITE, PLOT_PARTICIPANTS)
    print(f"Wrote participants plot -> {PLOT_PARTICIPANTS}")

    plot_single_series(series, "volunteers", "Volunteers per Event", NEAR_WHITE, PLOT_VOLUNTEERS)
    print(f"Wrote volunteers plot -> {PLOT_VOLUNTEERS}")


if __name__ == "__main__":
    main()

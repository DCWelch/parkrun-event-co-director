#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parkrun_config.py
Common configuration file for the various scripts in parkrun-event-co-director
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# ============================================================
#   EVENT DEFAULTS
# ============================================================

# Default root event URL used by the organizer script
ROOT_URL_DEFAULT: str = "https://www.parkrun.us/farmpond/"

# Human-friendly event name (used to title visualizations)
#     Note: Exclude "parkrun"... that is added automatically
EVENT_NAME_DEFAULT: str = "Farm Pond"

# Defaults for event range used by the organizer script
START_EVENT_DEFAULT: int | None = 1     # First parkrun event to consider
END_EVENT_DEFAULT: int | None   = None  # Last parkrun event to consider

# Default "Top N" size for leaderboard tables
LEADERBOARD_TOP_N_DEFAULT: int = 10

# ============================================================
#   VISUAL STYLE CONSTANTS
# ============================================================

# Brand-ish colors
PARKRUN_PURPLE: str = "#4B2E83"   # background
NEAR_WHITE: str     = "#F4F4F6"   # labels/grid/text
PARKRUN_YELLOW: str = "#FFA300"   # male highlight line / cell
PARKRUN_TEAL: str   = "#10ECCC"   # female highlight line / cell

# Typographic & layout constants
TITLE_SIZE: int  = 26
LABEL_SIZE: int  = 18
TICK_SIZE: int   = 16
HEADER_SIZE: int = 14
CELL_SIZE: int   = 12

# Axes / grid / line widths
AXIS_XY_LW: float = 2.8
AXIS_TR_LW: float = 1.2
GRID_LW: float    = 1.6
LINE_LW: float    = 3.2
BORDER_LW: float  = 1.2

# Alphas
GRID_ALPHA: float = 0.55
LOGO_ALPHA: float = 0.12
BBOX_ALPHA: float = 0.50   # for annotation boxes on progression charts

# Small vertical tweak used by table scripts
CELL_Y_OFFSET: float = 0.0

# ============================================================
#   PROJECT PATHS & FILENAMES
# ============================================================

# Root of the project (directory that contains parkrun_event_data_organizer.py)
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Data + results
DATA_DIR: Path           = PROJECT_ROOT / "data"
EVENT_DIR: Path          = DATA_DIR / "event_results"
AGEGROUP_DIR: Path       = DATA_DIR / "age_group_summaries"
EVENT_SERIES_CSV: Path   = DATA_DIR / "event_series_summary.csv"
MASTER_PARTICIPANTS_CSV: Path = DATA_DIR / "participants_master.csv"
PARKRUNS_MASTER_CSV      = DATA_DIR / "parkruns_master.csv"
VOLUNTEERS_MASTER_CSV    = DATA_DIR / "volunteers_master.csv"

# Visualizations + assets
VIS_DIR: Path            = PROJECT_ROOT / "visualizations"
LEADERBOARD_DIR: Path    = VIS_DIR / "leaderboards"
AGEGROUP_VIS_DIR: Path   = VIS_DIR / "agegroup_course_records"
ASSETS_DIR: Path         = PROJECT_ROOT / "assets"
PARKRUN_LOGO: Path       = ASSETS_DIR / "parkrun_logo_white.png"

# Scripts directory (used by parkrun_event_data_organizer)
SCRIPTS_DIR: Path        = PROJECT_ROOT / "scripts"

# Ensure key directories exist when importing this module in viz scripts
VIS_DIR.mkdir(parents=True, exist_ok=True)
LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
AGEGROUP_VIS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
#   Scraper / HTTP configuration
# ============================================================

# Alias for backward-compat naming (used in parkrun_event_data_organizer)
SERIES_SUMMARY_CSV: Path = EVENT_SERIES_CSV

# HTTP headers for scraping
HEADERS: dict = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# Regex patterns used when scraping
ID_HREF_RE = re.compile(r"/parkrunner/(\d+)(?:/|$)")
PERCENT_RE = re.compile(r"(\d{1,3}\.\d{2})\s*%")

# Results URL template:
# (see sanitize_base_url in parkrun_event_data_organizer.py)
# Example: base_url="https://www.parkrun.us/farmpond" ->
#          "https://www.parkrun.us/farmpond/results/1/"
RESULTS_URL_TEMPLATE: str = "{base_url}/results/{event_no}/"


# Volunteers URL template:
#   volunteers/?eventNumber=<event_no>
VOLUNTEERS_URL_TEMPLATE: str = "{base_url}/volunteers/?eventNumber={event_no}"

# ============================================================
#   TIME PARSING / FORMATTING HELPERS
# ============================================================

TIME_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2})\s*$")

def parse_time_to_seconds(val) -> Optional[int]:
    """
    Parse a parkrun-style time string into an integer number of seconds.

    Accepts:
      - 'mm:ss'
      - 'h:mm:ss'
    Returns None for blanks, DNF-like markers, or unparsable input.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if s in ("", "-", "—", "DNF", "None", "nan", "NaN"):
        return None
    m = TIME_RE.match(s)
    if not m:
        return None
    h = int(m.group(1)) if m.group(1) else 0
    mm = int(m.group(2))
    ss = int(m.group(3))
    return h * 3600 + mm * 60 + ss

def fmt_sec_mmss(sec: Optional[float]) -> str:
    """
    Format seconds as 'mm:ss' or 'h:mm:ss' if h > 0.
    """
    if sec is None or pd.isna(sec):
        return ""
    sec = int(round(float(sec)))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"

# ============================================================
#   PLOT HELPERS (LOGO + BASIC STYLING)
# ============================================================

def load_event_name(default_name: str = EVENT_NAME_DEFAULT) -> str:
    """
    Read event_name from data/event_series_summary.csv, if present.
    Used by leaderboard + course-record table scripts.
    """
    if EVENT_SERIES_CSV.exists():
        try:
            df = pd.read_csv(EVENT_SERIES_CSV)
            if "event_name" in df.columns:
                s = df["event_name"].dropna().astype(str)
                s = s[s.str.strip() != ""]
                if not s.empty:
                    return s.iloc[0].strip()
        except Exception:
            pass
    return default_name

def add_centered_background_logo(fig: plt.Figure, alpha: float = LOGO_ALPHA) -> None:
    """
    Add the parkrun logo faintly centered behind content on a Figure.
    Safe to call even if the logo file is missing.
    """
    if not PARKRUN_LOGO.exists():
        return
    try:
        img = Image.open(PARKRUN_LOGO).convert("RGBA")
    except Exception:
        return

    fig_w, fig_h = fig.get_size_inches()
    fig_h_px = fig_h * fig.dpi
    zoom = fig_h_px / img.height  # height of logo ~= figure height

    oi = OffsetImage(img, zoom=zoom, alpha=alpha)
    ab = AnnotationBbox(
        oi,
        (0.5, 0.5),
        xycoords=fig.transFigure,
        frameon=False,
        zorder=0,
    )
    fig.add_artist(ab)

def apply_standard_axes_style(ax: plt.Axes) -> None:
    """
    Apply the standard Farm Pond / parkrun visual style to a Matplotlib Axes
    for line charts: purple background, white grid, bold tick labels, etc.
    """
    fig = ax.figure
    fig.patch.set_facecolor(PARKRUN_PURPLE)
    ax.set_facecolor(PARKRUN_PURPLE)

    # Grid
    ax.grid(
        True,
        linestyle="--",
        linewidth=GRID_LW,
        color=NEAR_WHITE,
        alpha=GRID_ALPHA,
    )
    ax.set_axisbelow(True)

    # Spines
    ax.spines["bottom"].set_color(NEAR_WHITE)
    ax.spines["left"].set_color(NEAR_WHITE)
    ax.spines["bottom"].set_linewidth(AXIS_XY_LW)
    ax.spines["left"].set_linewidth(AXIS_XY_LW)
    for sp in ("top", "right"):
        ax.spines[sp].set_color(NEAR_WHITE)
        ax.spines[sp].set_alpha(0.4)
        ax.spines[sp].set_linewidth(AXIS_TR_LW)

    # Ticks
    ax.tick_params(
        colors=NEAR_WHITE,
        which="both",
        width=1.6,
        length=6,
        labelsize=TICK_SIZE,
    )
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

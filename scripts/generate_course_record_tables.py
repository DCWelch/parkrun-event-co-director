#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate course-record summary tables (PNG images) and a CSV of age-group
best times.

Inputs (expected to exist):
  visualizations/course_record_progression_series.csv
  data/event_series_summary.csv
  visualizations/agegroup_course_records/agegroup_*_course_record_progression_series.csv

Outputs:
  visualizations/course_record_best_times_table.png
  visualizations/course_record_best_agegrades_table.png
  visualizations/course_record_best_overall_table.png  (Times + Age Grades stacked)
  visualizations/agegroup_course_record_best_times_table.png
  visualizations/agegroup_course_record_best_times.csv
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, Any, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

import requests

# ------------------------- Paths -------------------------

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]            # parkrun_event_data_organizer/
DATA_DIR = PROJECT_ROOT / "data"
VIS_DIR = PROJECT_ROOT / "visualizations"
ASSETS_DIR = PROJECT_ROOT / "assets"

AGEGROUP_VIS_DIR = VIS_DIR / "agegroup_course_records"

CR_SERIES_CSV       = VIS_DIR / "course_record_progression_series.csv"
EVENT_SERIES_CSV    = DATA_DIR / "event_series_summary.csv"

BEST_TIMES_TABLE_PNG     = VIS_DIR / "course_record_best_times_table.png"
BEST_AGEGRADES_TABLE_PNG = VIS_DIR / "course_record_best_agegrades_table.png"
BEST_OVERALL_TABLE_PNG   = VIS_DIR / "course_record_best_overall_table.png"
AGEGROUP_BEST_TIMES_PNG  = VIS_DIR / "agegroup_course_record_best_times_table.png"
AGEGROUP_BEST_TIMES_CSV  = VIS_DIR / "agegroup_course_record_best_times.csv"

PARKRUN_LOGO = ASSETS_DIR / "parkrun_logo_white.png"

VIS_DIR.mkdir(parents=True, exist_ok=True)
AGEGROUP_VIS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------- Style constants (match CR charts) -------------------------

PARKRUN_PURPLE = "#4B2E83"     # background
NEAR_WHITE     = "#F4F4F6"     # labels/grid
PARKRUN_YELLOW = "#FFA300"     # male highlight
PARKRUN_TEAL   = "#10ECCC"     # female highlight

TITLE_SIZE   = 26
HEADER_SIZE  = 14
CELL_SIZE    = 12
BORDER_LW    = 1.2
LOGO_ALPHA   = 0.12

# Small vertical tweak to push text down inside cells
CELL_Y_OFFSET = 0 # in table cell coordinates

# ------------------------- Utilities -------------------------

TIME_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2})\s*$")

def parse_time_to_seconds(val) -> Optional[int]:
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
    if sec is None or pd.isna(sec):
        return ""
    sec = int(round(float(sec)))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"

def _add_centered_background_logo(fig: plt.Figure, alpha: float = LOGO_ALPHA):
    if not PARKRUN_LOGO.exists():
        return
    try:
        img = Image.open(PARKRUN_LOGO).convert("RGBA")
    except Exception:
        return

    fig_w, fig_h = fig.get_size_inches()
    fig_h_px = fig_h * fig.dpi
    zoom = fig_h_px / img.height

    oi = OffsetImage(img, zoom=zoom, alpha=alpha)
    ab = AnnotationBbox(
        oi, (0.5, 0.5),
        xycoords=fig.transFigure,
        frameon=False,
        zorder=0
    )
    fig.add_artist(ab)

def _style_table_cells(table, df: pd.DataFrame, highlight_gender: bool):
    """Common styling + vertical-centering tweak."""
    n_rows, n_cols = df.shape

    for (row, col), cell in table.get_celld().items():
        txt = cell.get_text()

        if row == 0:
            # header
            cell.set_text_props(color=PARKRUN_PURPLE, fontweight="bold", fontsize=HEADER_SIZE)
            cell.set_facecolor(NEAR_WHITE)
        else:
            cell.set_facecolor(PARKRUN_PURPLE)
            cell.set_edgecolor(NEAR_WHITE)
            cell.set_linewidth(BORDER_LW)
            txt.set_color(NEAR_WHITE)

        # vertical centering tweak: nudge text slightly downward
        x, y = txt.get_position()
        txt.set_position((x, y + CELL_Y_OFFSET))

    # Optional gender highlighting (only if Gender column exists)
    if highlight_gender and "Gender" in df.columns:
        gender_col_idx = list(df.columns).index("Gender")
        for row_idx in range(1, n_rows + 1):
            gender_val = str(df.iloc[row_idx - 1, gender_col_idx]).strip().lower()
            color = None
            if gender_val.startswith("m"):
                color = PARKRUN_YELLOW
            elif gender_val.startswith("f"):
                color = PARKRUN_TEAL
            if color:
                gcell = table[row_idx, gender_col_idx]
                gcell.set_facecolor(color)
                gcell.get_text().set_color(PARKRUN_PURPLE)
                gcell.set_linewidth(BORDER_LW)

def _draw_table(df, title, outpath, highlight_gender=True, extra_top_margin=0.0):
    n_rows, n_cols = df.shape

    height = max(3.0, 0.40 * n_rows + 0.8)
    base_width = max(7.0, 0.9 * n_cols + 2.0)
    width = base_width * 1.3 * 1.2

    fig, ax = plt.subplots(figsize=(width, height), dpi=160)
    fig.patch.set_facecolor(PARKRUN_PURPLE)
    ax.set_facecolor(PARKRUN_PURPLE)

    _add_centered_background_logo(fig, alpha=LOGO_ALPHA)
    ax.set_axis_off()

    display_df = df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].astype(str)

    col_widths = None
    if "parkrunner" in display_df.columns:
        base = [1.0] * len(display_df.columns)
        p_idx = list(display_df.columns).index("parkrunner")

        base[p_idx] *= 2.0

        total = sum(base)
        col_widths = [b / total for b in base]

    table_kwargs = dict(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.16, 0.96, 0.78],
    )
    if col_widths is not None:
        table_kwargs["colWidths"] = col_widths

    table = ax.table(**table_kwargs)
    table.auto_set_font_size(False)
    table.set_fontsize(CELL_SIZE)
    table.scale(1.1, 0.5)

    ax.set_title(
        title,
        color=NEAR_WHITE,
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=10,
    )

    for (row, col), cell in table.get_celld().items():
        txt = cell.get_text()

        cell.PAD = 0#0.15

        if row == 0:
            cell.set_text_props(color=PARKRUN_PURPLE, fontweight="bold", fontsize=HEADER_SIZE)
            cell.set_facecolor(NEAR_WHITE)
            cell.set_edgecolor(NEAR_WHITE)
        else:
            cell.set_facecolor(PARKRUN_PURPLE)
            cell.set_edgecolor(NEAR_WHITE)
            cell.set_linewidth(BORDER_LW)
            txt.set_color(NEAR_WHITE)

        txt.set_verticalalignment("center")
        x, y = txt.get_position()
        txt.set_position((x, y + CELL_Y_OFFSET))

    if highlight_gender and "Gender" in df.columns:
        gender_col = list(df.columns).index("Gender")
        for i in range(1, n_rows + 1):
            gval = display_df.iloc[i - 1, gender_col].strip().lower()
            color = PARKRUN_YELLOW if gval.startswith("m") else PARKRUN_TEAL if gval.startswith("f") else None
            if color:
                gcell = table[i, gender_col]
                gcell.set_facecolor(color)
                gcell.get_text().set_color(PARKRUN_PURPLE)

    fig.subplots_adjust(
        left=0.04,
        right=0.96,
        bottom=0.01,
        top=0.88 - extra_top_margin,
    )

    fig.savefig(outpath, dpi=160, facecolor=PARKRUN_PURPLE)
    plt.close(fig)

def _draw_combined_course_records(best_times_df: pd.DataFrame,
                                  best_agegrades_df: pd.DataFrame,
                                  title: str,
                                  outpath: Path):
    """
    Draw a single image with two stacked tables:
      - "Times"
      - "Age Grades"
    with corrected, tight margins and no overlapping titles.
    """

    max_cols = max(best_times_df.shape[1], best_agegrades_df.shape[1])
    height = 6.6
    base_width  = max(8.0, 1.0 * max_cols + 4.0)
    width = base_width * 1.2

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(width, height),
        dpi=160,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0},
    )

    fig.patch.set_facecolor(PARKRUN_PURPLE)
    _add_centered_background_logo(fig, alpha=LOGO_ALPHA)

    # --- Overall title ---
    fig.suptitle(
        title,
        color=NEAR_WHITE,
        fontsize=TITLE_SIZE,
        fontweight="bold",
        y=0.95,
    )

    # --- Helper for each subtable ---
    def draw_subtable(ax, df, subtitle: str, bbox):
        ax.set_facecolor(PARKRUN_PURPLE)
        ax.set_axis_off()

        ax.set_title(
            subtitle,
            color=NEAR_WHITE,
            fontsize=HEADER_SIZE + 2,
            fontweight="bold",
            pad=22,
        )

        display_df = df.copy().astype(str)
        
        # Widen parkrunner column by ~50%
        col_widths = None
        if "parkrunner" in display_df.columns:
            base = [1.0] * len(display_df.columns)
            p_idx = list(display_df.columns).index("parkrunner")
            base[p_idx] *= 2.0
            total = sum(base)
            col_widths = [b / total for b in base]
        
        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            cellLoc="center",
            loc="center",
            bbox=bbox,
            colWidths=col_widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(CELL_SIZE)
        table.scale(1.1, 0.5)

        _style_table_cells(table, display_df, highlight_gender=True)

    # Draw the two tables
    draw_subtable(
        axes[0],
        best_times_df,
        "Times",
        bbox=[0.02, 0.55, 0.96, 0.40],
    )

    draw_subtable(
        axes[1],
        best_agegrades_df,
        "Age Grades",
        bbox=[0.02, 0.55, 0.96, 0.40],
    )

    fig.subplots_adjust(
        left=0.04,
        right=0.96,
        bottom=0,
        top=0.70,
        hspace=1.2,
    )

    fig.savefig(outpath, dpi=160, facecolor=PARKRUN_PURPLE)
    plt.close(fig)

# ------------------------- Load course-record progression -------------------------

def load_course_record_series() -> pd.DataFrame:
    if not CR_SERIES_CSV.exists():
        raise FileNotFoundError(f"Missing course-record series CSV: {CR_SERIES_CSV}")
    pe = pd.read_csv(CR_SERIES_CSV)
    if "event" not in pe.columns:
        raise ValueError("course_record_progression_series.csv does not have 'event' column.")
    return pe

def load_event_name(default_name: str = "Unknown") -> str:
    """
    Read event_name from data/event_series_summary.csv, if present.
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

# ------------------------- Build overall tables -------------------------

def build_overall_best_tables(pe: pd.DataFrame):
    """
    From the progression series, extract overall best times and age-grades
    (male & female) and return three DataFrames:
      best_times_df, best_agegrades_df, combined_df.
    Column names use consistent title casing.
    """
    last = pe.sort_values("event").iloc[-1]

    # ---- Best times table (overall course records by time) ----
    best_times_rows: List[Dict[str, Any]] = [
        {
            "Gender": "Male",
            "Time": last.get("cr_male_time", ""),
            "Set at Event": int(last.get("cr_male_time_set_at_event"))
                            if pd.notna(last.get("cr_male_time_set_at_event")) else "",
            "parkrunner": last.get("cr_male_time_name", ""),
            "Age Group": last.get("cr_male_time_agegroup", ""),
            "Age Grade": (
                f"{float(last['cr_male_time_agegrade']):.2f}"
                if pd.notna(last.get("cr_male_time_agegrade")) else ""
            ),
        },
                {
            "Gender": "Female",
            "Time": last.get("cr_female_time", ""),
            "Set at Event": int(last.get("cr_female_time_set_at_event"))
                            if pd.notna(last.get("cr_female_time_set_at_event")) else "",
            "parkrunner": last.get("cr_female_time_name", ""),
            "Age Group": last.get("cr_female_time_agegroup", ""),
            "Age Grade": (
                f"{float(last['cr_female_time_agegrade']):.2f}"
                if pd.notna(last.get("cr_female_time_agegrade")) else ""
            ),
        },
    ]
    best_times_df = pd.DataFrame(best_times_rows)

    # ---- Best age-grade table (overall course records by age-grade) ----
    best_ag_rows: List[Dict[str, Any]] = [
        {
            "Gender": "Male",
            "Age Grade": (
                f"{float(last['cr_male_agegrade']):.2f}"
                if pd.notna(last.get("cr_male_agegrade")) else ""
            ),
            "Set at Event": int(last.get("cr_male_agegrade_set_at_event"))
                            if pd.notna(last.get("cr_male_agegrade_set_at_event")) else "",
            "parkrunner": last.get("cr_male_agegrade_name", ""),
            "Age Group": last.get("cr_male_agegrade_agegroup", ""),
            "Time": last.get("cr_male_agegrade_time", ""),
        },
        {
            "Gender": "Female",
            "Age Grade": (
                f"{float(last['cr_female_agegrade']):.2f}"
                if pd.notna(last.get("cr_female_agegrade")) else ""
            ),
            "Set at Event": int(last.get("cr_female_agegrade_set_at_event"))
                            if pd.notna(last.get("cr_female_agegrade_set_at_event")) else "",
            "parkrunner": last.get("cr_female_agegrade_name", ""),
            "Age Group": last.get("cr_female_agegrade_agegroup", ""),
            "Time": last.get("cr_female_agegrade_time", ""),
        },
    ]
    best_agegrades_df = pd.DataFrame(best_ag_rows)

    # ---- Combined DF (kept for compatibility / future use) ----
    combined_rows: List[Dict[str, Any]] = []
    for sex in ["male", "female"]:
        gender_label = "Male" if sex == "male" else "Female"
        combined_rows.append({
            "Gender": gender_label,
            "Time": last.get(f"cr_{sex}_time", ""),
            "Time Event": last.get(f"cr_{sex}_time_set_at_event", ""),
            "Time Parkrunner": last.get(f"cr_{sex}_time_name", ""),
            "Time Age Group": last.get(f"cr_{sex}_time_agegroup", ""),
            "Time Age Grade (%)": (
                f"{float(last[f'cr_{sex}_time_agegrade']):.2f}"
                if pd.notna(last.get(f"cr_{sex}_time_agegrade")) else ""
            ),
            "Record Age Grade (%)": (
                f"{float(last[f'cr_{sex}_agegrade']):.2f}"
                if pd.notna(last.get(f"cr_{sex}_agegrade")) else ""
            ),
            "Age Grade Event": last.get(f"cr_{sex}_agegrade_set_at_event", ""),
            "Age Grade Parkrunner": last.get(f"cr_{sex}_agegrade_name", ""),
            "Age Grade Age Group": last.get(f"cr_{sex}_agegrade_agegroup", ""),
            "Age Grade Time": last.get(f"cr_{sex}_agegrade_time", ""),
        })
    combined_df = pd.DataFrame(combined_rows)

    return best_times_df, best_agegrades_df, combined_df

# ------------------------- Age-group best times table -------------------------

def load_agegroup_best_times() -> pd.DataFrame:
    """
    For each agegroup_*_course_record_progression_series.csv file,
    take the final row (current record) and build a summary of best times.

    ONE row per age group (no Gender column).
    """
    rows: List[Dict[str, Any]] = []

    if not AGEGROUP_VIS_DIR.exists():
        return pd.DataFrame()

    for csv_path in sorted(AGEGROUP_VIS_DIR.glob("agegroup_*_course_record_progression_series.csv")):
        try:
            ag_df = pd.read_csv(csv_path)
        except Exception:
            continue
        if ag_df.empty or "age_group" not in ag_df.columns:
            continue

        ag_df = ag_df.sort_values("event")
        last = ag_df.iloc[-1]
        age_group = last.get("age_group", "")

        # Choose the single best time overall (male OR female)
        malesec = parse_time_to_seconds(last.get("cr_male_time", ""))
        femsec  = parse_time_to_seconds(last.get("cr_female_time", ""))
        if malesec is None and femsec is None:
            continue
        if femsec is None or (malesec is not None and malesec <= femsec):
            # pick male
            record_time   = last.get("cr_male_time", "")
            event_no      = last.get("cr_male_time_set_at_event", "")
            name          = last.get("cr_male_time_name", "")
            age_grade_val = last.get("cr_male_time_agegrade", np.nan)
        else:
            # pick female
            record_time   = last.get("cr_female_time", "")
            event_no      = last.get("cr_female_time_set_at_event", "")
            name          = last.get("cr_female_time_name", "")
            age_grade_val = last.get("cr_female_time_agegrade", np.nan)

        rows.append({
            "Age group": age_group,
            "Time": record_time,
            "Time set at event": int(event_no) if pd.notna(event_no) and str(event_no) != "" else "",
            "parkrunner": name,
            "Age grade (%)": f"{float(age_grade_val):.2f}" if pd.notna(age_grade_val) else "",
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("Age group").reset_index(drop=True)
    return df

# ------------------------- Main -------------------------

def main():
    # Overall course-record series
    pe = load_course_record_series()
    event_name = load_event_name(default_name="Unknown")

    # Build overall best tables
    best_times_df, best_agegrades_df, combined_df = build_overall_best_tables(pe)

    # 1) Times-only table
    _draw_table(
        best_times_df,
        title=f"{event_name} parkrun Course Records",
        outpath=BEST_TIMES_TABLE_PNG,
        highlight_gender=True,
        extra_top_margin=0.1,
    )
    print(f"Wrote best times table -> {BEST_TIMES_TABLE_PNG}")

    # 2) Age-grade-only table
    _draw_table(
        best_agegrades_df,
        title=f"{event_name} parkrun Age Grade Course Records",
        outpath=BEST_AGEGRADES_TABLE_PNG,
        highlight_gender=True,
        extra_top_margin=0.1,
    )
    print(f"Wrote best age-grade table -> {BEST_AGEGRADES_TABLE_PNG}")

    # 3) Combined image with Times + Age Grades stacked
    _draw_combined_course_records(
        best_times_df,
        best_agegrades_df,
        title=f"{event_name} parkrun Course Records",
        outpath=BEST_OVERALL_TABLE_PNG,
    )
    print(f"Wrote combined course-records table -> {BEST_OVERALL_TABLE_PNG}")

    # 4) Age-group best times
    ag_best_df = load_agegroup_best_times()
    if ag_best_df.empty:
        print("No age-group course-record series CSVs found; skipping age-group table.")
        return

    # Header text tweaks for display
    ag_best_df = ag_best_df.rename(columns={
        "Age group": "Age Group",
        "Time": "Time",
        "Time set at event": "Set at Event",
        "Age grade (%)": "Age Grade",
    })

    ag_best_df.to_csv(AGEGROUP_BEST_TIMES_CSV, index=False)
    print(f"Wrote age-group best-times CSV -> {AGEGROUP_BEST_TIMES_CSV}")

    _draw_table(
        ag_best_df,
        title=f"{event_name} parkrun\nAge Group Course Records",
        outpath=AGEGROUP_BEST_TIMES_PNG,
        highlight_gender=False,
        extra_top_margin=0,
    )
    print(f"Wrote age-group best-times table -> {AGEGROUP_BEST_TIMES_PNG}")

if __name__ == "__main__":
    main()

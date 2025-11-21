#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate leaderboard tables (PNG images) for a single parkrun event:

  1. Top-X Course Times
  2. Top-X Course Times (male)
  3. Top-X Course Times (female)
  4. Top-X Age Grades
  5. Top-X Age Grades (male)
  6. Top-X Age Grades (female)
  7. Top-X Most Volunteers
  8. Top-X Most parkruns
  9. Top-X Most Participations

Usage:
  python generate_leaderboard_tables.py           # uses default TOP_N (10)
  python generate_leaderboard_tables.py -n 25     # Top-25 instead of Top-10
"""

from __future__ import annotations
import argparse
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

# ========================= Config & Paths =========================

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
VIS_DIR = PROJECT_ROOT / "visualizations"
ASSETS_DIR = PROJECT_ROOT / "assets"

LEADERBOARD_DIR = VIS_DIR / "leaderboards"
EVENT_SERIES_CSV = DATA_DIR / "event_series_summary.csv"

# ---- Input parkrunner summary CSV ----
PARKRUNNER_SUMMARY_CSV = DATA_DIR / "participants_master.csv"

# ---- Column names in parkrunner summary CSV ----
NAME_COL            = "parkrunner"
GENDER_COL          = "Gender"
AGE_GROUP_COL       = "Age Group"
BEST_TIME_STR_COL   = "Best Time"          # mm:ss or hh:mm:ss
BEST_TIME_SEC_COL   = "Best Time (sec)"    # numeric; optional, computed if missing
BEST_AGEGRADE_COL   = "Best Age Grade"     # numeric percent
PARKRUNS_COL        = "parkruns"           # total runs at this event
VOLUNTEER_ROLES_COL = "Volunteers"         # total volunteers
PARTICIPATIONS_COL  = "Participations"     # runs + volunteers, if available

# ---- Default Top-N ----
DEFAULT_TOP_N = 10

# ---- Style (copied from Age Group Course Records styling) ----

PARKRUN_PURPLE = "#4B2E83"  # background
NEAR_WHITE     = "#F4F4F6"  # labels/grid
PARKRUN_YELLOW = "#FFA300"  # male highlight
PARKRUN_TEAL   = "#10ECCC"  # female highlight

TITLE_SIZE   = 26
HEADER_SIZE  = 14
CELL_SIZE    = 12
BORDER_LW    = 1.2
LOGO_ALPHA   = 0.12

CELL_Y_OFFSET = 0  # in table cell coordinates

PARKRUN_LOGO = ASSETS_DIR / "parkrun_logo_white.png"

VIS_DIR.mkdir(parents=True, exist_ok=True)
LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)

# ========================= Utilities =========================

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
        zorder=0,
    )
    fig.add_artist(ab)

def _draw_table(df: pd.DataFrame,
                title: str,
                outpath: Path,
                highlight_gender: bool = True,
                extra_top_margin: float = 0.0):

    n_rows, n_cols = df.shape

    height = max(3.0, 0.40 * n_rows + 0.8)
    base_width = max(7.0, 0.9 * n_cols + 2.0)
    width = base_width * 1.3 * 1.2  # slightly widened

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
        pad=30,
    )

    for (row, col), cell in table.get_celld().items():
        txt = cell.get_text()
        cell.PAD = 0
        if row == 0:
            cell.set_text_props(color=PARKRUN_PURPLE,
                                fontweight="bold",
                                fontsize=HEADER_SIZE)
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
        gender_col_idx = list(df.columns).index("Gender")
        for i in range(1, n_rows + 1):
            gval = display_df.iloc[i - 1, gender_col_idx].strip().lower()
            color = PARKRUN_YELLOW if gval.startswith("m") else (
                    PARKRUN_TEAL if gval.startswith("f") else None)
            if color:
                gcell = table[i, gender_col_idx]
                gcell.set_facecolor(color)
                gcell.get_text().set_color(PARKRUN_PURPLE)

    fig.subplots_adjust(
        left=0.04,
        right=0.96,
        bottom=0.01,
        top=0.70 - extra_top_margin,
    )

    fig.savefig(outpath, dpi=160, facecolor=PARKRUN_PURPLE)
    plt.close(fig)

def load_event_name(default_name: str = "Unknown") -> str:
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

# ========================= Data loading & prep =========================

def load_parkrunner_summary() -> pd.DataFrame:
    """
    Load the master participants file produced by parkrun_event_data_organizer.py
    (participants_master.csv) and reshape it into the columns expected by the
    leaderboard logic.
    """
    if not PARKRUNNER_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Missing master participants CSV: {PARKRUNNER_SUMMARY_CSV}"
        )

    df = pd.read_csv(PARKRUNNER_SUMMARY_CSV)

    # Build full name (used as "parkrunner" in the tables)
    df["parkrunner"] = (
        df["name_first"].fillna("") + " " + df["name_last"].fillna("")
    ).str.strip()

    # ---- Remove placeholder / unknown parkrunners ----
    name_lower = df["parkrunner"].str.strip().str.lower()
    is_unknown = name_lower.isin(
        [
            "unknown",
            "unknown parkrunner",
            "unknown runner",
            "unknown volunteer",
        ]
    )
    is_empty = df["parkrunner"].str.strip().eq("")
    df = df[~(is_unknown | is_empty)].copy()

    # Rename to the canonical display names that the leaderboard code uses
    df = df.rename(
        columns={
            "gender": "Gender",
            "age_group": "Age Group",
            "pb_time": "Best Time",
            "pb_age_grade": "Best Age Grade",
            "num_runs": "parkruns",
            "num_volunteers": "Volunteers",
            "num_participations": "Participations",
        }
    )

    # Ensure these exist even if older CSVs are missing some columns
    for col in ["Best Time", "Best Age Grade", "parkruns", "Volunteers", "Participations"]:
        if col not in df.columns:
            df[col] = pd.NA
    
    # Numeric best time for sorting
    df["Best Time (sec)"] = df["Best Time"].apply(parse_time_to_seconds)

    return df

# ========================= Leaderboard builders =========================

def make_top_times(df: pd.DataFrame, top_n: int, gender: Optional[str] = None) -> pd.DataFrame:
    """
    Top course times (overall or gender-filtered).

    Desired columns:
      Rank, parkrunner, Time
    """
    sub = df.copy()
    if gender is not None:
        g_norm = gender.lower()[0]  # 'm' or 'f'
        sub = sub[sub[GENDER_COL].astype(str).str.lower().str.startswith(g_norm)]

    sub = sub.dropna(subset=[BEST_TIME_SEC_COL])
    sub = sub.sort_values(BEST_TIME_SEC_COL, ascending=True).head(top_n)

    # Pretty time column
    if BEST_TIME_STR_COL in sub.columns:
        time_str = sub[BEST_TIME_STR_COL].apply(str)
    else:
        time_str = sub[BEST_TIME_SEC_COL].apply(fmt_sec_mmss)

    rows = []
    for rank, (idx, row) in enumerate(sub.iterrows(), start=1):
        rows.append({
            "Rank": rank,
            "parkrunner": row.get(NAME_COL, ""),
            "Time": time_str.loc[idx],
        })

    return pd.DataFrame(rows)

def make_top_agegrades(df: pd.DataFrame, top_n: int, gender: Optional[str] = None) -> pd.DataFrame:
    """
    Top age grades (overall or gender-filtered).

    Desired columns:
      Rank, parkrunner, Age Grade, Time, Age Group
    """
    sub = df.copy()
    if gender is not None:
        g_norm = gender.lower()[0]
        sub = sub[sub[GENDER_COL].astype(str).str.lower().str.startswith(g_norm)]

    sub = sub.dropna(subset=[BEST_AGEGRADE_COL])
    sub = sub.sort_values(BEST_AGEGRADE_COL, ascending=False).head(top_n)

    if BEST_TIME_STR_COL in sub.columns:
        time_str = sub[BEST_TIME_STR_COL].apply(str)
    else:
        time_str = sub[BEST_TIME_SEC_COL].apply(fmt_sec_mmss)

    agegrade_str = (sub[BEST_AGEGRADE_COL]
                    .apply(lambda x: f"{float(x):.2f}" if pd.notna(x) else ""))

    rows = []
    for rank, (idx, row) in enumerate(sub.iterrows(), start=1):
        rows.append({
            "Rank": rank,
            "parkrunner": row.get(NAME_COL, ""),
            "Age Grade": agegrade_str.loc[idx],
            "Time": time_str.loc[idx],
            "Age Group": row.get(AGE_GROUP_COL, ""),
        })

    return pd.DataFrame(rows)

def make_top_parkruns(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Top parkruns:

      Rank, parkrunner, parkruns
    """
    sub = df.copy()
    sub = sub.dropna(subset=[PARKRUNS_COL])
    sub = sub.sort_values(PARKRUNS_COL, ascending=False).head(top_n)

    rows = []
    for rank, (_, row) in enumerate(sub.iterrows(), start=1):
        rows.append({
            "Rank": rank,
            "parkrunner": row.get(NAME_COL, ""),
            "parkruns": int(row.get(PARKRUNS_COL, 0)),
        })

    return pd.DataFrame(rows)

def make_top_volunteers(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Top volunteers:

      Rank, parkrunner, Volunteers
    """
    sub = df.copy()
    sub = sub.dropna(subset=[VOLUNTEER_ROLES_COL])
    sub = sub.sort_values(VOLUNTEER_ROLES_COL, ascending=False).head(top_n)

    rows = []
    for rank, (_, row) in enumerate(sub.iterrows(), start=1):
        rows.append({
            "Rank": rank,
            "parkrunner": row.get(NAME_COL, ""),
            "Volunteers": int(row.get(VOLUNTEER_ROLES_COL, 0)),
        })

    return pd.DataFrame(rows)

def make_top_participations(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Top participations:

      Rank, parkrunner, Participations, parkruns, Volunteers
    """
    sub = df.copy()
    sub = sub.dropna(subset=[PARTICIPATIONS_COL])
    sub = sub.sort_values(PARTICIPATIONS_COL, ascending=False).head(top_n)

    rows = []
    for rank, (_, row) in enumerate(sub.iterrows(), start=1):
        rows.append({
            "Rank": rank,
            "parkrunner": row.get(NAME_COL, ""),
            "Participations": int(row.get(PARTICIPATIONS_COL, 0)),
            "parkruns": int(row.get(PARKRUNS_COL, 0)),
            "Volunteers": int(row.get(VOLUNTEER_ROLES_COL, 0)),
        })

    return pd.DataFrame(rows)

# ========================= Main =========================

def main(top_n: int):
    event_name = load_event_name(default_name="Unknown")
    df = load_parkrunner_summary()

    # ---- 1–3: Age Grades ----
    top_ag_all    = make_top_agegrades(df, top_n, gender=None)
    top_ag_female = make_top_agegrades(df, top_n, gender="F")
    top_ag_male   = make_top_agegrades(df, top_n, gender="M")

    _draw_table(
        top_ag_all,
        title=f"{event_name} parkrun\nTop-{top_n} Age Grades",
        outpath=LEADERBOARD_DIR / f"top_{top_n}_agegrades.png",
        highlight_gender=False,
        extra_top_margin=0.0,
    )
    _draw_table(
        top_ag_female,
        title=f"{event_name} parkrun\nTop-{top_n} Age Grades (Female)",
        outpath=LEADERBOARD_DIR / f"top_{top_n}_agegrades_female.png",
        highlight_gender=False,
        extra_top_margin=0.0,
    )
    _draw_table(
        top_ag_male,
        title=f"{event_name} parkrun\nTop-{top_n} Age Grades (Male)",
        outpath=LEADERBOARD_DIR / f"top_{top_n}_agegrades_male.png",
        highlight_gender=False,
        extra_top_margin=0.0,
    )

    # ---- 4: parkruns ----
    if PARKRUNS_COL in df.columns:
        top_runs = make_top_parkruns(df, top_n)
        _draw_table(
            top_runs,
            title=f"{event_name} parkrun\nTop-{top_n} Most parkruns",
            outpath=LEADERBOARD_DIR / f"top_{top_n}_parkruns.png",
            highlight_gender=False,
            extra_top_margin=0.0,
        )
        print("Wrote parkruns leaderboard")

    # ---- 5: participations ----
    if PARTICIPATIONS_COL in df.columns:
        top_participations = make_top_participations(df, top_n)
        _draw_table(
            top_participations,
            title=f"{event_name} parkrun\nTop-{top_n} Most Participations",
            outpath=LEADERBOARD_DIR / f"top_{top_n}_participations.png",
            highlight_gender=False,
            extra_top_margin=0.0,
        )
        print("Wrote participations leaderboard")

    # ---- 9: volunteers ----
    if VOLUNTEER_ROLES_COL in df.columns:
        top_volunteers = make_top_volunteers(df, top_n)
        _draw_table(
            top_volunteers,
            title=f"{event_name} parkrun\nTop-{top_n} Most Volunteers",
            outpath=LEADERBOARD_DIR / f"top_{top_n}_volunteers.png",
            highlight_gender=False,
            extra_top_margin=0.0,
        )
        print("Wrote volunteer leaderboard")

    # ---- 6–8: Course Times ----
    top_times_all    = make_top_times(df, top_n, gender=None)
    top_times_female = make_top_times(df, top_n, gender="F")
    top_times_male   = make_top_times(df, top_n, gender="M")

    _draw_table(
        top_times_all,
        title=f"{event_name} parkrun\nTop-{top_n} Course Times",
        outpath=LEADERBOARD_DIR / f"top_{top_n}_times.png",
        highlight_gender=False,
        extra_top_margin=0.0,
    )
    _draw_table(
        top_times_female,
        title=f"{event_name} parkrun\nTop-{top_n} Course Times (Female)",
        outpath=LEADERBOARD_DIR / f"top_{top_n}_times_female.png",
        highlight_gender=False,
        extra_top_margin=0.0,
    )
    _draw_table(
        top_times_male,
        title=f"{event_name} parkrun\nTop-{top_n} Course Times (Male)",
        outpath=LEADERBOARD_DIR / f"top_{top_n}_times_male.png",
        highlight_gender=False,
        extra_top_margin=0.0,
    )

    print(f"Leaderboards written to: {LEADERBOARD_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate parkrun leaderboard tables.")
    parser.add_argument(
        "-n", "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of entries per leaderboard (default: {DEFAULT_TOP_N})",
    )
    args = parser.parse_args()
    main(args.top_n)

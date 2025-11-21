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
from pathlib import Path
from typing import Optional, Any, List, Dict

import pandas as pd
import matplotlib.pyplot as plt

from parkrun_config import (
    # Paths
    VIS_DIR,
    AGEGROUP_VIS_DIR,
    # Style constants
    PARKRUN_PURPLE,
    NEAR_WHITE,
    PARKRUN_YELLOW,
    PARKRUN_TEAL,
    TITLE_SIZE,
    HEADER_SIZE,
    CELL_SIZE,
    BORDER_LW,
    LOGO_ALPHA,
    CELL_Y_OFFSET,
    # Time helpers
    parse_time_to_seconds,
    fmt_sec_mmss,  # (not used here yet, but imported for completeness)
    # Plot helpers
    add_centered_background_logo,
    load_event_name,
)

# ------------------------- File paths -------------------------

CR_SERIES_CSV       = VIS_DIR / "course_record_progression_series.csv"

BEST_TIMES_TABLE_PNG     = VIS_DIR / "course_record_best_times_table.png"
BEST_AGEGRADES_TABLE_PNG = VIS_DIR / "course_record_best_agegrades_table.png"
BEST_OVERALL_TABLE_PNG   = VIS_DIR / "course_record_best_overall_table.png"
AGEGROUP_BEST_TIMES_PNG  = VIS_DIR / "agegroup_course_record_best_times_table.png"
AGEGROUP_BEST_TIMES_CSV  = VIS_DIR / "agegroup_course_record_best_times.csv"


# ------------------------- Table styling helpers -------------------------

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


def _draw_table(df: pd.DataFrame,
                title: str,
                outpath: Path,
                highlight_gender: bool = True,
                extra_top_margin: float = 0.0) -> None:
    n_rows, n_cols = df.shape

    height = max(3.0, 0.40 * n_rows + 0.8)
    base_width = max(7.0, 0.9 * n_cols + 2.0)
    width = base_width * 1.3 * 1.2

    fig, ax = plt.subplots(figsize=(width, height), dpi=160)
    fig.patch.set_facecolor(PARKRUN_PURPLE)
    ax.set_facecolor(PARKRUN_PURPLE)

    add_centered_background_logo(fig, alpha=LOGO_ALPHA)
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

        cell.PAD = 0

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
                                  outpath: Path) -> None:
    """
    Draw a single image with two stacked tables:
      - "Times"
      - "Age Grades"
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
    add_centered_background_logo(fig, alpha=LOGO_ALPHA)

    # --- Overall title ---
    fig.suptitle(
        title,
        color=NEAR_WHITE,
        fontsize=TITLE_SIZE,
        fontweight="bold",
        y=0.95,
    )

    # --- Helper for each subtable ---
    def draw_subtable(ax, df: pd.DataFrame, subtitle: str, bbox):
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


# ------------------------- Build overall tables -------------------------

def build_overall_best_tables(pe: pd.DataFrame):
    """
    From the progression series, extract overall best times and age-grades
    (male & female) and return three DataFrames:
      best_times_df, best_agegrades_df, combined_df.

    Display-column targets:

      best_times_df (for tables 1 & times part of 3):
        Gender, Time, parkrunner

      best_agegrades_df (for tables 2 & age-grade part of 3):
        Gender, Age Grade, parkrunner, Age Group, Time
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

    # Keep only the columns you want to display, in your desired order:
    # Gender, Time, parkrunner
    best_times_df = best_times_df[["Gender", "Time", "parkrunner"]]

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

    # Keep only the display columns, in the desired order:
    # Gender, Age Grade, parkrunner, Age Group, Time
    best_agegrades_df = best_agegrades_df[
        ["Gender", "Age Grade", "parkrunner", "Age Group", "Time"]
    ]

    # ---- Combined DF (still returned, not used in plotting right now) ----
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
                if pd.notna(last.get(f'cr_{sex}_agegrade')) else ""
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
            age_grade_val = last.get("cr_male_time_agegrade", pd.NA)
        else:
            # pick female
            record_time   = last.get("cr_female_time", "")
            event_no      = last.get("cr_female_time_set_at_event", "")
            name          = last.get("cr_female_time_name", "")
            age_grade_val = last.get("cr_female_time_agegrade", pd.NA)

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
        title=f"{event_name} parkrun Course Records (Times)",
        outpath=BEST_TIMES_TABLE_PNG,
        highlight_gender=True,
        extra_top_margin=0.1,
    )

    # 2) Age-grade-only table
    _draw_table(
        best_agegrades_df,
        title=f"{event_name} parkrun Course Records (Age Grades)",
        outpath=BEST_AGEGRADES_TABLE_PNG,
        highlight_gender=True,
        extra_top_margin=0.1,
    )

    # 3) Combined image (Times + Age Grades)
    _draw_combined_course_records(
        best_times_df,
        best_agegrades_df,
        title=f"{event_name} parkrun Course Records",
        outpath=BEST_OVERALL_TABLE_PNG,
    )

    # 4) Age-group best times
    ag_best_df = load_agegroup_best_times()
    if ag_best_df.empty:
        print("No age-group course-record series CSVs found; skipping age-group table.")
        return

    # Rename headers for CSV + table usage
    ag_best_df = ag_best_df.rename(columns={
        "Age group": "Age Group",
        "Time": "Time",
        "Time set at event": "Set at Event",
        "Age grade (%)": "Age Grade",
    })

    # --- CSV (keep all columns) ---
    ag_best_df.to_csv(AGEGROUP_BEST_TIMES_CSV, index=False)
    print(f"Wrote age-group best-times CSV -> {AGEGROUP_BEST_TIMES_CSV}")

    # --- PNG table ---
    # (User preference: show only Age Group, parkrunner, Time — no Set at Event, no Age Grade)
    ag_best_display = ag_best_df[["Age Group", "parkrunner", "Time"]]

    _draw_table(
        ag_best_display,
        title=f"{event_name} parkrun\nAge Group Course Records",
        outpath=AGEGROUP_BEST_TIMES_PNG,
        highlight_gender=False,
        extra_top_margin=0,
    )
    print(f"Wrote age-group best-times table -> {AGEGROUP_BEST_TIMES_PNG}")


if __name__ == "__main__":
    main()

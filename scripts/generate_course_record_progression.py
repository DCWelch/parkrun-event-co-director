#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate course-record progression data + visuals.

Outputs to:
  visualizations/course_record_progression_series.csv
  visualizations/course_record_progression_times.png
  visualizations/course_record_progression_agegrades.png
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from parkrun_config import (
    # Paths
    EVENT_DIR,
    VIS_DIR,
    # Style constants
    PARKRUN_PURPLE,
    NEAR_WHITE,
    PARKRUN_YELLOW,
    PARKRUN_TEAL,
    TITLE_SIZE,
    LABEL_SIZE,
    TICK_SIZE,
    GRID_LW,
    LINE_LW,
    BBOX_ALPHA,
    GRID_ALPHA,
    LOGO_ALPHA,
    # Time helpers
    parse_time_to_seconds,
    fmt_sec_mmss,
    # Plot helpers
    add_centered_background_logo,
    apply_standard_axes_style,
)

# ------------------------- Output paths -------------------------

SERIES_CSV = VIS_DIR / "course_record_progression_series.csv"
PLOT_TIMES = VIS_DIR / "course_record_progression_times.png"
PLOT_AGEGRADES = VIS_DIR / "course_record_progression_agegrades.png"


# ------------------------- Gender helpers -------------------------

def is_male(g: Any) -> bool:
    return str(g).strip().lower().startswith("m")


def is_female(g: Any) -> bool:
    return str(g).strip().lower().startswith("f")


# ------------------------- Load all events -------------------------

def load_all_events() -> pd.DataFrame:
    """
    Load all event_XXXX.csv files from EVENT_DIR and normalize columns.
    """
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

    df = pd.concat(rows, ignore_index=True)

    # Normalize / parse times
    if "time" in df.columns:
        df["time_sec"] = df["time"].apply(parse_time_to_seconds)
    else:
        df["time_sec"] = np.nan

    # Ensure expected columns exist
    for col in ["gender", "age_group", "age_grade", "name_first", "name_last", "id", "runner"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Runner mask (only runners count for time/age-grade CRs)
    df["runner_num"] = pd.to_numeric(df["runner"], errors="coerce").fillna(0).astype(int)

    return df


# ------------------------- Record tracking -------------------------

def best_row_time(dfe: pd.DataFrame, gender: str) -> Optional[pd.Series]:
    if dfe.empty:
        return None
    if gender == "male":
        gmask = dfe["gender"].apply(is_male)
    else:
        gmask = dfe["gender"].apply(is_female)
    cand = dfe[(dfe["runner_num"] == 1) & gmask & dfe["time_sec"].notna()]
    if cand.empty:
        return None
    idx = cand["time_sec"].idxmin()
    return cand.loc[idx]


def best_row_agegrade(dfe: pd.DataFrame, gender: str) -> Optional[pd.Series]:
    if dfe.empty:
        return None
    if gender == "male":
        gmask = dfe["gender"].apply(is_male)
    else:
        gmask = dfe["gender"].apply(is_female)
    cand = dfe[(dfe["runner_num"] == 1) & gmask & pd.to_numeric(dfe["age_grade"], errors="coerce").notna()]
    if cand.empty:
        return None
    ag = pd.to_numeric(cand["age_grade"], errors="coerce")
    idx = ag.idxmax()
    return cand.loc[idx]


def person_name(r: pd.Series) -> str:
    f = str(r.get("name_first") or "").strip()
    l = str(r.get("name_last") or "").strip()
    return (f"{f} {l}").strip()


# ------------------------- Build progression series -------------------------

def build_course_record_progression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-event progression of course records (time and age grade, by gender).
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "event",
            "cr_male_time","cr_male_time_parkrunner_id","cr_male_time_name","cr_male_time_agegroup","cr_male_time_agegrade","cr_male_time_set_at_event",
            "cr_female_time","cr_female_time_parkrunner_id","cr_female_time_name","cr_female_time_agegroup","cr_female_time_agegrade","cr_female_time_set_at_event",
            "cr_male_agegrade","cr_male_agegrade_time","cr_male_agegrade_parkrunner_id","cr_male_agegrade_name","cr_male_agegrade_agegroup","cr_male_agegrade_set_at_event",
            "cr_female_agegrade","cr_female_agegrade_time","cr_female_agegrade_parkrunner_id","cr_female_agegrade_name","cr_female_agegrade_agegroup","cr_female_agegrade_set_at_event"
        ])

    events = sorted(df["event"].unique())
    out_rows: list[Dict[str, Any]] = []

    rec_m_time: Optional[int] = None
    rec_m_time_meta: Dict[str, Any] = {}
    rec_f_time: Optional[int] = None
    rec_f_time_meta: Dict[str, Any] = {}

    rec_m_ag: Optional[float] = None
    rec_m_ag_meta: Dict[str, Any] = {}
    rec_f_ag: Optional[float] = None
    rec_f_ag_meta: Dict[str, Any] = {}

    for ev in events:
        dfe = df[df["event"] == ev].copy()

        # --- candidates this event ---
        m_time_row = best_row_time(dfe, "male")
        f_time_row = best_row_time(dfe, "female")
        m_ag_row = best_row_agegrade(dfe, "male")
        f_ag_row = best_row_agegrade(dfe, "female")

        # --- male time record update ---
        if m_time_row is not None:
            t = int(m_time_row["time_sec"])
            if rec_m_time is None or t < rec_m_time:
                rec_m_time = t
                rec_m_time_meta = {
                    "id": m_time_row.get("id"),
                    "name": person_name(m_time_row),
                    "age_group": m_time_row.get("age_group"),
                    "age_grade": float(pd.to_numeric(m_time_row.get("age_grade"), errors="coerce")) if pd.notna(m_time_row.get("age_grade")) else None,
                    "set_event": ev,
                }

        # --- female time record update ---
        if f_time_row is not None:
            t = int(f_time_row["time_sec"])
            if rec_f_time is None or t < rec_f_time:
                rec_f_time = t
                rec_f_time_meta = {
                    "id": f_time_row.get("id"),
                    "name": person_name(f_time_row),
                    "age_group": f_time_row.get("age_group"),
                    "age_grade": float(pd.to_numeric(f_time_row.get("age_grade"), errors="coerce")) if pd.notna(f_time_row.get("age_grade")) else None,
                    "set_event": ev,
                }

        # --- male age-grade record update ---
        if m_ag_row is not None:
            ag = float(pd.to_numeric(m_ag_row["age_grade"], errors="coerce"))
            if rec_m_ag is None or ag > rec_m_ag:
                rec_m_ag = ag
                rec_m_ag_meta = {
                    "id": m_ag_row.get("id"),
                    "name": person_name(m_ag_row),
                    "age_group": m_ag_row.get("age_group"),
                    "time_sec": int(m_ag_row["time_sec"]) if pd.notna(m_ag_row.get("time_sec")) else None,
                    "set_event": ev,
                }

        # --- female age-grade record update ---
        if f_ag_row is not None:
            ag = float(pd.to_numeric(f_ag_row["age_grade"], errors="coerce"))
            if rec_f_ag is None or ag > rec_f_ag:
                rec_f_ag = ag
                rec_f_ag_meta = {
                    "id": f_ag_row.get("id"),
                    "name": person_name(f_ag_row),
                    "age_group": f_ag_row.get("age_group"),
                    "time_sec": int(f_ag_row["time_sec"]) if pd.notna(f_ag_row.get("time_sec")) else None,
                    "set_event": ev,
                }

        row = {
            "event": ev,

            "cr_male_time": fmt_sec_mmss(rec_m_time) if rec_m_time is not None else "",
            "cr_male_time_parkrunner_id": rec_m_time_meta.get("id", ""),
            "cr_male_time_name": rec_m_time_meta.get("name", ""),
            "cr_male_time_agegroup": rec_m_time_meta.get("age_group", ""),
            "cr_male_time_agegrade": rec_m_time_meta.get("age_grade", np.nan),
            "cr_male_time_set_at_event": rec_m_time_meta.get("set_event", ""),

            "cr_female_time": fmt_sec_mmss(rec_f_time) if rec_f_time is not None else "",
            "cr_female_time_parkrunner_id": rec_f_time_meta.get("id", ""),
            "cr_female_time_name": rec_f_time_meta.get("name", ""),
            "cr_female_time_agegroup": rec_f_time_meta.get("age_group", ""),
            "cr_female_time_agegrade": rec_f_time_meta.get("age_grade", np.nan),
            "cr_female_time_set_at_event": rec_f_time_meta.get("set_event", ""),

            "cr_male_agegrade": rec_m_ag if rec_m_ag is not None else np.nan,
            "cr_male_agegrade_time": fmt_sec_mmss(rec_m_ag_meta.get("time_sec")) if rec_m_ag_meta.get("time_sec") is not None else "",
            "cr_male_agegrade_parkrunner_id": rec_m_ag_meta.get("id", ""),
            "cr_male_agegrade_name": rec_m_ag_meta.get("name", ""),
            "cr_male_agegrade_agegroup": rec_m_ag_meta.get("age_group", ""),
            "cr_male_agegrade_set_at_event": rec_m_ag_meta.get("set_event", ""),

            "cr_female_agegrade": rec_f_ag if rec_f_ag is not None else np.nan,
            "cr_female_agegrade_time": fmt_sec_mmss(rec_f_ag_meta.get("time_sec")) if rec_f_ag_meta.get("time_sec") is not None else "",
            "cr_female_agegrade_parkrunner_id": rec_f_ag_meta.get("id", ""),
            "cr_female_agegrade_name": rec_f_ag_meta.get("name", ""),
            "cr_female_agegrade_agegroup": rec_f_ag_meta.get("age_group", ""),
            "cr_female_agegrade_set_at_event": rec_f_ag_meta.get("set_event", ""),
        }

        out_rows.append(row)

    pe = pd.DataFrame(out_rows)

    cols = [
        "event",
        "cr_male_time","cr_male_time_parkrunner_id","cr_male_time_name","cr_male_time_agegroup","cr_male_time_agegrade","cr_male_time_set_at_event",
        "cr_female_time","cr_female_time_parkrunner_id","cr_female_time_name","cr_female_time_agegroup","cr_female_time_agegrade","cr_female_time_set_at_event",
        "cr_male_agegrade","cr_male_agegrade_time","cr_male_agegrade_parkrunner_id","cr_male_agegrade_name","cr_male_agegrade_agegroup","cr_male_agegrade_set_at_event",
        "cr_female_agegrade","cr_female_agegrade_time","cr_female_agegrade_parkrunner_id","cr_female_agegrade_name","cr_female_agegrade_agegroup","cr_female_agegrade_set_at_event",
    ]
    pe = pe.reindex(columns=cols)
    return pe


# ------------------------- Annotation helper -------------------------

def _annotate_record_points(ax,
                            df: pd.DataFrame,
                            value_col: str,
                            set_event_col: str,
                            name_col: str,
                            value_formatter,
                            x_offset_px: int = 10,
                            y_offset_px: int = 10,
                            color_for_box: str = PARKRUN_YELLOW):
    """
    Add label boxes at events where a new record is *set*.
    """
    import matplotlib.transforms as mtransforms

    for _, row in df.iterrows():
        ev = row.get("event")
        if pd.isna(ev):
            continue
        set_ev = row.get(set_event_col)
        if pd.isna(set_ev) or set_ev == "":
            continue
        if int(set_ev) != int(ev):
            continue

        val = row.get(value_col)
        if pd.isna(val) or val == "":
            continue

        name = (row.get(name_col) or "").strip()
        label = f"{name} — {value_formatter(val)}" if name else f"{value_formatter(val)}"

        trans_offset = mtransforms.ScaledTranslation(
            x_offset_px / 72.0, y_offset_px / 72.0, ax.figure.dpi_scale_trans
        )

        ax.annotate(
            label,
            xy=(ev, val if isinstance(val, (int, float, np.floating)) else np.nan),
            xytext=(0, 0),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=14,
            color=NEAR_WHITE,
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc=color_for_box,
                ec=color_for_box,
                alpha=BBOX_ALPHA,
            ),
            transform=ax.transData + trans_offset,
        )


# ------------------------- Plotting -------------------------

def plot_times(pe: pd.DataFrame, outpath: Path) -> None:
    male_sec = pe["cr_male_time"].apply(parse_time_to_seconds).astype("float")
    female_sec = pe["cr_female_time"].apply(parse_time_to_seconds).astype("float")

    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    add_centered_background_logo(fig, alpha=LOGO_ALPHA)

    # lines
    ax.plot(pe["event"], male_sec, label="Male time", color=PARKRUN_YELLOW, linewidth=LINE_LW)
    ax.plot(pe["event"], female_sec, label="Female time", color=PARKRUN_TEAL, linewidth=LINE_LW)

    # include all values in y-lims (with a little pad)
    ymin = np.nanmin([np.nanmin(male_sec), np.nanmin(female_sec)])
    ymax = np.nanmax([np.nanmax(male_sec), np.nanmax(female_sec)])
    if np.isfinite(ymin) and np.isfinite(ymax):
        pad = max(5.0, 0.05 * (ymax - ymin))
        ax.set_ylim(ymin - pad, ymax + pad)

    # labels / title
    ax.set_xlabel(
        "Event number",
        color=NEAR_WHITE,
        fontweight="bold",
        fontsize=LABEL_SIZE,
        labelpad=10,
    )
    ax.set_ylabel(
        "Time (mm:ss)",
        color=NEAR_WHITE,
        fontweight="bold",
        fontsize=LABEL_SIZE,
        labelpad=10,
    )
    ax.set_title(
        "Course Record Progression — Times",
        color=NEAR_WHITE,
        fontweight="bold",
        fontsize=TITLE_SIZE,
        pad=22,
    )

    # Apply shared axes style (background, grid, spines, ticks)
    apply_standard_axes_style(ax)
    # Time formatting on y-axis
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: fmt_sec_mmss(x)))

    # annotate when a new record is set
    pe_plot = pe.copy()
    pe_plot["cr_male_time_sec"] = male_sec
    pe_plot["cr_female_time_sec"] = female_sec

    _annotate_record_points(
        ax,
        pe_plot,
        value_col="cr_male_time_sec",
        set_event_col="cr_male_time_set_at_event",
        name_col="cr_male_time_name",
        value_formatter=fmt_sec_mmss,
        x_offset_px=10,
        y_offset_px=10,
        color_for_box=PARKRUN_YELLOW,
    )
    _annotate_record_points(
        ax,
        pe_plot,
        value_col="cr_female_time_sec",
        set_event_col="cr_female_time_set_at_event",
        name_col="cr_female_time_name",
        value_formatter=fmt_sec_mmss,
        x_offset_px=10,
        y_offset_px=10,
        color_for_box=PARKRUN_TEAL,
    )

    # legend
    leg = ax.legend(facecolor=PARKRUN_PURPLE, edgecolor=NEAR_WHITE, fontsize=14)
    for txt in leg.get_texts():
        txt.set_color(NEAR_WHITE)
        txt.set_fontweight("bold")
    leg.get_frame().set_alpha(0.0)

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    fig.savefig(outpath, dpi=160, facecolor=PARKRUN_PURPLE)
    plt.close(fig)


def plot_agegrades(pe: pd.DataFrame, outpath: Path) -> None:
    male_ag = pd.to_numeric(pe["cr_male_agegrade"], errors="coerce")
    female_ag = pd.to_numeric(pe["cr_female_agegrade"], errors="coerce")

    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    add_centered_background_logo(fig, alpha=LOGO_ALPHA)

    ax.plot(pe["event"], male_ag, label="Male age grade", color=PARKRUN_YELLOW, linewidth=LINE_LW)
    ax.plot(pe["event"], female_ag, label="Female age grade", color=PARKRUN_TEAL, linewidth=LINE_LW)

    # y-lims
    ymin = np.nanmin([np.nanmin(male_ag), np.nanmin(female_ag)])
    ymax = np.nanmax([np.nanmax(male_ag), np.nanmax(female_ag)])
    if np.isfinite(ymin) and np.isfinite(ymax):
        pad = max(0.3, 0.05 * (ymax - ymin))
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xlabel(
        "Event number",
        color=NEAR_WHITE,
        fontweight="bold",
        fontsize=LABEL_SIZE,
        labelpad=10,
    )
    ax.set_ylabel(
        "Age grade (%)",
        color=NEAR_WHITE,
        fontweight="bold",
        fontsize=LABEL_SIZE,
        labelpad=10,
    )
    ax.set_title(
        "Course Record Progression — Age Grades",
        color=NEAR_WHITE,
        fontweight="bold",
        fontsize=TITLE_SIZE,
        pad=22,
    )

    apply_standard_axes_style(ax)

    _annotate_record_points(
        ax,
        pe,
        value_col="cr_male_agegrade",
        set_event_col="cr_male_agegrade_set_at_event",
        name_col="cr_male_agegrade_name",
        value_formatter=lambda v: f"{float(v):.2f}%",
        x_offset_px=10,
        y_offset_px=-10,
        color_for_box=PARKRUN_YELLOW,
    )
    _annotate_record_points(
        ax,
        pe,
        value_col="cr_female_agegrade",
        set_event_col="cr_female_agegrade_set_at_event",
        name_col="cr_female_agegrade_name",
        value_formatter=lambda v: f"{float(v):.2f}%",
        x_offset_px=10,
        y_offset_px=-10,
        color_for_box=PARKRUN_TEAL,
    )

    leg = ax.legend(facecolor=PARKRUN_PURPLE, edgecolor=NEAR_WHITE, fontsize=14)
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

    pe = build_course_record_progression(df)
    pe.to_csv(SERIES_CSV, index=False)
    print(f"Wrote series CSV -> {SERIES_CSV}")

    plot_times(pe, PLOT_TIMES)
    print(f"Wrote times plot -> {PLOT_TIMES}")

    plot_agegrades(pe, PLOT_AGEGRADES)
    print(f"Wrote age-grades plot -> {PLOT_AGEGRADES}")


if __name__ == "__main__":
    main()

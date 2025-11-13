#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate per-age-group course-record time progression data + visuals.

For each age group (e.g. VM40-44, SW30-34, etc.), we compute the
course-record *time* progression for male and female runners, and
generate:

  visualizations/agegroup_course_records/
    agegroup_<slug>_course_record_progression_series.csv
    agegroup_<slug>_course_record_progression_times.png

Notes:
  - Only rows with runner == 1 are considered (actual runners).
  - Age-grade records are NOT used here; we only track time records
    for each age group. Age-graded fairness is implicit in the age
    group itself.
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# ------------------------- Paths -------------------------

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]            # parkrun_event_data_organizer/
DATA_DIR = PROJECT_ROOT / "data"
EVENT_DIR = DATA_DIR / "event_results"
ASSETS_DIR = PROJECT_ROOT / "assets"
VIS_DIR = PROJECT_ROOT / "visualizations"
AGEGROUP_VIS_DIR = VIS_DIR / "agegroup_course_records"
AGEGROUP_VIS_DIR.mkdir(parents=True, exist_ok=True)

PARKRUN_LOGO = ASSETS_DIR / "parkrun_logo_white.png"

# ------------------------- Style constants (match main CR script) -------------------------

PARKRUN_PURPLE = "#4B2E83"     # background
NEAR_WHITE     = "#F4F4F6"     # labels/grid
PARKRUN_YELLOW = "#FFA300"     # male line
PARKRUN_TEAL   = "#10ECCC"     # female line

TITLE_SIZE   = 26
LABEL_SIZE   = 18
TICK_SIZE    = 16
AXIS_XY_LW   = 2.8
AXIS_TR_LW   = 1.2
GRID_LW      = 1.6
LINE_LW      = 3.2
LEGEND_FS    = 14
BBOX_ALPHA   = 0.20
GRID_ALPHA   = 0.55
LOGO_ALPHA   = 0.12

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

def is_male(g: Any) -> bool:
    return str(g).strip().lower().startswith("m")

def is_female(g: Any) -> bool:
    return str(g).strip().lower().startswith("f")

def _slugify_age_group(s: str) -> str:
    """
    Turn an age-group label like 'VM40-44' into 'vm40_44'
    suitable for filenames.
    """
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    return s.strip("_").lower()

def load_all_events() -> pd.DataFrame:
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

    # Normalize / parse
    if "time" in df.columns:
        df["time_sec"] = df["time"].apply(parse_time_to_seconds)
    else:
        df["time_sec"] = np.nan

    # Ensure expected columns exist
    for col in ["gender", "age_group", "age_grade", "name_first", "name_last", "id", "runner"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Runner mask (only runners count for time CRs)
    df["runner_num"] = pd.to_numeric(df["runner"], errors="coerce").fillna(0).astype(int)

    # Clean age_group
    df["age_group_clean"] = df["age_group"].astype("string").str.strip().replace({"": pd.NA})

    return df

def person_name(r: pd.Series) -> str:
    f = str(r.get("name_first") or "").strip()
    l = str(r.get("name_last") or "").strip()
    return (f"{f} {l}").strip()

# ------------------------- Age-group record tracking -------------------------

def best_row_time_agegroup(dfe: pd.DataFrame, gender: str) -> Optional[pd.Series]:
    """
    Best time row within an age-group-limited DataFrame dfe for a given gender.
    Only considers runner_num == 1 and non-null time_sec.
    """
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

def build_agegroup_course_record_progression(df: pd.DataFrame, age_group: str) -> pd.DataFrame:
    """
    Build a per-event course record time progression for a single age group.
    Returns a DataFrame with columns:

      event,
      age_group,
      cr_male_time, cr_male_time_parkrunner_id, cr_male_time_name, cr_male_time_agegrade, cr_male_time_set_at_event,
      cr_female_time, cr_female_time_parkrunner_id, cr_female_time_name, cr_female_time_agegrade, cr_female_time_set_at_event
    """
    dfg = df[df["age_group_clean"] == age_group].copy()
    if dfg.empty:
        return pd.DataFrame()

    events = sorted(dfg["event"].unique())
    out_rows: List[Dict[str, Any]] = []

    rec_m_time: Optional[int] = None
    rec_m_meta: Dict[str, Any] = {}
    rec_f_time: Optional[int] = None
    rec_f_meta: Dict[str, Any] = {}

    for ev in events:
        dfe = dfg[dfg["event"] == ev].copy()

        m_row = best_row_time_agegroup(dfe, "male")
        f_row = best_row_time_agegroup(dfe, "female")

        if m_row is not None:
            t = int(m_row["time_sec"])
            if rec_m_time is None or t < rec_m_time:
                rec_m_time = t
                rec_m_meta = {
                    "id": m_row.get("id"),
                    "name": person_name(m_row),
                    "age_grade": float(pd.to_numeric(m_row.get("age_grade"), errors="coerce"))
                                 if pd.notna(m_row.get("age_grade")) else None,
                    "set_event": ev,
                }

        if f_row is not None:
            t = int(f_row["time_sec"])
            if rec_f_time is None or t < rec_f_time:
                rec_f_time = t
                rec_f_meta = {
                    "id": f_row.get("id"),
                    "name": person_name(f_row),
                    "age_grade": float(pd.to_numeric(f_row.get("age_grade"), errors="coerce"))
                                 if pd.notna(f_row.get("age_grade")) else None,
                    "set_event": ev,
                }

        row = {
            "event": ev,
            "age_group": age_group,

            "cr_male_time": fmt_sec_mmss(rec_m_time) if rec_m_time is not None else "",
            "cr_male_time_parkrunner_id": rec_m_meta.get("id", ""),
            "cr_male_time_name": rec_m_meta.get("name", ""),
            "cr_male_time_agegrade": rec_m_meta.get("age_grade", np.nan),
            "cr_male_time_set_at_event": rec_m_meta.get("set_event", ""),

            "cr_female_time": fmt_sec_mmss(rec_f_time) if rec_f_time is not None else "",
            "cr_female_time_parkrunner_id": rec_f_meta.get("id", ""),
            "cr_female_time_name": rec_f_meta.get("name", ""),
            "cr_female_time_agegrade": rec_f_meta.get("age_grade", np.nan),
            "cr_female_time_set_at_event": rec_f_meta.get("set_event", ""),
        }
        out_rows.append(row)

    if not out_rows:
        return pd.DataFrame()

    pe = pd.DataFrame(out_rows)
    cols = [
        "event",
        "age_group",
        "cr_male_time","cr_male_time_parkrunner_id","cr_male_time_name","cr_male_time_agegrade","cr_male_time_set_at_event",
        "cr_female_time","cr_female_time_parkrunner_id","cr_female_time_name","cr_female_time_agegrade","cr_female_time_set_at_event",
    ]
    pe = pe.reindex(columns=cols)
    return pe

# ------------------------- Plot helpers (mirror main CR styling) -------------------------

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

def _apply_axes_style(ax: plt.Axes, y_is_time: bool):
    fig = ax.figure
    fig.patch.set_facecolor(PARKRUN_PURPLE)
    ax.set_facecolor(PARKRUN_PURPLE)

    ax.grid(True, linestyle="--", linewidth=GRID_LW, color=NEAR_WHITE, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)

    ax.spines["bottom"].set_color(NEAR_WHITE)
    ax.spines["left"].set_color(NEAR_WHITE)
    ax.spines["bottom"].set_linewidth(AXIS_XY_LW)
    ax.spines["left"].set_linewidth(AXIS_XY_LW)

    for sp in ["top", "right"]:
        ax.spines[sp].set_color(NEAR_WHITE)
        ax.spines[sp].set_alpha(0.4)
        ax.spines[sp].set_linewidth(AXIS_TR_LW)

    ax.tick_params(colors=NEAR_WHITE, which="both", width=1.6, length=6, labelsize=TICK_SIZE)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    if y_is_time:
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: fmt_sec_mmss(x)))

def _annotate_record_points(ax, df: pd.DataFrame, value_col: str, set_event_col: str, name_col: str,
                            value_formatter, x_offset_px: int = 10, y_offset_px: int = 10,
                            color_for_box: str = PARKRUN_YELLOW):
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
            xytext=(0, 0), textcoords="offset points",
            ha="left", va="bottom", fontsize=14, color=NEAR_WHITE,
            bbox=dict(boxstyle="round,pad=0.2", fc=color_for_box, ec=color_for_box, alpha=BBOX_ALPHA),
            transform=ax.transData + trans_offset,
        )

# ------------------------- Plotting per age-group -------------------------

def plot_agegroup_times(pe: pd.DataFrame, age_group: str, outpath: Path):
    """
    Plot course-record time progression for a single age group.
    """
    # parse times into seconds
    male_sec = pe["cr_male_time"].apply(parse_time_to_seconds).astype("float")
    female_sec = pe["cr_female_time"].apply(parse_time_to_seconds).astype("float")

    # if no valid times at all, skip plot
    vals = np.concatenate([male_sec.values, female_sec.values])
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        print(f"[age-group {age_group}] No valid CR times; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    _add_centered_background_logo(fig, alpha=LOGO_ALPHA)

    # lines
    ax.plot(pe["event"], male_sec, label="Male", color=PARKRUN_YELLOW, linewidth=LINE_LW)
    ax.plot(pe["event"], female_sec, label="Female", color=PARKRUN_TEAL, linewidth=LINE_LW)

    # y-lims from combined values
    ymin = float(np.nanmin(vals))
    ymax = float(np.nanmax(vals))
    pad = max(5.0, 0.05 * (ymax - ymin))
    ax.set_ylim(ymin - pad, ymax + pad)

    # labels / title
    ax.set_xlabel("Event number", color=NEAR_WHITE, fontweight="bold", fontsize=LABEL_SIZE, labelpad=10)
    ax.set_ylabel("Time (mm:ss)", color=NEAR_WHITE, fontweight="bold", fontsize=LABEL_SIZE, labelpad=10)
    ax.set_title(
        f"Course Record Progression — Times ({age_group})",
        color=NEAR_WHITE, fontweight="bold", fontsize=TITLE_SIZE, pad=22
    )

    _apply_axes_style(ax, y_is_time=True)

    # annotate record-setting events
    pe_plot = pe.copy()
    pe_plot["cr_male_time_sec"] = male_sec
    pe_plot["cr_female_time_sec"] = female_sec

    _annotate_record_points(
        ax, pe_plot,
        value_col="cr_male_time_sec",
        set_event_col="cr_male_time_set_at_event",
        name_col="cr_male_time_name",
        value_formatter=fmt_sec_mmss,
        x_offset_px=10, y_offset_px=10,
        color_for_box=PARKRUN_YELLOW
    )
    _annotate_record_points(
        ax, pe_plot,
        value_col="cr_female_time_sec",
        set_event_col="cr_female_time_set_at_event",
        name_col="cr_female_time_name",
        value_formatter=fmt_sec_mmss,
        x_offset_px=10, y_offset_px=10,
        color_for_box=PARKRUN_TEAL
    )

    # legend
    leg = ax.legend(facecolor=PARKRUN_PURPLE, edgecolor=NEAR_WHITE, fontsize=LEGEND_FS)
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

    age_groups = sorted(df["age_group_clean"].dropna().unique())
    if not age_groups:
        print("No age_group values found in event data; nothing to do.")
        return

    print(f"Found {len(age_groups)} age groups. Generating CR progressions…")

    for ag in age_groups:
        slug = _slugify_age_group(ag)
        series_csv = AGEGROUP_VIS_DIR / f"agegroup_{slug}_course_record_progression_series.csv"
        plot_png   = AGEGROUP_VIS_DIR / f"agegroup_{slug}_course_record_progression_times.png"

        pe = build_agegroup_course_record_progression(df, ag)
        if pe.empty:
            print(f"[age-group {ag}] no data; skipping.")
            continue

        pe.to_csv(series_csv, index=False)
        print(f"[age-group {ag}] Wrote series CSV -> {series_csv}")

        plot_agegroup_times(pe, ag, plot_png)
        print(f"[age-group {ag}] Wrote times plot -> {plot_png}")

if __name__ == "__main__":
    main()

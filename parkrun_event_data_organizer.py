"""
parkrun_event_data_organizer.py

Scrapes per-event results (participants and volunteers) and writes:
  data/
    event_results/event_0001.csv ...
    participants_master.csv
    event_series_summary.csv
    age_group_summaries/<agegroup>.csv

Then calls every Python script inside ./scripts/ one-by-one.

Usage examples:
  python parkrun_event_data_organizer.py
  python parkrun_event_data_organizer.py --base-url https://www.parkrun.org.uk/holyrood/ --end 148

Requirements:
  pip install requests beautifulsoup4 pandas python-dateutil
"""

from __future__ import annotations
import argparse
import logging
import os
import re
import sys
import math
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
import pandas as pd
from parkrun_config import (
    PROJECT_ROOT,
    DATA_DIR,
    EVENT_DIR,
    MASTER_PARTICIPANTS_CSV,
    AGEGROUP_DIR,
    SCRIPTS_DIR,
    SERIES_SUMMARY_CSV,
    ROOT_URL_DEFAULT,
    EVENT_NAME_DEFAULT,
    START_EVENT_DEFAULT,
    END_EVENT_DEFAULT,
    HEADERS,
    ID_HREF_RE,
    PERCENT_RE,
    RESULTS_URL_TEMPLATE,
    VOLUNTEERS_URL_TEMPLATE,
    parse_time_to_seconds,
    PARKRUNS_MASTER_CSV,
    VOLUNTEERS_MASTER_CSV,
)

# logger (not really config; cheap to keep local)
log = logging.getLogger("parkrun")

# Local alias for convenience (comes from config)
EVENT_NAME = EVENT_NAME_DEFAULT

# ------------------------- Models --------------------------------------

@dataclass
class EventRow:
    position: Optional[int]
    name_first: str
    name_last: str
    id: Optional[str]
    gender: Optional[str]
    age_group: Optional[str]
    age_grade: Optional[float]
    club: Optional[str]
    first_time_participant: int
    first_time_volunteer: int
    pb: int
    runner: int
    participant: int
    volunteer: int
    time: Optional[str]

@dataclass
class EventData:
    event_no: int
    url: str
    participants: List[EventRow] = field(default_factory=list)
    volunteers: List[Tuple[str, str, Optional[str]]] = field(default_factory=list)  # (first,last,id)

# ------------------------- Helpers -------------------------------------


def soup_text(node) -> str:
    """
    Extract concatenated text from a BeautifulSoup element, stripping
    whitespace and replacing \xa0 with regular spaces.
    """
    if not node:
        return ""
    s = " ".join(node.stripped_strings)
    s = s.replace("\xa0", " ").strip()
    return s


def normalize_root_url(raw: Optional[str]) -> str:
    """
    Normalize and lightly sanity-check the base event URL.
    Example valid forms:
      https://www.parkrun.us/farmpond/
      https://www.parkrun.org.uk/holyrood/
    """
    if not raw:
        raise ValueError("A --base-url is required (e.g., https://www.parkrun.us/farmpond/)")
    u = raw.strip()
    # quick sanity check
    if not re.search(r"https?://(www\.)?parkrun\.(us|org\.uk)/[^/\s]+/?$", u):
        log.warning("The provided --base-url doesn't look like a standard parkrun event URL: %s", u)
    return u.rstrip("/")


def seconds_to_time_str(sec: int) -> str:
    if sec is None:
        return ""
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    else:
        return f"{m}:{s:02d}"


def parse_age_grade(cell_text: str) -> Optional[float]:
    """
    Extract numeric age grade from text like '75.23 %' or return None.
    """
    if not cell_text:
        return None
    m = PERCENT_RE.search(cell_text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def split_name(full_name: str) -> Tuple[str, str]:
    """
    Very simple heuristic splitter: everything except the last token
    is first name; last token is last name.
    """
    full_name = (full_name or "").strip()
    if not full_name:
        return "", ""
    parts = full_name.split()
    if len(parts) == 1:
        return parts[0], ""
    first = " ".join(parts[:-1])
    last = parts[-1]
    return first, last


def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(EVENT_DIR, exist_ok=True)
    os.makedirs(AGEGROUP_DIR, exist_ok=True)


import time
from random import uniform

def get_http_session(cookie_header: Optional[str] = None) -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    if cookie_header:
        s.headers["Cookie"] = cookie_header
    return s


def fetch_html(url: str, session: Optional[requests.Session] = None) -> Optional[str]:
    s = session or get_http_session()
    tries = 3
    for i in range(tries):
        try:
            log.info(f"GET {url}")
            r = s.get(url, timeout=30)
            log.debug(f" -> HTTP {r.status_code}")
            if r.status_code == 200 and r.text:
                return r.text
            if r.status_code in (403, 429, 503):
                sleep_s = 1.5 + uniform(0, 1.5)
                log.warning(f"HTTP {r.status_code} for {url}; retrying in {sleep_s:.1f}s (attempt {i+1}/{tries})")
                time.sleep(sleep_s)
            else:
                log.warning(f"Unexpected HTTP {r.status_code} for {url}; content may be empty.")
        except requests.RequestException as e:
            sleep_s = 1.5 + uniform(0, 1.5)
            log.warning(f"Error fetching {url}: {e}; retrying in {sleep_s:.1f}s (attempt {i+1}/{tries})")
            time.sleep(sleep_s)
    log.error(f"Failed to fetch HTML after {tries} attempts: {url}")
    return None

# ------------------------- Parsing per-event results -------------------------

def parse_results_table(event_no: int, html: str) -> EventData:
    """
    Parse the main results table for one event using header-based detection.
    Volunteers are handled separately from the lower-page block.
    """
    soup = BeautifulSoup(html, "html.parser")

    # find correct table by headers
    table = None
    for tbl in soup.find_all("table"):
        ths = [soup_text(th).lower() for th in tbl.find_all("th")]
        if not ths:
            continue
        if ("position" in ths and "parkrunner" in ths and "time" in ths) or \
           ("pos" in ths and "parkrunner" in ths):
            table = tbl
            break

    if not table:
        log.warning(f"[event {event_no}] No results table found.")
        return EventData(event_no=event_no, url="", participants=[], volunteers=[])

    rows: list[EventRow] = []
    body = table.find("tbody") or table

    for tr in body.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue

        # position
        pos_txt = soup_text(tds[0])
        try:
            pos = int(pos_txt.strip())
        except Exception:
            pos = None

        # parkrunner cell
        runner_cell = tds[1]
        link = runner_cell.find("a", href=True)
        full_name = soup_text(link) if link else soup_text(runner_cell)
        first, last = split_name(full_name)
        pid = None
        if link:
            href = link.get("href", "")
            m = ID_HREF_RE.search(href)
            if m:
                pid = m.group(1)

        row_full_text = soup_text(tr)
        first_time_participant = 1 if re.search(r"First\s*Timer", row_full_text, re.I) else 0
        pb_flag = 1 if re.search(r"New\s*PB", row_full_text, re.I) else 0
        first_time_volunteer = 0

        # gender
        gender = None
        if len(tds) > 2:
            gender = soup_text(tds[2]) or None
            if gender:
                gender = gender.split()[0]

        # age group + age grade from same cell
        age_group = None
        age_grade = None
        if len(tds) > 3:
            ag_cell = soup_text(tds[3])
            m_group = re.search(r"\b[A-Z]{1,3}\d{1,2}-\d{1,2}\b", ag_cell)
            if m_group:
                age_group = m_group.group(0)
            else:
                age_group = ag_cell.split()[0] if ag_cell else None
            age_grade = parse_age_grade(ag_cell)

        # club
        club = None
        if len(tds) > 4:
            club_txt = soup_text(tds[4]).strip()
            club = club_txt or None

        # time is last cell; strip PB / first timer annotations
        time_cell = tds[-1]
        time_str = soup_text(time_cell) if time_cell else None
        if time_str:
            time_str = re.sub(r"(New\s*PB!?|First\s*Timer!?|PB\s*)", "", time_str, flags=re.I).strip()

        rows.append(
            EventRow(
                position=pos,
                name_first=first,
                name_last=last,
                id=pid,
                gender=gender,
                age_group=age_group,
                age_grade=age_grade,
                club=club,
                first_time_participant=first_time_participant,
                first_time_volunteer=first_time_volunteer,
                pb=pb_flag,
                runner=1,
                participant=1,  # runners are participants
                volunteer=0,
                time=time_str,
            )
        )

    return EventData(event_no=event_no, url="", participants=rows, volunteers=[])

def _debug_volunteer_scan(soup: BeautifulSoup, event_no: int, dump_html: bool = False) -> None:
    try:
        hits = soup.find_all(string=re.compile(r"volunteer", re.I))
    except Exception:
        hits = []
    log.debug(f"[event {event_no}][vol-debug] text hits containing 'volunteer': {len(hits)}")
    for i, txt in enumerate(hits[:10]):
        el = txt.parent if hasattr(txt, "parent") else None
        tag = el.name if getattr(el, "name", None) else "?"
        snippet = soup_text(el)[:140].replace("\n", " ") if el else ""
        anchors = el.find_all("a", href=ID_HREF_RE) if el else []
        log.debug(f"[event {event_no}][vol-debug] hit#{i} tag=<{tag}> anchors={len(anchors)} text='{snippet}'")
        sib = el
        for j in range(1, 5):
            sib = sib.find_next_sibling() if sib else None
            if not sib:
                break
            a2 = sib.find_all("a", href=ID_HREF_RE)
            if a2:
                log.debug(f"[event {event_no}][vol-debug]   sibling+{j} <{sib.name}> anchors={len(a2)}")
    if dump_html:
        os.makedirs("debug_html", exist_ok=True)
        p = os.path.join("debug_html", f"event_{event_no:04d}_full.html")
        with open(p, "w", encoding="utf-8") as f:
            f.write(str(soup))
        log.info(f"[event {event_no}][vol-debug] wrote full page -> {p}")


def extract_volunteers_from_page(soup: BeautifulSoup, event_no: int) -> list[tuple[str, str, Optional[str]]]:
    """
    Match the original behavior: find the 'Thanks to the volunteers' block
    on the results page and extract (first, last, id).
    """
    volunteers: list[tuple[str, str, Optional[str]]] = []

    def collect_from(container):
        if not container:
            return
        for a in container.find_all("a", href=True):
            href = a.get("href", "")
            if "/parkrunner/" not in href:
                continue
            m = ID_HREF_RE.search(href)
            pid = m.group(1) if m else None
            full = soup_text(a)
            f, l = split_name(full)
            if (f or l) and (f + l).strip():
                volunteers.append((f, l, pid))

    # main pattern: "Thanks to the volunteers"
    header_pat = re.compile(r"thanks\s*to\s*the\s*volunteers", re.I)
    header_node = soup.find(string=header_pat)
    if header_node:
        hdr = header_node.parent
        collect_from(hdr)
        sib = hdr
        # scan a few siblings in case the names are in following <p> or <div>
        for _ in range(6):
            sib = sib.find_next_sibling() if sib else None
            if not sib:
                break
            if getattr(sib, "name", "").lower() in {"h2", "h3", "h4"}:
                break
            collect_from(sib)

    # fallback: any block mentioning "volunteer" with anchor tags
    if not volunteers:
        for block in soup.find_all(string=re.compile(r"volunteer", re.I)):
            el = getattr(block, "parent", None)
            if not el:
                continue
            if el.find("table"):
                continue
            collect_from(el)
            sib = el
            for _ in range(4):
                sib = sib.find_next_sibling() if sib else None
                if not sib:
                    break
                if sib.find("table"):
                    continue
                collect_from(sib)
            if volunteers:
                break

    # de-dup
    seen = set()
    out: list[tuple[str, str, Optional[str]]] = []
    for f, l, pid in volunteers:
        key = pid if pid else (f"{f} {l}").strip().lower()
        if key and key not in seen:
            out.append((f, l, pid))
            seen.add(key)

    log.info(f"[volunteers] found {len(out)} anchor(s) in volunteers section for event {event_no}")
    return out

# ------------------------- Event CSV IO -------------------------

def load_existing_event_csvs() -> Dict[int, pd.DataFrame]:
    """
    Load already-scraped event CSVs from DATA_DIR/event_results as a dict:
      { event_no: DataFrame }
    If none exist, returns {}.
    """
    out: Dict[int, pd.DataFrame] = {}
    if not os.path.isdir(EVENT_DIR):
        return out

    for fn in os.listdir(EVENT_DIR):
        if not fn.startswith("event_") or not fn.endswith(".csv"):
            continue
        m = re.search(r"event_(\d{4})\.csv$", fn)
        if not m:
            continue
        evno = int(m.group(1))
        try:
            df = pd.read_csv(os.path.join(EVENT_DIR, fn), dtype={"id": "string"})
            out[evno] = df
        except Exception as e:
            log.warning(f"Failed to read {fn}: {e}")
    return out


def normalize_columns(df: pd.DataFrame, event_no: int) -> pd.DataFrame:
    """
    Ensure a standard set of columns for event result CSVs.
    """
    expected_cols = [
        "position",
        "name_first",
        "name_last",
        "id",
        "gender",
        "age_group",
        "age_grade",
        "club",
        "first_time_participant",
        "first_time_volunteer",
        "pb",
        "runner",
        "participant",
        "volunteer",
        "time",
        "event",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df["event"] = event_no
    if "id" in df.columns:
        df["id"] = df["id"].astype("string")
    return df


def save_event_df(event_no: int, df: pd.DataFrame) -> str:
    fp = os.path.join(EVENT_DIR, f"event_{event_no:04d}.csv")
    df.to_csv(fp, index=False)
    return fp

# ------------------------- Aggregation helpers -------------------------

def autodiscover_max_event(
    base_url: str,
    start: int = 1,
    ceiling: int = 2000,
    session: Optional[requests.Session] = None,
) -> int:
    """
    Autodiscover the last event number by walking forward from `start`
    until we stop seeing a valid results table / volunteers block.

    This is the same logic as the original script, just wired to the
    refactored config (RESULTS_URL_TEMPLATE).
    """
    lo = start
    hi = start

    while hi <= ceiling:
        url = RESULTS_URL_TEMPLATE.format(base_url=base_url.rstrip("/"), event_no=hi)
        html = fetch_html(url, session=session)
        if not html:
            log.info(f"[autodiscover] stop at {hi} (no html)")
            break

        soup = BeautifulSoup(html, "html.parser")

        # Does this page have a normal results table?
        table_exists = any(
            "Position" in (th.get_text() if th else "")
            for th in soup.find_all("th")
        )

        # Or at least some “thanks to the volunteers” text?
        has_any_result = table_exists or soup.find(
            string=re.compile(r"Thanks to the volunteers", re.I)
        )

        log.debug(
            f"[autodiscover] event {hi}: table={bool(table_exists)} "
            f"volunteers_block={bool(has_any_result)}"
        )

        if not has_any_result:
            break

        hi += 1

    last = hi - 1 if hi > lo else start - 1
    if last < start:
        last = start

    log.info(f"[autodiscover] last event detected: {last}")
    return last

def update_master_participants(all_event_dfs: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Build / update the participants_master.csv summary.
    One row per unique parkrunner (by id or name), with:
      - gender, age_group (latest known)
      - pb_time, pb_position, pb_age_grade
      - num_runs, num_volunteers, num_participations
      - volunteer_percentage

    Additionally writes:
      - parkruns_master.csv  (one row per *run* where runner == 1)
      - volunteers_master.csv (one row per *volunteer instance* where volunteer == 1)
    """
    if not all_event_dfs:
        log.warning("No event data frames provided; cannot update master participants.")
        return pd.DataFrame()

    os.makedirs(DATA_DIR, exist_ok=True)

    # Full concatenated dataset of all events
    all_events_concat = pd.concat(
        [df.assign(event=evno) for evno, df in all_event_dfs.items()],
        ignore_index=True
    )

    # Helper boolean columns
    all_events_concat["time_sec"] = all_events_concat["time"].apply(parse_time_to_seconds)
    all_events_concat["is_run"] = pd.to_numeric(all_events_concat["runner"], errors="coerce").fillna(0) == 1
    all_events_concat["is_volunteer"] = pd.to_numeric(all_events_concat["volunteer"], errors="coerce").fillna(0) == 1
    all_events_concat["is_participant"] = pd.to_numeric(all_events_concat["participant"], errors="coerce").fillna(0) == 1

    # -----------------------------
    # NEW: parkruns_master.csv
    # -----------------------------
    try:
        parkruns_master_df = all_events_concat.loc[all_events_concat["is_run"], [
            "position",
            "name_first",
            "name_last",
            "id",
            "gender",
            "age_group",
            "age_grade",
            "club",
            "first_time_participant",
            "pb",
            "time",
            "event",
        ]].copy()

        parkruns_master_df["id"] = parkruns_master_df["id"].astype("string")
        parkruns_master_df.to_csv(PARKRUNS_MASTER_CSV, index=False)
        log.info(
            f"Wrote parkruns master CSV -> {PARKRUNS_MASTER_CSV} "
            f"(rows={len(parkruns_master_df)})"
        )
    except KeyError as e:
        log.error(f"Error building parkruns_master.csv (missing column): {e}")

    # -----------------------------
    # NEW: volunteers_master.csv
    # -----------------------------
    try:
        volunteers_master_df = all_events_concat.loc[all_events_concat["is_volunteer"], [
            "name_first",
            "name_last",
            "id",
            "first_time_volunteer",
        ]].copy()

        volunteers_master_df["id"] = volunteers_master_df["id"].astype("string")
        volunteers_master_df.to_csv(VOLUNTEERS_MASTER_CSV, index=False)
        log.info(
            f"Wrote volunteers master CSV -> {VOLUNTEERS_MASTER_CSV} "
            f"(rows={len(volunteers_master_df)})"
        )
    except KeyError as e:
        log.error(f"Error building volunteers_master.csv (missing column): {e}")

    # -----------------------------
    # Existing participants_master logic
    # -----------------------------

    def identity_key(row):
        return row["id"] if pd.notnull(row.get("id")) and str(row.get("id")).strip() != "" \
               else (f"{row.get('name_first','')} {row.get('name_last','')}").strip().lower()

    all_events_concat["identity"] = all_events_concat.apply(identity_key, axis=1)

    master = []
    for identity, g in all_events_concat.groupby("identity"):
        g_sorted = g.sort_values(["event", "position"], ascending=[True, True])

        runs = g_sorted[g_sorted["is_run"]]
        vols = g_sorted[g_sorted["is_volunteer"]]
        parts = g_sorted[g_sorted["is_participant"]]

        num_runs = runs["event"].nunique()
        num_vols = vols["event"].nunique()
        num_parts = parts["event"].nunique()
        total_events = max(num_runs + num_vols, 1)
        volunteer_percentage = 100.0 * num_vols / total_events

        pb_run = runs.dropna(subset=["time_sec"]).sort_values("time_sec").head(1)
        if not pb_run.empty:
            pb_row = pb_run.iloc[0]
            pb_time_sec = int(pb_row["time_sec"])
            pb_time_str = seconds_to_time_str(pb_time_sec)
            pb_position = pb_row.get("position")
            pb_age_grade = pb_row.get("age_grade")
        else:
            pb_time_str = ""
            pb_time_sec = None
            pb_position = None
            pb_age_grade = None

        latest_row = g_sorted.iloc[-1]
        gender = latest_row.get("gender")
        age_group = latest_row.get("age_group")
        name_first = latest_row.get("name_first")
        name_last = latest_row.get("name_last")
        pid = latest_row.get("id")

        master.append({
            "name_first": name_first,
            "name_last": name_last,
            "id": pid,
            "gender": gender,
            "age_group": age_group,
            "pb_position": pb_position,
            "pb_age_grade": pb_age_grade,
            "pb_time": pb_time_str,
            "num_runs": num_runs,
            "num_volunteers": num_vols,
            "num_participations": num_parts,
            "volunteer_percentage": volunteer_percentage,
        })

    master_df = pd.DataFrame(master)
    master_df.to_csv(MASTER_PARTICIPANTS_CSV, index=False)
    log.info(f"Wrote master participants CSV -> {MASTER_PARTICIPANTS_CSV} (rows={len(master_df)})")
    return master_df

def build_event_series_summary(
    all_event_dfs: Dict[int, pd.DataFrame],
    event_name: str,
) -> pd.DataFrame:
    """
    Build event_series_summary.csv with one row per event, aggregating:
      - total runners
      - total participants
      - total volunteers
      - course record progression info (filled by other scripts later)
    """
    if not all_event_dfs:
        log.warning("No event data frames available; cannot build series summary.")
        cols = ["event", "date", "total_runners", "total_participants",
                "total_volunteers", "course_record_male_time",
                "course_record_female_time", "course_record_male_parkrunner_id",
                "course_record_female_parkrunner_id", "event_name"]
        pd.DataFrame(columns=cols).to_csv(SERIES_SUMMARY_CSV, index=False)
        return pd.DataFrame(columns=cols)

    per_event = []
    for evno, d in sorted(all_event_dfs.items()):
        d = d.copy()
        d["runner"] = pd.to_numeric(d["runner"], errors="coerce").fillna(0).astype(int)
        d["volunteer"] = pd.to_numeric(d["volunteer"], errors="coerce").fillna(0).astype(int)
        d["participant"] = pd.to_numeric(d["participant"], errors="coerce").fillna(0).astype(int)

        total_runners = d[d["runner"] == 1]["id"].nunique()
        total_volunteers = d[d["volunteer"] == 1]["id"].nunique()
        total_participants = d[d["participant"] == 1]["id"].nunique()

        per_event.append({
            "event": evno,
            "date": None,
            "total_runners": total_runners,
            "total_participants": total_participants,
            "total_volunteers": total_volunteers,
            "course_record_male_time": None,
            "course_record_female_time": None,
            "course_record_male_parkrunner_id": None,
            "course_record_female_parkrunner_id": None,
            "event_name": event_name,
        })

    out = pd.DataFrame(per_event)
    out.to_csv(SERIES_SUMMARY_CSV, index=False)
    log.info(f"Wrote series summary -> {SERIES_SUMMARY_CSV} (rows={len(out)})")
    return out


def build_agegroup_summaries(all_event_dfs: Dict[int, pd.DataFrame]) -> Dict[str, str]:
    """
    Writes one CSV per age group under AGEGROUP_DIR.
    Each CSV has rows with columns:
      metric,value,event,parkrunner_id,name_first,name_last,gender
    Includes time & age-grade extremes (with who+event), per-event min/max of avg/median,
    and most_runners_agegroup_<slug>.
    """
    os.makedirs(AGEGROUP_DIR, exist_ok=True)
    if not all_event_dfs:
        return {}

    full = pd.concat([df.assign(event=evno) for evno, df in all_event_dfs.items()], ignore_index=True)

    for col in ["runner", "volunteer", "participant"]:
        if col in full.columns:
            full[col] = pd.to_numeric(full[col], errors="coerce").fillna(0).astype(int)
        else:
            full[col] = 0

    full["time_sec"] = full["time"].apply(parse_time_to_seconds)

    ag = full.dropna(subset=["age_group"])
    written: Dict[str, str] = {}

    def add_metric(rows_list, metric_name, value, event=None, parkrunner_id=None, row=None):
        name_first = row.get("name_first") if row is not None else None
        name_last = row.get("name_last") if row is not None else None
        gender = row.get("gender") if row is not None else None
        rows_list.append({
            "metric": metric_name,
            "value": value,
            "event": event,
            "parkrunner_id": parkrunner_id,
            "name_first": name_first,
            "name_last": name_last,
            "gender": gender,
        })

    for age_group, g in ag.groupby("age_group"):
        ag_slug = re.sub(r"[^A-Za-z0-9]+", "_", str(age_group)).strip("_").lower()

        rows = []

        runners = g[g["runner"] == 1].dropna(subset=["time_sec"])
        if not runners.empty:
            best_idx = runners["time_sec"].idxmin()
            best_row = runners.loc[best_idx]
            add_metric(
                rows,
                "best_time",
                seconds_to_time_str(int(best_row["time_sec"])),
                event=best_row["event"],
                parkrunner_id=best_row.get("id"),
                row=best_row,
            )

        if "age_grade" in g.columns:
            ag_valid = g.dropna(subset=["age_grade"])
            if not ag_valid.empty:
                best_idx = ag_valid["age_grade"].idxmax()
                best_row = ag_valid.loc[best_idx]
                add_metric(
                    rows,
                    "best_age_grade",
                    float(best_row["age_grade"]),
                    event=best_row["event"],
                    parkrunner_id=best_row.get("id"),
                    row=best_row,
                )

        for ev, evg in g.groupby("event"):
            run_evg = evg[evg["runner"] == 1].dropna(subset=["time_sec"])
            if not run_evg.empty:
                min_time = run_evg["time_sec"].min()
                max_time = run_evg["time_sec"].max()
                add_metric(rows, "min_time_event", seconds_to_time_str(int(min_time)), event=ev)
                add_metric(rows, "max_time_event", seconds_to_time_str(int(max_time)), event=ev)

        for ev, evg in g.groupby("event"):
            run_evg = evg[evg["runner"] == 1].dropna(subset=["time_sec"])
            if not run_evg.empty:
                avg_time = run_evg["time_sec"].mean()
                med_time = run_evg["time_sec"].median()
                add_metric(rows, "avg_time_event", seconds_to_time_str(int(avg_time)), event=ev)
                add_metric(rows, "median_time_event", seconds_to_time_str(int(med_time)), event=ev)

        for ev, evg in g.groupby("event"):
            num_runners = int((evg["runner"] == 1).sum())
            add_metric(rows, "num_runners_event", num_runners, event=ev)

        evg = g.groupby("event").agg(num_runners=("runner", lambda x: int((x == 1).sum())))
        if not evg.empty:
            ev_max = evg["num_runners"].idxmax()
            add_metric(rows, f"most_runners_agegroup_{ag_slug}", int(evg.loc[ev_max, "num_runners"]), event=ev_max)

        df_out = pd.DataFrame(rows)
        out_path = os.path.join(AGEGROUP_DIR, f"{ag_slug}.csv")
        df_out.to_csv(out_path, index=False)
        written[ag_slug] = out_path
        log.info(f"[age-groups] wrote {age_group} -> {out_path} (rows={len(df_out)})")

    return written

# ------------------------- Run scripts from ./scripts/ -------------------------

def run_all_scripts(scripts_dir: str) -> None:
    """
    Executes every *.py in scripts_dir (non-recursive), in sorted order.
    Skips files that start with '_' or are not files.
    Uses current Python interpreter.
    """
    if not os.path.isdir(scripts_dir):
        log.info("No scripts directory found at %s; skipping.", scripts_dir)
        return
    scripts = [
        os.path.join(scripts_dir, f)
        for f in sorted(os.listdir(scripts_dir))
        if f.endswith(".py") and not f.startswith("_") and os.path.isfile(os.path.join(scripts_dir, f))
    ]
    if not scripts:
        log.info("No scripts to run in %s; skipping.", scripts_dir)
        return

    log.info("Running %d script(s) from %s …", len(scripts), scripts_dir)
    for script in scripts:
        log.info(" -> running: %s", script)
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), env.get("PYTHONPATH", "")])
            subprocess.run([sys.executable, script], check=True, cwd=PROJECT_ROOT, env=env)
        except subprocess.CalledProcessError as e:
            log.error("Script failed: %s (exit=%s)", script, e.returncode)
        except Exception as e:
            log.error("Error running script %s: %s", script, e)

# ------------------------- Main -------------------------

def main():
    parser = argparse.ArgumentParser(description="parkrun event data organizer")

    # Event configuration (now with default root URL)
    parser.add_argument(
        "--base-url",
        default=ROOT_URL_DEFAULT,
        help=(
            "Full parkrun event URL, e.g. "
            "https://www.parkrun.us/farmpond/ or https://www.parkrun.org.uk/holyrood/ "
            f"(default: {ROOT_URL_DEFAULT})"
        ),
    )
    
    parser.add_argument(
        "--event-name",
        default=EVENT_NAME,
        help=f"Human-readable event name, e.g. 'Farm Pond' (default: '{EVENT_NAME}')",
    )

    # Range configuration
    parser.add_argument("--start", type=int, default=START_EVENT_DEFAULT,
                        help=f"First event number to include (default: {START_EVENT_DEFAULT})")
    parser.add_argument("--end", type=int, default=END_EVENT_DEFAULT,
                        help="Last event number to include (inclusive). If not given, autodetect latest event.")
    parser.add_argument("--max", type=int, default=None,
                        help="Shortcut for --start 1 --end <max>")

    parser.add_argument(
        "--cookie",
        default=None,
        help="Optional raw Cookie header value to include with requests (copy from your browser if needed)."
    )

    parser.add_argument("--refresh", action="store_true", help="Ignore local event CSVs and re-scrape the requested range.")

    # Logging
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--verbose", action="store_true", help="Verbose (DEBUG) logging")
    g.add_argument("--quiet", action="store_true", help="Only warnings and errors")

    args = parser.parse_args()
    extra_cookie = args.cookie

    session = get_http_session(extra_cookie)

    event_name = args.event_name

    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    if args.quiet:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    base = args.base_url or ROOT_URL_DEFAULT
    base_url = normalize_root_url(base)

    start = args.start
    end = args.end

    if args.max is not None:
        start = 1
        end = args.max

    log.info(f"Collecting data for: {base}")
    log.info(f"Events: {start} → {end if end is not None else 'auto'}")

    ensure_dirs()

    session = get_http_session(args.cookie)

    existing = load_existing_event_csvs()
    log.info(f"Found existing local events: {sorted(existing.keys())}")

    if end is None:
        seed = max(1, max(existing.keys(), default=0) + 0)
        end = autodiscover_max_event(base_url=base_url, start=seed, session=session)
        if end < start:
            end = start

    all_event_dfs: Dict[int, pd.DataFrame] = {}

    def identity_key(row):
        return row["id"] if pd.notnull(row.get("id")) and str(row.get("id")).strip() != "" \
               else (f"{row.get('name_first','')} {row.get('name_last','')}").strip().lower()
    volunteer_history: set = set()

    for evno in sorted(k for k in existing.keys() if k < start):
        df = existing[evno]
        for _, r in df[df["volunteer"] == 1].iterrows():
            volunteer_history.add(identity_key(r))

    if not args.refresh:
        for evno in sorted(k for k in existing.keys() if start <= k <= end):
            all_event_dfs[evno] = existing[evno]
            for _, r in existing[evno][existing[evno]["volunteer"] == 1].iterrows():
                volunteer_history.add(identity_key(r))
            log.info(f"[event {evno}] loaded from disk (rows={len(existing[evno])})")

    for evno in range(start, end + 1):
        if evno in all_event_dfs:
            continue

        url = RESULTS_URL_TEMPLATE.format(base_url=base_url, event_no=evno)
        html = fetch_html(url, session=session)
        if not html:
            log.warning(f"[event {evno}] No HTML received; skipping.")
            continue

        ed = parse_results_table(evno, html)
        ed.url = url

        # Volunteers come from the same results page, in the "Thanks to the volunteers" block.
        soup = BeautifulSoup(html, "html.parser")
        vols = extract_volunteers_from_page(soup, evno)
        if not vols:
            _debug_volunteer_scan(soup, evno, dump_html=False)
        ed.volunteers = vols

        # Mark runners who are also volunteers
        volunteer_ids = {pid for _, _, pid in ed.volunteers if pid}
        volunteer_names = {(f"{f} {l}").strip().lower() for f, l, _ in ed.volunteers}

        for r in ed.participants:
            namekey = (f"{r.name_first} {r.name_last}").strip().lower()
            if (r.id and r.id in volunteer_ids) or (namekey in volunteer_names):
                r.volunteer = 1
                r.participant = 1  # ensure they’re marked as participants if they weren’t already

        # Add volunteers who didn't appear in the results table (e.g. non-running volunteers)
        known_ids = {r.id for r in ed.participants if r.id}
        known_namekeys = {(f"{r.name_first} {r.name_last}").strip().lower()
                          for r in ed.participants}

        for f, l, pid in ed.volunteers:
            namekey = (f"{f} {l}").strip().lower()
            if (pid and pid in known_ids) or (namekey in known_namekeys):
                continue
            ed.participants.append(
                EventRow(
                    position=None,
                    name_first=f,
                    name_last=l,
                    id=pid,
                    gender=None,
                    age_group=None,
                    age_grade=None,
                    club=None,
                    first_time_participant=0,
                    first_time_volunteer=0,
                    pb=0,
                    runner=0,
                    participant=1,  # volunteers-only are still participants
                    volunteer=1,
                    time=None,
                )
            )

        # compute first-time-volunteer flag using cross-event history
        for r in ed.participants:
            key = r.id if r.id else (f"{r.name_first} {r.name_last}").strip().lower()
            if r.volunteer == 1:
                r.first_time_volunteer = 1 if key not in volunteer_history else 0
                # once they’ve volunteered here, add to history so future events know
                volunteer_history.add(key)

        rows_for_df = []
        for row in ed.participants:
            rows_for_df.append({
                "position": row.position,
                "name_first": row.name_first,
                "name_last": row.name_last,
                "id": row.id,
                "gender": row.gender,
                "age_group": row.age_group,
                "age_grade": row.age_grade,
                "club": row.club,
                "first_time_participant": row.first_time_participant,
                "first_time_volunteer": row.first_time_volunteer,
                "pb": row.pb,
                "runner": row.runner,
                "participant": row.participant,
                "volunteer": row.volunteer,
                "time": row.time,
            })

        for f, l, pid in ed.volunteers:
            rows_for_df.append({
                "position": None,
                "name_first": f,
                "name_last": l,
                "id": pid,
                "gender": None,
                "age_group": None,
                "age_grade": None,
                "club": None,
                "first_time_participant": 0,
                "first_time_volunteer": 0,
                "pb": 0,
                "runner": 0,
                "participant": 1,
                "volunteer": 1,
                "time": None,
            })

        df_event = pd.DataFrame(rows_for_df)
        df_event = normalize_columns(df_event, evno)
        save_event_df(evno, df_event)
        all_event_dfs[evno] = df_event
        log.info(f"[event {evno}] scraped & saved (rows={len(df_event)})")

    if not all_event_dfs:
        log.error("No event data collected; exiting.")
        return 1

    master_df = update_master_participants(all_event_dfs)
    series_df = build_event_series_summary(all_event_dfs, event_name=event_name)
    ag_written = build_agegroup_summaries(all_event_dfs)

    log.info(f"Master participants: {len(master_df)} rows")
    log.info(f"Series summary: {len(series_df)} rows")
    log.info(f"Age-group summaries: {len(ag_written)} files written")

    log.info("Now running visualization scripts from ./scripts/ …")
    run_all_scripts(SCRIPTS_DIR)

    return 0


if __name__ == "__main__":
    sys.exit(main())

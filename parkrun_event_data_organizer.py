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

# ------------------------- Config & Paths -------------------------

# Default root event URL
ROOT_URL_DEFAULT = "https://www.parkrun.us/farmpond/"
EVENT_NAME = "Farm Pond"

# Defaults for event range
START_EVENT_DEFAULT = 1
END_EVENT_DEFAULT   = None

# Folders
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EVENT_DIR = os.path.join(DATA_DIR, "event_results")
MASTER_PARTICIPANTS_CSV = os.path.join(DATA_DIR, "participants_master.csv")
SERIES_SUMMARY_CSV = os.path.join(DATA_DIR, "event_series_summary.csv")
AGEGROUP_DIR = os.path.join(DATA_DIR, "age_group_summaries")  # <- matches README
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

# HTTP headers
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

ID_HREF_RE = re.compile(r"/parkrunner/(\d+)(?:/|$)")
TIME_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2})\s*$")
PERCENT_RE = re.compile(r"(\d{1,3}\.\d{2})\s*%")

# logger
log = logging.getLogger("parkrun")

# Results URL template: base_url already ends WITHOUT trailing slash
RESULTS_URL_TPL = "{base}/results/{event_no}/"

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
                log.warning(f"HTTP {r.status_code} for {url}; retrying in {sleep_s:.1f}s ({i+1}/{tries})")
                time.sleep(sleep_s)
                continue
            if r.status_code != 200:
                log.warning(f"Non-200 for {url}: {r.status_code}")
                return None
        except requests.RequestException as e:
            log.error(f"Request failed for {url}: {e}")
            time.sleep(1.0)
    return None

def sanitize_base_url(raw: str) -> str:
    """
    Accepts things like:
      https://www.parkrun.us/farmpond/
      https://www.parkrun.us/farmpond
      https://www.parkrun.org.uk/holyrood/
      https://www.parkrun.org.uk/holyrood
    Returns without trailing slash, e.g. 'https://www.parkrun.us/farmpond'
    """
    if not raw:
        raise ValueError("A --base-url is required (e.g., https://www.parkrun.us/farmpond/)")
    u = raw.strip()
    # quick sanity check
    if not re.search(r"https?://(www\.)?parkrun\.(us|org\.uk)/[^/\s]+/?$", u):
        log.warning("The provided --base-url doesn't look like a standard parkrun event URL: %s", u)
    return u.rstrip("/")

def parse_time_to_seconds(t) -> Optional[int]:
    if t is None:
        return None
    try:
        import pandas as _pd
        if _pd.isna(t):
            return None
    except Exception:
        if isinstance(t, float) and math.isnan(t):
            return None
    if not isinstance(t, str):
        t = str(t)
    t = t.strip()
    if t in ("", "-", "—", "DNF", "nan", "NaN", "None"):
        return None
    m = TIME_RE.match(t)
    if not m:
        return None
    h = m.group(1)
    m_ = m.group(2)
    s = m.group(3)
    hours = int(h) if h else 0
    minutes = int(m_)
    seconds = int(s)
    return hours * 3600 + minutes * 60 + seconds

def seconds_to_time_str(sec: int) -> str:
    if sec is None:
        return ""
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"

def split_name(full: str) -> Tuple[str, str]:
    parts = full.strip().split()
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return " ".join(parts[:-1]), parts[-1]

def extract_id_from_href(href: Optional[str]) -> Optional[str]:
    if not href:
        return None
    m = ID_HREF_RE.search(href)
    return m.group(1) if m else None

def soup_text(el) -> str:
    return el.get_text(" ", strip=True) if el else ""

def clean_pct(text: str) -> Optional[float]:
    if not text:
        return None
    m = PERCENT_RE.search(text)
    if not m:
        text = text.replace("%", " %")
        m = PERCENT_RE.search(text)
    return float(m.group(1)) if m else None

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
            if not sib: break
            a2 = sib.find_all("a", href=ID_HREF_RE)
            if a2:
                log.debug(f"[event {event_no}][vol-debug]   sibling+{j} <{sib.name}> anchors={len(a2)}")
    if dump_html:
        os.makedirs("debug_html", exist_ok=True)
        p = os.path.join("debug_html", f"event_{event_no:04d}_full.html")
        with open(p, "w", encoding="utf-8") as f:
            f.write(str(soup))
        log.info(f"[event {event_no}][vol-debug] wrote full page -> {p}")

# ------------------------- Parsing -------------------------------------

def extract_volunteers_from_page(soup: BeautifulSoup) -> List[Tuple[str, str, Optional[str]]]:
    volunteers: List[Tuple[str, str, Optional[str]]] = []

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

    header_pat = re.compile(r"thanks\s*to\s*the\s*volunteers", re.I)
    header_node = soup.find(string=header_pat)
    if header_node:
        hdr = header_node.parent
        collect_from(hdr)
        sib = hdr
        for _ in range(6):
            sib = sib.find_next_sibling() if sib else None
            if not sib:
                break
            if getattr(sib, "name", "").lower() in {"h2", "h3", "h4"}:
                break
            collect_from(sib)

    gratitude_pat = re.compile(r"(grateful.*volunteer|made this event happen|thanks.*volunteer)", re.I)
    gratitude_node = soup.find(string=gratitude_pat)
    if gratitude_node:
        collect_from(gratitude_node.parent)
        collect_from(gratitude_node.parent.parent if gratitude_node.parent else None)
        sib = gratitude_node.parent
        for _ in range(6):
            sib = sib.find_next_sibling() if sib else None
            if not sib:
                break
            if getattr(sib, "name", "").lower() in {"h2", "h3", "h4"}:
                break
            collect_from(sib)

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

    if not volunteers and gratitude_node:
        text = soup_text(gratitude_node.parent)
        parts = text.split(":", 1)
        if len(parts) == 2:
            names_blob = parts[1]
            for raw in names_blob.split(","):
                name = raw.strip()
                if not name or "volunteer" in name.lower():
                    continue
                f, l = split_name(name)
                if (f or l) and (f + l).strip():
                    volunteers.append((f, l, None))

    seen = set()
    out: List[Tuple[str, str, Optional[str]]] = []
    for f, l, pid in volunteers:
        key = pid if pid else (f"{f} {l}").strip().lower()
        if key and key not in seen:
            out.append((f, l, pid))
            seen.add(key)

    log.info(f"[volunteers] found {len(out)} anchor(s) in volunteers section")
    no_pid = sum(1 for _, _, pid in out if not pid)
    if no_pid:
        log.debug(f"[volunteers] {no_pid} volunteer(s) had no parkrunner id (name-only fallback).")
    for sample in out[:5]:
        log.debug(f"[volunteers] sample: {sample}")
    return out

def parse_event_page(event_no: int, base_url: str, session: Optional[requests.Session] = None) -> Optional[EventData]:
    url = RESULTS_URL_TPL.format(base=base_url.rstrip("/"), event_no=event_no)
    html = fetch_html(url, session=session)
    if not html:
        log.warning(f"[event {event_no}] No HTML returned; skipping.")
        return None

    soup = BeautifulSoup(html, "html.parser")

    table = None
    for tbl in soup.find_all("table"):
        ths = [soup_text(th).lower() for th in tbl.find_all("th")]
        if not ths:
            continue
        if ("position" in ths and "parkrunner" in ths and "time" in ths) or \
           ("pos" in ths and "parkrunner" in ths):
            table = tbl
            break

    data = EventData(event_no=event_no, url=url)

    vol_list = extract_volunteers_from_page(soup)
    if not vol_list:
        _debug_volunteer_scan(soup, event_no, dump_html=False)
    for f, l, pid in vol_list:
        data.volunteers.append((f, l, pid))

    if table:
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 3:
                continue

            pos_txt = soup_text(tds[0])
            position = None
            try:
                position = int(pos_txt.strip())
            except Exception:
                position = None

            runner_cell = tds[1]
            a = runner_cell.find("a", href=ID_HREF_RE)
            full_name = soup_text(a) if a else soup_text(runner_cell)
            f, l = split_name(full_name)
            pid = extract_id_from_href(a.get("href") if a else None)

            row_full_text = soup_text(tr)
            is_first_timer = 1 if re.search(r"First\s*Timer", row_full_text, re.I) else 0
            is_pb = 1 if re.search(r"New\s*PB", row_full_text, re.I) else 0

            gender = soup_text(tds[2]) if len(tds) > 2 else None
            gender = gender.split()[0] if gender else None

            age_group = None
            age_grade = None
            if len(tds) > 3:
                ag_cell = soup_text(tds[3])
                m_group2 = re.search(r"\b[A-Z]{1,3}\d{1,2}-\d{1,2}\b", ag_cell)
                if m_group2:
                    age_group = m_group2.group(0)
                else:
                    age_group = ag_cell.split()[0] if ag_cell else None
                age_grade = clean_pct(ag_cell)

            club = None
            if len(tds) > 4:
                club_txt = soup_text(tds[4]).strip()
                club = club_txt if club_txt else None

            t_cell = tds[-1] if tds else None
            t_str = soup_text(t_cell) if t_cell else None
            if t_str:
                t_str = re.sub(r"(New\s*PB!?|First\s*Timer!?|PB\s*)", "", t_str, flags=re.I).strip()

            data.participants.append(EventRow(
                position=position,
                name_first=f, name_last=l, id=pid,
                gender=gender, age_group=age_group, age_grade=age_grade,
                club=club, first_time_participant=is_first_timer, first_time_volunteer=0,
                pb=is_pb,
                runner=1,
                participant=0,
                volunteer=0,
                time=t_str
            ))
    else:
        log.warning(f"[event {event_no}] No results table found.")

    volunteer_ids = set(pid for _,_,pid in data.volunteers if pid)
    volunteer_names = set((f"{f} {l}").strip().lower() for f,l,_ in data.volunteers)

    known_ids = set(r.id for r in data.participants if r.id)
    known_namekeys = set((f"{r.name_first} {r.name_last}").strip().lower() for r in data.participants)

    for r in data.participants:
        namekey = (f"{r.name_first} {r.name_last}").strip().lower()
        if (r.id and r.id in volunteer_ids) or (namekey in volunteer_names):
            r.volunteer = 1

    for f,l,pid in data.volunteers:
        namekey = (f"{f} {l}").strip().lower()
        if (pid and pid in known_ids) or (namekey in known_namekeys):
            continue
        data.participants.append(EventRow(
            position=None,
            name_first=f, name_last=l, id=pid,
            gender=None, age_group=None, age_grade=None, club=None,
            first_time_participant=0, first_time_volunteer=0, pb=0,
            runner=0,
            participant=0,
            volunteer=1,
            time=None
        ))

    for r in data.participants:
        r.participant = 1 if (r.runner == 1 or r.volunteer == 1) else 0

    log.info(f"[event {event_no}] parsed participants={sum(1 for r in data.participants if r.participant==1)} "
             f"volunteers_listed={len(data.volunteers)} rows_total={len(data.participants)}")
    return data

# ------------------------- Aggregation ---------------------------------

def load_existing_event_csvs() -> Dict[int, pd.DataFrame]:
    out = {}
    if not os.path.isdir(EVENT_DIR):
        return out
    for fn in os.listdir(EVENT_DIR):
        if not fn.lower().endswith(".csv"):
            continue
        m = re.match(r"event_(\d{4})\.csv$", fn)
        if not m:
            continue
        evno = int(m.group(1))
        try:
            df = pd.read_csv(os.path.join(EVENT_DIR, fn), dtype={"id": "string"})
            out[evno] = df
        except Exception as e:
            log.warning(f"Failed to read {fn}: {e}")
    return dict(sorted(out.items()))

def to_dataframe(event: EventData, first_time_volunteer_ids_prev: set) -> pd.DataFrame:
    rows = []
    for r in event.participants:
        key = r.id if r.id else (f"{r.name_first} {r.name_last}").strip().lower()
        r.first_time_volunteer = 1 if (r.volunteer == 1 and key not in first_time_volunteer_ids_prev) else 0
        rows.append({
            "position": r.position,
            "name_first": r.name_first,
            "name_last": r.name_last,
            "id": r.id,
            "gender": r.gender,
            "age_group": r.age_group,
            "age_grade": r.age_grade,
            "club": r.club,
            "first_time_participant": r.first_time_participant,
            "first_time_volunteer": r.first_time_volunteer,
            "pb": r.pb,
            "runner": r.runner,
            "participant": 1 if (r.runner or r.volunteer) else 0,
            "volunteer": r.volunteer,
            "time": r.time
        })
    df = pd.DataFrame(rows)
    if "id" in df.columns:
        df["id"] = df["id"].astype("string")
    return df

def save_event_df(event_no: int, df: pd.DataFrame) -> str:
    fp = os.path.join(EVENT_DIR, f"event_{event_no:04d}.csv")
    df.to_csv(fp, index=False)
    return fp

def update_master_participants(all_event_dfs: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    if not all_event_dfs:
        pd.DataFrame(columns=[
            "name_first","name_last","id","gender","age_group",
            "pb_position","pb_age_grade","pb_time",
            "num_runs","num_participations","num_volunteers","volunteer_percentage"
        ]).to_csv(MASTER_PARTICIPANTS_CSV, index=False)
        return pd.DataFrame()

    df = pd.concat(
        [d.assign(event=evno) for evno, d in all_event_dfs.items()],
        ignore_index=True
    )

    df["time_sec"] = df["time"].apply(parse_time_to_seconds)

    df["identity"] = df["id"]
    no_id_mask = df["identity"].isna() | (df["identity"] == "")
    df.loc[no_id_mask, "identity"] = (
        (df["name_first"].fillna("") + " " + df["name_last"].fillna(""))
        .str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    )

    grouped = df.groupby("identity", dropna=False)

    def most_recent(series):
        s = series.dropna()
        return s.iloc[-1] if len(s) else None

    agg = grouped.agg(
        name_first=("name_first", most_recent),
        name_last=("name_last", most_recent),
        id=("id", most_recent),
        gender=("gender", most_recent),
        age_group=("age_group", most_recent),
        pb_position=("position", lambda s: pd.to_numeric(s, errors="coerce").min()),
        pb_age_grade=("age_grade", "max"),
        pb_time_sec=("time_sec", lambda s: pd.to_numeric(s, errors="coerce").min()),
        num_runs=("runner", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
        num_volunteers=("volunteer", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
    ).reset_index(drop=True)

    num_part_series = grouped.apply(
        lambda g: int(g.loc[
            (pd.to_numeric(g["participant"], errors="coerce").fillna(0) > 0) |
            (pd.to_numeric(g["volunteer"], errors="coerce").fillna(0) > 0),
            "event"
        ].nunique())
    ).rename("num_participations").reset_index(drop=True)

    agg["num_participations"] = num_part_series

    agg["volunteer_percentage"] = agg.apply(
        lambda r: int(round((r["num_volunteers"] / r["num_participations"]) * 100))
        if r["num_participations"] else 0,
        axis=1
    )

    agg["pb_time"] = agg["pb_time_sec"].apply(lambda x: seconds_to_time_str(int(x)) if pd.notnull(x) else None)
    agg.drop(columns=["pb_time_sec"], inplace=True)

    agg = agg.sort_values(["name_last", "name_first"], na_position="last").reset_index(drop=True)

    agg.to_csv(MASTER_PARTICIPANTS_CSV, index=False)
    log.info(f"Wrote master participants -> {MASTER_PARTICIPANTS_CSV} (rows={len(agg)})")
    return agg

def build_series_summary(all_event_dfs: Dict[int, pd.DataFrame], event_name: str) -> pd.DataFrame:
    """
    Build overall series-level summary and write event_series_summary.csv

    Now includes an 'event_name' column on every row so downstream scripts
    (tables, charts) can pick up the human-readable event name.
    """
    if not all_event_dfs:
        cols = ["metric", "value", "event",
                "parkrunner_id", "name_first", "name_last",
                "gender", "event_name"]
        pd.DataFrame(columns=cols).to_csv(SERIES_SUMMARY_CSV, index=False)
        return pd.DataFrame(columns=cols)

    # ---------- Per-event aggregation ----------
    per_event = []
    for evno, df in all_event_dfs.items():
        d = df.copy()
        d["time_sec"] = d["time"].apply(parse_time_to_seconds)

        # identity (prefer id; else normalized name)
        d["identity"] = d["id"]
        no_id_mask = d["identity"].isna() | (d["identity"] == "")
        d.loc[no_id_mask, "identity"] = (
            (d["name_first"].fillna("") + " " + d["name_last"].fillna(""))
            .str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
        )

        runs = int(pd.to_numeric(d["runner"], errors="coerce").fillna(0).sum())
        vols = int(pd.to_numeric(d["volunteer"], errors="coerce").fillna(0).sum())
        participants_unique = int(
            d.loc[pd.to_numeric(d["participant"], errors="coerce").fillna(0) > 0, "identity"].nunique()
        )

        # runners only for time/age-grade stats
        d_part = d[pd.to_numeric(d["runner"], errors="coerce").fillna(0) > 0]

        def gender_filter(g):
            if "gender" not in d_part.columns:
                return d_part.iloc[0:0]
            return d_part[d_part["gender"].astype("string").str.lower() == g.lower()]

        e = {
            "event": evno,
            "runs": runs,
            "volunteers": vols,
            "participants": participants_unique,
            "min_time": d_part["time_sec"].min(skipna=True),
            "max_time": d_part["time_sec"].max(skipna=True),
            "avg_time": d_part["time_sec"].mean(skipna=True),
            "median_time": d_part["time_sec"].median(skipna=True),
        }

        grades_all = pd.to_numeric(d_part["age_grade"], errors="coerce")

        def _num_or_none(x, fn):
            x = fn(x) if hasattr(x, "__array__") else float("nan")
            return float(round(x, 2)) if pd.notna(x) else None

        e["min_age_grade"] = _num_or_none(grades_all, lambda s: s.min(skipna=True))
        e["max_age_grade"] = _num_or_none(grades_all, lambda s: s.max(skipna=True))
        e["avg_age_grade"] = _num_or_none(grades_all, lambda s: s.mean(skipna=True))
        e["median_age_grade"] = _num_or_none(grades_all, lambda s: s.median(skipna=True))

        for label, g in [("male", "Male"), ("female", "Female")]:
            gdf = gender_filter(g)
            e[f"min_time_{label}"] = gdf["time_sec"].min(skipna=True)
            e[f"max_time_{label}"] = gdf["time_sec"].max(skipna=True)
            e[f"avg_time_{label}"] = gdf["time_sec"].mean(skipna=True)
            e[f"median_time_{label}"] = gdf["time_sec"].median(skipna=True)

            e[f"runs_{label}"] = int(pd.to_numeric(gdf["runner"], errors="coerce").fillna(0).sum())

            g_grades = pd.to_numeric(gdf["age_grade"], errors="coerce")
            e[f"min_age_grade_{label}"] = _num_or_none(g_grades, lambda s: s.min(skipna=True))
            e[f"max_age_grade_{label}"] = _num_or_none(g_grades, lambda s: s.max(skipna=True))
            e[f"avg_age_grade_{label}"] = _num_or_none(g_grades, lambda s: s.mean(skipna=True))
            e[f"median_age_grade_{label}"] = _num_or_none(g_grades, lambda s: s.median(skipna=True))

        per_event.append(e)

    pe = pd.DataFrame(per_event)

    def event_of(stat, reducer):
        s = pe[stat]
        idx = reducer(s)
        ev = int(pe.loc[idx, "event"])
        val = s.loc[idx]
        return ev, val

    series_rows = []

    def add_metric(metric, value, event=None, who=None):
        row = {
            "metric": metric,
            "value": value,
            "event": event if event is not None else "",
            "parkrunner_id": who[0] if who else "",
            "name_first": who[1] if who else "",
            "name_last": who[2] if who else "",
            "gender": who[3] if who else "",
            "event_name": event_name,
        }
        series_rows.append(row)

    # extremes (single pass)
    for stat, reducer, label in [
        ("participants", pd.Series.idxmax, "max_participants"),
        ("participants", pd.Series.idxmin, "min_participants"),
        ("volunteers",  pd.Series.idxmax, "max_volunteers"),
        ("volunteers",  pd.Series.idxmin, "min_volunteers"),
        ("runs",        pd.Series.idxmax, "max_runners"),
        ("runs",        pd.Series.idxmin, "min_runners"),
    ]:
        ev, val = event_of(stat, reducer)
        add_metric(label, int(val), event=ev)

    ev, val = event_of("volunteers",  pd.Series.idxmax); add_metric("most_volunteers",      int(val), event=ev)
    ev, val = event_of("runs",        pd.Series.idxmax); add_metric("most_runs",            int(val), event=ev)
    ev, val = event_of("participants",pd.Series.idxmax); add_metric("most_participations",  int(val), event=ev)
    ev, val = event_of("volunteers",  pd.Series.idxmin); add_metric("least_volunteers",     int(val), event=ev)
    ev, val = event_of("runs",        pd.Series.idxmin); add_metric("least_runs",           int(val), event=ev)
    ev, val = event_of("participants",pd.Series.idxmin); add_metric("least_participations", int(val), event=ev)

    full = pd.concat(
        [df.assign(event=evno) for evno, df in all_event_dfs.items()],
        ignore_index=True
    )
    full["time_sec"] = full["time"].apply(parse_time_to_seconds)
    part = full[pd.to_numeric(full["runner"], errors="coerce").fillna(0) == 1].dropna(subset=["time_sec"])

    def extrema(dframe, reducer, label):
        if dframe.empty:
            return
        idx = reducer(dframe["time_sec"])
        row = dframe.loc[idx]
        who = (row.get("id", ""), row.get("name_first", ""), row.get("name_last", ""), row.get("gender", ""))
        add_metric(label, seconds_to_time_str(int(row["time_sec"])), event=int(row["event"]), who=who)

    extrema(part, pd.Series.idxmin, "min_time")
    extrema(part, pd.Series.idxmax, "max_time")

    for g, suffix in [("Male", "male"), ("Female", "female")]:
        gdf = part[part["gender"].astype("string").str.lower() == g.lower()]
        if not gdf.empty:
            extrema(gdf, pd.Series.idxmin, f"min_time_{suffix}")
            extrema(gdf, pd.Series.idxmax, f"max_time_{suffix}")

    for dframe, lab in [
        (part, ""),
        (part[part["gender"].astype("string").str.lower() == "male"], "_male"),
        (part[part["gender"].astype("string").str.lower() == "female"], "_female"),
    ]:
        if dframe.empty:
            continue
        add_metric(f"average_time{lab}", seconds_to_time_str(int(dframe["time_sec"].mean())), event="")
        add_metric(f"median_time{lab}",  seconds_to_time_str(int(dframe["time_sec"].median())), event="")

    part_grades = pd.to_numeric(part["age_grade"], errors="coerce")
    if not part_grades.empty:
        add_metric("average_age_grade", float(round(part_grades.mean(skipna=True), 2)), event="")
        add_metric("median_age_grade",  float(round(part_grades.median(skipna=True), 2)), event="")
        for g, suffix in [("Male", "male"), ("Female", "female")]:
            gdf = part[part["gender"].astype("string").str.lower() == g.lower()]
            g_grades = pd.to_numeric(gdf["age_grade"], errors="coerce")
            if not g_grades.empty:
                add_metric(f"average_age_grade_{suffix}", float(round(g_grades.mean(skipna=True), 2)), event="")
                add_metric(f"median_age_grade_{suffix}",  float(round(g_grades.median(skipna=True), 2)), event="")

    add_metric("average_num_participants", int(round(pe["participants"].mean())))
    add_metric("median_num_participants",  float(pe["participants"].median()))
    add_metric("average_num_volunteers",   int(round(pe["volunteers"].mean())))
    add_metric("median_num_volunteers",    float(pe["volunteers"].median()))
    add_metric("average_num_runners",      int(round(pe["runs"].mean())))
    add_metric("median_num_runners",       float(pe["runs"].median()))

    out = pd.DataFrame(series_rows)
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
    full["time_sec"] = full["time"].apply(parse_time_to_seconds)
    runners = full[pd.to_numeric(full["runner"], errors="coerce").fillna(0) == 1].copy()
    runners["age_grade_num"] = pd.to_numeric(runners["age_grade"], errors="coerce")
    runners["age_group_clean"] = runners["age_group"].astype("string").str.strip().replace({"": pd.NA})

    age_groups = sorted(runners["age_group_clean"].dropna().unique())

    def _slugify_age_group(s: str) -> str:
        import re
        return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()

    written = {}

    for ag in age_groups:
        g = runners[runners["age_group_clean"] == ag]
        if g.empty:
            continue

        ag_slug = _slugify_age_group(ag)
        rows = []

        def add_metric(metric, value, event=None, who=None):
            rows.append({
                "metric": metric,
                "value": value,
                "event": event if event is not None else "",
                "parkrunner_id": who[0] if who else "",
                "name_first": who[1] if who else "",
                "name_last": who[2] if who else "",
                "gender": who[3] if who else "",
            })

        t_nonnull = g.dropna(subset=["time_sec"])
        if not t_nonnull.empty:
            i_min = t_nonnull["time_sec"].idxmin()
            r_min = t_nonnull.loc[i_min]
            who_min = (r_min.get("id",""), r_min.get("name_first",""), r_min.get("name_last",""), r_min.get("gender",""))
            add_metric(f"min_time_agegroup_{ag_slug}", seconds_to_time_str(int(r_min["time_sec"])), event=int(r_min["event"]), who=who_min)

            i_max = t_nonnull["time_sec"].idxmax()
            r_max = t_nonnull.loc[i_max]
            who_max = (r_max.get("id",""), r_max.get("name_first",""), r_max.get("name_last",""), r_max.get("gender",""))
            add_metric(f"max_time_agegroup_{ag_slug}", seconds_to_time_str(int(r_max["time_sec"])), event=int(r_max["event"]), who=who_max)

        g_nonnull = g.dropna(subset=["age_grade_num"])
        if not g_nonnull.empty:
            j_min = g_nonnull["age_grade_num"].idxmin()
            rg_min = g_nonnull.loc[j_min]
            who_gmin = (rg_min.get("id",""), rg_min.get("name_first",""), rg_min.get("name_last",""), rg_min.get("gender",""))
            add_metric(f"min_age_grade_agegroup_{ag_slug}", float(round(rg_min["age_grade_num"], 2)), event=int(rg_min["event"]), who=who_gmin)

            j_max = g_nonnull["age_grade_num"].idxmax()
            rg_max = g_nonnull.loc[j_max]
            who_gmax = (rg_max.get("id",""), rg_max.get("name_first",""), rg_max.get("name_last",""), rg_max.get("gender",""))
            add_metric(f"max_age_grade_agegroup_{ag_slug}", float(round(rg_max["age_grade_num"], 2)), event=int(rg_max["event"]), who=who_gmax)

        evg = g.groupby("event", as_index=True).agg(
            avg_time=("time_sec", lambda s: pd.to_numeric(s, errors="coerce").dropna().mean()),
            med_time=("time_sec", lambda s: pd.to_numeric(s, errors="coerce").dropna().median()),
            avg_grade=("age_grade_num", lambda s: pd.to_numeric(s, errors="coerce").dropna().mean()),
            med_grade=("age_grade_num", lambda s: pd.to_numeric(s, errors="coerce").dropna().median()),
            num_runners=("runner", "sum"),
        )

        def add_event_ext(stat_col: str, metric_label: str, fmt):
            col = evg[stat_col].dropna()
            if col.empty:
                return
            ev_min = int(col.idxmin())
            ev_max = int(col.idxmax())
            v_min = col.loc[ev_min]
            v_max = col.loc[ev_max]
            add_metric(f"min_{metric_label}_agegroup_{ag_slug}", fmt(v_min), event=ev_min)
            add_metric(f"max_{metric_label}_agegroup_{ag_slug}", fmt(v_max), event=ev_max)

        add_event_ext("avg_time", "average_time", lambda x: seconds_to_time_str(int(x)))
        add_event_ext("med_time", "median_time",  lambda x: seconds_to_time_str(int(x)))
        add_event_ext("avg_grade", "average_age_grade", lambda x: float(round(x, 2)))
        add_event_ext("med_grade", "median_age_grade",  lambda x: float(round(x, 2)))

        if not evg["num_runners"].dropna().empty:
            ev_max = int(evg["num_runners"].idxmax())
            add_metric(f"most_runners_agegroup_{ag_slug}", int(evg.loc[ev_max, "num_runners"]), event=ev_max)

        df_out = pd.DataFrame(rows)
        out_path = os.path.join(AGEGROUP_DIR, f"{ag_slug}.csv")
        df_out.to_csv(out_path, index=False)
        written[ag_slug] = out_path
        log.info(f"[age-groups] wrote {ag} -> {out_path} (rows={len(df_out)})")

    return written

# ------------------------- Auto-discover event max ----------------------

def autodiscover_max_event(base_url: str, start: int = 1, ceiling: int = 2000, session: Optional[requests.Session] = None) -> int:
    lo = start
    hi = start
    while hi <= ceiling:
        url = RESULTS_URL_TPL.format(base=base_url.rstrip("/"), event_no=hi)
        html = fetch_html(url, session=session)
        if not html:
            log.info(f"[autodiscover] stop at {hi} (no html)")
            break
        soup = BeautifulSoup(html, "html.parser")
        table_exists = any("Position" in (th.get_text() if th else "") for th in soup.find_all("th"))
        has_any_result = table_exists or soup.find(string=re.compile("Thanks to the volunteers", re.I))
        log.debug(f"[autodiscover] event {hi}: table={bool(table_exists)} volunteers_block={bool(has_any_result)}")
        if not has_any_result:
            break
        hi += 1
    last = hi - 1 if hi > lo else start - 1
    log.info(f"[autodiscover] last event detected: {last}")
    return last

# ------------------------- Run post-scripts -----------------------------

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
            env["PYTHONPATH"] = os.pathsep.join([PROJECT_ROOT, env.get("PYTHONPATH", "")])
            subprocess.run([sys.executable, script], check=True, cwd=PROJECT_ROOT, env=env)
        except subprocess.CalledProcessError as e:
            log.error("Script failed: %s (exit=%s)", script, e.returncode)

# ------------------------- Main ----------------------------------------

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

    try:
        base = sanitize_base_url(args.base_url)
    except ValueError as e:
        log.error(str(e))
        sys.exit(2)

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
        end = autodiscover_max_event(base_url=base, start=seed, session=session)
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
    else:
        log.info("Refresh mode: ignoring local CSVs and re-scraping.")

    for evno in range(start, end + 1):
        if evno in all_event_dfs:
            continue
        log.info(f"[event {evno}] fetching…")
        event_data = parse_event_page(evno, base_url=base, session=session)
        if event_data is None:
            log.warning(f"[event {evno}] no data parsed; stopping fetch loop.")
            break

        df_ev = to_dataframe(event_data, first_time_volunteer_ids_prev=volunteer_history)
        for _, r in df_ev[df_ev["volunteer"] == 1].iterrows():
            key = r["id"] if pd.notnull(r["id"]) and str(r["id"]).strip() != "" \
                  else (f"{r['name_first']} {r['name_last']}").strip().lower()
            volunteer_history.add(key)

        path = save_event_df(evno, df_ev)
        all_event_dfs[evno] = df_ev
        log.info(f"[event {evno}] saved -> {path} (rows={len(df_ev)})")

    master = update_master_participants(all_event_dfs)
    summary = build_series_summary(all_event_dfs, event_name=event_name)
    _ = build_agegroup_summaries(all_event_dfs)

    log.info(f"Data build complete. participants_master rows={len(master)}; series_summary rows={len(summary)}")

    # Run post-processing / visualization scripts
    run_all_scripts(SCRIPTS_DIR)

if __name__ == "__main__":
    sys.exit(main())

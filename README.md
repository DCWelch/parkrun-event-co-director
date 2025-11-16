

# parkrun Event Co-Director

<img width="1920" height="1056" alt="course_record_best_overall_table" src="https://github.com/user-attachments/assets/f38abbf4-23ba-4054-950d-1a0ae7e470b6" />

parkrun event data organizer and summarizer:
 - Pulls data from the parkrun website for a specific parkrun
 - Saves event-specific, event summary, per-parkrunner, and age group data breakdowns in .csv files on your local machine
 - Creates some visualizations

## How to Use

If you are an event director who wants the outputs from this tool, but doesn't want to run it themselves, feel free to email farmpond@parkrun.com. I would be happy to generate all the files for you :)

If you would like to run things yourself, I tried to make it as simple as I could:

`git clone https://github.com/DCWelch/parkrun-event-co-director.git`

`cd parkrun-event-co-director`

`pip install -r requirements.txt`

`python parkrun_event_data_organizer.py`

## Directory Structure

- `parkrun_event_data_organizer/` — top-level project folder  
  - `parkrun_event_data_organizer.py` — main script 

  - `data/`
    - `event_results/` — per-event results (scraped from parkrun website)  
      - `event_0001.csv`  
      - `event_0002.csv`  
      - `...`
    - `age_group_summaries/` — per-age-group summaries for each event (generated)  
      - `event_0001_agegroup_summary.csv`  
      - `event_0002_agegroup_summary.csv`  
      - `...`
    - `participants_master.csv` — summary of all parkrunners across events (generated)  
    - `event_series_summary.csv` — overall statistics across events (generated)  

  - `visualizations/` — visualizations (generated)  
    - `course_record_progression_series.csv`  
    - `course_record_progression_times.png`  
    - `course_record_progression_agegrades.png`  
    - `event_counts_series.csv`  
    - `runners_per_event.png`  
    - `participants_per_event.png`  
    - `volunteers_per_event.png`  

  - `scripts/` — helper scripts  
    - `generate_course_record_progression.py`  
    - `generate_event_counts.py`  
    - `(other analysis scripts)`  

  - `assets/` — various assets  
    - `parkrun_logo_white.png`  

  - `README.md`

## Other Helpful Resources

The following links send you to other helpful tools for parkrun Event Directors, Volunteer Coordinators and/or Social Media Managers:
 - https://chromewebstore.google.com/detail/parkrun-event-summary/nfdbgfodockojbhmenjohphggbokgmaf
 - https://github.com/leoz0214/Parkrun-Data-Scraper

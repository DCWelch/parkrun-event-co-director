
# parkrun Event Co-Director

parkrun event data organizer and summarizer:
 - Pulls data from the parkrun website for a specific parkrun
 - Saves event-specific, event summary, per-parkrunner, and age group data breakdowns in .csv files on your local machine
 - Creates some visualizations

<img width="1920" height="1056" alt="course_record_best_overall_table" src="https://github.com/user-attachments/assets/f38abbf4-23ba-4054-950d-1a0ae7e470b6" />

## Running the Tool

If you're a parkrun Event Director who wants the outputs from this tool, but doesn't want to run it themselves, feel free to email farmpond@parkrun.com. I would be happy to generate everything and send it to you :)

If you would like to run the tool locally:

### Requirements

Python 3.8+

### Setup

`git clone https://github.com/DCWelch/parkrun-event-co-director.git`

`cd parkrun-event-co-director`

`pip install -r requirements.txt`

### Config

Modify the "EVENT DEFAULTS" in parkrun_config.py to match the parkrun you want to analyze, namely:
 - ROOT_URL_DEFAULT (e.g. "https://www.parkrun.us/farmpond/")
 - EVENT_NAME_DEFAULT (e.g. "Farm Pond"... Exclude "parkrun", that word is added automatically)
 - Others are less critical, but can modify the results to suit your specific needs

### Run

`python parkrun_event_data_organizer.py`

## Outputs

- `parkrun_event_data_organizer/` — top-level project folder

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

## Other Helpful Resources

The following links send you to other helpful tools for parkrun Event Directors, Volunteer Coordinators and/or Social Media Managers:
 - https://github.com/AlanLyttonJones/Age-Grade-Tables
 - https://chromewebstore.google.com/detail/parkrun-event-summary/nfdbgfodockojbhmenjohphggbokgmaf
 - https://github.com/leoz0214/Parkrun-Data-Scraper
 - https://github.com/FingerLakesRunnersClub/AgeGradeCalculator

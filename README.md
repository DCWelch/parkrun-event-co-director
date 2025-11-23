
# parkrun Event Co-Director

parkrun event data organizer and summarizer:
 - Pulls data from the parkrun website
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
After running `parkrun_event_data_organizer.py`, the tool produces the following structure:
### data/
- event_series_summary.csv
- parkruns_master.csv
- participants_master.csv
- volunteers_master.csv
#### data/event_results/ — Per-event scraped results
- event_0001.csv
- event_0002.csv
- …
#### data/age_group_summaries/ — Per-age group summaries
- sm20_24.csv
- vw40_44.csv
- …
### visualizations/
- agegroup_course_record_best_times.csv
- agegroup_course_record_best_times_table.png
- course_record_best_agegrades_table.png
- course_record_best_overall_table.png
- course_record_best_times_table.png
- course_record_progression_agegrades.png
- course_record_progression_series.csv
- course_record_progression_times.png
- event_counts_series.csv
- participants_per_event.png
- runners_per_event.png
- volunteers_per_event.png
#### visualizations/leaderboards/ — Various Top-N Leaderboards
- top_10_agegrades.png
- top_10_volunteers.png
- …
#### visualizations/agegroup_course_records/ — Age Group Course Record Progressions
- agegroup_jm10_course_record_progression_times.csv
- agegroup_jm10_course_record_progression_times.png
- agegroup_sw30_34_course_record_progression_times.csv
- agegroup_sw30_34_course_record_progression_times.png
- …

## Other Helpful Resources

The following links send you to other helpful tools for parkrun Event Directors, Volunteer Coordinators and/or Social Media Managers:
 - https://github.com/AlanLyttonJones/Age-Grade-Tables (Source for Age Grading)
 - https://chromewebstore.google.com/detail/parkrun-event-summary/nfdbgfodockojbhmenjohphggbokgmaf (parkrun event summaries Google Chrome Extension)
 - https://github.com/leoz0214/Parkrun-Data-Scraper (Another parkrun data tool similar to this one)
 - https://github.com/FingerLakesRunnersClub/AgeGradeCalculator (.NET Age Grading Tool)
 - http://howardgrubb.co.uk/athletics/mldrroad25.html (2025 Age Grading Tool)
 - http://www.howardgrubb.co.uk/athletics/wmaroad15.html (2010 Age Grading Tool... Use the 2010 Factors for parkrun Age Grading)

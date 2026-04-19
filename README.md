# IPL Batsman Comparison Dashboard

A public Streamlit dashboard for comparing two IPL batters across current-season batting phases, model-based Runs Above Expected views, and model-free strike-rate context views. https://ipl-dashboard-comparative-ku4yz8ukhhqkhdsbtb66re.streamlit.app/

The app is designed for quick cricket analysis for the IPL 2026 season. It reads prepared dashboard CSV files from `data/current/` and renders the comparison UI in `app.py`. This repository is dashboard-only; data refreshes are prepared outside this public repo and published here as CSV files.

## What The Dashboard Shows

- Side-by-side season snapshots for any two batters.
- Own 15-ball phase comparisons through `76-90` balls.
- Runs Above Expected and Runs Above Expected rate views.
- Strike-rate context views:
  - SR points vs match
  - SR points vs teammates
  - SR points vs league phase
  - difficulty-adjusted strike rate
- Powerplay and death-over comparison charts.
- Worm charts showing cumulative runs gained or lost versus league phase rate.
- Shot and control views:
  - Control % by own 15-ball phase
  - Most Productive Shots
  - Line-Length Matchup Grid
- Match-level phase detail table with CSV download.
- Built-in Light and Dark dashboard themes. The app opens in Dark mode by default.

## Important Metric Notes

### Runs Above Expected

Runs Above Expected compares what a batter actually scored with what the model expected an average IPL batter to score from the same batting situations.

```text
Runs Above Expected = actual batter runs - expected runs
```

Positive values mean the batter scored more than expected for the situations faced. Negative values mean the batter scored less than expected.

### Own 15-Ball Phases

Own phases are based on the batter's personal balls faced in an innings, not team overs.

Examples:

- `1-15` means the batter's first 15 legal balls in each innings.
- `16-30` means the batter's 16th to 30th legal balls in each innings.
- Phase totals are summed across innings, so 49 balls in `1-15` means 49 total balls across all innings where the batter was within balls 1-15 of that innings.

### Model-Free Metrics

The strike-rate context views do not use model output. They compare batting strike rate against match, teammate, or league phase baselines.

The worm chart is also model-free. It shows cumulative runs gained or lost versus the league phase rate.

The shot and control views are descriptive and model-free. Control %, shot-type rates, boundary %, dot %, and line-length matchup rates are calculated directly from the prepared current-season ball-by-ball data.

## Data Files

The app reads dashboard-ready CSVs from:

```text
data/current/
```

Expected files include:

- `player_season.csv`
- `player_match.csv`
- `phase_15_ball.csv`
- `context_15_ball.csv`
- `control_profile.csv`
- `shot_profile.csv`
- `line_length_matchups.csv`
- `powerplay_death_summary.csv`
- `powerplay_death_worms.csv`
- `last_updated.json`

`last_updated.json` is used to show the latest match covered by the dashboard data.

## What Is Not In This Repository

This repository is intentionally dashboard-only.

It does not include:

- training code
- model artifacts
- raw data processing scripts
- the source analytics/model project

The Streamlit app does not train, load, or modify the xR model. It only reads prepared public dashboard CSVs.

## Running Locally

Install the dashboard dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

The app expects the `data/current/` CSV files to already exist. It does not rebuild the source data locally.

## Deployment

This repo is suitable for Streamlit Community Cloud.

Recommended settings:

- App file: `app.py`
- Python dependencies: `requirements.txt`
- Data source: committed CSV files in `data/current/`

When new prepared dashboard CSVs are pushed to this repository, Streamlit Cloud should pick up the new commit and refresh/redeploy the public app. The data build and model scoring pipeline live outside this public repository.

## Current Season

The dashboard is currently built for the IPL 2026 season.

## Repository Structure

```text
.
|-- app.py
|-- requirements.txt
|-- data/
|   `-- current/
|       |-- context_15_ball.csv
|       |-- control_profile.csv
|       |-- last_updated.json
|       |-- line_length_matchups.csv
|       |-- phase_15_ball.csv
|       |-- player_match.csv
|       |-- player_season.csv
|       |-- powerplay_death_summary.csv
|       |-- powerplay_death_worms.csv
|       `-- shot_profile.csv
`-- README.md
```

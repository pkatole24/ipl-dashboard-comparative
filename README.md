# IPL Batsman Comparison Dashboard

A public Streamlit dashboard for comparing two IPL batters across current-season batting phases, model-based Runs Above Expected views, and model-free strike-rate context views.

The app is designed for quick cricket analysis rather than model development. It reads prepared dashboard CSV files from `data/current/` and renders the comparison UI in `app.py`.

## What The Dashboard Shows

- Side-by-side season snapshots for any two batters.
- Own 15-ball phase comparisons through `76-90` balls.
- Runs Above Expected and Runs Above Expected rate views.
- Model-free strike-rate context views:
  - SR points vs match
  - SR points vs teammates
  - SR points vs league phase
  - difficulty-adjusted strike rate
- Powerplay and death-over comparison charts.
- Worm charts showing cumulative runs gained or lost versus league phase rate.
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
- `powerplay_death_summary.csv`
- `powerplay_death_worms.csv`
- `last_updated.json`

`last_updated.json` is used to show the latest match covered by the dashboard data.

## What Is Not In This Repository

This repository is intentionally dashboard-only.

It does not include:

- training code
- xR model artifacts
- private notebooks
- raw Cricsheet ZIP processing scripts
- model dependencies such as TensorFlow
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

The app expects the `data/current/` CSV files to already exist.

## Deployment

This repo is suitable for Streamlit Community Cloud.

Recommended settings:

- App file: `app.py`
- Python dependencies: `requirements.txt`
- Data source: committed CSV files in `data/current/`

When new dashboard CSVs are pushed to this repository, Streamlit Cloud should pick up the new commit and refresh/redeploy the public app.

## Data Refresh Workflow

The public dashboard repo should only receive prepared dashboard files.

The intended refresh flow is:

1. A private/local analytics pipeline downloads the latest IPL ball-by-ball data.
2. The private pipeline scores and aggregates the current season.
3. Only the dashboard-safe outputs are copied into this repo:
   - `app.py` when UI changes are made
   - `requirements.txt` when dashboard dependencies change
   - `data/current/*.csv`
   - `data/current/last_updated.json`
4. The public repo is committed and pushed.
5. Streamlit Cloud refreshes the public app from the new commit.

This keeps the public dashboard lightweight while keeping model code and model artifacts separate.

## Current Season

The dashboard is currently built for the IPL 2026 season.

## Repository Structure

```text
.
├── app.py
├── requirements.txt
├── data/
│   └── current/
│       ├── context_15_ball.csv
│       ├── last_updated.json
│       ├── phase_15_ball.csv
│       ├── player_match.csv
│       ├── player_season.csv
│       ├── powerplay_death_summary.csv
│       └── powerplay_death_worms.csv
└── README.md
```

## Notes For Future Changes

- Keep this repository dashboard-only.
- Do not add model artifacts or training dependencies here.
- Keep model-based metrics labelled as Runs Above Expected in user-facing text.
- Keep model-free strike-rate context metrics clearly described as model-free.
- If the data schema changes, update the README and app explanations together.

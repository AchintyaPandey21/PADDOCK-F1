# Paddock F1

Paddock F1 is a modern Streamlit dashboard for exploring Formula 1 session data with FastF1, Pandas, and Plotly. It is designed as a portfolio-quality analytics project with a broadcast-inspired dark UI, modular architecture, and interactive workflows for pace analysis, tyre strategy, and telemetry comparison.

## Features

- Dynamic session loading by season, Grand Prix, and session type
- Driver analytics for fastest laps, average pace, sector times, compounds, stint lengths, and consistency
- Interactive Plotly visuals for pace evolution, fastest laps, position changes, heatmaps, sector comparisons, and stint strategy
- Telemetry comparison between two drivers with speed, throttle, brake, gear, RPM, and racing line overlays
- FastF1 cache support with automatic `cache/` directory creation
- CSV downloads for lap data, leaderboards, strategy tables, and telemetry exports
- ML-powered what-if strategy simulator with tyre plans, pit windows, projected race time, and counterfactual position
- Responsive dark theme inspired by Formula 1 broadcast graphics

## Stack

- Python
- Streamlit
- FastF1
- Pandas
- Plotly
- Matplotlib

## Project Structure

```text
paddock-f1/
├── app.py
├── assets/
├── cache/
├── data/
├── pages/
├── utils/
├── visualizations/
├── .streamlit/
├── README.md
└── requirements.txt
```

## Getting Started

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure PostgreSQL in `.env` by copying `.env.example`. The `.env` file is ignored by Git; use the real password configured for your local PostgreSQL role.

```dotenv
DB_HOST=localhost
DB_PORT=5432
DB_NAME=f1_analytics
DB_USER=postgres
DB_PASSWORD=your_local_postgres_password
```

4. Initialize the schema:

```bash
python init_db.py
```

The initialization script creates `DB_NAME` automatically when it does not exist, then creates the required tables without deleting existing data.

5. Run the dashboard:

```bash
streamlit run app.py
```

6. Open the local Streamlit URL shown in the terminal, usually `http://localhost:8501`.

## How It Works

- `app.py` loads session controls, builds the shared FastF1 context, and routes page views with Streamlit navigation.
- `utils/` contains session loading, cache setup, styling helpers, constants, and analytics transforms.
- `visualizations/` contains reusable Plotly chart builders.
- `pages/` contains the dashboard views for Race Centre, Driver Analytics, Tyre Strategy, the What-if Strategy Simulator, and Telemetry Lab.
- `ml/` contains the PostgreSQL training query, Scikit-learn pipeline, persisted model artifact, and counterfactual simulation engine.

## Data Notes

- The first load for a new season or session can be slow because FastF1 needs to fetch and cache data.
- Live data access requires internet connectivity the first time a session is requested.
- Once a session is cached, Paddock F1 can reuse that local cache on later runs.
- Session data is stored in PostgreSQL after its first FastF1 load, allowing later requests to use the local analytics store.

## Suggested Demo Flow

For a strong walkthrough, open a recent completed race session and explore:

1. `Race Centre` for session context and pace leaderboard.
2. `Driver Analytics` for detailed lap and consistency analysis.
3. `Tyre Strategy` for compounds, stint timelines, and pit-window storytelling.
4. `Telemetry Lab` for fastest-lap overlays between two drivers.
5. `Strategy Simulator` to select a driver, define compounds and pit windows, then compare the projected plan against the recorded race.

## Strategy Simulator

The simulator trains on clean green-flag race laps stored in PostgreSQL. It predicts lap time from the circuit reference pace, driver, team, compound, tyre age, lap number, and weather fields, then adds a circuit-specific pit-loss estimate and compares the resulting delta to the recorded competitor timing order.

Retrain after importing more historical races:

```bash
python -m ml.train
```

The simulator is intentionally a decision-support tool, not a race oracle. It does not model safety cars, traffic, changing weather within a stint, tyre allocation rules, or rival teams responding to the counterfactual.

## Future Enhancements

- Weather deep dives
- DRS effectiveness analysis
- Pit stop prediction and undercut modeling
- Exportable chart images
- Race simulation and custom ranking models

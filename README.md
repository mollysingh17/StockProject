\# Stock ML Trading Backtest Engine



This is a Python project that ingests historical stock prices, stores them locally in SQLite, computes features/signals, and evaluates trading strategies with a reproducible backtest pipeline. It includes SMA baselines and an ML based in/out policy trained on price derived features and evaluated using a strict time-based train/test split.



\## Project layout

\- `backend/` — core engine (DB, ingestion, signals/features, backtest)

\- `experiments/` — runnable experiments (SMA baselines, ML policy)

\- `scripts/` — one-off utilities



\## Setup

Create/activate a virtual environment, then install dependencies:



```bash

pip install -r requirements.txt






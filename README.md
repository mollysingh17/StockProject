Python backtesting engine for equity trading strategies. It downloads daily prices for the S&P 500, generates trading signals, and simulates them with next-day execution and transaction costs. Results are reported as CAGR, Sharpe ratio, and max drawdown, compared against buy-and-hold.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Data is stored in a local SQLite file (stocks.db) unless DATABASE_URL is set.

## Usage

Run from the project root.

```
python -m backend.universe
python -m backend.main
python -m scripts.inspectdb
python -m experiments.sma_train_test
python -m experiments.ml_baseline
pytest
```

Universe fetches the current S&P 500 ticker list. Main downloads 10 years of daily prices for those tickers and SPY. Inspectdb shows row counts and date ranges. The experiments run single-stock SMA and ML strategies.

## Method

Signals computed on day t are traded on day t+1. Parameters are tuned on earlier data and evaluated once on later held-out data. Transaction costs are charged whenever the position changes.

## Limitations

The ticker list is today's S&P 500 applied to the past, so backtests include companies that joined the index because they did well and leave out companies that were removed. Historical results are optimistic because of this.

Prices come from yfinance, an unofficial Yahoo Finance API.

Python backtesting engine for equity trading strategies. It downloads daily prices for the S&P 500, generates trading signals, and simulates them with next-day execution and transaction costs. Results are reported as CAGR, Sharpe ratio, and max drawdown, compared against buy-and-hold.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Start the Postgres database (requires Docker Desktop):

```
docker compose up -d
```

The code connects to the local database by default. Set DATABASE_URL to use a different one.

## Usage

Run from the project root.

```
python -m backend.data.tickers
python -m backend.data.membership
python -m backend.data.update
python -m backend.strategy.portfolio
pytest
```

tickers fetches the current S&P 500 list. membership loads historical index membership. update downloads daily prices, only fetching what is missing. portfolio runs the monthly momentum backtest against equal-weight and SPY benchmarks.

The original single-stock experiments are in experiments/single_stock.
## Method

Signals computed on day t are traded on day t+1. Parameters are tuned on earlier data and evaluated once on later held-out data. Transaction costs are charged whenever the position changes.

## Findings

Each month, the strategy buys the 20 S&P 500 stocks with the highest 12-month return (skipping the most recent month). Using today's index list for the whole history, it returned 36.5% per year from 2017 to 2026. Using each month's actual members, it returned 12.1%, below both an equal-weighted portfolio of all members (12.4%) and SPY (14.1%). Almost all of the apparent edge came from survivorship bias.

## Limitations

Backtests use point-in-time S&P 500 membership (fja05680/sp500), so each month only ranks stocks that were members at the time. Renamed companies are mapped to their current tickers. Prices for acquired and bankrupt companies are not available from yfinance, so those stocks cannot be picked in the backtest.

Prices come from yfinance, an unofficial Yahoo Finance API.

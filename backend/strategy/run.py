import json
import sys
from datetime import datetime, timezone

import pandas as pd

from backend.data.membership import current_tickers, load_membership_matrix
from backend.db import engine
from backend.strategy.metrics import summarize
from backend.strategy.portfolio import month_end_dates, portfolio_returns, top_n_weights
from backend.strategy.prices import load_close_matrix
from backend.strategy.scores import momentum_score

COST_BPS = 10.0
BACKTEST_START = "2017-12-01"

STRATEGIES = [
    {"strategy": "momentum", "n": 20},
    {"strategy": "momentum", "n": 50},
    {"strategy": "momentum", "n": 167},
    {"strategy": "equal_weight", "n": None},
]


def run_all(universe: str) -> pd.DataFrame:
    closes = load_close_matrix()
    stocks = closes.drop(columns=["SPY"])
    rebalance_dates = month_end_dates(stocks.index)

    scores = momentum_score(stocks)
    if universe == "point_in_time":
        scores = scores.where(load_membership_matrix(stocks.index, stocks.columns))
    else:
        scores = scores[[t for t in current_tickers() if t in scores.columns]]

    weights = {}
    for params in STRATEGIES:
        n = params["n"] or len(scores.columns)
        weights[json.dumps(params)] = top_n_weights(scores, rebalance_dates, n=n)

    start = max(
        max(w.index[w.sum(axis=1) > 0][0] for w in weights.values()),
        pd.Timestamp(BACKTEST_START),
    )
    returns = {
        key: portfolio_returns(stocks.reindex(columns=w.columns), w, cost_bps=COST_BPS).loc[start:].iloc[1:]
        for key, w in weights.items()
    }
    returns[json.dumps({"strategy": "spy", "n": None})] = closes["SPY"].pct_change().loc[start:].iloc[1:]

    run_at = datetime.now(timezone.utc)
    rows = []
    for key, r in returns.items():
        params = json.loads(key)
        rows.append({
            "run_at": run_at,
            "strategy": params["strategy"],
            "params": json.dumps({"n": params["n"], "cost_bps": COST_BPS}),
            "universe": universe,
            "start_date": r.index[0].date(),
            "end_date": r.index[-1].date(),
            **summarize(r),
        })
    return pd.DataFrame(rows)


def save_runs(results: pd.DataFrame) -> None:
    results.to_sql("backtest_runs", engine, if_exists="append", index=False)


if __name__ == "__main__":
    universe = "today" if "--today" in sys.argv else "point_in_time"
    results = run_all(universe)
    save_runs(results)
    print(f"Saved {len(results)} runs ({universe}) from {results['start_date'].iloc[0]} to {results['end_date'].iloc[0]}\n")
    print(results[["strategy", "params", "final", "cagr", "sharpe", "max_dd"]].round(3).to_string(index=False))
import pandas as pd


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = pd.Series(index, index=index)
    return pd.DatetimeIndex(dates.groupby(index.to_period("M")).max().values)


def top_n_weights(scores: pd.DataFrame, rebalance_dates: pd.DatetimeIndex, n: int = 20) -> pd.DataFrame:
    rows = {}
    for date in rebalance_dates:
        top = scores.loc[date].dropna().nlargest(n).index
        if len(top) > 0:
            rows[date] = pd.Series(1.0 / len(top), index=top)
    targets = pd.DataFrame(rows).T.reindex(columns=scores.columns).fillna(0.0)
    return targets.reindex(scores.index).ffill().fillna(0.0)


def portfolio_returns(closes: pd.DataFrame, weights: pd.DataFrame, cost_bps: float = 10.0) -> pd.Series:
    returns = closes.pct_change(fill_method=None).fillna(0.0)
    held = weights.shift(1).fillna(0.0)
    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()
    return (held * returns).sum(axis=1) - turnover * cost_bps / 10000


if __name__ == "__main__":
    from backend.data.membership import current_tickers, load_membership_matrix
    from backend.strategy.metrics import summarize
    from backend.strategy.prices import load_close_matrix
    from backend.strategy.scores import momentum_score

    closes = load_close_matrix()
    stocks = closes.drop(columns=["SPY"])
    rebalance_dates = month_end_dates(stocks.index)

    scores = momentum_score(stocks)
    today_list = scores[[t for t in current_tickers() if t in scores.columns]]
    point_in_time = scores.where(load_membership_matrix(stocks.index, stocks.columns))

    weights = {
        "Momentum top 20 (today's list)": top_n_weights(today_list, rebalance_dates, n=20),
        "Momentum top 20 (point-in-time)": top_n_weights(point_in_time, rebalance_dates, n=20),
        "Equal weight (today's list)": top_n_weights(today_list, rebalance_dates, n=len(stocks.columns)),
        "Equal weight (point-in-time)": top_n_weights(point_in_time, rebalance_dates, n=len(stocks.columns)),
    }

    start = max(w.index[w.sum(axis=1) > 0][0] for w in weights.values())
    results = {
        name: portfolio_returns(stocks.reindex(columns=w.columns), w).loc[start:].iloc[1:]
        for name, w in weights.items()
    }
    results["SPY"] = closes["SPY"].pct_change().loc[start:].iloc[1:]

    table = pd.DataFrame({name: summarize(r) for name, r in results.items()}).T
    print(f"Backtest from {start.date()} to {stocks.index[-1].date()}\n")
    print(table.round(3).to_string())
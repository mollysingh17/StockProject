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
    from backend.metrics import summarize
    from backend.prices import load_close_matrix
    from backend.scores import momentum_score

    closes = load_close_matrix()
    stocks = closes.drop(columns=["SPY"])
    scores = momentum_score(stocks)
    rebalance_dates = month_end_dates(stocks.index)
    momentum = top_n_weights(scores, rebalance_dates, n=20)
    equal = top_n_weights(scores, rebalance_dates, n=len(stocks.columns))

    start = momentum.index[momentum.sum(axis=1) > 0][0]
    results = {
        "Momentum top 20": portfolio_returns(stocks, momentum).loc[start:].iloc[1:],
        "Equal weight all": portfolio_returns(stocks, equal).loc[start:].iloc[1:],
        "SPY": closes["SPY"].pct_change().loc[start:].iloc[1:],
    }
    table = pd.DataFrame({name: summarize(r) for name, r in results.items()}).T
    print(f"Backtest from {start.date()} to {stocks.index[-1].date()}\n")
    print(table.round(3).to_string())
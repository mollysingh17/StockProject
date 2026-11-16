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



import pandas as pd

from backend.strategy.scores import momentum_score, value_score

FEATURES = ["mom_12_1", "mom_6_1", "mom_1", "value_5y", "vol_3m", "dist_200"]


def raw_features(closes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    returns = closes.pct_change(fill_method=None)
    return {
        "mom_12_1": momentum_score(closes),
        "mom_6_1": closes.shift(21) / closes.shift(126) - 1,
        "mom_1": closes / closes.shift(21) - 1,
        "value_5y": value_score(closes),
        "vol_3m": returns.rolling(63).std(),
        "dist_200": closes / closes.rolling(200).mean() - 1,
    }


def build_dataset(closes: pd.DataFrame, eligible: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    members = eligible.loc[dates]
    monthly = closes.loc[dates]

    columns = {}
    for name, frame in raw_features(closes).items():
        columns[name] = frame.loc[dates].where(members).rank(axis=1, pct=True).stack()
    columns["fwd_ret"] = (monthly.shift(-1) / monthly - 1).where(members).stack()

    data = pd.concat(columns, axis=1)
    data.index.names = ["date", "ticker"]
    data = data.dropna(subset=FEATURES)

    median = data.groupby(level="date")["fwd_ret"].transform("median")
    data["label"] = (data["fwd_ret"] > median).astype(float).where(data["fwd_ret"].notna())
    return data


if __name__ == "__main__":
    from backend.data.membership import load_membership_matrix
    from backend.strategy.portfolio import month_end_dates
    from backend.strategy.prices import load_close_matrix

    closes = load_close_matrix().drop(columns=["SPY"])
    dates = month_end_dates(closes.index)
    eligible = load_membership_matrix(closes.index, closes.columns)
    data = build_dataset(closes, eligible, dates)

    per_month = data.groupby(level="date").size()
    print(f"{len(data)} rows: {per_month.size} months from {per_month.index[0].date()} to {per_month.index[-1].date()}")
    print(f"Stocks per month: min {per_month.min()}, median {int(per_month.median())}, max {per_month.max()}")
    print(f"Share of labels = 1: {data['label'].mean():.3f}\n")
    print(data.tail(5).round(3).to_string())
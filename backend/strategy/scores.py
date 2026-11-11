import pandas as pd


def momentum_score(closes: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    return closes.shift(skip) / closes.shift(lookback) - 1


if __name__ == "__main__":
    from backend.strategy.prices import load_close_matrix

    closes = load_close_matrix()
    scores = momentum_score(closes.drop(columns=["SPY"]))
    latest = scores.iloc[-1].dropna().sort_values(ascending=False)
    print(f"Scored {len(latest)} stocks on {scores.index[-1].date()}")
    print(latest.head(20).round(3).to_string())
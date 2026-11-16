import numpy as np
import pandas as pd


def momentum_score(closes: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    return closes.shift(skip) / closes.shift(lookback) - 1


def value_score(closes: pd.DataFrame, years: int = 5, window_years: int = 1) -> pd.DataFrame:
    days = 252
    past_avg = closes.rolling(window_years * days).mean().shift(int((years - window_years / 2) * days))
    return np.log(past_avg / closes)


def combined_score(*scores: pd.DataFrame) -> pd.DataFrame:
    valid = scores[0].notna()
    for s in scores[1:]:
        valid &= s.notna()
    ranks = [s.where(valid).rank(axis=1, pct=True) for s in scores]
    return sum(ranks) / len(ranks)


if __name__ == "__main__":
    from backend.data.membership import current_tickers
    from backend.strategy.prices import load_close_matrix

    closes = load_close_matrix()
    stocks = closes[[t for t in current_tickers() if t in closes.columns]]
    latest = value_score(stocks).iloc[-1].dropna().sort_values(ascending=False)
    print(f"Value scores for {len(latest)} current members on {stocks.index[-1].date()}\n")
    print("Cheapest (fell most over 5 years):")
    print(latest.head(10).round(2).to_string())
    print("\nMost expensive (rose most):")
    print(latest.tail(10).round(2).to_string())
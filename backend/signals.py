import pandas as pd
from sqlalchemy import text

from backend.db import engine


def load_price_history(ticker: str = "AAPL") -> pd.DataFrame:
   
    query = text("SELECT date, close FROM price_history WHERE ticker = :ticker ORDER BY date")
    df = pd.read_sql_query(query, engine, params={"ticker": ticker})
    df = df.rename(columns={"close": "Close"})

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["Close"])
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df.sort_values("date").reset_index(drop=True)


def add_sma(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    df[f"sma_{window}"] = df["Close"].rolling(window=window).mean()
    return df


def add_sma_signal(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = add_sma(df, window=window)
    df["signal"] = (df["Close"] > df[f"sma_{window}"]).astype(int)
    return df


def add_sma_crossover_signal(
    df: pd.DataFrame, short_window: int = 20, long_window: int = 50
) -> pd.DataFrame:
    df = df.copy()
    df[f"sma_{short_window}"] = df["Close"].rolling(short_window).mean()
    df[f"sma_{long_window}"] = df["Close"].rolling(long_window).mean()
    # 1 when short SMA is above long SMA (uptrend). During the warm-up period
    # the SMAs are NaN, NaN > NaN is False, so the signal is 0 (out).
    df["signal"] = (df[f"sma_{short_window}"] > df[f"sma_{long_window}"]).astype(int)
    return df


def add_sma_crossover_signal_confirmed(
    df: pd.DataFrame, short_window: int, long_window: int, confirm_days: int = 3
) -> pd.DataFrame:
    """Crossover signal that must hold for `confirm_days` days in a row.

    BUG FIX: the old version did raw.rolling(0) when confirm_days=0. An empty
    window sums to 0 == confirm_days, so the signal was 1 on EVERY day,
    ignoring the SMAs completely. confirm_days <= 1 now means "no confirmation".
    """
    df = add_sma_crossover_signal(df, short_window, long_window)
    if confirm_days <= 1:
        return df

    df["signal"] = df["signal"].rolling(confirm_days).min().fillna(0).astype(int)
    return df

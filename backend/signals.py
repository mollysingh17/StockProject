import pandas as pd
from .db import engine


def load_price_history(ticker: str = "AAPL") -> pd.DataFrame:
    query = f"""
        SELECT date, Close
        FROM price_history
        WHERE ticker = '{ticker}'
        ORDER BY date;
    """
    df = pd.read_sql(query, engine)
    return df


def add_sma(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    df[f"sma_{window}"] = df["Close"].rolling(window=window).mean()
    return df

def add_sma_signal(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = add_sma(df, window=window).copy()
    col = f"sma_{window}"
    df["signal"] = (df["Close"] > df[col]).astype(int)
    return df

def add_sma_crossover_signal(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50
) -> pd.DataFrame:
    df = df.copy()
    # short-term SMA
    df[f"sma_{short_window}"] = df["Close"].rolling(window=short_window).mean()
    # long-term SMA
    df[f"sma_{long_window}"] = df["Close"].rolling(window=long_window).mean()

    # Signal: 1 when short SMA is above long SMA (uptrend), else 0
    df["signal"] = (df[f"sma_{short_window}"] > df[f"sma_{long_window}"]).astype(int)
    return df

def add_sma_crossover_signal_confirmed(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
    confirm_days: int = 3,
):
    df = df.copy()

    df[f"sma_{short_window}"] = df["Close"].rolling(short_window).mean()
    df[f"sma_{long_window}"] = df["Close"].rolling(long_window).mean()

    # raw crossover signal: 1 if bullish, 0 if not
    raw = (df[f"sma_{short_window}"] > df[f"sma_{long_window}"]).astype(int)

    # require confirm_days of bullish condition before entering
    df["signal"] = (
        raw.rolling(confirm_days)
           .apply(lambda x: 1 if x.sum() == confirm_days else 0, raw=True)
           .fillna(0)
           .astype(int)
    )

    return df

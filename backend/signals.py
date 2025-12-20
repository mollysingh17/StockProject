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
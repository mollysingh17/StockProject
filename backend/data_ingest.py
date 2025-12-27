import yfinance as yf
import pandas as pd
from sqlalchemy import text

from .db import engine 

def fetch_price_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.reset_index(inplace=True)      
    df.rename(columns={"Date": "date"}, inplace=True)
    df["ticker"] = ticker
    return df

def store_price_history(df: pd.DataFrame, table_name: str = "price_history") -> None:
    """Append price rows while preventing duplicate (ticker, date) records.

    DB-agnostic approach: delete any existing rows for the same ticker in the incoming
    date range, then append the new rows.
    """
    if df is None or df.empty:
        return

    df = df.copy()

    if "ticker" not in df.columns:
        raise ValueError("store_price_history expects a 'ticker' column in df")
    if "date" not in df.columns:
        raise ValueError("store_price_history expects a 'date' column in df")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    # Drop duplicates inside the incoming batch
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")

    # (Your fetch function returns one ticker at a time, so this is safe)
    ticker = str(df["ticker"].iloc[0])
    start = df["date"].min()
    end = df["date"].max()

    delete_sql = text(f"""
        DELETE FROM {table_name}
        WHERE ticker = :ticker AND date >= :start_date AND date <= :end_date
    """)

    with engine.begin() as conn:
        conn.execute(delete_sql, {"ticker": ticker, "start_date": start, "end_date": end})

    df.to_sql(table_name, engine, if_exists="append", index=False)
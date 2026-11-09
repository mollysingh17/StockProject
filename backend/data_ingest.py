import pandas as pd
from sqlalchemy import Date, bindparam, inspect, text

from backend.db import engine

PRICE_COLUMNS = ["date", "open", "high", "low", "close", "volume", "ticker"]


def fetch_price_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    import yfinance as yf 

    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}")

  
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()
    
    df.columns = [str(c).lower() for c in df.columns]
    df["ticker"] = ticker
    return df[[c for c in PRICE_COLUMNS if c in df.columns]]


def store_price_history(df: pd.DataFrame, table_name: str = "price_history") -> None:
    """Idempotent write: replace any existing rows for this ticker and date range.

    The delete and the insert run in ONE transaction, so a crash mid-write can't
    leave the ticker with its old rows deleted and no new rows.
    """
    if df is None or df.empty:
        return

    df = df.copy()
    for col in ("ticker", "date"):
        if col not in df.columns:
            raise ValueError(f"store_price_history expects a '{col}' column")
    if df["ticker"].nunique() != 1:
        raise ValueError("store_price_history expects one ticker per call")

   
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").drop_duplicates(subset=["ticker", "date"], keep="last")

    ticker = str(df["ticker"].iloc[0])
    params = {"ticker": ticker, "start_date": df["date"].min(), "end_date": df["date"].max()}

    delete_sql = text(
        f"DELETE FROM {table_name} "
        "WHERE ticker = :ticker AND date >= :start_date AND date <= :end_date"
    ).bindparams(bindparam("start_date", type_=Date), bindparam("end_date", type_=Date))

    with engine.begin() as conn:
        if inspect(conn).has_table(table_name):
            conn.execute(delete_sql, params)
        df.to_sql(table_name, conn, if_exists="append", index=False, dtype={"date": Date()})

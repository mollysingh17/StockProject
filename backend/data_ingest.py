import yfinance as yf
import pandas as pd

from db import engine  

def fetch_price_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.reset_index(inplace=True)      
    df.rename(columns={"Date": "date"}, inplace=True)
    df["ticker"] = ticker
    return df

def store_price_history(df: pd.DataFrame, table_name: str = "price_history") -> None:
    df.to_sql(table_name, engine, if_exists="append", index=False)
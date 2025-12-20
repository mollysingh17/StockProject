import yfinance as yf
import pandas as pd

from db import engine  

def fetch_price_history(ticker: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

    # 🔹 If yfinance gives MultiIndex columns like ('Close', 'AAPL'), flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.reset_index(inplace=True)      # turn Date index into a column
    df.rename(columns={"Date": "date"}, inplace=True)
    df["ticker"] = ticker
    return df

def store_price_history(df: pd.DataFrame, table_name: str = "price_history") -> None:
    # append to table or create if it doesn't exist
    df.to_sql(table_name, engine, if_exists="append", index=False)
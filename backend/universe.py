from io import StringIO

import pandas as pd
import requests

from backend.db import engine

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_sp500() -> pd.DataFrame:
    html = requests.get(SP500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    table = pd.read_html(StringIO(html))[0]
    df = table[["Symbol", "Security", "GICS Sector"]].rename(
        columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"}
    )
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df


def store_universe(df: pd.DataFrame) -> None:
    df.to_sql("tickers", engine, if_exists="replace", index=False)


if __name__ == "__main__":
    df = fetch_sp500()
    store_universe(df)
    print(f"Stored {len(df)} tickers")
    print(df["sector"].value_counts().to_string())
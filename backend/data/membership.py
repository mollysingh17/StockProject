from io import StringIO

import pandas as pd
import requests
from sqlalchemy import Date, text

from backend.db import engine

MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
RENAMES = {
    "ANTM": "ELV",
    "ABC": "COR",
    "BLL": "BALL",
    "BK": "BNY",
    "PKI": "RVTY",
    "WLTW": "WTW",
    "RE": "EG",
    "JEC": "J",
    "SYMC": "GEN",
    "NLOK": "GEN",
    "FLT": "CPAY",
    "FI": "FISV",
    "MMC": "MRSH",
    "HRS": "LHX",
    "UTX": "RTX",
    "DWDP": "DD",
    "KORS": "CPRI",
    "LB": "BBWI",
    "FB": "META",
    "TMK": "GL",
    "ARNC": "HWM",
    "CTL": "LUMN",
    "MYL": "VTRS",
}


def fetch_membership(since: str = "2015-01-01") -> pd.DataFrame:
    csv_text = requests.get(MEMBERSHIP_URL, timeout=60).text
    df = pd.read_csv(StringIO(csv_text), parse_dates=["date"])
    df = df[df["date"] >= since]
    df["ticker"] = df["tickers"].str.split(",")
    df = df.explode("ticker")[["date", "ticker"]]
    df["ticker"] = df["ticker"].str.strip().str.replace(".", "-", regex=False)
    df["ticker"] = df["ticker"].replace(RENAMES)
    df = df.drop_duplicates(subset=["date", "ticker"])
    df["date"] = df["date"].dt.date
    return df.reset_index(drop=True)


def store_membership(df: pd.DataFrame) -> None:
    df.to_sql("index_membership", engine, if_exists="replace", index=False, dtype={"date": Date()})

def current_tickers() -> list[str]:
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(text("SELECT ticker FROM tickers")).fetchall()]


def load_membership_matrix(dates: pd.DatetimeIndex, tickers: pd.Index) -> pd.DataFrame:
    with engine.connect() as conn:
        snaps = pd.read_sql_query(
            text("SELECT date, ticker FROM index_membership"), conn, parse_dates=["date"]
        )
    current = current_tickers()

    matrix = pd.crosstab(snaps["date"], snaps["ticker"]).astype(float)
    matrix = matrix.reindex(columns=matrix.columns.union(current), fill_value=0.0)
    after_last = matrix.index.max() + pd.Timedelta(days=1)
    matrix.loc[after_last] = matrix.columns.isin(current).astype(float)

    matrix = matrix.reindex(matrix.index.union(dates)).ffill()
    return matrix.reindex(index=dates, columns=tickers).fillna(0.0) > 0


if __name__ == "__main__":
    df = fetch_membership()
    store_membership(df)
    per_date = df.groupby("date")["ticker"].count()
    print(f"Stored {len(df)} rows: {per_date.size} snapshots from {per_date.index.min()} to {per_date.index.max()}")
    print(f"Members per snapshot: min {per_date.min()}, max {per_date.max()}")
    print(f"Distinct tickers since 2015: {df['ticker'].nunique()}")
import pandas as pd
from sqlalchemy import Date, text

from backend.data.ingest import FULL_HISTORY_START
from backend.db import engine

CHUNK_SIZE = 50


def stored_tickers() -> list[str]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT ticker FROM price_history ORDER BY ticker")).fetchall()
    return [r[0] for r in rows if r[0] != "SPY"]


def fetch_splits_batch(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers, start=FULL_HISTORY_START, actions=True, auto_adjust=True,
        group_by="ticker", threads=True, progress=False,
    )
    frames = []
    available = set(raw.columns.get_level_values(0))
    for ticker in tickers:
        if ticker not in available or "Stock Splits" not in raw[ticker].columns:
            continue
        events = raw[ticker]["Stock Splits"]
        events = events[events > 0]
        frames.append(pd.DataFrame({"ticker": ticker, "date": events.index, "ratio": events.values}))
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "ratio"])
    return pd.concat(frames, ignore_index=True)


def store_splits(df: pd.DataFrame) -> None:
    df = df.assign(date=pd.to_datetime(df["date"]).dt.date)
    df.to_sql("splits", engine, if_exists="replace", index=False, dtype={"date": Date()})


def main():
    tickers = stored_tickers()
    frames = []
    for i in range(0, len(tickers), CHUNK_SIZE):
        frames.append(fetch_splits_batch(tickers[i : i + CHUNK_SIZE]))
        print(f"  {min(i + CHUNK_SIZE, len(tickers))}/{len(tickers)} done")

    splits = pd.concat(frames, ignore_index=True)
    store_splits(splits)
    print(f"\nStored {len(splits)} splits for {splits['ticker'].nunique()} tickers\n")
    for ticker in ["AAPL", "NVDA", "TSLA"]:
        print(splits[splits["ticker"] == ticker].to_string(index=False))
        print()


if __name__ == "__main__":
    main()
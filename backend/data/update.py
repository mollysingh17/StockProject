import pandas as pd
from sqlalchemy import inspect, text

from backend.data.ingest import (
    ensure_price_index,
    fetch_price_history_batch,
    history_was_adjusted,
    store_price_history,
)
from backend.db import engine

BENCHMARKS = ["SPY"]
CHUNK_SIZE = 50
OVERLAP_DAYS = 7


def load_tickers() -> list[str]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT ticker FROM tickers ORDER BY ticker")).fetchall()
    return [r[0] for r in rows] + BENCHMARKS


def stored_tickers() -> set[str]:
    with engine.connect() as conn:
        if not inspect(conn).has_table("price_history"):
            return set()
        rows = conn.execute(text("SELECT DISTINCT ticker FROM price_history")).fetchall()
    return {r[0] for r in rows}


def update_start_date() -> str:
    with engine.connect() as conn:
        latest = conn.execute(text("SELECT MAX(date) FROM price_history")).scalar()
    return (pd.to_datetime(latest) - pd.Timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d")


def ingest(tickers: list[str], start: str | None = None) -> list[str]:
    failed, readjusted = [], []
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i : i + CHUNK_SIZE]
        frames, chunk_failed = fetch_price_history_batch(chunk, start=start)
        for ticker, df in frames.items():
            if start and history_was_adjusted(df):
                readjusted.append(ticker)
                continue
            store_price_history(df)
        failed += chunk_failed
        print(f"  {min(i + CHUNK_SIZE, len(tickers))}/{len(tickers)} done, {len(failed)} failed so far")

    if readjusted:
        print(f"  Prices were re-adjusted for {readjusted}, re-downloading full history")
        failed += ingest(readjusted)
    return failed


def main():
    ensure_price_index()
    tickers = load_tickers()
    existing = stored_tickers()
    new = [t for t in tickers if t not in existing]
    old = [t for t in tickers if t in existing]

    failed = []
    if new:
        print(f"Downloading full history for {len(new)} new tickers")
        failed += ingest(new)
        ensure_price_index()
    if old:
        start = update_start_date()
        print(f"Updating {len(old)} tickers from {start}")
        failed += ingest(old, start=start)

    if failed:
        print(f"\nFailed ({len(failed)}): {failed}")
    else:
        print("\nAll tickers up to date.")


if __name__ == "__main__":
    main()
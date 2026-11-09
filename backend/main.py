from sqlalchemy import text

from backend.data_ingest import fetch_price_history_batch, store_price_history
from backend.db import engine

BENCHMARKS = ["SPY"]
CHUNK_SIZE = 50


def load_tickers() -> list[str]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT ticker FROM tickers ORDER BY ticker")).fetchall()
    return [r[0] for r in rows] + BENCHMARKS


def main():
    tickers = load_tickers()
    stored, failed = 0, []
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i : i + CHUNK_SIZE]
        frames, chunk_failed = fetch_price_history_batch(chunk)
        for df in frames.values():
            store_price_history(df)
        stored += len(frames)
        failed += chunk_failed
        print(f"{min(i + CHUNK_SIZE, len(tickers))}/{len(tickers)} done, {len(failed)} failed so far")

    print(f"\nStored {stored} tickers.")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
from sqlalchemy import text
from backend.data.ingest import fetch_price_history_batch, store_price_history
from backend.db import engine

CHUNK_SIZE = 50


def missing_member_tickers() -> list[str]:
    query = text("""
        SELECT DISTINCT ticker FROM index_membership
        WHERE date >= (
            SELECT MAX(date) FROM index_membership
            WHERE date <= (SELECT MIN(date) FROM price_history)
        )
        AND ticker NOT IN (SELECT DISTINCT ticker FROM price_history)
        ORDER BY ticker
    """)
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(query).fetchall()]


def main():
    tickers = missing_member_tickers()
    print(f"{len(tickers)} past members have no prices stored\n")

    found, failed = [], []
    for i in range(0, len(tickers), CHUNK_SIZE):
        frames, chunk_failed = fetch_price_history_batch(tickers[i : i + CHUNK_SIZE])
        for ticker, df in frames.items():
            store_price_history(df)
            found.append(ticker)
        failed += chunk_failed

    print(f"\nyfinance had data for {len(found)}: {sorted(found)}")
    print(f"\nNo data for {len(failed)}: {sorted(failed)}")


if __name__ == "__main__":
    main()
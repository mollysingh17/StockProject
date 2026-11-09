"""Ingest daily prices. Run from the project root:  python -m backend.main"""

from backend.data_ingest import fetch_price_history, store_price_history

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


def main():
    failed = []
    for ticker in TICKERS:
        try:
            df = fetch_price_history(ticker, period="5y", interval="1d")
            store_price_history(df)
            print(f"Stored {len(df)} rows for {ticker}.")
        except Exception as e:  
            print(f"FAILED {ticker}: {e}")
            failed.append(ticker)
    if failed:
        print(f"\n{len(failed)} ticker(s) failed: {failed}")


if __name__ == "__main__":
    main()

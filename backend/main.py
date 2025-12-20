from data_ingest import fetch_price_history, store_price_history


def main():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    for ticker in tickers:
        df = fetch_price_history(ticker, period="5y", interval="1d")
        store_price_history(df)
        print(f"Stored {len(df)} rows for {ticker}.")


if __name__ == "__main__":
    main()
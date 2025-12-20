from data_ingest import fetch_price_history, store_price_history


def main():
    ticker = "AAPL"
    df = fetch_price_history(ticker, period="6mo", interval="1d")
    store_price_history(df)
    print(f"Stored {len(df)} rows for {ticker}.")


if __name__ == "__main__":
    main()
from .signals import load_price_history, add_sma_crossover_signal
from .backtest import run_backtest


def main():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    short = 20
    long = 50

    for ticker in tickers:
        df = load_price_history(ticker)
        df = add_sma_crossover_signal(df, short, long)
        results = run_backtest(df, short, long)
        print(ticker, results)


if __name__ == "__main__":
    main()
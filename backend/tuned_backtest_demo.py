from .signals import (
    load_price_history,
    add_sma_crossover_signal_confirmed,
)
from .backtest import run_backtest

# Best SMA params found from the grid search
BEST_SMA_PARAMS = {
    "AAPL":  (5, 100),
    "MSFT":  (5, 100),
    "GOOGL": (20, 100),
    "AMZN":  (20, 200),
    "NVDA":  (10, 100),
}


def main():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    for ticker in tickers:
        short, long = BEST_SMA_PARAMS[ticker]

        df = load_price_history(ticker)
        df = add_sma_crossover_signal_confirmed(df, short_window=short, long_window=long)
        
        results = run_backtest(df, short_window=short, long_window=long)

        print(f"\n=== {ticker} (short={short}, long={long}) ===")
        print(f"Buy & Hold:")
        print(f"  Final:   {results['buy_hold_final']:.3f}x")
        print(f"  CAGR:    {results['buy_hold_cagr']:.3f}")
        print(f"  Max DD:  {results['buy_hold_max_dd']:.3f}")

        print(f"SMA Strategy (tuned):")
        print(f"  Final:   {results['strategy_final_net']:.3f}x")
        print(f"  CAGR:    {results['strategy_cagr']:.3f}")
        print(f"  Max DD:  {results['strategy_max_dd']:.3f}")
        print(f"  Sharpe:  {results['strategy_sharpe']:.3f}")


if __name__ == "__main__":
    main()
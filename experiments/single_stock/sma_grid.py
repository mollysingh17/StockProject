"""In-sample SMA grid search over the FULL history (exploration only: these
numbers are optimistic because the same data picks and scores the params).
Run:  python -m experiments.sma_grid"""

from experiments.single_stock.backtest import run_backtest_from_signal
from experiments.single_stock.signals import add_sma_crossover_signal, load_price_history

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
SHORT_WINDOWS = [5, 10, 20]
LONG_WINDOWS = [50, 100, 200]
COST_BPS = 5.0  


def main():
    for ticker in TICKERS:
        print(f"\n=== {ticker} (in-sample) ===")
        df = load_price_history(ticker)

        rows = []
        for short in SHORT_WINDOWS:
            for long in LONG_WINDOWS:
                if short >= long:
                    continue
                res = run_backtest_from_signal(
                    add_sma_crossover_signal(df, short, long), cost_bps=COST_BPS
                )
                rows.append((short, long, res))
                print(
                    f"short={short:>3}, long={long:>3} -> CAGR={res['strategy_cagr']:.3f} "
                    f"Sharpe={res['strategy_sharpe']:.3f} Final={res['strategy_final_net']:.3f} "
                    f"InMkt={res['in_market_frac']:.2f}"
                )

        s, l, r = max(rows, key=lambda x: x[2]["strategy_sharpe"])
        print(f"  -> Best by Sharpe: {s}/{l}  Sharpe={r['strategy_sharpe']:.3f}  CAGR={r['strategy_cagr']:.3f}")
        s, l, r = max(rows, key=lambda x: x[2]["strategy_cagr"])
        print(f"  -> Best by CAGR:   {s}/{l}  CAGR={r['strategy_cagr']:.3f}  Sharpe={r['strategy_sharpe']:.3f}")
        print(f"  Buy & hold:        CAGR={r['buy_hold_cagr']:.3f}  Sharpe={r['buy_hold_sharpe']:.3f}")


if __name__ == "__main__":
    main()

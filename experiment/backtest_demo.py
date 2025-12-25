from backend.signals import load_price_history, add_sma_crossover_signal
from backend.backtest import run_backtest

def main():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    short_windows = [5, 10, 20]
    long_windows = [50, 100, 200]

    for ticker in tickers:
        print(f"\n=== {ticker} ===")
        df = load_price_history(ticker)

        best_sharpe = None
        best_sharpe_cfg = None

        best_cagr = None
        best_cagr_cfg = None

        for short in short_windows:
            for long in long_windows:
                if short >= long:
                    continue

                df_sig = add_sma_crossover_signal(df, short, long)
                results = run_backtest(df_sig, short, long)

                cagr = results["strategy_cagr"]
                sharpe = results["strategy_sharpe"]
                final = results["strategy_final_net"]

                print(
                    f"short={short:>3}, long={long:>3} -> "
                    f"CAGR={cagr:.3f}, Sharpe={sharpe:.3f}, Final={final:.3f}"
                )

                # Track best by Sharpe
                if best_sharpe is None or sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_sharpe_cfg = (short, long, cagr, final)

                # Track best by CAGR
                if best_cagr is None or cagr > best_cagr:
                    best_cagr = cagr
                    best_cagr_cfg = (short, long, sharpe, final)

        # Summary for this ticker
        if best_sharpe_cfg:
            s_short, s_long, s_cagr, s_final = best_sharpe_cfg
            print(
                f"  -> Best by Sharpe: short={s_short}, long={s_long}, "
                f"CAGR={s_cagr:.3f}, Final={s_final:.3f}, Sharpe={best_sharpe:.3f}"
            )

        if best_cagr_cfg:
            c_short, c_long, c_sharpe, c_final = best_cagr_cfg
            print(
                f"  -> Best by CAGR:   short={c_short}, long={c_long}, "
                f"CAGR={best_cagr:.3f}, Final={c_final:.3f}, Sharpe={c_sharpe:.3f}"
            )


if __name__ == "__main__":
    main()
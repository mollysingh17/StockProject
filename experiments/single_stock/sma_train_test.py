"""Tune SMA windows on TRAIN only, evaluate on TEST.
Run:  python -m experiments.sma_train_test

BUG FIX (warm-up): the old code split first and then computed SMAs on each
segment, so the first `long_window` days of the test set (200 of ~375 days for
a 200-day SMA) had NaN SMAs and the strategy sat out of the market, while
buy-and-hold was scored over the full test set. SMAs only look backward, so we
compute signals on the full history and THEN slice. No leakage.
"""

from experiments.single_stock.backtest import run_backtest_from_signal, train_test_split_by_time
from experiments.single_stock.signals import add_sma_crossover_signal_confirmed, load_price_history

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
SHORT_WINDOWS = [5, 10, 15, 20, 30, 40]
LONG_WINDOWS = [50, 100, 150, 200]
COST_BPS = 5.0


def signals_for(df, short, long, confirm_days):
    return add_sma_crossover_signal_confirmed(df, short, long, confirm_days)


def segment(df_sig, start, end):
    """Rows of df_sig between two dates (inclusive)."""
    return df_sig[(df_sig["date"] >= start) & (df_sig["date"] <= end)]


def find_best_sma_params_on_train(
    df, train_range, confirm_days, use_penalty, exposure_target=0.6, exposure_lambda=0.5
):
    best_score, best_params = None, None
    for short in SHORT_WINDOWS:
        for long in LONG_WINDOWS:
            if short >= long:
                continue
            train_sig = segment(signals_for(df, short, long, confirm_days), *train_range)
            res = run_backtest_from_signal(train_sig, cost_bps=COST_BPS)
            in_market = res["in_market_frac"]

            score = res["strategy_sharpe"]
            if use_penalty:
                score -= exposure_lambda * abs(in_market - exposure_target)
            # discourage near-always-in / near-always-out solutions
            score -= 2.0 * max(0.0, in_market - 0.97) + 2.0 * max(0.0, 0.03 - in_market)

            if best_score is None or score > best_score:
                best_score, best_params = score, (short, long)
    return best_params, best_score


def main():
    for ticker in TICKERS:
        df = load_price_history(ticker)
        train_df, test_df = train_test_split_by_time(df, train_frac=0.7)
        train_range = (train_df["date"].min(), train_df["date"].max())
        test_range = (test_df["date"].min(), test_df["date"].max())

        for confirm_days in [0, 3]:
            print(f"\n{ticker} | confirm_days={confirm_days}")
            for label, use_penalty in [("PLAIN", False), ("PENALIZED", True)]:
                (s, l), score = find_best_sma_params_on_train(df, train_range, confirm_days, use_penalty)
                test = run_backtest_from_signal(
                    segment(signals_for(df, s, l, confirm_days), *test_range), cost_bps=COST_BPS
                )
                print(
                    f"{label:<9} {s}/{l} | TRAIN score={score:.3f} | TEST "
                    f"Final={test['strategy_final_net']:.3f} CAGR={test['strategy_cagr']:.3f} "
                    f"Sharpe={test['strategy_sharpe']:.3f} MaxDD={test['strategy_max_dd']:.3f} "
                    f"InMkt={test['in_market_frac']:.2f}"
                )
            print(
                f"BUY&HOLD  TEST Final={test['buy_hold_final']:.3f} CAGR={test['buy_hold_cagr']:.3f} "
                f"Sharpe={test['buy_hold_sharpe']:.3f} MaxDD={test['buy_hold_max_dd']:.3f}"
            )


if __name__ == "__main__":
    main()

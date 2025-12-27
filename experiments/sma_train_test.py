
from backend.signals import load_price_history, add_sma_crossover_signal_confirmed
from backend.backtest import run_backtest

# Best SMA params found from the grid search
BEST_SMA_PARAMS = {
    "AAPL":  (5, 100),
    "MSFT":  (5, 100),
    "GOOGL": (20, 100),
    "AMZN":  (20, 200),
    "NVDA":  (10, 100),
}

def train_test_split_by_time(df, train_frac=0.7):
    """
    Split a price history DataFrame into train/test segments by time order.

    df must be sorted by date ascending.
    """
    n = len(df)
    split_idx = int(n * train_frac)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df

def run_sma_backtest_on_segment(df_segment, short_window, long_window, confirm_days):
    df_segment = df_segment.copy()

    df_segment = add_sma_crossover_signal_confirmed(
        df_segment,
        short_window=short_window,
        long_window=long_window,
        confirm_days=confirm_days,
    )
    
    in_market_frac = df_segment["signal"].mean()
    print(f"In-market fraction: {in_market_frac:.3f}")

    # IMPORTANT: call run_backtest using YOUR actual signature
    results = run_backtest(
        df_segment,
        short_window=short_window,
        long_window=long_window,
        fee_per_trade=0.0005,  # keep default, or set explicitly
        long_only=True,
    )
    return results

def find_best_sma_params_on_train(
    df_train,
    short_windows,
    long_windows,
    confirm_days,
    use_penalty: bool,
    exposure_target=0.6,
    exposure_lambda=0.5,
):
    best_score = None
    best_params = None

    for short in short_windows:
        for long in long_windows:
            if short >= long:
                continue

            df_sig = add_sma_crossover_signal_confirmed(
                df_train,
                short_window=short,
                long_window=long,
                confirm_days=confirm_days,
            )

            in_market = df_sig["signal"].mean()

            # 1) compute sharpe first (so it exists)
            res = run_backtest(df_sig, short_window=short, long_window=long)
            sharpe = res["strategy_sharpe"]

            # 2) base score
            score = sharpe

            # 3) optional exposure-target penalty (your "penalized objective")
            if use_penalty:
                score = score - exposure_lambda * abs(in_market - exposure_target)

            # 4) extra penalty for near-always-long / near-always-out
            # (instead of hard-skipping)
            extreme_penalty = 2.0 * max(0.0, in_market - 0.97) + 2.0 * max(0.0, 0.03 - in_market)
            score = score - extreme_penalty

            if best_score is None or score > best_score:
                best_score = score
                best_params = (short, long)

    # If literally nothing was evaluated (should be rare), fall back
    if best_params is None:
        print(
            f"[tune] confirm_days={confirm_days}: no candidates evaluated. "
            f"Falling back to {short_windows[0]}/{long_windows[0]}."
        )
        return (short_windows[0], long_windows[0]), float("-inf")

    return best_params, best_score

def main():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    #tickers = ["AAPL"]
    
    for ticker in tickers:
        # Load full price history
        df = load_price_history(ticker)

        # Split by time
        train_df, test_df = train_test_split_by_time(df, train_frac=0.7)

        # Use already-known baseline params
        #short_window, long_window = BEST_SMA_PARAMS[ticker]
        #confirm_days = 3
        for confirm_days in [0, 3]:
            short_windows = [5, 10, 15, 20, 30, 40]
            long_windows  = [50, 100, 150, 200]
        

            (best_s_plain, best_l_plain), score_plain = find_best_sma_params_on_train(
                train_df,
                short_windows=short_windows,
                long_windows=long_windows,
                confirm_days=confirm_days,
                use_penalty=False,
            )

            (best_s_pen, best_l_pen), score_pen = find_best_sma_params_on_train(
                train_df,
                short_windows=short_windows,
                long_windows=long_windows,
                confirm_days=confirm_days,
                use_penalty=True,
                exposure_target=0.6,
                exposure_lambda=0.5,
            )

            plain_train = run_sma_backtest_on_segment(
                train_df,
                short_window=best_s_plain,
                long_window=best_l_plain,
                confirm_days=confirm_days,
            )

            pen_train = run_sma_backtest_on_segment(
                train_df,
                short_window=best_s_pen,
                long_window=best_l_pen,
                confirm_days=confirm_days,
            )

            plain_test = run_sma_backtest_on_segment(
                test_df,
                best_s_plain,
                best_l_plain,
                confirm_days,
            )

            pen_test = run_sma_backtest_on_segment(
                test_df,
                best_s_pen,
                best_l_pen,
                confirm_days,
            )

            print(f"\n{ticker} | confirm_days={confirm_days}")
            print(
                f"PLAIN Sharpe-tune : short={best_s_plain}, long={best_l_plain} | "
                f"TRAIN score={score_plain:.3f} | "
                f"TEST Final={plain_test['strategy_final_net']:.3f} "
                f"CAGR={plain_test['strategy_cagr']:.3f} "
                f"Sharpe={plain_test['strategy_sharpe']:.3f}"
                )

            print(
                f"PENALIZED tune   : short={best_s_pen}, long={best_l_pen} | "
                f"TRAIN score={score_pen:.3f} | "
                f"TEST Final={pen_test['strategy_final_net']:.3f} "
                f"CAGR={pen_test['strategy_cagr']:.3f} "
                f"Sharpe={pen_test['strategy_sharpe']:.3f}"
            )

            print(
                f"BUY & HOLD (TEST): "
                f"Final={plain_test['buy_hold_final']:.3f} "
                f"CAGR={plain_test['buy_hold_cagr']:.3f}"
            )
            
            print(f"{ticker} confirm={confirm_days} | plain={best_s_plain}/{best_l_plain} | pen={best_s_pen}/{best_l_pen}")
        
    '''
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

        '''
if __name__ == "__main__":
    main()
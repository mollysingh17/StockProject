
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

def find_best_sma_params_on_train(df_train, short_windows, long_windows, confirm_days):
    best_final = None
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

            res = run_backtest(
                df_sig,
                short_window=short,
                long_window=long,
            )

            score = res["strategy_sharpe"]
            if best_final is None or score > best_final:
                best_final = score
                best_params = (short, long)

    return best_params, best_final

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
        

            (best_short, best_long), best_train_final = find_best_sma_params_on_train(
                train_df,
                short_windows=short_windows,
                long_windows=long_windows,
                confirm_days=confirm_days,
            )

            train_res = run_sma_backtest_on_segment(
                train_df,
                short_window=best_short,
                long_window=best_long,
                confirm_days=confirm_days,
            )

            test_res = run_sma_backtest_on_segment(
                test_df,
                short_window=best_short,
                long_window=best_long,
                confirm_days=confirm_days,
            )

            print(f"\n{ticker}")
            print(f"Best params on TRAIN: short={best_short}, long={best_long}, confirm={confirm_days}")
            print(f"TRAIN BuyHold : {train_res['buy_hold_final']:.3f}")
            print(f"TRAIN Strategy: {train_res['strategy_final_net']:.3f}")
            print(f"TEST  BuyHold : {test_res['buy_hold_final']:.3f}")
            print(f"TEST  Strategy: {test_res['strategy_final_net']:.3f}")

            print(f"TRAIN CAGR   : {train_res['strategy_cagr']:.3f} | Sharpe: {train_res['strategy_sharpe']:.3f} | MaxDD: {train_res['strategy_max_dd']:.3f}")
            print(f"TEST  CAGR   : {test_res['strategy_cagr']:.3f} | Sharpe: {test_res['strategy_sharpe']:.3f} | MaxDD: {test_res['strategy_max_dd']:.3f}")
            print(f"TRAIN BH CAGR: {train_res['buy_hold_cagr']:.3f} | MaxDD: {train_res['buy_hold_max_dd']:.3f}")
            print(f"TEST  BH CAGR: {test_res['buy_hold_cagr']:.3f} | MaxDD: {test_res['buy_hold_max_dd']:.3f}")
        
        
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
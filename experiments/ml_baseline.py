import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier

from backend.signals import load_price_history
from backend.backtest import run_backtest_from_signal


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1"] = df["Close"].pct_change()
    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_20"] = df["Close"].pct_change(20)
    df["vol_20"] = df["Close"].pct_change().rolling(20).std()

    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["sma_200"] = df["Close"].rolling(200).mean()

    df["close_over_sma50"] = df["Close"] / df["sma_50"] - 1.0
    df["sma10_over_sma50"] = df["sma_10"] / df["sma_50"] - 1.0
    df["sma20_over_sma200"] = df["Close"].rolling(20).mean() / df["sma_200"] - 1.0

    return df


def main():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    horizon = 63            # predict direction over next 5 trading days
    cost_bps = 1.0         # 1bp per trade (set 0.0 if you want)
    thresholds = np.arange(0.50, 0.71, 0.02)

    feature_cols = [
        "ret_1", "ret_5", "ret_20",
        "vol_20",
        "close_over_sma50",
        "sma10_over_sma50",
        "sma20_over_sma200",
    ]

    for t in tickers:
        df = load_price_history(ticker=t)  # returns date, Close
        df = make_features(df)

        # label: up over next horizon days?
        fwd_ret = df["Close"].shift(-horizon) / df["Close"] - 1.0
        df["y"] = (fwd_ret > 0).astype(int)

        # drop rows with NaNs from rolling features + last horizon rows
        df = df.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)

        # time split
        n = len(df)
        cut = int(n * 0.7)
        train = df.iloc[:cut].copy()
        test = df.iloc[cut:].copy()

        X_train = train[feature_cols].values
        y_train = train["y"].values
        X_test = test[feature_cols].values

        model = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=400,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # tune threshold on TRAIN by Sharpe (NOT accuracy)
        p_train = model.predict_proba(X_train)[:, 1]

        best_thr = None
        best_train_sh = None
        for thr in thresholds:
            train_sig = train[["date", "Close"]].copy()

            # size position from 0..1 based on confidence above thr
            pos = (p_train - thr) / (1.0 - thr)
            train_sig["signal"] = np.clip(pos, 0.0, 1.0)
            res = run_backtest_from_signal(train_sig, cost_bps=cost_bps)
            
            sh = res["strategy_sharpe"]
            if best_train_sh is None or sh > best_train_sh:
                best_train_sh = sh
                best_thr = thr

        # evaluate on TEST
        p_test = model.predict_proba(X_test)[:, 1]
        test_sig = test[["date", "Close"]].copy()
        pos = (p_test - best_thr) / (1.0 - best_thr)
        test_sig["signal"] = np.clip(pos, 0.0, 1.0) ** 0.5

        out = run_backtest_from_signal(test_sig, cost_bps=cost_bps)

        print(f"\n=== {t} | horizon={horizon}d | cost={cost_bps}bps ===")
        print(f"Chosen threshold (by TRAIN Sharpe): {best_thr:.2f} (train_sharpe={best_train_sh:.3f})")
        print(
            f"ML (TEST): Final={out['strategy_final_net']:.3f} "
            f"CAGR={out['strategy_cagr']:.3f} Sharpe={out['strategy_sharpe']:.3f} "
            f"MaxDD={out['strategy_max_dd']:.3f} InMkt={out['in_market_frac']:.3f}"
        )
        print(
            f"BH (TEST): Final={out['buy_hold_final']:.3f} "
            f"CAGR={out['buy_hold_cagr']:.3f} Sharpe={out['buy_hold_sharpe']:.3f} "
            f"MaxDD={out['buy_hold_max_dd']:.3f}"
        )


if __name__ == "__main__":
    main()
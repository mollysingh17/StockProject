"""Gradient-boosted direction classifier with confidence-based position sizing.
Run:  python -m experiments.single_stock.ml_baseline

Time layout per ticker (no shuffling, ever):

  |---- FIT (train model) ----|gap|-- VAL (pick threshold) --|------ TEST ------|
  0                        55% - h  55%                     70%               100%

Fixes vs. the old version:
  1. Rows whose 63-day future isn't known yet used to get y=0 (NaN > 0 is False).
     Now y is NaN there, and those rows are only dropped from the FIT set.
  2. A gap of `horizon` rows sits between FIT and VAL so FIT labels can't
     peek at the period used to choose the threshold.
  3. The threshold (and sizing curve) is chosen on VAL predictions, not on the
     model's own training data, where its probabilities are overconfident.
  4. One sizing function is used for VAL and TEST (the old code tuned with
     linear sizing and tested with **0.5).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from experiments.single_stock.backtest import run_backtest_from_signal
from experiments.single_stock.signals import load_price_history

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
HORIZON = 63  # predict direction over the next ~3 months (63 trading days)
COST_BPS = 1.0
THRESHOLDS = np.arange(0.50, 0.71, 0.02)

SIZING_POWERS = [0.5, 1.0, 2.0]
FIT_END, VAL_END = 0.55, 0.70

FEATURE_COLS = [
    "ret_1", "ret_5", "ret_20", "vol_20",
    "close_over_sma50", "sma10_over_sma50", "sma20_over_sma200",
]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    df["ret_1"] = c.pct_change()
    df["ret_5"] = c.pct_change(5)
    df["ret_20"] = c.pct_change(20)
    df["vol_20"] = c.pct_change().rolling(20).std()
    sma_10, sma_50, sma_200 = (c.rolling(w).mean() for w in (10, 50, 200))
    df["close_over_sma50"] = c / sma_50 - 1.0
    df["sma10_over_sma50"] = sma_10 / sma_50 - 1.0
    df["sma20_over_sma200"] = c.rolling(20).mean() / sma_200 - 1.0
    return df


def make_label(df: pd.DataFrame, horizon: int) -> pd.Series:
    fwd_ret = df["Close"].shift(-horizon) / df["Close"] - 1.0
    
    return (fwd_ret > 0).astype(float).where(fwd_ret.notna())


def size_position(p: np.ndarray, thr: float, power: float) -> np.ndarray:
    """0 below the threshold, rising to 1 at p=1, shaped by `power`."""
    return np.clip((p - thr) / (1.0 - thr), 0.0, 1.0) ** power


def backtest_probs(seg: pd.DataFrame, p: np.ndarray, thr: float, power: float) -> dict:
    sig = seg[["date", "Close"]].copy()
    sig["signal"] = size_position(p, thr, power)
    return run_backtest_from_signal(sig, cost_bps=COST_BPS)


def run_ticker(df: pd.DataFrame, ticker: str) -> dict:
    df = make_features(df)
    df["y"] = make_label(df, HORIZON)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    n = len(df)
    fit_end, val_end = int(n * FIT_END), int(n * VAL_END)
    fit = df.iloc[: fit_end - HORIZON].dropna(subset=["y"])
    val = df.iloc[fit_end:val_end]
    test = df.iloc[val_end:]  

    model = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=400, random_state=42
    )
    model.fit(fit[FEATURE_COLS].values, fit["y"].astype(int).values)

    p_val = model.predict_proba(val[FEATURE_COLS].values)[:, 1]
    best = max(
        ((thr, pw, backtest_probs(val, p_val, thr, pw)["strategy_sharpe"])
         for thr in THRESHOLDS for pw in SIZING_POWERS),
        key=lambda x: x[2],
    )
    thr, power, val_sharpe = best

    p_test = model.predict_proba(test[FEATURE_COLS].values)[:, 1]
    out = backtest_probs(test, p_test, thr, power)

    print(f"\n=== {ticker} | horizon={HORIZON}d | cost={COST_BPS}bps ===")
    print(f"rows: fit={len(fit)} val={len(val)} test={len(test)}")
    print(f"Chosen on VAL: threshold={thr:.2f} power={power} (val_sharpe={val_sharpe:.3f})")
    print(
        f"ML (TEST): Final={out['strategy_final_net']:.3f} CAGR={out['strategy_cagr']:.3f} "
        f"Sharpe={out['strategy_sharpe']:.3f} MaxDD={out['strategy_max_dd']:.3f} "
        f"InMkt={out['in_market_frac']:.3f}"
    )
    print(
        f"BH (TEST): Final={out['buy_hold_final']:.3f} CAGR={out['buy_hold_cagr']:.3f} "
        f"Sharpe={out['buy_hold_sharpe']:.3f} MaxDD={out['buy_hold_max_dd']:.3f}"
    )
    return {"ticker": ticker, "threshold": thr, "power": power, **out}


def main():
    for t in TICKERS:
        run_ticker(load_price_history(ticker=t), t)


if __name__ == "__main__":
    main()

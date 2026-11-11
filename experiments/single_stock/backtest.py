import numpy as np
import pandas as pd


def run_backtest_from_signal(
    df: pd.DataFrame,
    cost_bps: float = 0.0,  
    annual_trading_days: int = 252,
) -> dict:
    """Backtest a long-only strategy from a precomputed signal.

    Required columns: date, Close, signal.
    signal is the target exposure for the NEXT day, from 0.0 (out) to 1.0
    (fully in); fractional values are allowed (position sizing).
    Next-day execution: position = signal.shift(1), so today's close is never
    used to trade today (no look-ahead).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "signal" not in df.columns:
        raise ValueError("df must contain a 'signal' column")

    df["ret"] = df["Close"].pct_change().fillna(0.0)
    df["position"] = df["signal"].shift(1).fillna(0.0).clip(0.0, 1.0)

    # cost charged on turnover |pos_t - pos_{t-1}|
    df["turnover"] = df["position"].diff().abs().fillna(0.0)
    df["cost"] = df["turnover"] * (cost_bps / 10000.0)

    df["strategy_ret"] = df["position"] * df["ret"] - df["cost"]
    df["buy_hold_ret"] = df["ret"]

    df["strategy_eq"] = (1.0 + df["strategy_ret"]).cumprod()
    df["buy_hold_eq"] = (1.0 + df["buy_hold_ret"]).cumprod()

    def cagr(eq: pd.Series) -> float:
        n = len(eq)
        if n < 2:
            return 0.0
        return float(eq.iloc[-1] ** (annual_trading_days / n) - 1.0)

    def sharpe(r: pd.Series) -> float:
        sd = r.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return 0.0
        return float(r.mean() / sd * np.sqrt(annual_trading_days))

    def max_dd(eq: pd.Series) -> float:
        return float((eq / eq.cummax() - 1.0).min())

    return {
        "strategy_final_net": float(df["strategy_eq"].iloc[-1]),
        "strategy_cagr": cagr(df["strategy_eq"]),
        "strategy_sharpe": sharpe(df["strategy_ret"]),
        "strategy_max_dd": max_dd(df["strategy_eq"]),
        "buy_hold_final": float(df["buy_hold_eq"].iloc[-1]),
        "buy_hold_cagr": cagr(df["buy_hold_eq"]),
        "buy_hold_sharpe": sharpe(df["buy_hold_ret"]),
        "buy_hold_max_dd": max_dd(df["buy_hold_eq"]),
        "in_market_frac": float(df["position"].mean()),
        "n_days": int(len(df)),
    }


def train_test_split_by_time(df: pd.DataFrame, train_frac: float = 0.7, gap: int = 0):
    """Split by time. `gap` rows are dropped between train and test (an
    "embargo") so labels that look `gap` days ahead can't peek into test."""
    split_idx = int(len(df) * train_frac)
    return df.iloc[: max(split_idx - gap, 0)].copy(), df.iloc[split_idx:].copy()

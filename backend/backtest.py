import numpy as np
import pandas as pd

def run_backtest_from_signal(
    df: pd.DataFrame,
    cost_bps: float = 0.0,   # e.g. 1.0 = 1 basis point per trade (0.01%)
    annual_trading_days: int = 252,
):
    """
    Backtest a simple in/out strategy from a precomputed signal.
    Required columns: date, Close, signal (0/1)
    Uses next-day execution: position = signal.shift(1)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "signal" not in df.columns:
        raise ValueError("df must contain a 'signal' column (0/1)")

    # returns
    df["ret"] = df["Close"].pct_change().fillna(0.0)

    # position (trade on next bar)
    df["position"] = df["signal"].shift(1).fillna(0.0).clip(0.0, 1.0)

    # transaction costs: charge when position changes
    # turnover is |pos_t - pos_{t-1}|
    df["turnover"] = df["position"].diff().abs().fillna(0.0)
    cost_rate = cost_bps / 10000.0
    df["cost"] = df["turnover"] * cost_rate

    df["strategy_ret"] = df["position"] * df["ret"] - df["cost"]
    df["buy_hold_ret"] = df["ret"]

    # equity curves
    df["strategy_eq"] = (1.0 + df["strategy_ret"]).cumprod()
    df["buy_hold_eq"] = (1.0 + df["buy_hold_ret"]).cumprod()

    def cagr(eq: pd.Series) -> float:
        n = len(eq)
        if n < 2:
            return 0.0
        years = n / annual_trading_days
        return float(eq.iloc[-1] ** (1.0 / years) - 1.0)

    def sharpe(r: pd.Series) -> float:
        # daily Sharpe annualized
        mu = r.mean()
        sd = r.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return 0.0
        return float((mu / sd) * np.sqrt(annual_trading_days))

    def max_dd(eq: pd.Series) -> float:
        peak = eq.cummax()
        dd = eq / peak - 1.0
        return float(dd.min())

    out = {
        "strategy_final_net": float(df["strategy_eq"].iloc[-1]),
        "strategy_cagr": cagr(df["strategy_eq"]),
        "strategy_sharpe": sharpe(df["strategy_ret"]),
        "strategy_max_dd": max_dd(df["strategy_eq"]),
        "buy_hold_final": float(df["buy_hold_eq"].iloc[-1]),
        "buy_hold_cagr": cagr(df["buy_hold_eq"]),
        "buy_hold_sharpe": sharpe(df["buy_hold_ret"]),
        "buy_hold_max_dd": max_dd(df["buy_hold_eq"]),
        "in_market_frac": float(df["position"].mean()),
    }
    return out
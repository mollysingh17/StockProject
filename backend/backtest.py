import numpy as np
import pandas as pd


def _compute_cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    n_periods = len(equity_curve)
    if n_periods <= 1:
        return 0.0
    final_value = equity_curve.iloc[-1]
    return float(final_value ** (periods_per_year / n_periods) - 1)


def _compute_max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1.0
    return float(drawdown.min())  # e.g. -0.35 = -35%


def _compute_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((returns.mean() / std) * np.sqrt(periods_per_year))


def run_backtest(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50,
    fee_per_trade: float = 0.0005,  # 0.05% per position change
    long_only: bool = True,         # NEW: if False, use long/short
):
    df = df.copy()

    # Only keep rows where both SMAs exist
    valid_mask = df[f"sma_{short_window}"].notna() & df[f"sma_{long_window}"].notna()
    df = df[valid_mask].copy()

    # Daily percent returns of the stock
    df["daily_return"] = df["Close"].pct_change().fillna(0)

    # Base position from signal (0/1)
    df["position"] = df["signal"].shift(1).fillna(0)

    # If long_only = False, treat 0 as -1 (short when not in uptrend)
    if not long_only:
        df["position"] = df["position"].replace(0, -1)

    # Strategy returns before costs
    df["strategy_return"] = df["daily_return"] * df["position"]

    # Trading cost: whenever position changes (enter/exit/flip)
    df["trade"] = df["position"].diff().abs().fillna(0)
    df["cost"] = df["trade"] * fee_per_trade

    # After costs
    df["strategy_return_after_costs"] = df["strategy_return"] - df["cost"]

    # Equity curves
    buy_hold_curve = (1 + df["daily_return"]).cumprod()
    strategy_curve_net = (1 + df["strategy_return_after_costs"]).cumprod()

    return {
        # final multiples
        "buy_hold_final": float(buy_hold_curve.iloc[-1]),
        "strategy_final_net": float(strategy_curve_net.iloc[-1]),

        # annualized growth
        "buy_hold_cagr": _compute_cagr(buy_hold_curve),
        "strategy_cagr": _compute_cagr(strategy_curve_net),

        # risk metrics
        "buy_hold_max_dd": _compute_max_drawdown(buy_hold_curve),
        "strategy_max_dd": _compute_max_drawdown(strategy_curve_net),
        "strategy_sharpe": _compute_sharpe(df["strategy_return_after_costs"]),
    }
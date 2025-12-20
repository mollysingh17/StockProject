import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50,
):
    df = df.copy()

    # Only keep rows where both SMAs exist 
    valid_mask = df[f"sma_{short_window}"].notna() & df[f"sma_{long_window}"].notna()
    df = df[valid_mask].copy()

    # Daily percent returns of the stock
    df["daily_return"] = df["Close"].pct_change().fillna(0)

    # Shift signal by 1 day
    df["position"] = df["signal"].shift(1).fillna(0)

    df["strategy_return"] = df["daily_return"] * df["position"]

    # equity curves
    buy_hold_curve = (1 + df["daily_return"]).cumprod()
    strategy_curve = (1 + df["strategy_return"]).cumprod()

    return {
        "buy_hold_final": float(buy_hold_curve.iloc[-1]),
        "strategy_final": float(strategy_curve.iloc[-1]),
    }
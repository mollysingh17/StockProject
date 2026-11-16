import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(returns: pd.Series) -> float:
    growth = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    return float(growth ** (1 / years) - 1)


def sharpe(returns: pd.Series) -> float:
    sd = returns.std()
    if sd == 0:
        return 0.0
    return float(returns.mean() / sd * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min())


def summarize(returns: pd.Series) -> dict:
    return {
        "final": float((1 + returns).prod()),
        "cagr": cagr(returns),
        "sharpe": sharpe(returns),
        "max_dd": max_drawdown(returns),
    }
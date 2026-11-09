"""Run:  pytest -q"""

import numpy as np
import pandas as pd
import pytest

from backend.backtest import run_backtest_from_signal, train_test_split_by_time
from experiments.ml_baseline import make_label, size_position
from backend.signals import (
    add_sma,
    add_sma_crossover_signal,
    add_sma_crossover_signal_confirmed,
)


def prices(closes):
    return pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=len(closes), freq="B"), "Close": closes}
    )




def test_sma_matches_hand_computed():
    df = add_sma(prices([1, 2, 3, 4, 5]), window=3)
    
    assert df["sma_3"].isna().tolist() == [True, True, False, False, False]
    assert df["sma_3"].iloc[2:].tolist() == [2.0, 3.0, 4.0]


def test_crossover_is_out_during_warmup_and_in_on_uptrend():
    df = add_sma_crossover_signal(prices(list(range(1, 11))), short_window=2, long_window=4)
    assert df["signal"].iloc[:3].tolist() == [0, 0, 0]  
    assert df["signal"].iloc[3:].eq(1).all() 


def test_confirm_days_zero_does_not_force_always_in():
    
    falling = prices(list(range(20, 0, -1)))
    df = add_sma_crossover_signal_confirmed(falling, 2, 4, confirm_days=0)
    assert df["signal"].sum() == 0


def test_confirmation_delays_entry():
    raw = add_sma_crossover_signal(prices(list(range(1, 11))), 2, 4)["signal"]
    conf = add_sma_crossover_signal_confirmed(prices(list(range(1, 11))), 2, 4, confirm_days=3)["signal"]
    first_raw, first_conf = raw.idxmax(), conf.idxmax()
    assert first_conf == first_raw + 2


def test_always_in_equals_buy_and_hold():
    df = prices([100, 110, 99, 120])
    df["signal"] = 1
    res = run_backtest_from_signal(df)
  
    assert res["strategy_final_net"] == pytest.approx(res["buy_hold_final"])
    assert res["buy_hold_final"] == pytest.approx(1.2)


def test_next_day_execution_no_lookahead():
    df = prices([100, 110, 110])
    df["signal"] = [0, 1, 0]
    res = run_backtest_from_signal(df)
    assert res["strategy_final_net"] == pytest.approx(1.0)


def test_costs_charged_on_turnover():
    df = prices([100, 100, 100, 100])
    df["signal"] = [1, 0, 1, 0]  
    res = run_backtest_from_signal(df, cost_bps=10)  
    assert res["strategy_final_net"] == pytest.approx(0.999 ** 3)


def test_max_drawdown():
    df = prices([100, 120, 60, 90])
    df["signal"] = 1
    assert run_backtest_from_signal(df)["buy_hold_max_dd"] == pytest.approx(-0.5)


def test_split_gap():
    train, test = train_test_split_by_time(prices(list(range(100))), 0.7, gap=10)
    assert len(train) == 60 and len(test) == 30



def test_label_is_nan_when_future_unknown():
    y = make_label(prices([1, 2, 3, 2, 1]), horizon=2)
 
    assert y.iloc[:3].tolist() == [1.0, 0.0, 0.0]
    assert y.iloc[3:].isna().all()


def test_size_position():
    p = np.array([0.4, 0.6, 0.8, 1.0])
    assert size_position(p, 0.6, 1.0).tolist() == pytest.approx([0, 0, 0.5, 1.0])
    assert size_position(p, 0.6, 2.0)[2] == pytest.approx(0.25) 

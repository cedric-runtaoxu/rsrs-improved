"""rsrs_improved 单元测试。

直接运行::

    python tests/test_improved.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsrs_improved.backtest import backtest, max_drawdown
from rsrs_improved.indicators import make_score, rsrs_accel, rsrs_signal
from rsrs_improved.sensitivity import grid_search, overfitting_diagnosis
from rsrs_improved.strategy import VARIANTS, run_variant


def _df(n=800, seed=0):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0002, 0.012, n)
    close = pd.Series(1000 * np.cumprod(1 + ret),
                      index=pd.bdate_range("2018-01-01", periods=n))
    high = close * (1 + np.abs(rng.normal(0.004, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0.004, 0.003, n)))
    # 开盘价带隔夜跳空，这样才能区分「开盘成交」与「收盘成交」
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, n))
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1.0})
    df["ret"] = df["close"] / df["close"].shift(1) - 1.0
    return df


def test_open_vs_close_execution_differ():
    df = _df()
    z = make_score(df, "baseline")
    sig = rsrs_signal(z)
    close_res = backtest(df, sig, execution="close")
    open_res = backtest(df, sig, execution="open")
    assert not np.allclose(close_res.ret, open_res.ret)


def test_open_execution_no_future_info():
    """开盘成交的收益只用当日 open->close，不偷用前一日信息。"""
    df = _df()
    z = make_score(df, "baseline")
    sig = rsrs_signal(z)
    res = backtest(df, sig, execution="open")
    # 开盘成交下，切换当天的收益应等于 close/open-1（不含隔夜跳空）
    changed = res.position.ne(res.position.shift(1).fillna(0))
    for i in range(1, len(df)):
        if changed.iloc[i]:
            expected = res.position.iloc[i] * (df["close"].iloc[i] / df["open"].iloc[i] - 1)
            # 扣除成本后的策略收益应 ≤ 毛收益
            assert res.ret.iloc[i] <= expected + 1e-12


def test_accel_defined():
    df = _df()
    beta = (df["high"].rolling(18).cov(df["low"]) / df["low"].rolling(18).var())
    accel = rsrs_accel(beta, lag=10, m=600)
    assert accel.isna().sum() >= 600 + 10 - 1


def test_all_variants_run():
    df = _df()
    for name in VARIANTS:
        res = run_variant(df, name)
        assert len(res.ret) == len(df)
        assert res.n_trades >= 0


def test_grid_search_shape_and_diagnosis():
    df = _df(n=1500)
    grid = grid_search(df, split_date="2019-12-31",
                       n_list=(10, 18), m_list=(300,), s_buy_list=(0.7,),
                       s_sell_list=(-0.7,))
    assert len(grid) == 2
    assert set(["train_sharpe", "test_sharpe"]).issubset(grid.columns)
    diag = overfitting_diagnosis(grid)
    assert "样本外正夏普占比" in diag


def test_max_drawdown_bounded():
    df = _df()
    z = make_score(df, "baseline")
    res = backtest(df, rsrs_signal(z), execution="open")
    assert -1.0 < max_drawdown(res.equity) <= 0.0


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed.append((fn.__name__, repr(e)))
            print(f"  [FAIL] {fn.__name__}: {e!r}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())

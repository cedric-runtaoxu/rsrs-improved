"""策略变体：把「信号方法 × 执行方式 × 阈值」组合成可复现的命名配置。"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .backtest import BacktestResult, backtest
from .indicators import make_score, rsrs_signal

# 命名变体。每个字段对应 make_score / rsrs_signal / backtest 的参数。
VARIANTS: Dict[str, dict] = {
    "baseline_close": {
        "label": "原始版（收盘成交）",
        "method": "baseline", "execution": "close",
        "n": 18, "m": 600, "s_buy": 0.7, "s_sell": -0.7,
    },
    "baseline_open": {
        "label": "原始版（开盘成交）",
        "method": "baseline", "execution": "open",
        "n": 18, "m": 600, "s_buy": 0.7, "s_sell": -0.7,
    },
    "smoothed_open": {
        "label": "信号平滑（3 日）",
        "method": "smoothed", "execution": "open", "smooth_window": 3,
        "n": 18, "m": 600, "s_buy": 0.7, "s_sell": -0.7,
    },
    "asym_open": {
        "label": "不对称阈值（买 0.7 / 卖 -0.5）",
        "method": "baseline", "execution": "open",
        "n": 18, "m": 600, "s_buy": 0.7, "s_sell": -0.5,
    },
    "combined_open": {
        "label": "RSRS + 加速项",
        "method": "combined", "execution": "open",
        "n": 18, "m": 600, "s_buy": 0.7, "s_sell": -0.7, "k": 0.5, "accel_lag": 10,
    },
}


def run_variant(df: pd.DataFrame, variant: str, **overrides) -> BacktestResult:
    """按命名变体跑一次回测，``overrides`` 可覆盖任意参数。"""
    cfg = dict(VARIANTS[variant])
    cfg.update(overrides)

    method = cfg["method"]
    execution = cfg["execution"]
    s_buy = cfg.get("s_buy", 0.7)
    s_sell = cfg.get("s_sell", -0.7)
    cost = cfg.get("cost", 0.001)
    lag = cfg.get("lag", 1)

    score_kw = {k: cfg[k] for k in ("n", "m", "k", "accel_lag", "smooth_window") if k in cfg}
    score = make_score(df, method=method, **score_kw)
    signal = rsrs_signal(score, s_buy=s_buy, s_sell=s_sell)
    result = backtest(df, signal, execution=execution, lag=lag, cost=cost)
    result.params.update({"variant": variant, "label": VARIANTS[variant]["label"],
                          "method": method, "s_buy": s_buy, "s_sell": s_sell,
                          **score_kw})
    return result

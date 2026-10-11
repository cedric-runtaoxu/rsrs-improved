"""RSRS 指标：原始版 + 改进项（加速项 / 平滑）。

符号与时间轴同 ``rsrs-timing``：信号在 t 日收盘才可知，回测中滞后生效。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsrs_beta(high: pd.Series, low: pd.Series, n: int = 18) -> pd.Series:
    """滚动 OLS 斜率：``high = alpha + beta * low``。"""
    cov = high.rolling(n, min_periods=n).cov(low)
    var = low.rolling(n, min_periods=n).var()
    beta = (cov / var).replace([np.inf, -np.inf], np.nan)
    return beta.rename("beta")


def rsrs_zscore(beta: pd.Series, m: int = 600) -> pd.Series:
    """把 ``beta`` 用过去 ``m`` 日均值/标准差标准化。"""
    mu = beta.rolling(m, min_periods=m).mean()
    sd = beta.rolling(m, min_periods=m).std(ddof=1)
    return ((beta - mu) / sd.replace(0.0, np.nan)).rename("zscore")


def rsrs_accel(beta: pd.Series, lag: int = 10, m: int = 600) -> pd.Series:
    """斜率加速项（「右偏 RSRS」的简化实现）。

    思想：RSRS 原版只看斜率的**水平**，而斜率的**变化**（加速上涨/加速下跌）
    是更强的趋势确认信号。这里用 ``beta_t - beta_{t-lag}`` 衡量斜率变化，
    再标准化到与 z 相同量纲，便于线性合成。
    """
    accel = beta - beta.shift(lag)
    mu = accel.rolling(m, min_periods=m).mean()
    sd = accel.rolling(m, min_periods=m).std(ddof=1)
    return ((accel - mu) / sd.replace(0.0, np.nan)).rename("accel")


def smooth(z: pd.Series, window: int = 3) -> pd.Series:
    """信号平滑，降低阈值附近的高频抖动，从而减少交易次数。"""
    return z.rolling(window, min_periods=1).mean()


def rsrs_signal(
    z: pd.Series,
    s_buy: float = 0.7,
    s_sell: float = -0.7,
    initial: float = 0.0,
) -> pd.Series:
    """由得分序列产生 0/1 仓位（状态机：越过阈值才切换）。"""
    zvals = z.to_numpy(dtype="float64")
    out = np.empty(len(z), dtype="float64")
    state = float(initial)
    for i in range(len(zvals)):
        v = zvals[i]
        if not np.isnan(v):
            if v > s_buy:
                state = 1.0
            elif v < s_sell:
                state = 0.0
        out[i] = state
    return pd.Series(out, index=z.index, name="position")


def make_score(
    df: pd.DataFrame,
    method: str = "baseline",
    n: int = 18,
    m: int = 600,
    k: float = 0.5,
    accel_lag: int = 10,
    smooth_window: int = 3,
) -> pd.Series:
    """按方法生成可阈值化的得分序列。

    method
        "baseline"   原始 RSRS 标准分
        "smoothed"   原始标准分做短期平滑
        "combined"   标准分 + k*加速项
    """
    beta = rsrs_beta(df["high"], df["low"], n)
    z = rsrs_zscore(beta, m)
    if method == "baseline":
        score = z
    elif method == "smoothed":
        score = smooth(z, smooth_window)
    elif method == "combined":
        score = z + k * rsrs_accel(beta, accel_lag, m)
    else:
        raise ValueError(f"未知方法: {method!r}")
    return score.rename("score")

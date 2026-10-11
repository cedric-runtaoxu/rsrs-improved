"""回测引擎（支持收盘成交 / 开盘成交两种执行方式）与绩效统计。

执行方式
--------
* ``execution="close"``：信号 t 日收盘产生，t+1 日**收盘**成交，吃满 t+1 全天收益。
* ``execution="open"``：信号 t 日收盘产生，t+1 日**开盘**成交，进一步消除前视
  （更贴近「收盘才知道信号、次日一早才能下单」的真实约束）。

两种方式的差异本身就是研究对象：开盘成交通常更保守，能看出收盘成交
是否在偷跑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    ret: pd.Series
    benchmark_ret: pd.Series
    position: pd.Series
    equity: pd.Series
    benchmark_equity: pd.Series
    n_trades: int
    params: dict = field(default_factory=dict)


def backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    execution: str = "close",
    lag: int = 1,
    cost: float = 0.001,
) -> BacktestResult:
    """按 0/1 仓位信号回测。

    Parameters
    ----------
    signal : t 日收盘决策的目标仓位（0/1）。
    execution : ``"close"`` 次日收盘成交 | ``"open"`` 次日开盘成交。
    lag : 信号日到执行日间隔，**必须 >= 1**。
    cost : 单边交易费率，仓位切换当天扣除。
    """
    if lag < 1:
        raise ValueError("lag 必须 >= 1，否则存在未来函数")

    sig = signal.reindex(df.index).ffill().fillna(0.0)
    pos = sig.shift(lag).fillna(0.0)          # 当日实际持有的仓位
    prev = pos.shift(1).fillna(0.0)
    changed = pos.ne(prev)                     # 当日是否发生切换

    close_ret = df["ret"].fillna(0.0)
    if execution == "close":
        daily = close_ret
    elif execution == "open":
        open_ret = (df["close"] / df["open"] - 1.0).fillna(0.0)
        daily = pd.Series(
            np.where(changed.to_numpy(), open_ret.to_numpy(), close_ret.to_numpy()),
            index=df.index,
        )
    else:
        raise ValueError(f"未知执行方式: {execution!r}")

    cost_series = changed.astype("float64") * cost
    strat = pos * daily - cost_series
    bench = close_ret

    return BacktestResult(
        ret=strat,
        benchmark_ret=bench,
        position=pos,
        equity=(1.0 + strat).cumprod(),
        benchmark_equity=(1.0 + bench).cumprod(),
        n_trades=int((sig.diff().ne(0)).sum()),
        params={"execution": execution, "lag": lag, "cost": cost},
    )


# ------------------------------------------------------------------ 绩效指标


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def annualized_return(ret: pd.Series, periods: int = TRADING_DAYS) -> float:
    years = len(ret) / periods
    growth = float((1.0 + ret).prod())
    return growth ** (1.0 / years) - 1.0 if years > 0 and growth > 0 else np.nan


def annualized_vol(ret: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(ret.std(ddof=1) * np.sqrt(periods))


def sharpe(ret: pd.Series, periods: int = TRADING_DAYS) -> float:
    sd = ret.std(ddof=1)
    return float(ret.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def calmar(ret: pd.Series, equity: pd.Series) -> float:
    mdd = max_drawdown(equity)
    return annualized_return(ret) / abs(mdd) if mdd < 0 else np.nan


def summarize(result: BacktestResult) -> Dict[str, float]:
    ann = annualized_return(result.ret)
    return {
        "累计收益": total_return(result.equity),
        "年化收益": ann,
        "年化波动": annualized_vol(result.ret),
        "夏普比率": sharpe(result.ret),
        "最大回撤": max_drawdown(result.equity),
        "Calmar": calmar(result.ret, result.equity),
        "交易次数": float(result.n_trades),
        "基准年化收益": annualized_return(result.benchmark_ret),
        "基准最大回撤": max_drawdown(result.benchmark_equity),
        "超额年化收益": ann - annualized_return(result.benchmark_ret),
    }

"""评估：样本内外切分、分年度收益、变体对比表。"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .backtest import BacktestResult, annualized_return, max_drawdown, sharpe


def split_returns(result: BacktestResult, split_date: str):
    """按日期把收益序列切成样本内 / 样本外。"""
    split = pd.Timestamp(split_date)
    train = result.ret[result.ret.index <= split]
    test = result.ret[result.ret.index > split]
    train_bench = result.benchmark_ret[result.benchmark_ret.index <= split]
    test_bench = result.benchmark_ret[result.benchmark_ret.index > split]
    return (train, test, train_bench, test_bench)


def period_metrics(ret: pd.Series, bench: pd.Series) -> Dict[str, float]:
    """一段收益的核心指标（策略 vs 基准）。"""
    eq = (1.0 + ret).cumprod()
    beq = (1.0 + bench).cumprod()
    return {
        "策略年化": annualized_return(ret),
        "策略夏普": sharpe(ret),
        "策略最大回撤": max_drawdown(eq),
        "基准年化": annualized_return(bench),
        "基准最大回撤": max_drawdown(beq),
        "超额年化": annualized_return(ret) - annualized_return(bench),
    }


def train_test_metrics(result: BacktestResult, split_date: str) -> pd.DataFrame:
    """样本内 / 样本外对比表（判断是否过拟合）。"""
    train, test, tb, teb = split_returns(result, split_date)
    rows = {
        f"样本内 (≤{split_date})": period_metrics(train, tb),
        f"样本外 (>{split_date})": period_metrics(test, teb),
    }
    df = pd.DataFrame(rows).T
    df.index.name = "区间"
    return df


def yearly_returns(result: BacktestResult) -> pd.DataFrame:
    """分年度收益表：策略 vs 基准，行=年。"""
    s = pd.DataFrame({"strategy": result.ret, "benchmark": result.benchmark_ret})
    s = 1.0 + s
    s["year"] = result.ret.index.year
    out = s.groupby("year")[["strategy", "benchmark"]].prod() - 1.0
    out.columns = ["策略", "基准"]
    out["超额"] = out["策略"] - out["基准"]
    return out


def comparison_table(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    """多个变体的绩效对比表，行=变体。"""
    rows = []
    for name, res in results.items():
        ann = annualized_return(res.ret)
        rows.append({
            "变体": res.params.get("label", name),
            "年化收益": ann,
            "年化波动": res.ret.std(ddof=1) * np.sqrt(252),
            "夏普": sharpe(res.ret),
            "最大回撤": max_drawdown(res.equity),
            "Calmar": ann / abs(max_drawdown(res.equity)),
            "交易次数": res.n_trades,
            "超额年化": ann - annualized_return(res.benchmark_ret),
        })
    return pd.DataFrame(rows).set_index("变体")

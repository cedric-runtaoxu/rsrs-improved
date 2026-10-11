"""参数敏感性：网格搜索 + 样本内外过拟合诊断。

这是「改进版」最重要的模块 —— 它回答的不再是「这个参数好不好」，
而是「**这个好结果里有多少是运气**」。

做法
----
1. 在样本内（≤ split_date）网格搜索 (n, m, s_buy, s_sell)；
2. 按样本内夏普排序，看样本外夏普是否同步 —— 如果样本内最好的一批
   在样本外明显变差，说明参数被过拟合了。
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .backtest import annualized_return, backtest, max_drawdown, sharpe
from .indicators import rsrs_beta, rsrs_signal, rsrs_zscore


def grid_search(
    df: pd.DataFrame,
    split_date: str = "2014-12-31",
    execution: str = "open",
    cost: float = 0.001,
    n_list: Iterable[int] = (10, 18, 24, 30, 42),
    m_list: Iterable[int] = (250, 600, 900),
    s_buy_list: Iterable[float] = (0.5, 0.7, 1.0),
    s_sell_list: Iterable[float] = (-0.5, -0.7, -1.0),
) -> pd.DataFrame:
    """网格搜索，返回每个参数组合的样本内 / 样本外指标。"""
    split = pd.Timestamp(split_date)
    rows = []
    for n in n_list:
        beta = rsrs_beta(df["high"], df["low"], n)
        for m in m_list:
            z = rsrs_zscore(beta, m)
            for sb in s_buy_list:
                for ss in s_sell_list:
                    signal = rsrs_signal(z, s_buy=sb, s_sell=ss)
                    res = backtest(df, signal, execution=execution, cost=cost)
                    tr = res.ret[res.ret.index <= split]
                    te = res.ret[res.ret.index > split]
                    rows.append({
                        "n": n, "m": m, "s_buy": sb, "s_sell": ss,
                        "train_sharpe": sharpe(tr) if len(tr) > 0 else np.nan,
                        "test_sharpe": sharpe(te) if len(te) > 0 else np.nan,
                        "train_ann": annualized_return(tr),
                        "test_ann": annualized_return(te),
                        "test_mdd": max_drawdown((1.0 + te).cumprod()) if len(te) > 0 else np.nan,
                        "n_trades": res.n_trades,
                    })
    return pd.DataFrame(rows)


def overfitting_diagnosis(grid: pd.DataFrame, k: int = 10) -> dict:
    """过拟合诊断：样本内选优在样本外还成立吗？"""
    g = grid.dropna(subset=["train_sharpe", "test_sharpe"])
    if g.empty:
        return {}
    top = g.nlargest(k, "train_sharpe")
    corr = float(g["train_sharpe"].corr(g["test_sharpe"]))
    return {
        "参数组合数": len(g),
        "样本内最优 K 组的样本外夏普均值": float(top["test_sharpe"].mean()),
        "全网格样本外夏普均值": float(g["test_sharpe"].mean()),
        "样本内外夏普相关性": corr,
        "样本内正夏普占比": float((g["train_sharpe"] > 0).mean()),
        "样本外正夏普占比": float((g["test_sharpe"] > 0).mean()),
        # 关键指标：最优组 vs 随机组的样本外差距
        "最优K组-全网格样本外差距": float(top["test_sharpe"].mean() - g["test_sharpe"].mean()),
    }


def top_parameters(grid: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """按样本内夏普排前 k 的参数组合。"""
    cols = ["n", "m", "s_buy", "s_sell", "train_sharpe", "test_sharpe",
            "train_ann", "test_ann", "test_mdd", "n_trades"]
    return grid.sort_values("train_sharpe", ascending=False).head(k)[cols].reset_index(drop=True)

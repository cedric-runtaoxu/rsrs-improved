"""RSRS 改进与稳健性分析 —— 主入口。

运行::

    python run_all.py

产出写入 report/：
- variant_comparison.csv        变体绩效对比
- train_test.csv                样本内外对比
- grid.csv / top_params.csv     参数网格与最优参数
- overfitting.csv               过拟合诊断
- yearly.csv                    分年度收益
- multi_index.csv               多指数稳健性
- *.png                         图表（需 matplotlib）
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rsrs_improved.backtest import summarize
from rsrs_improved.data import SYMBOLS, load_index
from rsrs_improved.evaluation import (
    comparison_table,
    train_test_metrics,
    yearly_returns,
)
from rsrs_improved.plot import plot_comparison, plot_drawdown, plot_overfitting
from rsrs_improved.sensitivity import (
    grid_search,
    overfitting_diagnosis,
    top_parameters,
)
from rsrs_improved.strategy import VARIANTS, run_variant

REPORT = Path(__file__).resolve().parent / "report"
SPLIT = "2014-12-31"


def _save(df: pd.DataFrame, name: str):
    df.to_csv(REPORT / name, encoding="utf-8-sig")


def _print_df(df: pd.DataFrame):
    print(df.to_string(float_format=lambda x: f"{x:,.4f}"))


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    df = load_index("sh000300")
    print(f"标的: 沪深300（sh000300），{df.index[0].date()} ~ {df.index[-1].date()}，{len(df)} 个交易日\n")

    # ================================================================ 1. 变体对比
    print("=" * 78)
    print("1) 变体对比（全样本 2005-2024，默认参数 n=18 m=600）")
    print("=" * 78)
    results = {name: run_variant(df, name) for name in VARIANTS}
    comp = comparison_table(results)
    _save(comp, "variant_comparison.csv")
    _print_df(comp)

    # ================================================================ 2. 样本内外
    print("\n" + "=" * 78)
    print(f"2) 样本内 / 样本外对比（baseline_open，切分点 {SPLIT}）")
    print("=" * 78)
    tt = train_test_metrics(results["baseline_open"], SPLIT)
    _save(tt, "train_test.csv")
    _print_df(tt)

    # ================================================================ 3. 参数敏感性
    print("\n" + "=" * 78)
    print("3) 参数敏感性网格搜索（样本内 2005-2014 选参，样本外 2015-2024 验证）")
    print("=" * 78)
    grid = grid_search(df, split_date=SPLIT)
    _save(grid, "grid.csv")
    top = top_parameters(grid, k=10)
    _save(top, "top_params.csv")
    print("\n样本内夏普排名前 10 的参数（注意样本外是否同步）：")
    _print_df(top)

    diag = overfitting_diagnosis(grid, k=10)
    _save(pd.DataFrame([diag]), "overfitting.csv")
    print("\n过拟合诊断：")
    for k, v in diag.items():
        print(f"  {k:<24s} {v:,.4f}" if isinstance(v, float) else f"  {k:<24s} {v}")

    # ================================================================ 4. 多指数稳健性
    print("\n" + "=" * 78)
    print("4) 多指数稳健性（baseline_open，n=18 m=600）")
    print("=" * 78)
    mi_rows = []
    for symbol in ("sh000300", "sh000001", "sz399001"):
        d = load_index(symbol)
        r = run_variant(d, "baseline_open")
        s = summarize(r)
        mi_rows.append({"指数": SYMBOLS[symbol], **{k: v for k, v in s.items()
                       if k in ("年化收益", "夏普比率", "最大回撤", "超额年化收益", "交易次数")}})
    mi = pd.DataFrame(mi_rows).set_index("指数")
    _save(mi, "multi_index.csv")
    _print_df(mi)

    # ================================================================ 5. 分年度
    print("\n" + "=" * 78)
    print("5) 分年度收益（baseline_open）")
    print("=" * 78)
    yearly = yearly_returns(results["baseline_open"])
    _save(yearly, "yearly.csv")
    _print_df(yearly)

    # ================================================================ 6. 图表
    fig1 = plot_comparison(results, REPORT)
    fig2 = plot_overfitting(grid, REPORT)
    fig3 = plot_drawdown(results["baseline_open"], REPORT)
    for f in (fig1, fig2, fig3):
        if f:
            print(f"\n图表: {f}")

    print(f"\n全部结果已写入: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

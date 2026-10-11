"""可视化（matplotlib 可选）。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .backtest import BacktestResult


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        _setup_cjk_font()
        import matplotlib.pyplot as plt
        return plt
    except Exception:  # pragma: no cover
        return None


def _setup_cjk_font() -> None:
    """注册中文字体，避免图表中文标签显示成方框。"""
    import os

    import matplotlib
    from matplotlib import font_manager

    win = os.environ.get("WINDIR", r"C:\Windows")
    candidates = [os.path.join(win, "Fonts", f) for f in ("msyh.ttc", "simhei.ttf", "simsun.ttc")]
    candidates += [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
                name = font_manager.FontProperties(fname=p).get_name()
                matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                matplotlib.rcParams["axes.unicode_minus"] = False
                return
            except Exception:  # pragma: no cover
                continue


def plot_comparison(
    results: Dict[str, BacktestResult],
    outdir: str | Path,
    title: str = "RSRS 变体对比",
) -> Optional[Path]:
    plt = _mpl()
    if plt is None:
        return None
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, res in results.items():
        ax.plot(res.equity.index, res.equity, lw=1.1, label=res.params.get("label", name))
    ax.plot(res.benchmark_equity.index, res.benchmark_equity, lw=1.4, color="#333",
            ls="--", label="买入持有基准")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("净值（对数轴）")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "variant_comparison.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_overfitting(
    grid: pd.DataFrame,
    outdir: str | Path,
) -> Optional[Path]:
    """样本内夏普 vs 样本外夏普散点。散点越贴近 45° 线，过拟合越轻。"""
    plt = _mpl()
    if plt is None:
        return None
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    g = grid.dropna(subset=["train_sharpe", "test_sharpe"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(g["train_sharpe"], g["test_sharpe"], s=18, alpha=0.5, color="#1f77b4")
    lo = min(g["train_sharpe"].min(), g["test_sharpe"].min())
    hi = max(g["train_sharpe"].max(), g["test_sharpe"].max())
    ax.plot([lo, hi], [lo, hi], ls="--", color="#999", label="45° 线（样本内外一致）")
    ax.axhline(0, color="crimson", lw=0.8)
    ax.axvline(0, color="crimson", lw=0.8)
    ax.set_xlabel("样本内夏普")
    ax.set_ylabel("样本外夏普")
    ax.set_title("参数敏感性：样本内 vs 样本外夏普")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "overfitting_scatter.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_drawdown(
    result: BacktestResult,
    outdir: str | Path,
    title: str = "策略回撤",
) -> Optional[Path]:
    plt = _mpl()
    if plt is None:
        return None
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    dd = result.equity / result.equity.cummax() - 1.0
    ax.fill_between(dd.index, dd.to_numpy(), 0, color="crimson", alpha=0.4)
    ax.set_title(title)
    ax.set_ylabel("回撤")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "drawdown.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p

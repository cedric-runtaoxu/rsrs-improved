"""数据加载与校验（同 rsrs-timing，见其说明）。

数据来源：腾讯行情接口（免费、无需 token），字段 date,open,high,low,close,volume。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SYMBOLS = {
    "sh000300": "沪深300",
    "sh000001": "上证综指",
    "sz399001": "深证成指",
}


def load_index(
    symbol: str = "sh000300",
    start: Optional[str] = None,
    end: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    base = Path(data_dir) if data_dir else DATA_DIR
    path = base / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少数据文件 {path}，请先运行 _fetch_data.py")

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"数据缺少必要列: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df.dropna(subset=["open", "high", "low", "close"])

    # 源数据取整/错误处理：容差 0.01 以内的取整差异保留；超出则丢弃并告警。
    tol = 0.01
    invalid = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1) - tol)
        | (df["low"] > df[["open", "close", "high"]].min(axis=1) + tol)
    )
    if invalid.any():
        print(f"[data] 丢弃 {int(invalid.sum())} 行 OHLC 自相矛盾的数据："
              f"{df.index[invalid].date.tolist()[:10]}")
        df = df[~invalid]
    df = df[df["close"] > 0]

    df["ret"] = df["close"] / df["close"].shift(1) - 1.0
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df

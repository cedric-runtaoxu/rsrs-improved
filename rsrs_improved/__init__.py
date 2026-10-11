"""rsrs_improved —— RSRS 择时指标的改进与稳健性分析。

项目定位（对应学习计划 W3）
----------------------------
在 ``rsrs-timing`` 的原始版复现基础上，回答三个问题：

1. **能不能做得更诚实？** 原始版用收盘信号+次日收盘成交；改成次日开盘成交
   （open execution），进一步消除「当日高低价」带来的前视。
2. **能不能做得更好？** 引入斜率加速项（右偏 RSRS）、不对称阈值、信号平滑，
   逐项检验是否带来增量。
3. **这些改进靠不靠谱？** 参数敏感性、样本内外切分、多指数稳健性，
   回答「好结果里有多少是运气」。
"""

from . import backtest, data, evaluation, indicators, sensitivity, strategy

__version__ = "0.2.0"
__all__ = ["backtest", "data", "evaluation", "indicators", "sensitivity",
           "strategy", "__version__"]

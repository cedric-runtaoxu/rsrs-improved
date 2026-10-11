-- ============================================================
-- 量化研究中高频用到的 SQL（窗口函数）练习
-- 对应学习计划 W3 的「SQL 补强」：这 6 道题覆盖了因子计算最常用的
-- 窗口函数用法（LAG / 窗口聚合 / RANK / PARTITION BY / gaps-and-islands）。
--
-- 环境：任何支持标准 SQL 的数据库（PostgreSQL / MySQL 8+ / SQLite 3.25+）。
-- 本文件可直接在 SQLite 里执行：sqlite3 quant.db < quant_sql.sql
-- ============================================================


-- ------------------------------------------------------------ 建表
DROP TABLE IF EXISTS daily_price;
CREATE TABLE daily_price (
    stock_code TEXT NOT NULL,
    trade_date DATE NOT NULL,
    close      REAL NOT NULL,
    high       REAL NOT NULL,
    low        REAL NOT NULL,
    volume     REAL NOT NULL,          -- 成交量（手）
    amount     REAL NOT NULL,          -- 成交额（元）
    PRIMARY KEY (stock_code, trade_date)
);

DROP TABLE IF EXISTS stock_meta;
CREATE TABLE stock_meta (
    stock_code TEXT PRIMARY KEY,
    industry   TEXT NOT NULL,          -- 行业（如「银行」「电子」）
    list_date  DATE NOT NULL           -- 上市日
);

-- 造几条示例数据（真实场景用你导出的日行情表替换）
INSERT INTO stock_meta VALUES
    ('000001','银行','2010-01-01'),
    ('000002','电子','2010-01-01'),
    ('000003','银行','2015-01-01'),
    ('000004','电子','2015-01-01');

INSERT INTO daily_price VALUES
    ('000001','2024-01-02',10.0,10.2, 9.9,100,100000),
    ('000001','2024-01-03',10.1,10.3,10.0,120,121000),
    ('000001','2024-01-04',10.4,10.5,10.1,150,156000),
    ('000001','2024-01-05',10.3,10.4,10.2,110,113000),
    ('000002','2024-01-02',20.0,20.5,19.8, 80,160000),
    ('000002','2024-01-03',20.4,20.6,20.0, 90,184000),
    ('000002','2024-01-04',19.9,20.4,19.7,100,199000),
    ('000002','2024-01-05',20.2,20.3,19.9, 85,172000),
    ('000003','2024-01-02', 5.0, 5.1, 4.9,200,100000),
    ('000003','2024-01-03', 5.1, 5.2, 5.0,210,107000),
    ('000003','2024-01-04', 5.0, 5.1, 4.9,190, 95000),
    ('000003','2024-01-05', 5.2, 5.3, 5.0,220,114000),
    ('000004','2024-01-02',15.0,15.3,14.8, 60, 90000),
    ('000004','2024-01-03',15.5,15.7,15.0, 70,109000),
    ('000004','2024-01-04',15.3,15.6,15.1, 65, 99500),
    ('000004','2024-01-05',15.8,15.9,15.4, 75,119000);


-- ------------------------------------------------------------ Q1 日收益率
-- 用 LAG 取前一交易日的收盘价，计算 close/prev_close - 1。
-- 注意 PARTITION BY stock_code：LAG 只在同一只股票内部向前取，不会串行。
SELECT
    stock_code, trade_date, close,
    ROUND(close / LAG(close) OVER (PARTITION BY stock_code ORDER BY trade_date) - 1, 4) AS daily_ret
FROM daily_price
ORDER BY stock_code, trade_date;


-- ------------------------------------------------------------ Q2 动量因子
-- 过去 N 日动量 = close / N 日前收盘 - 1。这是多因子模型里最基础的价量因子。
-- 用窗口聚合的 ROWS BETWEEN 写法，比 LAG 更灵活（可搭配其他聚合）。
SELECT
    stock_code, trade_date, close,
    ROUND(close / MIN(close) OVER (
        PARTITION BY stock_code
        ORDER BY trade_date
        ROWS BETWEEN 2 PRECEDING AND 2 PRECEDING   -- 这里演示 N=2 的窗口
    ) - 1, 4) AS mom_2d
FROM daily_price
ORDER BY stock_code, trade_date;


-- ------------------------------------------------------------ Q3 流动性因子
-- 过去 N 日日均成交额（流动性代理）。窗口聚合 + AVG。
SELECT
    stock_code, trade_date,
    ROUND(AVG(amount) OVER (
        PARTITION BY stock_code
        ORDER BY trade_date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2) AS avg_amount_3d
FROM daily_price
ORDER BY stock_code, trade_date;


-- ------------------------------------------------------------ Q4 行业内排名
-- 每日「成交额」在**同行业内**的截面排名。RANK() + PARTITION BY (date, industry)
-- 需要 JOIN 股票元数据。这是「行业中性化」的 SQL 原型。
SELECT
    p.trade_date, m.industry, p.stock_code, p.amount,
    RANK() OVER (
        PARTITION BY p.trade_date, m.industry
        ORDER BY p.amount DESC
    ) AS amount_rank_in_industry
FROM daily_price p
JOIN stock_meta m ON p.stock_code = m.stock_code
ORDER BY p.trade_date, m.industry, amount_rank_in_industry;


-- ------------------------------------------------------------ Q5 连续上涨天数
-- 经典 gaps-and-islands：标记「上涨/下跌」状态，用行号差给连续区间分组，
-- 再统计每组长度。这是择时信号、事件研究里非常常用的操作。
WITH flagged AS (
    SELECT stock_code, trade_date, close,
           CASE WHEN close > LAG(close) OVER (PARTITION BY stock_code ORDER BY trade_date)
                THEN 1 ELSE 0 END AS is_up
    FROM daily_price
),
grp AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY stock_code ORDER BY trade_date)
         - ROW_NUMBER() OVER (PARTITION BY stock_code, is_up ORDER BY trade_date) AS g
    FROM flagged
)
SELECT stock_code, MIN(trade_date) AS start_date, MAX(trade_date) AS end_date,
       COUNT(*) AS consecutive_up_days
FROM grp
WHERE is_up = 1
GROUP BY stock_code, g
ORDER BY stock_code, start_date;


-- ------------------------------------------------------------ Q6 月末截面：换手率排名
-- 每月最后一个交易日，按「当日成交额/流通市值代理」即换手率给全市场排名，
-- 取换手率最低的 20%（低换手因子，A 股常见）。
-- 关键点：先定位每月末日，再做截面 RANK。
WITH month_end AS (
    SELECT stock_code, trade_date, close, amount,
           MAX(trade_date) OVER (PARTITION BY stock_code, strftime('%Y-%m', trade_date)) AS me_date
    FROM daily_price
),
monthly AS (
    SELECT * FROM month_end WHERE trade_date = me_date
),
ranked AS (
    SELECT *,
           RANK() OVER (PARTITION BY trade_date ORDER BY amount ASC) AS turnover_rank,
           COUNT(*) OVER (PARTITION BY trade_date) AS total_n
    FROM monthly
)
SELECT trade_date, stock_code, amount,
       ROUND(1.0 * turnover_rank / total_n, 3) AS turnover_pctile
FROM ranked
WHERE 1.0 * turnover_rank / total_n <= 0.2   -- 换手最低的 20%
ORDER BY trade_date, turnover_pctile;

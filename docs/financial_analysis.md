# 季报景气 × 错定价（`financial_analysis.py`）

## 数据源

- **主表**：`data/financial_quarter_bs.parquet`。
- **行情对齐**：需先有 `features.build_features` 产出的 **`ret_5d`**；若缺 `financial_quarter_bs.parquet`，财务列填空 + `financial_has_triple=0`。

## 三道防火墙（实现约定）

| 防火墙 | 含义 | 实现要点 |
|--------|------|-----------|
| **1. 公告对齐** | \(G_t,G_{t-1},G_{t-2}\) 只能来自**已公告**财报；不得用报表期当「已上市面」。 | **仅** `NOTICE_DATE` 作为可见日 `_vis`；**不再**用 `REPORT_DATE` 填补公告。面板行 `date=T` 时，只允许 **NOTICE_DATE ≤ T−1** 的财年行进入三连季（日历上 `T−1` 为全市场相邻上一交易日）。 |
| **2. T−1 / 冷冻** | 决策在 **T**（或 T 开盘）；**T−1 收盘**前信息才合法。 | **`financial_ret_5d_rank`** 对的是 **T−1** 行上的 `ret_5d`（按股 **`shift(1)`**）；与基本面秩同一 `date=T` 行对齐后用於算 **`financial_gap`**。面板列 **`ret_5d` 未被改写**（仍为当日收盘价定义的技术特征）。 |
| **3. 逐日截面秩** | 排名不得跨多年历史混排。 | `groupby("date")` 对 `financial_delta`、`financial_accel`、合成与 `financial_gap` 相关秩 **每日重算**。 |

## XGBoost

常量 **`FINANCIAL_FEATURE_COLUMNS`**；已并进 **`features.FEATURE_COLUMNS`**；除 **`financial_has_triple`** 外 **`FEATURES_NA_OK`** 允许 NaN。

## 入口

`merge_financial_features(panel, …)` · 或由 **`features.build_features(prices)`** 带出。

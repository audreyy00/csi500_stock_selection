# Analyst 预期结构特征（`consensus_logic.py`）

## 数据源

`data/profit_forecast_csi500.parquet`（东财截面快照）。

- **减持 / 卖出**：不进入 consensus 分母；若在整表上**列恒为 0** 则读入后从内存表中**物理删除**。
- **无公告日时序**：仍为按 `(date, stock_code)` **广播截面值**。

## 设计原则（与建模规范对齐）

| 层级 | 内容 |
|------|------|
| 基底 | **正交**：`analyst_growth`、`analyst_consensus`、`analyst_coverage`（= `log1p(研报数)`）；**不做**单一的 `growth×consensus×sqrt(c)` 作主因子。 |
| 结构 | `analyst_mispricing`：每日截面 `rankpct(growth) − rankpct(consensus)`；`analyst_dispersion`：买/增持/中性比例的 **Shannon 熵**；`analyst_imbalance`：`(买入−中性)/(买+增持+中性)`。 |
| 弱交互 | `analyst_ix_growth_consensus` 等乘积列 — 供树模型学习，**不设唯一合成打分**。 |
| coverage | **`has_coverage`** 与 log 密度：**闸门/条件变量**，非「乘在核心分上放大收益」的杠杆。 |
| Legacy | `analyst_legacy_score`：旧式三乘积 — **不在 `FEATURE_COLUMNS`**。 |

`features.FEATURE_COLUMNS` 引用 **`ANALYST_FEATURE_COLUMNS`**；除 `has_coverage` 外，其余 analyst 列在 `FEATURES_NA_OK` 中允许 NaN（无覆盖或分母失效）。

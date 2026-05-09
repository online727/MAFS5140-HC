# Strategy Ver2 Research Workflow

本文档说明 momentum feature research 到 `strategy_ver2` 组合信号的完整流程，包括 feature 计算、feature analysis、组合权重生成、自动回测，以及相关命令行参数。

## 1. Feature 计算

脚本：`tools/mom_features.py`

该脚本从 `data/{dataset}/{split}.parquet` 读取行情数据，提取 `close` 和 `volume` 面板，批量计算 momentum 相关 features，并对每个 feature 同时评估两个方向：

- `raw`: 原始 feature 方向。
- `neg`: feature 取反方向，用于捕捉短周期反转类信号。

当前 feature families 包括：

- `return`: `ret`, `logret`, rolling mean/sum return, time-series return zscore。
- `trend`: SMA/EMA distance, SMA/EMA slope, MA cross, price time-series zscore。
- `oscillator`: ROC, RSI, stochastic close, MACD histogram。
- `volume`: volume change, relative volume, volume zscore, dollar volume zscore, price-volume corr, OBV slope proxy。
- `volatility`: realized vol, downside vol, return range, volatility-adjusted return。

对每个 feature variant，脚本会：

1. 保存 raw feature panel。
2. 用单个 feature 横截面 zscore 后生成 top-N long-only weights。
3. 对该单 feature weights 做向量化回测，计算 PnL 指标。
4. 计算 feature 与未来 `5min/15min/30min/60min` return 的横截面 IC、ICIR、RankIC、RankICIR。
5. 保存 IC series 和 summary。

默认输出路径按数据集和 split 分层：

```text
data/output/mom_features/{dataset}/{split}/features/{feature}.parquet
data/output/mom_features/{dataset}/{split}/ic_series/{variant}.csv
data/output/mom_features/{dataset}/{split}/feature_summary.csv
data/weights/mom_features/{dataset}/{split}/{variant}.parquet
```

示例：

```bash
python tools/mom_features.py --data-path data/mini1/validation.parquet
```

常用参数：

- `--data-path`: 必填，行情数据路径，例如 `data/mini1/train.parquet`。
- `--dataset`: 可选，数据集名；默认从 `--data-path` 的父目录推断，例如 `mini1`。
- `--split`: 可选，数据 split；默认从 `--data-path` 文件名推断，例如 `train` 或 `validation`。
- `--feature-format`: feature 保存格式，`parquet` 或 `csv`，默认 `parquet`。
- `--weights-format`: 单 feature weights 保存格式，`parquet` 或 `csv`，默认 `parquet`。
- `--directions`: 需要评估的方向，默认 `raw,neg`。
- `--top-selector`: 单 feature 回测选股数量；大于 1 表示 top N，小于 1 表示 top proportion，默认 `20`。
- `--max-weight`: 单资产权重上限，默认 `0.05`。
- `--periods-per-year`: 年化周期数，默认 `252 * 78`。
- `--feature-names`: 可选，只计算指定 feature，逗号分隔。
- `--max-features`: 可选，只计算前 N 个 feature，主要用于 smoke test。
- `--skip-save-features`: 不保存 feature panel。
- `--skip-save-weights`: 不保存单 feature weights。
- `--skip-save-ic-series`: 不保存 IC series。

## 2. Feature Analysis

脚本：`tools/feat_ana.py`

该脚本读取 `mom_features.py` 生成的 `feature_summary.csv`，从多个维度给 feature variants 排名并筛选 feature set。

主要分析维度：

- PnL: `sharpe_ratio`, `cumulative_return`, `max_drawdown`。
- IC quality: 多 horizon 的 `icir` 和 `rankicir`。
- Robustness: 多 horizon 平均 ICIR/RankICIR、最小 ICIR/RankICIR、正 IC horizon 数量。
- Risk adjusted score: Sharpe 和 drawdown 的组合。
- Composite score: PnL、IC、RankIC、drawdown 的综合 percentile score。
- Redundancy: 对 top composite pool 计算 feature value correlation 和 weights correlation，控制重复信号。

输出的 feature sets：

- `best_by_sharpe`
- `best_by_rankicir`
- `best_by_icir`
- `robust_multi_horizon`
- `low_drawdown`
- `family_balanced`
- `composite_best`
- `diversified_best`

默认输出路径：

```text
data/output/feat_ana/{dataset}/{split}/feature_scores.csv
data/output/feat_ana/{dataset}/{split}/top_by_metric.csv
data/output/feat_ana/{dataset}/{split}/selected_feature_sets.csv
data/output/feat_ana/{dataset}/{split}/family_summary.csv
data/output/feat_ana/{dataset}/{split}/horizon_summary.csv
data/output/feat_ana/{dataset}/{split}/redundancy_pairs.csv
data/output/feat_ana/{dataset}/{split}/analysis_report.md
data/output/feat_ana/{dataset}/{split}/*.png
```

其中 `selected_feature_sets.csv` 是 `strategy_ver2.py` 的直接输入。

示例：

```bash
python tools/feat_ana.py --data-path data/mini1/validation.parquet
```

常用参数：

- `--data-path`: 推荐提供，用于推断 `{dataset}/{split}` 并定位默认输入输出目录。
- `--dataset`: 可选，数据集名；未提供时从 `--data-path` 推断。
- `--split`: 可选，split 名；未提供时从 `--data-path` 推断。
- `--top-n`: 每个 feature set 选出的 feature 数量，默认 `30`。
- `--feature-corr-threshold`: feature value correlation 去重阈值，默认 `0.85`。
- `--weights-corr-threshold`: weights correlation 去重阈值，默认 `0.85`。
- `--redundancy-pool-size`: 用于冗余检查的 top composite candidate 数量；默认 `max(top_n * 3, 60)`。

## 3. Strategy Ver2 逻辑

脚本：`tools/strategy_ver2.py`

`strategy_ver2` 不重新计算 feature，而是读取 `feat_ana.py` 选出的 feature set 和 `mom_features.py` 保存的 feature panels。

对每个 `set_name`，策略流程如下：

1. 从 `selected_feature_sets.csv` 读取该 feature set 的 features。
2. 按 `direction` 读取 feature panel：
   - `raw`: 使用原始 feature。
   - `neg`: 使用 `-feature`。
3. 每个 feature panel 在每个 timestamp 做横截面 zscore。
4. 对所有 feature zscore 做等权平均，得到 combined score。
5. 每个 timestamp 选择 score 最高的 top-N 资产。
6. 只对正 score 分配权重，按 score 占比分配。
7. 对单资产权重应用 `max_weight` cap。
8. 无正 score 时保持 cash。
9. 保存 combined score、weights、manifest。
10. 默认立即对生成的 weights 做回测，保存 returns、equity curve 和 metrics。

默认输出路径：

```text
data/output/strategy_ver2/{dataset}/{split}/scores/{set_name}.parquet
data/weights/strategy_ver2/{dataset}/{split}/{set_name}.parquet
data/output/strategy_ver2/{dataset}/{split}/manifest.csv

data/output/backtest/strategy_ver2/{dataset}/{split}/portfolio_returns_{set_name}.csv
data/output/backtest/strategy_ver2/{dataset}/{split}/equity_curve_{set_name}.png
data/output/backtest/strategy_ver2/{dataset}/{split}/metrics_{set_name}.csv
data/output/backtest/strategy_ver2/{dataset}/{split}/metrics_summary.csv
```

示例：生成并回测全部 feature sets。

```bash
python tools/strategy_ver2.py --data-path data/mini1/validation.parquet --set-name all
```

示例：只生成并回测一个 feature set。

```bash
python tools/strategy_ver2.py \
  --data-path data/mini1/validation.parquet \
  --set-name composite_best
```

示例：只生成 score/weights，不回测。

```bash
python tools/strategy_ver2.py \
  --data-path data/mini1/validation.parquet \
  --set-name all \
  --skip-backtest
```

常用参数：

- `--data-path`: 行情数据路径；用于推断 `{dataset}/{split}`、对齐 index/columns，并在默认回测中作为 backtest data。
- `--dataset`: 可选，数据集名；默认从 `--data-path` 推断。
- `--split`: 可选，split 名；默认从 `--data-path` 推断。
- `--set-name`: feature set 名称，默认 `diversified_best`；可设为 `all` 批量生成所有 set。
- `--top-selector`: 选股数量；大于 1 表示 top N，小于 1 表示 top proportion，默认 `20`。
- `--max-weight`: 单资产权重上限，默认 `0.05`。
- `--feature-limit`: 可选，只使用 feature set 中前 N 个 features。
- `--skip-backtest`: 只生成 score 和 weights，不运行回测。
- `--periods-per-year`: 回测年化周期数，默认 `252 * 78`。

## 4. 推荐运行顺序

对单个数据集 split，例如 `mini1/validation`：

```bash
python tools/mom_features.py --data-path data/mini1/validation.parquet
python tools/feat_ana.py --data-path data/mini1/validation.parquet
python tools/strategy_ver2.py --data-path data/mini1/validation.parquet --set-name all
```

对所有数据集和 split，可以依次运行：

```bash
for dataset in mini1 mini2 final; do
  for split in train validation; do
    python tools/mom_features.py --data-path data/${dataset}/${split}.parquet
    python tools/feat_ana.py --data-path data/${dataset}/${split}.parquet
    python tools/strategy_ver2.py --data-path data/${dataset}/${split}.parquet --set-name all
  done
done
```

运行完成后，重点查看：

```text
data/output/feat_ana/{dataset}/{split}/analysis_report.md
data/output/feat_ana/{dataset}/{split}/selected_feature_sets.csv
data/output/strategy_ver2/{dataset}/{split}/manifest.csv
data/output/backtest/strategy_ver2/{dataset}/{split}/metrics_summary.csv
```

## 5. 注意事项

- 当前数据文件中的 validation split 命名为 `validation.parquet`，输出目录也使用 `validation`。
- `strategy_ver2.py` 是研究脚本，不会替换根目录的 event-driven `strategy.py`。
- `strategy_ver2.py` 默认会覆盖同一 `{dataset}/{split}/{set_name}` 下已有 score、weights 和 backtest 输出。
- 如果运行时看到 pyarrow 的 CPU info warning，通常是 sandbox 下读取系统 CPU 信息失败，不影响 parquet 读取和结果生成。

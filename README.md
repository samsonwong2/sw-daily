# sw-daily

申万行业指数日更：行业信息、二级行业日线、独立的 qlib 数据集、相关性选池、池子审查，以及冻结 HMM 的状态转移验证。后面的 HTML 和操作清单按同样的命令结构再加。

Qlib 不随 `pip` 安装。`dump_bin.py` 需要本机已有的 qlib；默认解释器找不到 `qlib.utils` 时，会改用 `paths.qlib_scripts_dir` 指向的源码树。

## Install

```bash
pip install -e .
cp config.env.example config.env
cp configs/sw_daily.json.example configs/sw_daily.json
```

`config.env` 里设 `PY`。路径写在 `configs/sw_daily.json` 的 `paths` 里，空字符串沿用默认值：

| key | 默认 |
| --- | --- |
| `info_dir` | `~/temp/sw` |
| `qlib_dir` | `~/data/qlib_data/sw_index_data` |
| `qlib_scripts_dir` | `~/python/qlib/scripts` |
| `log_dir` | 项目下的 `logs/` |

没有这份 json 也可以跑，程序直接用上表的默认路径。

## Daily order

按这个顺序跑。`pool` 读 qlib 里的二级行业行情，`cluster-review` 和 `regime` 读 `pool` 写出的入选表。没装 `pip install -e .` 时，把 `sw-daily` 换成 `python run.py` 即可。

```bash
sw-daily etl info
sw-daily etl history
sw-daily etl daily
sw-daily pool
sw-daily cluster-review
sw-daily regime
```

对应的 etf-daily 命令是 `etf-daily pool`、`etf-daily cluster-review`、`etf-daily regime --skip-rebuild --skip-html`。申万这边没有手工池、反转标签重建和 HTML，所以 `regime` 默认就是那条跳过重建和画图的链路。

首次要先跑 `etl info` 和 `etl history`。之后每个交易日只跑 `etl daily`，池子和状态按需要重跑。

`history` 已有 CSV 的代码会跳过，`--force` 才重下。`daily` 当天已有 bar 时不覆盖，`--force-today` 才替换。也可以指定日期：

```bash
sw-daily etl daily --date 2026-10-07 --force-today
```

每日 shell 等价于 `etl daily`，额外参数原样传下去：

```bash
./scripts/daily_sw_update.sh
./scripts/daily_sw_update.sh --force-today
PY=/home/huangtuo/python/envs/py312/bin/python ./scripts/daily_sw_update.sh
```

`etl daily` 和 `etl history` 写完 CSV 后都会调用 qlib 的 `dump_bin.py dump_all`，刷新 `qlib_dir` 下的 bin 数据。行情 CSV 在 `qlib_dir/csv`。

### `sw-daily pool`

对应 `etf-daily pool`。宇宙是 `etl info` 写出的二级行业表，行情是 `qlib_dir` 里的收盘和成交量，所以先跑 `etl history`。

对最近 252 个交易日的收益做 Ward 层级聚类（距离 = 1 - 相关系数，阈值 0.40）。簇内按年化 Sharpe 和 12-1 动量各一半打分，每个簇先留一名代表，再用更严的簇内相关阈值去掉近克隆。历史不足 120 个交易日的行业不进聚类。ETF 那边的基金类型过滤、成立日截止、手工白名单和黑名单在这里没有。

```bash
sw-daily pool
sw-daily pool --start-date 2020-01-01 --end-date 2026-08-31
```

`--start-date` / `--end-date` 是从 qlib 取行情的窗口，默认就是上面这一对。输出在 `info_dir/pool/`：

| 文件 | 内容 |
| --- | --- |
| `sw_cluster_mapping.csv` | 全部二级行业：簇号、是否入选、原因、上级行业 |
| `sw_cluster_mapping_selected.csv` | 只保留入选行业，后面的审查和 regime 都读这份 |
| `sw_cluster_mapping_selected_metadata.json` | 阈值、窗口和入选数量 |
| `dendrogram_selected_reps.png` / `.svg` | 谱系图，入选行业的叶子带 `_x` |

### `sw-daily cluster-review`

对应 `etf-daily cluster-review`。读上一节的两份 CSV，用截至审查日的 qlib 行情看入选质量。

两类问题会写进报告。漏族：规模至少为 2 的簇里没有入选代表，并标出该簇近 20 日最强的行业。代表落后：入选行业不是簇内近 20 日最强，且落后幅度达到 `--min-regret`（默认 1%）。漏掉的簇还会列出树上最近的几个已选簇。ETF 那边的手工池对比、白名单原因和扩展 audit 没有移植。

```bash
sw-daily cluster-review
sw-daily cluster-review --future-end 2026-10-07 --windows 5,20,60 --min-regret 0.01 --min-cluster-n 2 --top-n 15
```

`--future-end` 是收益锚点，默认今天。`--windows` 是回看的交易日窗口，报告正文用其中的 20 日。`--top-n` 是 Markdown 表里展示的行数。报告在 `info_dir/pool/review/daily_YYYYMMDD/`：

| 文件 | 内容 |
| --- | --- |
| `daily_cluster_review_YYYYMMDD.md` | 摘要、漏族、代表落后、替换建议 |
| `missed_clusters.csv` | 全部漏族 |
| `rep_lag.csv` | 全部代表落后 |

### `sw-daily regime`

对应 `etf-daily regime --skip-rebuild --skip-html`。宇宙是 `sw_cluster_mapping_selected.csv`，行情仍从 `qlib_dir` 读取，起点会在 `--eval-start` 之前再多取约 400 个交易日供模型训练。

每个月用该月之前的数据冻结一个 3 状态 student-t HMM（训练窗 252 日）。之后每个交易日只做前向滤波：预测分布双侧分位（q*=0.90）之外的跳变记为 OBSERVE。逐月用更早月份的样本做 logistic 交叉拟合，样本够再做 isotonic 校准，过 FDR 阈值的向上跳变升为 CONFIRM。当日跌幅达到或超过 7%、且模型没有给出向下跳变时，补一条极端下跌覆盖。

不跑 etf-daily 的后两步：不重建反转标签，不画全池 HTML，也不做盘中实时快照。

```bash
sw-daily regime
sw-daily regime --as-of 2026-10-07 --eval-start 2024-06-01 --horizon 10 --jobs 8
```

`--as-of` 是信号截止日，默认今天。`--eval-start` 是开始打分的日期，默认 `2024-06-01`。`--horizon` 是趋势标签往后看的交易日，默认 10。`--jobs` 是按行业并行拟合的进程数。同一天已经写进 `signals_oos.csv` 时会直接跳过；要整段重算加 `--no-resume`，不用模型缓存加 `--no-cache`。

决策包在 `info_dir/regime/decision_pack/`，冻结模型在 `info_dir/regime/model_cache/`：

| 文件 | 内容 |
| --- | --- |
| `signals_oos.csv` | 每个行业每个交易日的状态、跳变和 CONFIRM / OBSERVE / NONE |
| `shards/YYYY-MM.signals.csv` | 按月增量，配合同名 `.done` |
| `go_nogo.json` | 覆盖率、相对 z 分数基线的提升、校准和假阳性 |
| `REGIME_TRANSITION_VALIDATION.md` | 上面这项检查的可读摘要 |
| `event_metrics.csv` | CONFIRM（没有则用 OBSERVE）和趋势事件的匹配 |
| `run_manifest.json` | 宇宙、窗口、配置哈希 |

`go_nogo` 为 PASS 表示分位跳变覆盖率落在 8%–35%，校准和假阳性也过线。CONFIRM 相对基线没有稳定超额时会被抑制，这项本身不单独把结果打成失败。

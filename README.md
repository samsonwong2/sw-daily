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
sw-daily adaptive --as-of 2026-10-08
sw-daily listing --as-of 2026-10-08
sw-daily hrp --as-of 2026-10-08
sw-daily hrp --dist-t 0.8 --as-of 2026-10-08
sw-daily rules7 --listing-dir ~/temp/sw/adaptive/20261008_from_listing
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

### `sw-daily adaptive`

对应 `etf-daily adaptive --as-of`。每个入选行业在训练期（默认图窗起点之前）从 7 种因果分段法里选训练分最高的一种，样本外冻结该方法，只在上涨态做多：进入上涨买入，离开上涨卖出。对照是同一套交易规则打在固定的 `hybrid_ma_adx` 分段上。

不画 ETF 那边的 HRP 路由、恢复上涨诊断、公平路径锚和票卡。

```bash
sw-daily adaptive --as-of 2026-09-30
sw-daily adaptive --as-of 2026-09-30 --start-date 2026-04-01 --train-cutoff 2026-04-01 --jobs 8
```

同一训练截止日会复用已冻结的方法，缓存放在 `info_dir/adaptive/_train_cache`，换一个 `--as-of` 也不会重训。当天目录里已经有同一图窗的 HTML 时直接跳过。要重选方法并重画，加 `--retrain`。只跑部分行业用 `--code 801010 --code 801011`。行情在 `--start-date` 之前就结束的行业（例如 2024-06 停更的二级行业）仍会出图，窗口改成它自己的历史，日志里记一行 `[INFO]`，不算失败。

### `sw-daily listing`

对应 `etf-daily listing --as-of`。不重新选分段法，直接用当天 `adaptive` 目录里冻结的 `configs/{code}.json`。每只行业的图从它自己的第一根 K 线画到 `--as-of`，交易规则仍是进入上涨买入、离开上涨卖出。

```bash
sw-daily listing --as-of 2026-09-30
sw-daily listing --as-of 2026-09-30 --code 801010
```

先跑过 `sw-daily adaptive --as-of 2026-09-30`，否则找不到冻结配置。`sw-daily listing` 和 `sw-daily adaptive` 画的是同一套 12 图（价格到图11，加图12）。空日历日会丢掉。两根 K 线如果隔了 30 天以上，图从后面那段的第一天画起，例如煤炭开采从 2021-12-13 而不是 2014-02-21。文件名带中文名和实际起点，例如 `regime_transition_801951_煤炭开采_20211213_20260930_adaptive.html`。`sw-daily regime` 打分时同样把长缺口之前的行情去掉，避免停牌段被前值填平。输出在 `info_dir/adaptive/YYYYMMDD_from_listing/`，还有 `listing_dates.csv` 和 `batch_summary.csv`。行业起点缓存在 `info_dir/adaptive/_listing_dates.csv`。

如果已经有更早一天的 `*_from_listing`，收盘价没变就整页复制，日志是 `[REUSE]`。只多出一根 K 线时，在昨天的 12 图上补上这一根，并重画图12，日志是 `[INCR]`。中间缺了不止一根，或对不上昨日收盘价，才整页重画。当天目录里已经有同一终点的 HTML 会跳过。默认 `--jobs 8`。`--no-incremental` 强制全部重画。

### `sw-daily hrp`

对应 `etf-daily hrp --as-of`。对入选池最近 252 个自然日附近的收益做 Ward 聚类（距离 = 1 - 相关系数，默认阈值 0.40）。簇内两两相关低于 `--min-corr`（默认 0.55）的成员会被拆开。每个簇取相关中心（medoid），涨组和跌组再各取区间涨跌绝对值最大的一只，幅度不到 5% 则标成没有明确方向。另外用 skfolio 的 HRP-CVaR 画两张参考谱系图。

不读持仓表，也不做债券/海外/境内的域划分。

```bash
sw-daily hrp --as-of 2026-09-30
sw-daily hrp --as-of 2026-09-30 --dist-t 0.80 --min-corr 0.55
```

输出在 `info_dir/hrp/YYYYMMDD/`：`hrp_dendrogram_YYYYMMDD.html` 和 `cluster_representatives_YYYYMMDD.csv`。`--dist-t` 不是 0.40 时文件名会带 `_d080` 这样的后缀。

### `sw-daily rules7`

对应 `etf-daily rules7 --listing-dir`。对一个 `*_from_listing` 目录里的行业，用同一套买卖规则出当天清单：aux_edge 买点、可选 B1、快形态拦截、图9 或图8 卖点、G1 之后不再按轨道卖、以及从成交价起算的回落止损。申万行业没有 7 只 ETF 的专属规则，每只按自身历史自动分成平稳震荡、慢趋势、高波动主题、长期下跌或周期，历史不足一年的单独一类。

行情读 sw 的 qlib，不从 HTML 里拆 K 线。gap 的锚定仍是沪深300（`SH510300`），从 `~/data/qlib_data/all_fund_data` 读取。同簇当天有多只买入信号时，只标一只推荐买入，聚类表默认用 `info_dir/hrp/YYYYMMDD/cluster_representatives_YYYYMMDD.csv`。

```bash
sw-daily rules7 --listing-dir ~/temp/sw/adaptive/20260930_from_listing
```

`--as-of` 省略时从目录名里的 8 位日期取。输出写回该目录：`rules7_checklist_YYYYMMDD.csv` 和同名 markdown。最后一根行情不是 as-of 的行业（例如已经退市）记进 `rules7_checklist_YYYYMMDD_skipped.csv`。

输出在 `info_dir/adaptive/YYYYMMDDall_adaptive/`：`*_adaptive.html`、`configs/{code}.json`、`batch_summary.csv`、`trades/`、目录里的 `README.md`（平均 edge、胜率和相对固定方法的差）。

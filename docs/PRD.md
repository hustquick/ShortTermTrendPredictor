# ShortTermTrendPredictor PRD

版本：v1.0  
日期：2026-05-19  
适用范围：当前 `ShortTermTrendPredictor` 仓库的理解、维护、重构和后续迭代。

## 1. 背景与目标

`ShortTermTrendPredictor` 是一个针对 BTC/USDT 1 分钟 K 线的短周期方向预测系统。系统核心目标不是提高全量样本准确率，而是识别少量“值得通知和跟踪”的高置信方向信号，并通过实时验证持续评估策略质量。

当前项目已经从单一模型预测演进为多策略实时观察系统，包含：

- 真实市场数据接入与本地缓存。
- 未来 10 分钟涨跌二分类标签。
- LightGBM、XGBoost、CatBoost 集成模型。
- 双子模型输出：`up_signal_probability`、`down_signal_probability` 和 `direction_edge`。
- 多策略过滤器。
- 强制方向观察样本记录。
- 10 分钟后自动验证。
- 企业微信高置信通知。
- 自学习通知门控。
- Kronos 可选确认/领先策略。
- 每策略独立 CSV 记录。
- 每策略独立实时图表窗口。
- AlgoTrading 风格最小核心框架层。

本 PRD 的目标是把当前系统行为和重构目标写清楚，避免后续继续叠加补丁导致职责混乱。

## 2. 产品定位

### 2.1 产品一句话

基于 BTC/USDT 1m K 线，持续训练最近局部市场状态，输出未来 10 分钟涨跌方向信号，并只对满足策略和自学习门控的高置信信号进行通知。

### 2.2 核心用户

- 策略研发者：分析不同策略的实时预测、验证和胜率。
- 量化实验者：快速试验新的短周期过滤逻辑。
- 实时观察者：通过企业微信群和本地实时图表观察当前信号。

### 2.3 非目标

- 不做自动下单。
- 不承诺投资收益。
- 不以全局准确率作为核心指标。
- 不把历史回测结果批量写入实时预测 CSV。
- 不把低置信强制观察预测当成正式交易信号。

### 2.4 启动性能要求

实时策略进程重启不应每次重复执行所有重型任务。当前性能优化原则：

- 已保存主模型存在时，启动首轮优先复用本地模型，重训延后到 30 分钟周期。
- 历史匹配 walk-forward 样本池允许 30 分钟缓存复用。
- Kronos 只在双子模型产生足够方向候选时运行。

## 3. 当前运行模式

系统入口是 `main.py`。

| 模式 | 命令 | 用途 |
| --- | --- | --- |
| `train` | `python main.py --mode train` | 训练实时模型 |
| `realtime` | `python main.py --mode realtime` | 传统单模型实时预测 |
| `realtime_strategies` | `python main.py --mode realtime_strategies` | 多策略实时预测、记录、验证、通知 |
| `training_backtest` | `python main.py --mode training_backtest` | 允许未来数据泄露的训练回测 |
| `strict_backtest` | `python main.py --mode strict_backtest` | 严格时序 walk-forward 回测 |
| `tune_dual_model` | `python main.py --mode tune_dual_model` | 双子模型调参 |

多策略实时模式常用参数：

```bash
python -u main.py --mode realtime_strategies --observe-all --live-chart
```

参数语义：

- `--observe-all`：运行全部观察策略。
- `--strategies a,b,c`：指定策略集合。
- `--train-minutes 720`：指定训练和历史匹配窗口长度。
- `--once`：只运行一轮，便于测试。
- `--no-update-cache`：只使用本地 K 线缓存，不联网补齐。
- `--live-chart`：打开每策略独立 matplotlib 实时图表窗口。

严格回测已经接入同一套 core 流程：

- 使用 `FeaturePipeline` 构造特征。
- 使用和实时相同的多策略集合生成 `final_direction`。
- 使用和实时相同的生产质量门控与 `RiskGate` 判定正式信号。
- 使用回测内存版 `RollingLearningGate` 按时间推进更新自学习状态。
- 结果中 `is_valid_signal=True` 表示该行等价于实时模式的 `official_signals.csv` 正式信号。

默认严格回测使用训练窗口快速历史匹配池，以避免每次模型更新都重建多桶 walk-forward 历史池造成不可接受的运行时间。需要完全样本外历史匹配池时，可显式传入：

```bash
python main.py --mode strict_backtest --walk-forward-match-pool
```

## 3.1 最小 AlgoTrading 风格架构

实时策略模式按职责拆成以下核心对象：

| 层 | 文件 | 职责 |
| --- | --- | --- |
| DataFeed | `core/data_feed.py` | 加载 BTC/USDT 1m K 线和本地缓存 |
| FeaturePipeline | `core/feature_pipeline.py` | 把 K 线转换为模型特征 |
| AlphaModel | `core/alpha_model.py` | 加载、重训、预测 `up/down` 概率 |
| Strategy | `strategies/rules.py` | 将模型概率和特征转换成策略方向 |
| RiskGate | `core/risk_gate.py` | 判断信号是否具备正式通知资格 |
| OutputStore | `core/output_store.py` | 拆分写入所有预测和正式信号 |
| Notifier | `core/notifier.py` | 发送企业微信预测和验证通知 |
| Analyzer | `core/analyzer.py` | 面向正式信号统计胜率 |

架构目标不是直接提高预测准确率，而是让每个环节可审计、可替换、可独立回测，避免把低置信观察样本和正式信号混在一起评估。

## 4. 核心业务规则

### 4.1 预测对象

- 标的：BTC/USDT。
- K 线粒度：1 分钟。
- 预测周期：未来 10 分钟。
- 方向定义：
  - `up`：未来 10 分钟 close 高于当前 close。
  - `down`：未来 10 分钟 close 不高于当前 close，或低于中性灰区。

### 4.2 时间规则

- 内部所有核心逻辑必须使用 13 位 UTC 毫秒时间戳。
- 禁止使用 10 位秒级时间戳参与数据对齐、训练、验证。
- 展示、日志、CSV 时间统一为北京时间 `YYYY-MM-DD HH:MM:SS`。
- 未来价格必须通过 `current_timestamp + 10 * 60_000` 精确匹配，不允许用行号偏移替代。
- 最新 K 线以 `timestamp` 最大的已收盘 1m K 线为准。

### 4.3 训练窗口

- 默认训练数据：最近 48 小时 1m K 线。
- 实时策略可以通过 `--train-minutes` 临时缩短窗口做验证。
- 实时主模型默认每 30 分钟重训一次。
- 每 60 秒进行一轮实时预测和 pending 验证。

### 4.4 评估口径

系统存在两类预测记录：

- 正式信号：策略最终输出 `final_direction in {up, down}`，并通过自学习通知门控后才可通知。
- 强制观察预测：策略最终输出 `no_trade`，系统仍根据模型概率强制生成 `raw_direction` 记录和验证，用于积累样本与可视化，不应被当作正式信号。

高置信策略胜率应优先统计正式信号。强制观察预测可用于研发和校准，但不能混入正式策略胜率。

实时多策略模式的规范输出：

- `data/all_predictions.csv`：记录每个策略每轮预测，无论是否低置信、是否 `no_trade`、是否通知。
- `data/official_signals.csv`：只记录通过白名单、自学习和生产质量门槛的正式通知信号。

## 5. 数据源与缓存

### 5.1 数据源优先级

当前数据下载模块 `data_download.py` 支持：

- Binance REST API。
- Binance Vision 备用数据源。
- OKX BTC-USDT 备用数据源。
- 本地 CSV 缓存。

请求配置：

- `verify=False`，用于兼容 Mac SSL 问题。
- Header 携带 `X-MBX-APIKEY`，API Key 从环境变量读取。
- 不硬编码 API Key/Secret。

### 5.2 标准 K 线字段

标准 K 线 DataFrame 字段：

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `quote_asset_volume`
- `number_of_trades`
- `taker_buy_base_volume`
- `taker_buy_quote_volume`
- `close_time`

### 5.3 缓存文件

主历史缓存：

```text
data/BTCUSDT_1m_history.csv
```

缓存要求：

- 去重。
- 按 timestamp 升序。
- 保留 13 位毫秒时间戳。
- 不允许出现 1970 年异常时间。

## 6. 特征工程

特征工程由 `features.py` 负责。所有特征只能使用当前和过去数据。

主要特征族：

- 收益率：`ret_1`、`ret_2`、`ret_3`、`ret_5`、`ret_10`、`ret_15`、`ret_20`、`ret_30`。
- 均线/EMA：`ma_n_ratio`、`ema_n_ratio`、`ema_n_slope`、`ema_5_20_diff`、`ema_10_30_diff`、`ema_20_60_diff`。
- MACD：`macd`、`macd_signal`、`macd_hist`、`macd_hist_diff`。
- RSI：`rsi_6`、`rsi_14`。
- KDJ：`kdj_k`、`kdj_d`、`kdj_j`。
- 布林带：`boll_position`、`boll_width`。
- 波动率：`volatility_5`、`volatility_10`、`volatility_30`、`atr_14`。
- K 线结构：`body_ratio`、`upper_shadow_ratio`、`lower_shadow_ratio`、`close_position`。
- 成交量：`volume_ratio_n`、`volume_zscore`、`volume_change`。
- 主动买入：`taker_buy_ratio`、`taker_buy_ratio_change`、`taker_buy_ratio_ma_5`、`taker_buy_ratio_ma_10`。
- 趋势一致性：`trend_5`、`trend_10`、`trend_15`、`trend_agreement`。

标签构建：

- `future_price` 通过未来 10 分钟时间戳精确匹配。
- `future_return = future_price / close - 1`。
- 若开启中性灰区，则小于阈值的样本设为 NaN，训练时删除。

## 7. 模型训练

### 7.1 模型组成

训练模块为 `trainer.py`。

系统使用三个模型集成：

- LightGBM
- XGBoost
- CatBoost

当前核心形态是双子模型：

- up 子模型：输出 `up_signal_probability`。
- down 子模型：输出 `down_signal_probability`。
- `direction_edge = up_signal_probability - down_signal_probability`。

### 7.2 权重

训练样本使用时间指数衰减权重：

- 越新的样本权重越大。
- 目标是让模型更贴近最近市场局部状态。

### 7.3 模型文件

模型输出路径：

```text
models/dual_backtest_ensemble_model.pkl
models/dual_model_params.json
```

调参报告：

```text
data/dual_model_tuning_report.csv
data/strict_param_search_report.csv
```

### 7.4 训练风险

当前系统有意追求短期局部拟合，因此必须区分：

- 训练回测：允许泄露，只看拟合能力。
- 严格回测：禁止泄露，才代表可验证表现。
- 实时验证：真正线上表现。

## 8. 策略系统

策略接口位于 `strategies/base.py`：

```python
StrategyDecision(direction, confidence, reason)
```

方向只能是：

- `up`
- `down`
- `no_trade`

### 8.1 当前 realtime 策略集合

`realtime_strategy_runner.py` 当前可运行策略：

- `short_momentum`
- `adaptive_dual`
- `relaxed_scenario`
- `historical_match`
- `historical_match_long`
- `historical_match_short`
- `kronos_confirm`
- `kronos_lead`
- `finstar_scenario`

`run_realtime_strategies.py` 默认正式策略：

- `historical_match_short`

`--observe-all` 运行上述全部策略。

### 8.2 统一陷阱过滤

做空反弹陷阱拒绝：

- `ret_10 > 0 且 macd_hist > 0`
- `ret_30 > 0 且 macd_hist > 0`
- `close_position < 0.02`
- `close_position > 0.98 且 ret_10 > 0`
- `rsi_14 < 45 且 boll_position <= 0.15`

做多追高陷阱拒绝：

- `rsi_14 > 80`
- `boll_position > 0.84`
- `close_position > 0.98`

### 8.3 历史匹配策略

历史匹配策略使用 walk-forward 样本外概率池：

- 每个历史样本只能使用该时间点之前训练出的模型概率。
- 禁止当前模型回头给整段历史打分。
- 目标是减少 `success_rate=1.0000` 但实盘失败的过拟合乐观偏差。
- walk-forward 历史匹配池构建成本高，因为内部会按时间桶重复训练模型。
- 构建结果缓存到 `data/historical_match_walk_forward_cache.pkl`，默认 30 分钟有效。

### 8.4 Kronos 策略

Kronos 是可选确认/领先模型。

当前形态：

- `kronos_confirm`：双子模型有候选方向时，要求 Kronos 方向一致。
- `kronos_lead`：Kronos 可作为领先方向，但不能被双子模型强烈反向否定。

Kronos 风险：

- 加载和预测耗时高。
- 依赖本地模型与 tokenizer 缓存。
- 实时循环中应允许失败，不得阻塞整体系统。
- 只有当双子模型 `abs(direction_edge)` 与模型置信度达到最低运行门槛时才调用 Kronos，否则返回 skipped 结果。

### 8.5 FinStar 场景策略

`finstar_scenario` 将市场状态、历史相似样本和可选 Kronos 融合，生成场景判断。

当前要求：

- 不能只凭模型高置信放行。
- 需结合历史相似验证。

## 9. 实时多策略流程

多策略实时核心在 `realtime_strategy_runner.py`。

每一轮流程：

1. 拉取或读取最近 K 线。
2. 回填到期 pending 预测。
3. 判断是否需要重训模型。
4. 构造最新特征。
5. 如需要，刷新历史匹配样本池。
6. 如需要，运行 Kronos 预测。
7. 调用双子模型输出概率。
8. 对每个策略调用 `decide()`。
9. 无论策略是否 `no_trade`，都记录一条观察预测。
10. 对正式信号执行自学习通知门控。
11. 写入 pending。
12. 写入聚合 CSV 和单策略 CSV。
13. 生成静态 PNG 快照。
14. 如启用 `--live-chart`，刷新每策略实时窗口。

## 10. 强制观察预测

为了提高样本量，系统对每个策略每轮都记录预测。

规则：

- 如果策略输出 `up/down`，则：
  - `raw_direction = final_direction = up/down`
- 如果策略输出 `no_trade`，则：
  - `final_direction = no_trade`
  - `raw_direction = up` 当 `up_signal_probability >= down_signal_probability`
  - 否则 `raw_direction = down`

注意：

- `raw_direction` 用于观察和到期验证。
- `final_direction` 用于判断是否是正式策略信号。
- `notify_enabled` 只允许正式信号进入企业微信通知。

## 11. 自学习通知门控

自学习模块为 `strategy_learning.py`。

目标：

- 根据最近验证结果，动态决定某策略某方向是否继续通知。
- 低胜率方向停止通知，但仍继续记录和验证。

状态维度：

```text
strategy + direction
```

状态类型：

- `explore`：样本不足，只观察记录，不通知。
- `active`：滚动样本不少于 10 条，且胜率达到 70%，允许通知。
- `probation`：胜率中间状态，只观察记录，不通知。
- `disabled`：胜率低于 55%，只观察记录，不通知。
- `feature_blocked`：近期同类特征签名错误次数过多。

状态文件：

```text
data/strategy_learning_state.json
```

重要配置：

- `STRATEGY_LEARNING_ROLLING_WINDOW`
- `STRATEGY_LEARNING_MIN_SAMPLES`
- `STRATEGY_LEARNING_DISABLE_WIN_RATE`
- `STRATEGY_LEARNING_ENABLE_WIN_RATE`
- `STRATEGY_LEARNING_FEATURE_BLOCK_MIN_ERRORS`

当前通知原则：真正的正式信号以 `notify_enabled=True` 为准，而不是简单看 `final_direction in {up, down}`。策略可以继续给出方向用于观察，但未通过生产白名单、自学习门控和生产质量门槛时不应进入企业微信通知和正式信号胜率统计。当前生产白名单保留 `historical_match`、`historical_match_short`、`adaptive_dual`、`kronos_confirm`、`kronos_lead`；其中 adaptive 和 Kronos 额外收紧正式通知门槛。

## 12. 输出文件

### 12.1 传统实时预测

```text
data/predictions.csv
```

当前多策略模式也会写入该聚合文件，但字段已扩展为策略预测字段。

### 12.2 多策略聚合记录

```text
data/strategy_predictions.csv
data/strategy_predictions_latest.csv
data/pending_strategy_signals.jsonl
data/validated_strategy_signals.csv
```

### 12.3 每策略独立记录

```text
data/strategy_predictions/{strategy}.csv
```

字段：

- `prediction_id`
- `timestamp`
- `strategy`
- `current_price`
- `raw_direction`
- `final_direction`
- `confidence`
- `reason`
- `up_signal_probability`
- `down_signal_probability`
- `direction_edge`
- `validation_timestamp`
- `validation_status`
- `actual_direction`
- `future_price`
- `is_correct`
- `notify_enabled`

### 12.4 图表文件

```text
data/strategy_charts/{strategy}.png
```

PNG 是静态快照，不是实时窗口。

## 13. 企业微信通知

通知模块为 `strategy_notifier.py`。

通知类型：

- 预测通知。
- 10 分钟后验证通知。

通知触发条件：

- 策略最终方向是 `up/down`。
- 策略在 `OFFICIAL_SIGNAL_STRATEGY_ALLOWLIST` 中。
- 自学习门控返回 `notify=True`。

强制观察预测不通知。

当前生产白名单：

- `historical_match`
- `historical_match_short`
- `adaptive_dual`
- `kronos_confirm`
- `kronos_lead`

额外生产质量门槛：

- `adaptive_dual`：`confidence >= 0.75`、`abs(direction_edge) >= 0.50`，且同方向必须有历史匹配策略或高置信 Kronos 策略二次确认。
- `kronos_confirm` / `kronos_lead`：只允许做多正式通知，且 `kronos_conf >= 0.10`。
- `kronos_lead`：允许领先，但不能逆着明显双子模型方向；当前反向 edge 容忍度为 `0.05`。
- Kronos 做空通知当前禁用；做空结果继续记录和验证。

未通过额外门槛的策略方向只记录和验证，不推送企业微信。

当前风险：

- Webhook 写在 `config.py` 中，后续应迁移到环境变量或本地未跟踪配置。
- 通知格式与策略展示名有耦合，新增策略时容易漏改展示名。

## 14. 实时图表需求

启动参数：

```bash
python -u main.py --mode realtime_strategies --observe-all --live-chart
```

要求：

- 每个策略一个独立 matplotlib 窗口。
- 每个窗口两个子图：
  - 上图：最近 30 分钟滚动双 Y 图。
  - 下图：置信度分桶准确率直方图。
- 上图右边界固定为最新预测时间。
- 上图左 Y 轴：BTC 价格。
- 上图右 Y 轴：confidence，范围 0.0 到 1.0。
- grid 使用置信度轴网格线，不使用价格轴网格线。
- marker 语义：
  - `raw_direction=up`：上三角。
  - `raw_direction=down`：下三角。
  - `notify_enabled=True`：实心三角，表示正式通知信号。
  - `notify_enabled=False`：空心半透明三角，表示观察预测或未通过门控的策略方向。
  - 灰色：未验证。
  - 绿色：验证正确。
  - 红色：验证错误。
- 每个置信度点用虚线连接到置信度轴基线。

直方图：

- 横轴：置信度分桶，默认 0.1 宽度。
- 纵轴：该置信度区间内已验证正式通知信号准确率。
- 每个柱子显示准确率和样本数。

## 15. 回测需求

### 15.1 训练回测

允许未来数据泄露。

用途：

- 观察模型对最近局部市场的拟合能力。
- 不作为真实有效表现。

### 15.2 严格时序回测

必须满足：

- 预测点只能使用该时间点之前的数据训练模型。
- 模型按时间滚动更新。
- 预测频率固定为 1 分钟。
- 每个预测用未来 10 分钟真实价格验证。
- 不允许任何未来数据泄露。

### 15.3 策略回测未来要求

后续重构应把实时多策略逻辑与严格回测逻辑统一，做到：

- 同一策略接口可同时用于实时和历史 walk-forward 回测。
- 同一套输出 schema 可用于 CSV、通知和图表。
- 正式信号与强制观察预测分开统计。

## 16. 当前主要问题

### 16.1 职责混杂

`realtime_strategy_runner.py` 同时负责：

- 策略注册。
- CSV schema。
- pending 存储。
- 验证回填。
- 通知调用。
- 图表绘制。
- 自学习调用。
- 实时训练循环。

这是当前最大的重构对象。

### 16.2 输出文件语义混乱

`data/predictions.csv` 起初是单模型实时预测文件，现在多策略模式也写入扩展字段，容易破坏原始固定字段预期。

建议：

- 传统实时预测继续使用 `data/predictions.csv`。
- 多策略预测只使用 `data/strategy_predictions.csv` 和单策略目录。
- 如果必须兼容，明确 schema 版本。

### 16.3 正式信号与观察预测容易混淆

当前为提高样本量，每策略每轮都会强制生成 `raw_direction`。

风险：

- 用户可能把空心观察预测误解为正式信号。
- 胜率统计可能误把观察预测混入高置信策略信号。

建议：

- 所有统计 API 都明确 `scope=official|observed|all`。
- 图表和 CSV 均保留 `raw_direction`、`final_direction`、`notify_enabled`。

### 16.4 配置集中但缺少分层

`config.py` 同时包含：

- 项目路径。
- 数据源。
- 训练参数。
- 策略阈值。
- 通知 webhook。
- Kronos 配置。
- 自学习配置。

建议拆为：

- `settings/data.py`
- `settings/model.py`
- `settings/strategy.py`
- `settings/notification.py`
- `settings/runtime.py`

### 16.5 状态存储不适合长期运行

当前 pending 使用 JSONL，CSV 使用直接读写和 upsert 重写。

风险：

- 长期运行文件变大后性能下降。
- 并发进程可能写冲突。
- 异常退出可能造成中间状态不一致。

建议后续迁移到 SQLite：

- `predictions`
- `validations`
- `strategy_states`
- `model_runs`
- `notifications`

### 16.6 图表和核心策略耦合

图表绘制直接放在 runner 内部。

建议：

- 拆出 `strategy_charts.py`。
- runner 只发事件或调用独立 renderer。

## 17. 重构目标

### 17.1 第一阶段：整理边界

目标是不改变策略行为，只拆职责。

建议模块：

```text
runtime/
  strategy_registry.py
  strategy_loop.py
  prediction_store.py
  validation_service.py
  notification_service.py
  chart_service.py
  learning_gate.py
```

验收：

- 现有命令兼容。
- CSV 输出字段不变。
- 通知行为不变。
- 图表行为不变。
- `py_compile` 和一次 `--once --no-update-cache` 通过。

### 17.2 第二阶段：统一数据模型

引入明确实体：

- `MarketCandle`
- `ModelPrediction`
- `StrategyDecision`
- `RecordedPrediction`
- `ValidationResult`
- `NotificationDecision`

验收：

- 不再通过散乱 dict 在函数间传递核心业务对象。
- CSV writer 只负责序列化。
- 图表只依赖 `RecordedPrediction`。

### 17.3 第三阶段：SQLite 状态存储

迁移：

- pending JSONL。
- validated CSV。
- per-strategy CSV 可继续作为导出物。

验收：

- 进程重启后 pending 不丢。
- 同一 `prediction_id` 幂等 upsert。
- 查询最近 30 分钟图表数据不需要全量读 CSV。

### 17.4 第四阶段：严格回测复用实时策略

目标：

- 实时策略和回测策略不再两套逻辑。
- 所有策略可直接 walk-forward 回测。
- 统计结果区分 official 和 observed。

验收：

- 同一策略在实时和回测中产生一致决策。
- 回测输出可复用图表和统计模块。

## 18. 关键验收标准

### 18.1 实时运行

命令：

```bash
python -u main.py --mode realtime_strategies --observe-all --live-chart
```

验收：

- 每 60 秒执行一轮。
- 每个策略每轮有一条单策略 CSV 记录。
- `no_trade` 也有强制 `raw_direction`。
- 正式信号才可能通知。
- pending 到期后自动验证。
- 图表窗口持续刷新。

### 18.2 时间正确性

验收：

- 所有内部 timestamp 为 13 位毫秒。
- CSV 展示时间为北京时间。
- 验证价格通过精确未来 10 分钟 timestamp 匹配。
- 不出现 1970 时间。

### 18.3 通知正确性

验收：

- `final_direction=no_trade` 不通知。
- 自学习 disabled 不通知。
- 预测通知和验证通知成对出现。
- 验证通知包含累计策略准确率。

### 18.4 图表正确性

验收：

- 每策略一个窗口。
- 上图显示最近 30 分钟。
- 横轴最右侧是最新预测时间。
- 置信度网格线来自右轴。
- 上下三角区分方向。
- 实心/空心区分正式信号和观察预测。
- 下图显示已验证样本的置信度分桶准确率。

## 19. 推荐优先级

P0：

- 保持实时策略稳定运行。
- 保持时间戳和验证无泄露。
- 明确 official vs observed 统计口径。
- 拆出图表模块。

P1：

- 拆分 `realtime_strategy_runner.py`。
- 引入实体 dataclass。
- 将 webhook 移到环境变量。
- 增加策略级统计命令。

P2：

- SQLite 状态存储。
- 回测和实时共用策略接口。
- 图表支持策略筛选和窗口关闭自动处理。
- 增加测试覆盖。

## 20. 后续测试建议

最小测试集：

```bash
source ~/btc_env/bin/activate
python -m py_compile config.py data_download.py features.py trainer.py realtime_strategy_runner.py main.py
python main.py --mode realtime_strategies --strategies short_momentum,adaptive_dual --train-minutes 720 --once --no-update-cache
```

建议新增自动化测试：

- timestamp 秒级输入拒绝。
- feature 不使用未来数据。
- label 通过 timestamp 精确匹配。
- `no_trade` 强制方向逻辑。
- `notify_enabled` 只对正式信号开放。
- pending 到期回填。
- 单策略 CSV upsert 幂等。
- 置信度分桶准确率统计。

## 21. 术语表

- `raw_direction`：系统用于验证的强制方向。
- `final_direction`：策略最终输出方向，可能是 `no_trade`。
- `official signal`：`final_direction in {up, down}` 且通过通知门控的正式信号。
- `observed prediction`：为了积累样本强制记录的观察预测。
- `confidence`：策略内部置信度，不一定等于 up/down 模型概率最大值。
- `direction_edge`：`up_signal_probability - down_signal_probability`。
- `pending`：已预测但未到 10 分钟验证时间的记录。
- `validated`：已经匹配未来价格并完成对错判定的记录。

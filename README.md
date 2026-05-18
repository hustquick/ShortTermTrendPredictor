# ShortTermTrendPredictor

BTC/USDT 1 分钟 K 线未来 10 分钟涨跌二分类预测项目。

系统采用双回测机制：

- 训练回测：允许未来数据参与训练，故意过拟合最近 48 小时局部趋势，只用于观察拟合能力。
- 严格验证回测：按时间滚动更新模型和预测，每个模型只使用更新时间点之前的数据，预测点使用最临近的已训练模型，结果才代表真实有效表现。

## 项目结构

```text
ShortTermTrendPredictor/
├── config.py
├── data_download.py
├── features.py
├── trainer.py
├── run_strategy.py
├── realtime.py
├── main.py
├── requirements.txt
└── README.md
```

## 核心规则

- 预测标的：BTC/USDT 1m K 线。
- 预测目标：未来 10 分钟 close 是否高于当前 close。
- 高置信基础阈值：`up_probability >= 0.80` 或 `up_probability <= 0.20`。
- 连续确认过滤：做多要求最近 3 分钟内 3 次 `up_probability >= 0.95`，且当前 `up_probability >= 0.95`、`close_position > 0.70`、`ret_5 < 0`、`ret_30 > 0`、`rsi_14 < 60`；若 `body_ratio > 0.95` 且 `close_position > 0.99` 则跳过。
- 连续确认过滤：做空要求最近 15 分钟内至少 10 次 `up_probability <= 0.25`，且当前 `up_probability <= 0.20`、`close_position > 0.70`、`macd_hist < 0`、`0 < ret_30 < 0.0015`、`ret_10 > -0.0005`、`rsi_14 < 59`；若 `taker_buy_ratio >= 0.95`、`body_ratio >= 0.95` 且 `trend_agreement > 0` 则跳过，两次有效信号至少间隔 1 分钟。
- 当前实盘信号配置：启用之前的高置信做多 pullback 信号和当前高置信做空反弹失败信号。
- 训练标签灰区：训练时忽略未来 10 分钟涨跌幅绝对值小于 `0.03%` 的噪声样本。
- 评估口径：只统计高置信信号胜率，忽略全局准确率。
- 严格回测默认每 1 分钟检查一次信号，模型默认每 30 分钟滚动更新一次，模拟实时模式中“短期加密度、定期重训、连续确认过滤”的执行方式。
- 内部时间：仅使用币安 13 位 UTC 毫秒时间戳。
- 展示时间：日志和 CSV 统一输出东八区时间。
- API Key：从 `BINANCE_API_KEY` 环境变量读取，不硬编码。
- Mac SSL：请求使用 `verify=False` 并关闭相关 warning。

## 输出文件

实时预测只写入 `data/predictions.csv`，不会写入历史回测结果或批量回填数据。

字段顺序固定：

```csv
timestamp,current_price,future_price,predicted_direction,actual_direction,up_probability,confidence,is_valid_signal,is_correct
```

实时预测会先追加一条当前预测记录；到达未来 10 分钟后，再通过精确毫秒时间戳匹配真实 future price 并回填验证字段。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

可选：

```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

当前代码只需要公开 K 线接口，缺少 Key 时仍会携带空 `X-MBX-APIKEY` 请求头。

## 运行

训练最近 48 小时过拟合模型：

```bash
python3 main.py --mode train
```

第一阶段训练回测，允许未来数据泄露：

```bash
python3 main.py --mode training_backtest
```

第二阶段严格时序验证回测：

```bash
python3 main.py --mode strict_backtest --backtest-days 10 --step-minutes 1 --model-update-minutes 30
```

如果只想快速抽样，可以限制最大预测点数：

```bash
python3 main.py --mode strict_backtest --backtest-days 10 --step-minutes 1 --model-update-minutes 30 --max-steps 3000
```

启动实时预测：

```bash
python3 main.py --mode realtime
```

启动当前高胜率实时策略通知：

```bash
python3 main.py --mode realtime_strategies
```

当前默认只运行 `historical_match_short` 作为正式高置信策略。该策略在最新已验证样本中表现最好，`historical_match_long` 暂不作为正式信号；如果需要继续观察所有策略，可以运行：

```bash
python3 main.py --mode realtime_strategies --observe-all
```

`finstar_scenario` 已收紧为必须通过历史相似样本验证才允许出信号，避免只凭高模型置信度放行低质量场景信号。

企业微信通知白名单固定在 `config.py` 的 `OFFICIAL_SIGNAL_STRATEGY_ALLOWLIST`。当前所有实时策略都在白名单内；使用 `--observe-all` 运行时，任意策略产生 up/down 信号都会推送企业微信预测通知，并在 10 分钟后推送验证通知。

历史相似样本匹配使用 walk-forward 样本外概率池：每个历史样本只使用该时间桶之前的数据训练出来的模型概率，避免当前模型回头给整段历史打分造成 `success_rate=1.0000` 的乐观偏差。历史池默认每 120 分钟重训一次以控制实时启动耗时，实时主模型仍按原配置重训。

策略层增加了统一陷阱过滤：

- short 拒绝反弹陷阱：`ret_10 > 0 且 macd_hist > 0`、`ret_30 > 0 且 macd_hist > 0`、`close_position < 0.02`、`close_position > 0.98 且 ret_10 > 0`、或 `rsi_14 < 45 且 boll_position <= 0.15`。
- long 拒绝追高陷阱：`rsi_14 > 80`、`boll_position > 0.84`、或 `close_position > 0.98`。

如果网络无法访问币安，可先把 `data/BTCUSDT_1m_history.csv` 放入本地，并在严格回测时使用：

```bash
python3 main.py --mode strict_backtest --no-update-cache
```

## 注意

- `data/predictions.csv` 只服务实时预测，不保存训练回测或严格验证回测结果。
- 严格验证回测不得使用预测点之后的数据训练模型。
- 禁止把 10 位秒级时间戳传入核心逻辑；检测到会直接报错。
- 最新 K 线以 `timestamp` 最大的一条已收盘 1m K 线为准。

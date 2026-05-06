# config.py

from pathlib import Path

# =========================
# 项目基础配置
# =========================

PROJECT_NAME = "ShortTermTrendPredictor"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# =========================
# 本地数据缓存
# =========================

HISTORY_CSV = DATA_DIR / "BTCUSDT_1m_history.csv"
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
PENDING_FILE = DATA_DIR / "pending_predictions.jsonl"
MODEL_FILE = MODEL_DIR / "dual_backtest_ensemble_model.pkl"

# =========================
# 币安数据配置
# =========================

BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_VISION_BASE_URL = "https://data-api.binance.vision"
BINANCE_KLINES_ENDPOINT = "/api/v3/klines"

OKX_BASE_URL = "https://www.okx.com"
OKX_CANDLES_ENDPOINT = "/api/v5/market/candles"
OKX_HISTORY_CANDLES_ENDPOINT = "/api/v5/market/history-candles"
OKX_INST_ID = "BTC-USDT"
OKX_BAR = "1m"

SYMBOL = "BTCUSDT"
INTERVAL = "1m"

INTERVAL_MS = 60_000

# 实时训练使用最近 48 小时
TRAIN_HOURS = 48
TRAIN_MINUTES = TRAIN_HOURS * 60

# 严格回测拉取最近 N 天数据
BACKTEST_DAYS = 7
BACKTEST_MINUTES = BACKTEST_DAYS * 24 * 60

BINANCE_LIMIT = 1000

REQUEST_VERIFY_SSL = False
REQUEST_TIMEOUT = 15

# =========================
# 企业微信通知配置
# =========================

ENABLE_WECHAT_NOTIFICATIONS = True
WECHAT_WEBHOOK_URL = (
    "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?"
    "key=b614abd2-e508-447e-afc7-d9f86fe1edc0"
)
WECHAT_REQUEST_TIMEOUT = 10

# =========================
# 预测任务配置
# =========================

# 核心任务：
# 每 1 分钟滚动预测一次，判断未来第 10 分钟 close 是否高于当前 close。
PREDICT_HORIZON_MINUTES = 10
PREDICT_HORIZON_MS = PREDICT_HORIZON_MINUTES * INTERVAL_MS

# 是否对训练标签做轻度去噪。
# True：future_return > +阈值 为 up，future_return < -阈值 为 down，中间样本删除。
# False：future_close > current_close 即 up，否则 down。
USE_LABEL_NEUTRAL_ZONE = True

# 只有 USE_LABEL_NEUTRAL_ZONE=True 时生效。
LABEL_NEUTRAL_THRESHOLD = 0.0003

# =========================
# 信号阈值配置
# =========================

# 做多：p_up >= LONG_SIGNAL_THRESHOLD
# 做空：p_up <= SHORT_SIGNAL_THRESHOLD
#
# 单点概率只作为基础门槛，最终是否出信号还要通过连续确认过滤。
LONG_SIGNAL_THRESHOLD = 0.80
SHORT_SIGNAL_THRESHOLD = 0.20
ENABLE_LONG_SIGNALS = True
ENABLE_SHORT_SIGNALS = True

# 连续确认过滤：
# 单点高概率容易在 1m 噪声和趋势切换时过度自信。
# 只有最近若干分钟内同方向概率持续稳定，才允许高置信信号通过。
ENABLE_SIGNAL_STABILITY_FILTER = True
LONG_STABILITY_THRESHOLD = 0.95
LONG_STABILITY_WINDOW = 3
LONG_STABILITY_MIN_COUNT = 3
SHORT_STABILITY_THRESHOLD = 0.25
SHORT_STABILITY_WINDOW = 15
SHORT_STABILITY_MIN_COUNT = 10
SIGNAL_MIN_INTERVAL_MINUTES = 1

# 做多 regime 过滤：
# 连续高概率只代表模型稳定看多，还需要当前可见的回调后转强结构配合。
ENABLE_LONG_REGIME_FILTER = True
LONG_REGIME_REQUIRE_EMA_20_60_POSITIVE = False
LONG_REGIME_REQUIRE_MACD_HIST_POSITIVE = False
LONG_REGIME_MAX_RET_5 = 0.0
LONG_REGIME_MIN_CLOSE_POSITION = 0.70
LONG_REGIME_MIN_RET_30 = 0.0

# 做空 regime 过滤：
# 低上涨概率持续出现时，要求当前 K 线收在相对高位但 MACD 动能仍偏空。
ENABLE_SHORT_REGIME_FILTER = True
SHORT_REGIME_MIN_CLOSE_POSITION = 0.70
SHORT_REGIME_REQUIRE_MACD_HIST_NEGATIVE = True
SHORT_REGIME_MIN_RET_30 = 0.0
SHORT_REGIME_MAX_RET_30 = 0.0015
SHORT_REGIME_MIN_RET_10 = -0.0005
SHORT_REGIME_MAX_RSI_14 = 59.0
SHORT_REGIME_SKIP_AGGRESSIVE_BUY_CANDLE = True
SHORT_REGIME_AGGRESSIVE_BUY_MIN_TAKER_RATIO = 0.95
SHORT_REGIME_AGGRESSIVE_BUY_MIN_BODY_RATIO = 0.95
SHORT_REGIME_AGGRESSIVE_BUY_MIN_TREND = 0.0
SHORT_REGIME_SKIP_WEAK_MIXED_BULLISH_TREND = False
SHORT_REGIME_WEAK_TREND_MAX_RSI_6 = 50.0

# 高置信信号二次过滤：
# 概率阈值只说明模型有把握，但 1m BTC 容易在反向趋势中给出过度自信信号。
# 这里要求预测方向和当前可见短线趋势一致，否则降级为 no_trade。
ENABLE_SIGNAL_QUALITY_GATE = False
SIGNAL_MIN_TREND_AGREEMENT = 1.0 / 3.0

# 实时预测频率：60 秒执行一次
REALTIME_INTERVAL_SECONDS = 60

# 模型重训练频率：30 分钟执行一次
RETRAIN_INTERVAL_SECONDS = 30 * 60

# =========================
# 严格回测配置
# =========================

BACKTEST_TRAIN_WINDOW_MINUTES = 48 * 60
BACKTEST_STEP_MINUTES = 1
BACKTEST_MODEL_UPDATE_MINUTES = RETRAIN_INTERVAL_SECONDS // 60
BACKTEST_MAX_STEPS = None
BACKTEST_MIN_TRAIN_SAMPLES = 500
BACKTEST_PROGRESS_EVERY = 20

# 分桶统计设置
PROB_BIN_WIDTH = 0.05

# 自动阈值搜索：
# 至少多少个信号才认为该阈值组合有参考价值。
MIN_SIGNALS_FOR_THRESHOLD_SEARCH = 30

# =========================
# 模型配置
# =========================

RANDOM_STATE = 42

# 时间衰减权重，越新的样本权重越大
TIME_DECAY_STRENGTH = 4.0

LGB_WEIGHT = 1.0 / 3.0
XGB_WEIGHT = 1.0 / 3.0
CAT_WEIGHT = 1.0 / 3.0

# =========================
# 输出 CSV 字段
# =========================

CSV_COLUMNS = [
    "timestamp",
    "current_price",
    "future_price",
    "predicted_direction",
    "actual_direction",
    "up_probability",
    "confidence",
    "is_valid_signal",
    "is_correct",
]

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
ALL_PREDICTIONS_CSV = DATA_DIR / "all_predictions.csv"
OFFICIAL_SIGNALS_CSV = DATA_DIR / "official_signals.csv"
PENDING_FILE = DATA_DIR / "pending_predictions.jsonl"
MODEL_FILE = MODEL_DIR / "dual_backtest_ensemble_model.pkl"
DUAL_MODEL_PARAMS_FILE = MODEL_DIR / "dual_model_params.json"
DUAL_MODEL_TUNING_REPORT_CSV = DATA_DIR / "dual_model_tuning_report.csv"
STRICT_PARAM_SEARCH_CSV = DATA_DIR / "strict_param_search_report.csv"
STRATEGY_LEARNING_STATE_FILE = DATA_DIR / "strategy_learning_state.json"
HISTORICAL_MATCH_CACHE_FILE = DATA_DIR / "historical_match_walk_forward_cache.pkl"

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

TRAIN_HOURS = 48
TRAIN_MINUTES = TRAIN_HOURS * 60

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

PREDICT_HORIZON_MINUTES = 10
PREDICT_HORIZON_MS = PREDICT_HORIZON_MINUTES * INTERVAL_MS

USE_LABEL_NEUTRAL_ZONE = True
LABEL_NEUTRAL_THRESHOLD = 0.0005

# 方向正确只是基础验证；正式质量评估还要求未来涨跌幅超过该阈值。
# 0.0008 表示 0.08%，用于过滤方向对但波动太小、不值得通知和跟踪的信号。
TRADABLE_RETURN_THRESHOLD = 0.0002

# =========================
# 信号阈值配置
# =========================

LONG_SIGNAL_THRESHOLD = 0.80
SHORT_SIGNAL_THRESHOLD = 0.20
ENABLE_LONG_SIGNALS = True
ENABLE_SHORT_SIGNALS = True

DUAL_DIRECTION_MIN_EDGE = 0.20

ENABLE_SIGNAL_STABILITY_FILTER = True
LONG_STABILITY_THRESHOLD = 0.95
LONG_STABILITY_WINDOW = 3
LONG_STABILITY_MIN_COUNT = 3
SHORT_STABILITY_THRESHOLD = 0.25
SHORT_STABILITY_WINDOW = 15
SHORT_STABILITY_MIN_COUNT = 10
SIGNAL_MIN_INTERVAL_MINUTES = 1

ENABLE_LONG_REGIME_FILTER = True
LONG_REGIME_REQUIRE_EMA_20_60_POSITIVE = False
LONG_REGIME_REQUIRE_MACD_HIST_POSITIVE = False
LONG_REGIME_MAX_RET_5 = 0.0
LONG_REGIME_MIN_CLOSE_POSITION = 0.70
LONG_REGIME_MIN_RET_30 = 0.0
LONG_REGIME_MAX_RSI_14 = 60.0
LONG_REGIME_SKIP_FULL_HIGH_BODY = True
LONG_REGIME_FULL_HIGH_MIN_BODY_RATIO = 0.95
LONG_REGIME_FULL_HIGH_MIN_CLOSE_POSITION = 0.99

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

ENABLE_SIGNAL_QUALITY_GATE = False
SIGNAL_MIN_TREND_AGREEMENT = 1.0 / 3.0

# =========================
# 高胜率策略过滤配置
# =========================

ENABLE_HIGH_WIN_RATE_FILTER = True
HIGH_WIN_RATE_MIN_CONFIDENCE = 0.85
HIGH_WIN_RATE_MIN_EDGE = 0.25
HIGH_WIN_RATE_MIN_ATR_14 = 0.00025
HIGH_WIN_RATE_MIN_VOLUME_RATIO = 0.80
HIGH_WIN_RATE_REQUIRE_TREND_ALIGNMENT = True
HIGH_WIN_RATE_REQUIRE_MACD_DIRECTION = True
HIGH_WIN_RATE_MAX_RSI_LONG = 68.0
HIGH_WIN_RATE_MAX_BOLL_POSITION_LONG = 0.90
HIGH_WIN_RATE_MIN_BOLL_POSITION_SHORT = 0.10

REALTIME_INTERVAL_SECONDS = 60
RETRAIN_INTERVAL_SECONDS = 30 * 60

# 企业微信通知白名单。列入的策略产生 up/down 信号时会推送预测和验证通知。
OFFICIAL_SIGNAL_STRATEGY_ALLOWLIST = (
    "adaptive_rule_switch",
    "adaptive_dual",
    "historical_match",
    "historical_match_short",
    "kronos_confirm",
    "kronos_lead",
)
HISTORICAL_MATCH_WALK_FORWARD_MODEL_UPDATE_MINUTES = 120
HISTORICAL_MATCH_CACHE_MAX_AGE_MINUTES = 30
HISTORICAL_MATCH_CACHE_STALE_MAX_HOURS = 24

# Kronos 可选确认模型。默认只使用本地 Hugging Face 缓存，避免实时循环临时下载卡住。
KRONOS_MODEL_NAME = "NeoQuasar/Kronos-small"
KRONOS_TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
KRONOS_LOCAL_FILES_ONLY = True
KRONOS_DEVICE = "cpu"
KRONOS_LOOKBACK = 128
KRONOS_MAX_CONTEXT = 128
KRONOS_LOAD_TIMEOUT_SECONDS = 30
KRONOS_PREDICT_TIMEOUT_SECONDS = 45
KRONOS_USE_SUBPROCESS = True
KRONOS_RUN_MIN_EDGE = 0.12
KRONOS_RUN_MIN_CONFIDENCE = 0.20

# =========================
# 策略自学习配置
# =========================

ENABLE_STRATEGY_SELF_LEARNING = True
STRATEGY_LEARNING_ROLLING_WINDOW = 30
STRATEGY_LEARNING_MIN_SAMPLES = 3
STRATEGY_LEARNING_DISABLE_WIN_RATE = 0.55
STRATEGY_LEARNING_ENABLE_WIN_RATE = 0.70
STRATEGY_LEARNING_FEATURE_BLOCK_MIN_ERRORS = 2
ADAPTIVE_DUAL_MIN_EDGE = 0.15
ADAPTIVE_DUAL_MIN_CONFIDENCE = 0.20
KRONOS_LEAD_MIN_CONFIDENCE = 0.10
KRONOS_LEAD_MAX_OPPOSITE_EDGE = 0.05

# 正式通知质量门槛。策略仍可继续记录和验证，但只有满足这些条件才会推送。
ADAPTIVE_NOTIFY_MIN_CONFIDENCE = 0.30
ADAPTIVE_NOTIFY_MIN_EDGE = 0.15
ADAPTIVE_NOTIFY_REQUIRE_CONFIRMATION = False
ADAPTIVE_STRICT_FILTER_ENABLED = True
ADAPTIVE_STRICT_ALLOW_DOWN = False
ADAPTIVE_STRICT_LONG_MAX_BOLL_POSITION = 0.75
ADAPTIVE_STRICT_LONG_MAX_RSI_14 = 70.0
ADAPTIVE_STRICT_LONG_MAX_CLOSE_POSITION = 0.95
ADAPTIVE_STRICT_LONG_MIN_RET_30 = 0.0
HISTORICAL_MATCH_NOTIFY_MIN_MATCHED = 15
HISTORICAL_MATCH_NOTIFY_MIN_SUCCESS_RATE = 0.72
KRONOS_NOTIFY_MIN_CONFIDENCE = 0.10
KRONOS_NOTIFY_ALLOW_DOWN = False

# 自适应规则切换：先探索候选规则，只有候选规则近期验证胜率达标才正式通知。
ADAPTIVE_RULE_SWITCH_MIN_SAMPLES = 5
ADAPTIVE_RULE_SWITCH_MIN_WIN_RATE = 0.80
ADAPTIVE_RULE_SWITCH_ROLLING_WINDOW = 10
ADAPTIVE_RULE_SWITCH_MAX_UP_PROBABILITY = 0.285
ADAPTIVE_RULE_SWITCH_MIN_RSI_14 = 47.5
ADAPTIVE_RULE_SWITCH_MIN_RET_30 = -0.00013
ADAPTIVE_RULE_SWITCH_VOLUME_GATE_ENABLED = False
ADAPTIVE_RULE_SWITCH_MAX_VOLUME_RATIO_10 = 2.0
ADAPTIVE_RULE_SWITCH_MAX_QUOTE_VOLUME_RATIO_10 = 2.0
ADAPTIVE_RULE_SWITCH_MAX_TRADE_COUNT_RATIO_10 = 2.0
ADAPTIVE_RULE_SWITCH_MAX_VOLUME_ZSCORE = 2.5
ADAPTIVE_RULE_SWITCH_MAX_ABS_VOLUME_CHANGE = 2.0

# =========================
# 严格回测配置
# =========================

BACKTEST_TRAIN_WINDOW_MINUTES = 48 * 60
BACKTEST_STEP_MINUTES = 1
BACKTEST_MODEL_UPDATE_MINUTES = RETRAIN_INTERVAL_SECONDS // 60
BACKTEST_MAX_STEPS = None
BACKTEST_MIN_TRAIN_SAMPLES = 500
BACKTEST_PROGRESS_EVERY = 20

PROB_BIN_WIDTH = 0.05
MIN_SIGNALS_FOR_THRESHOLD_SEARCH = 30

STRICT_PARAM_SEARCH_ENABLED = True
STRICT_PARAM_SEARCH_LONG_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
STRICT_PARAM_SEARCH_SHORT_THRESHOLDS = [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
STRICT_PARAM_SEARCH_TOP_N = 20

STRICT_PARAM_RECOMMEND_MIN_SIGNALS = 300
STRICT_PARAM_RECOMMEND_MIN_WIN_RATE = 0.54
STRICT_PARAM_RECOMMEND_MIN_SIGNAL_RATIO = 0.10
STRICT_PARAM_RECOMMEND_MIN_SIDE_SIGNALS = 50

# =========================
# 双子模型自动调参配置
# =========================

DUAL_MODEL_TUNE_DAYS = 60
DUAL_MODEL_TUNE_VALID_RATIO = 0.30
DUAL_MODEL_TUNE_MAX_TRIALS_PER_SIDE = 30
DUAL_MODEL_TUNE_MIN_VALID_SIGNALS = 30
DUAL_MODEL_TUNE_MIN_WIN_RATE = 0.70
DUAL_MODEL_TUNE_SIGNAL_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

DUAL_MODEL_PARAM_GRID = [
    {"n_estimators": 120, "learning_rate": 0.06, "max_depth": 3, "num_leaves": 7, "min_child_samples": 120, "subsample": 0.75, "colsample_bytree": 0.75, "reg_alpha": 0.30, "reg_lambda": 3.00},
    {"n_estimators": 160, "learning_rate": 0.05, "max_depth": 3, "num_leaves": 7, "min_child_samples": 80, "subsample": 0.80, "colsample_bytree": 0.80, "reg_alpha": 0.20, "reg_lambda": 2.00},
    {"n_estimators": 220, "learning_rate": 0.04, "max_depth": 4, "num_leaves": 15, "min_child_samples": 60, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.15, "reg_lambda": 1.50},
    {"n_estimators": 320, "learning_rate": 0.035, "max_depth": 5, "num_leaves": 31, "min_child_samples": 40, "subsample": 0.90, "colsample_bytree": 0.90, "reg_alpha": 0.10, "reg_lambda": 1.00},
    {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 6, "num_leaves": 31, "min_child_samples": 25, "subsample": 0.95, "colsample_bytree": 0.95, "reg_alpha": 0.05, "reg_lambda": 0.50},
    {"n_estimators": 650, "learning_rate": 0.025, "max_depth": 7, "num_leaves": 63, "min_child_samples": 15, "subsample": 0.95, "colsample_bytree": 0.95, "reg_alpha": 0.02, "reg_lambda": 0.25},
    {"n_estimators": 800, "learning_rate": 0.035, "max_depth": 8, "num_leaves": 63, "min_child_samples": 10, "subsample": 1.00, "colsample_bytree": 1.00, "reg_alpha": 0.00, "reg_lambda": 0.00},
    {"n_estimators": 200, "learning_rate": 0.06, "max_depth": 4, "num_leaves": 20, "min_child_samples": 50, "subsample": 0.90, "colsample_bytree": 0.90, "reg_alpha": 0.10, "reg_lambda": 0.80},
    {"n_estimators": 350, "learning_rate": 0.04, "max_depth": 5, "num_leaves": 40, "min_child_samples": 30, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.08, "reg_lambda": 0.80},
    {"n_estimators": 500, "learning_rate": 0.02, "max_depth": 6, "num_leaves": 60, "min_child_samples": 20, "subsample": 0.90, "colsample_bytree": 0.90, "reg_alpha": 0.05, "reg_lambda": 0.50},
    {"n_estimators": 700, "learning_rate": 0.015, "max_depth": 8, "num_leaves": 80, "min_child_samples": 10, "subsample": 0.95, "colsample_bytree": 0.95, "reg_alpha": 0.05, "reg_lambda": 0.25},
]

# =========================
# 模型配置
# =========================

RANDOM_STATE = 42
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
    "future_return",
    "predicted_direction",
    "actual_direction",
    "up_probability",
    "confidence",
    "is_valid_signal",
    "is_correct",
    "is_tradable_correct",
]

# high_win_rate_filter.py

import pandas as pd

from config import (
    HIGH_WIN_RATE_MAX_BOLL_POSITION_LONG,
    HIGH_WIN_RATE_MAX_RSI_LONG,
    HIGH_WIN_RATE_MIN_ATR_14,
    HIGH_WIN_RATE_MIN_BOLL_POSITION_SHORT,
    HIGH_WIN_RATE_MIN_CONFIDENCE,
    HIGH_WIN_RATE_MIN_EDGE,
    HIGH_WIN_RATE_MIN_VOLUME_RATIO,
    HIGH_WIN_RATE_REQUIRE_MACD_DIRECTION,
    HIGH_WIN_RATE_REQUIRE_TREND_ALIGNMENT,
)


def _value(row: pd.Series, name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    if pd.isna(value):
        return default
    return float(value)


def passes_high_win_rate_filter(
    X: pd.DataFrame,
    direction: str,
    up_signal_probability: float,
    down_signal_probability: float,
    direction_edge: float,
) -> bool:
    """
    高胜率交易过滤器。

    该过滤器不负责产生方向，只负责判断模型给出的方向是否处于更适合交易的行情结构中。
    目标是牺牲信号数量，提高最终正式交易信号质量。
    """
    if X.empty or direction not in {"up", "down"}:
        return False

    row = X.iloc[0]

    atr_14 = _value(row, "atr_14")
    volume_ratio_20 = _value(row, "volume_ratio_20", 1.0)
    trend_agreement = _value(row, "trend_agreement")
    macd_hist = _value(row, "macd_hist")
    macd_hist_diff = _value(row, "macd_hist_diff")
    rsi_14 = _value(row, "rsi_14", 50.0)
    boll_position = _value(row, "boll_position", 0.5)
    close_position = _value(row, "close_position", 0.5)
    ret_5 = _value(row, "ret_5")
    ret_10 = _value(row, "ret_10")
    ret_30 = _value(row, "ret_30")
    ema_5_20_diff = _value(row, "ema_5_20_diff")
    ema_20_60_diff = _value(row, "ema_20_60_diff")
    taker_buy_ratio_diff_5_10 = _value(row, "taker_buy_ratio_diff_5_10")

    if atr_14 < HIGH_WIN_RATE_MIN_ATR_14:
        return False

    if volume_ratio_20 < HIGH_WIN_RATE_MIN_VOLUME_RATIO:
        return False

    if direction == "up":
        if up_signal_probability < HIGH_WIN_RATE_MIN_CONFIDENCE:
            return False
        if direction_edge < HIGH_WIN_RATE_MIN_EDGE:
            return False
        if HIGH_WIN_RATE_REQUIRE_TREND_ALIGNMENT and trend_agreement <= 0:
            return False
        if HIGH_WIN_RATE_REQUIRE_MACD_DIRECTION and (macd_hist <= 0 or macd_hist_diff <= 0):
            return False
        if rsi_14 >= HIGH_WIN_RATE_MAX_RSI_LONG:
            return False
        if boll_position >= HIGH_WIN_RATE_MAX_BOLL_POSITION_LONG:
            return False
        if close_position >= 0.98:
            return False
        if ret_5 <= 0 or ret_10 <= 0:
            return False
        if ret_30 < -0.0005:
            return False
        if ema_5_20_diff <= 0 or ema_20_60_diff < -0.0005:
            return False
        if taker_buy_ratio_diff_5_10 < -0.05:
            return False
        return True

    if down_signal_probability < HIGH_WIN_RATE_MIN_CONFIDENCE:
        return False
    if -direction_edge < HIGH_WIN_RATE_MIN_EDGE:
        return False
    if HIGH_WIN_RATE_REQUIRE_TREND_ALIGNMENT and trend_agreement >= 0:
        return False
    if HIGH_WIN_RATE_REQUIRE_MACD_DIRECTION and (macd_hist >= 0 or macd_hist_diff >= 0):
        return False
    if rsi_14 <= 100.0 - HIGH_WIN_RATE_MAX_RSI_LONG:
        return False
    if boll_position <= HIGH_WIN_RATE_MIN_BOLL_POSITION_SHORT:
        return False
    if close_position <= 0.02:
        return False
    if ret_5 >= 0 or ret_10 >= 0:
        return False
    if ret_30 > 0.0005:
        return False
    if ema_5_20_diff >= 0 or ema_20_60_diff > 0.0005:
        return False
    if taker_buy_ratio_diff_5_10 > 0.05:
        return False
    return True

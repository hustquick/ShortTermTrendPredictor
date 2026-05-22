# market_regime.py

from strategies.base import feature_value


def classify_market_regime(features) -> str:
    """Return a compact market regime label for strategy quality attribution."""
    ret_10 = feature_value(features, "ret_10")
    ret_30 = feature_value(features, "ret_30")
    macd_hist = feature_value(features, "macd_hist")
    trend = feature_value(features, "trend_agreement")
    rsi_14 = feature_value(features, "rsi_14", 50.0)
    boll_position = feature_value(features, "boll_position", 0.5)
    volatility = feature_value(features, "volatility", 0.0)
    atr_14 = feature_value(features, "atr_14", 0.0)

    vol_ref = max(abs(volatility), abs(atr_14))
    high_vol = vol_ref >= 0.0015
    low_vol = vol_ref <= 0.0005

    if rsi_14 >= 78 or boll_position >= 0.90:
        heat = "overheated"
    elif rsi_14 <= 28 or boll_position <= 0.10:
        heat = "oversold"
    else:
        heat = "neutral"

    if ret_30 > 0 and ret_10 > 0 and macd_hist > 0 and trend > 0:
        direction = "trend_up"
    elif ret_30 < 0 and ret_10 < 0 and macd_hist < 0 and trend < 0:
        direction = "trend_down"
    elif ret_30 > 0 and ret_10 < 0:
        direction = "up_pullback"
    elif ret_30 < 0 and ret_10 > 0:
        direction = "down_rebound"
    else:
        direction = "range"

    if high_vol:
        vol = "high_vol"
    elif low_vol:
        vol = "low_vol"
    else:
        vol = "mid_vol"

    return f"{direction}|{vol}|{heat}"

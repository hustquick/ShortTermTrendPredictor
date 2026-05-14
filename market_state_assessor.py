# market_state_assessor.py

from dataclasses import dataclass

import pandas as pd


@dataclass
class MarketState:
    trend_direction: str
    trend_strength: str
    volatility_regime: str
    momentum_state: str
    rsi_zone: str
    bollinger_zone: str
    market_state: str
    score: float
    reason: str


def _safe_float(row: pd.Series, name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    if pd.isna(value):
        return default
    return float(value)


def assess_market_state(features: pd.Series) -> MarketState:
    """Compute-in-CoT inspired deterministic market state assessment.

    This module only assesses the current market state. It does not predict direction.
    """
    trend = _safe_float(features, "trend_agreement")
    macd_hist = _safe_float(features, "macd_hist")
    macd_hist_diff = _safe_float(features, "macd_hist_diff")
    ret_5 = _safe_float(features, "ret_5")
    ret_10 = _safe_float(features, "ret_10")
    ema_5_20_diff = _safe_float(features, "ema_5_20_diff")
    rsi = _safe_float(features, "rsi_14", 50.0)
    boll = _safe_float(features, "boll_position", 0.5)
    atr = abs(_safe_float(features, "atr"))
    close = abs(_safe_float(features, "close", 1.0))
    volatility = atr / max(close, 1e-12)

    up_votes = 0
    down_votes = 0
    if trend > 0:
        up_votes += 1
    elif trend < 0:
        down_votes += 1
    if macd_hist > 0:
        up_votes += 1
    elif macd_hist < 0:
        down_votes += 1
    if ret_5 > 0:
        up_votes += 1
    elif ret_5 < 0:
        down_votes += 1
    if ret_10 > 0:
        up_votes += 1
    elif ret_10 < 0:
        down_votes += 1
    if ema_5_20_diff > 0:
        up_votes += 1
    elif ema_5_20_diff < 0:
        down_votes += 1

    if up_votes >= down_votes + 2:
        trend_direction = "up"
    elif down_votes >= up_votes + 2:
        trend_direction = "down"
    else:
        trend_direction = "neutral"

    vote_gap = abs(up_votes - down_votes)
    if vote_gap >= 4:
        trend_strength = "strong"
    elif vote_gap >= 2:
        trend_strength = "weak"
    else:
        trend_strength = "choppy"

    if volatility >= 0.003:
        volatility_regime = "expansion"
    elif volatility <= 0.001:
        volatility_regime = "contraction"
    else:
        volatility_regime = "normal"

    if macd_hist_diff > 0 and ret_5 > 0:
        momentum_state = "accelerating_up"
    elif macd_hist_diff < 0 and ret_5 < 0:
        momentum_state = "accelerating_down"
    elif trend_direction == "up" and ret_5 < 0:
        momentum_state = "up_fading"
    elif trend_direction == "down" and ret_5 > 0:
        momentum_state = "down_fading"
    else:
        momentum_state = "mixed"

    if rsi >= 75:
        rsi_zone = "extreme_overbought"
    elif rsi >= 65:
        rsi_zone = "overbought"
    elif rsi <= 25:
        rsi_zone = "extreme_oversold"
    elif rsi <= 35:
        rsi_zone = "oversold"
    else:
        rsi_zone = "neutral"

    if boll >= 0.90:
        bollinger_zone = "upper"
    elif boll <= 0.10:
        bollinger_zone = "lower"
    else:
        bollinger_zone = "middle"

    reversal_risk = (
        (trend_direction == "up" and (rsi_zone in {"overbought", "extreme_overbought"} or bollinger_zone == "upper"))
        or (trend_direction == "down" and (rsi_zone in {"oversold", "extreme_oversold"} or bollinger_zone == "lower"))
    )

    if trend_direction == "neutral" or trend_strength == "choppy":
        market_state = "choppy"
    elif reversal_risk:
        market_state = f"{trend_direction}_reversal_risk"
    else:
        market_state = f"trend_{trend_direction}"

    score = vote_gap / 5.0
    reason = (
        f"votes_up={up_votes};votes_down={down_votes};trend={trend_direction};"
        f"strength={trend_strength};momentum={momentum_state};rsi={rsi_zone};boll={bollinger_zone}"
    )
    return MarketState(
        trend_direction=trend_direction,
        trend_strength=trend_strength,
        volatility_regime=volatility_regime,
        momentum_state=momentum_state,
        rsi_zone=rsi_zone,
        bollinger_zone=bollinger_zone,
        market_state=market_state,
        score=float(score),
        reason=reason,
    )

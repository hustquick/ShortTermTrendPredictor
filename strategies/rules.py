# strategies/rules.py

import itertools
import math

import pandas as pd

from config import (
    ADAPTIVE_DUAL_MIN_CONFIDENCE,
    ADAPTIVE_DUAL_MIN_EDGE,
    ADAPTIVE_RULE_MINER_COOLDOWN_RECORDS,
    ADAPTIVE_RULE_MINER_ENABLED,
    ADAPTIVE_RULE_MINER_LOOKBACK_DAY_OPTIONS,
    ADAPTIVE_RULE_MINER_LOOKBACK_DAYS,
    ADAPTIVE_RULE_MINER_LOOKBACK_SIGNALS,
    ADAPTIVE_RULE_MINER_MAX_CLAUSES,
    ADAPTIVE_RULE_MINER_MIN_SAMPLES,
    ADAPTIVE_RULE_MINER_MIN_WILSON_LOWER,
    ADAPTIVE_RULE_MINER_MIN_WIN_RATE,
    ADAPTIVE_RULE_SWITCH_MIN_SAMPLES,
    ADAPTIVE_RULE_SWITCH_MIN_WILSON_LOWER,
    ADAPTIVE_RULE_SWITCH_MIN_WIN_RATE,
    ADAPTIVE_RULE_SWITCH_ROLLING_WINDOW,
    ADAPTIVE_RULE_SWITCH_ALLOW_GLOBAL_FALLBACK,
    ADAPTIVE_RULE_SWITCH_CONTEXT_DISABLE_WIN_RATE,
    ADAPTIVE_RULE_SWITCH_CONTEXT_ENABLED,
    ADAPTIVE_RULE_SWITCH_CONTEXT_MIN_SAMPLES,
    ADAPTIVE_RULE_SWITCH_MAX_RECENT_LOSS_STREAK,
    ADAPTIVE_RULE_SWITCH_MICRO_CONTEXT_ENABLED,
    ADAPTIVE_RULE_SWITCH_MICRO_CONTEXT_MIN_SAMPLES,
    ADAPTIVE_RULE_SWITCH_MAX_ABS_VOLUME_CHANGE,
    ADAPTIVE_RULE_SWITCH_MAX_QUOTE_VOLUME_RATIO_10,
    ADAPTIVE_RULE_SWITCH_MAX_TRADE_COUNT_RATIO_10,
    ADAPTIVE_RULE_SWITCH_MAX_UP_PROBABILITY,
    ADAPTIVE_RULE_SWITCH_MAX_VOLUME_RATIO_10,
    ADAPTIVE_RULE_SWITCH_MAX_VOLUME_ZSCORE,
    ADAPTIVE_RULE_SWITCH_MIN_RET_30,
    ADAPTIVE_RULE_SWITCH_MIN_RSI_14,
    ADAPTIVE_RULE_SWITCH_REGIME_ENABLED,
    ADAPTIVE_RULE_SWITCH_REGIME_MIN_SAMPLES,
    ADAPTIVE_RULE_SWITCH_VOLUME_GATE_ENABLED,
    DATA_DIR,
    ADAPTIVE_STRICT_ALLOW_DOWN,
    ADAPTIVE_STRICT_FILTER_ENABLED,
    ADAPTIVE_STRICT_LONG_MAX_BOLL_POSITION,
    ADAPTIVE_STRICT_LONG_MAX_CLOSE_POSITION,
    ADAPTIVE_STRICT_LONG_MAX_RSI_14,
    ADAPTIVE_STRICT_LONG_MIN_RET_30,
    KRONOS_LEAD_MAX_OPPOSITE_EDGE,
    KRONOS_LEAD_MIN_CONFIDENCE,
    LONG_SIGNAL_THRESHOLD,
    SHORT_SIGNAL_THRESHOLD,
)
from finstar_scenario_layer import evaluate_finstar_scenario
from high_win_rate_filter import passes_high_win_rate_filter
from historical_match_filter import evaluate_historical_match
from market_regime import classify_market_regime
from strategies.base import StrategyDecision, feature_value


VALIDATED_STRATEGY_SIGNALS = DATA_DIR / "validated_strategy_signals.csv"


def _short_rebound_trap_reason(features) -> str | None:
    ret_10 = feature_value(features, "ret_10")
    ret_30 = feature_value(features, "ret_30")
    macd_hist = feature_value(features, "macd_hist")
    close_position = feature_value(features, "close_position", 0.5)
    rsi_14 = feature_value(features, "rsi_14", 50.0)
    boll_position = feature_value(features, "boll_position", 0.5)

    if ret_10 > 0 and macd_hist > 0:
        return "short_rebound_trap_ret10_macd_positive"
    if ret_30 > 0 and macd_hist > 0:
        return "short_rebound_trap_ret30_macd_positive"
    if close_position < 0.02:
        return "short_rebound_trap_close_at_low"
    if close_position > 0.98 and ret_10 > 0:
        return "short_rebound_trap_close_at_high_with_positive_ret10"
    if rsi_14 < 45 and boll_position <= 0.15:
        return "short_rebound_trap_oversold_lower_band"
    return None


def _long_chase_trap_reason(features) -> str | None:
    rsi_14 = feature_value(features, "rsi_14", 50.0)
    boll_position = feature_value(features, "boll_position", 0.5)
    close_position = feature_value(features, "close_position", 0.5)

    if rsi_14 > 80:
        return "long_chase_trap_extreme_rsi"
    if boll_position > 0.84:
        return "long_chase_trap_upper_boll"
    if close_position > 0.98:
        return "long_chase_trap_close_at_high"
    return None


def _reject_trap(features, direction: str, confidence: float) -> StrategyDecision | None:
    if direction == "up":
        reason = _long_chase_trap_reason(features)
        if reason is not None:
            return StrategyDecision("no_trade", confidence, reason)
    if direction == "down":
        reason = _short_rebound_trap_reason(features)
        if reason is not None:
            return StrategyDecision("no_trade", confidence, reason)
    return None


def _reason_value(reason: str, key: str) -> str:
    prefix = f"{key}="
    for part in str(reason).split(";"):
        if part.startswith(prefix):
            return part[len(prefix):]
    return ""


def _beijing_session_from_timestamp(timestamp_ms: float) -> str:
    if pd.isna(timestamp_ms) or timestamp_ms <= 0:
        return "unknown"
    hour = int((int(timestamp_ms) // 3_600_000 + 8) % 24)
    if 8 <= hour < 15:
        return "asia_day"
    if 15 <= hour < 21:
        return "europe_overlap"
    if 21 <= hour or hour < 1:
        return "us_open"
    return "late_us"


def _bucket(value: float, thresholds: list[tuple[float, str]], default: str) -> str:
    if pd.isna(value):
        return "unknown"
    for threshold, label in thresholds:
        if value <= threshold:
            return label
    return default


def _adaptive_probability_context(prediction: dict) -> str:
    p_up_raw = float(prediction.get("up_probability", 0.5))
    edge = float(prediction.get("direction_edge", 0.0))
    p_up_signal = float(prediction.get("up_signal_probability", 0.0))
    p_down_signal = float(prediction.get("down_signal_probability", 0.0))
    up_bucket = _bucket(
        p_up_raw,
        [(0.10, "pup_xlow"), (0.20, "pup_low"), (0.35, "pup_midlow"), (0.50, "pup_mixed")],
        "pup_high",
    )
    edge_bucket = _bucket(
        abs(edge),
        [(0.15, "edge_small"), (0.35, "edge_mid"), (0.60, "edge_large")],
        "edge_extreme",
    )
    dominance = "up_dom" if p_up_signal > p_down_signal else "down_dom"
    return f"{up_bucket}|{edge_bucket}|{dominance}"


def _adaptive_feature_context(features, prediction: dict) -> str:
    rsi_14 = feature_value(features, "rsi_14", 50.0)
    ret_5 = feature_value(features, "ret_5")
    ret_10 = feature_value(features, "ret_10")
    ret_30 = feature_value(features, "ret_30")
    macd_hist = feature_value(features, "macd_hist")
    macd_hist_diff = feature_value(features, "macd_hist_diff")
    ema_5_20_diff = feature_value(features, "ema_5_20_diff")
    boll_position = feature_value(features, "boll_position", 0.5)
    boll_width = feature_value(features, "boll_width", 0.0)
    close_position = feature_value(features, "close_position", 0.5)
    volume_ratio_10 = feature_value(features, "volume_ratio_10", 1.0)
    quote_volume_ratio_10 = feature_value(features, "quote_volume_ratio_10", 1.0)
    trade_count_ratio_10 = feature_value(features, "trade_count_ratio_10", 1.0)
    taker_buy_ratio = feature_value(features, "taker_buy_ratio", 0.5)
    taker_buy_ratio_diff_5_10 = feature_value(features, "taker_buy_ratio_diff_5_10", 0.0)
    trend = feature_value(features, "trend_agreement")
    trend_long = feature_value(features, "trend_agreement_long", trend)
    atr_14 = feature_value(features, "atr_14", 0.0)
    volatility_30 = feature_value(features, "volatility_30", 0.0)
    volatility_ref = max(abs(atr_14), abs(volatility_30))

    rsi_bucket = _bucket(
        rsi_14,
        [(40, "rsi_cold"), (50, "rsi_soft"), (60, "rsi_mid"), (70, "rsi_warm")],
        "rsi_hot",
    )
    ret_bucket = _bucket(
        ret_30,
        [(-0.002, "ret30_drop"), (-0.0005, "ret30_down"), (0.0005, "ret30_flat"), (0.002, "ret30_up")],
        "ret30_surge",
    )
    ret_short_bucket = _bucket(
        ret_5 + ret_10,
        [(-0.002, "ret_short_drop"), (-0.0005, "ret_short_down"), (0.0005, "ret_short_flat"), (0.002, "ret_short_up")],
        "ret_short_surge",
    )
    macd_bucket = "macd_pos" if macd_hist > 0 else "macd_neg" if macd_hist < 0 else "macd_flat"
    macd_slope_bucket = (
        "macd_rising" if macd_hist_diff > 0 else "macd_falling" if macd_hist_diff < 0 else "macd_stable"
    )
    ema_bucket = (
        "ema_fast_above" if ema_5_20_diff > 0 else "ema_fast_below" if ema_5_20_diff < 0 else "ema_flat"
    )
    boll_bucket = _bucket(
        boll_position,
        [(0.10, "boll_lower"), (0.35, "boll_lowmid"), (0.65, "boll_mid"), (0.90, "boll_highmid")],
        "boll_upper",
    )
    boll_width_bucket = _bucket(
        boll_width,
        [(0.004, "boll_narrow"), (0.010, "boll_normal"), (0.020, "boll_wide")],
        "boll_extreme_wide",
    )
    position_bucket = _bucket(
        close_position,
        [(0.20, "pos_low"), (0.50, "pos_midlow"), (0.80, "pos_midhigh"), (0.95, "pos_high")],
        "pos_extreme",
    )
    volume_bucket = _bucket(
        volume_ratio_10,
        [(0.80, "vol_quiet"), (1.20, "vol_normal"), (1.80, "vol_active")],
        "vol_shock",
    )
    taker_bucket = _bucket(
        taker_buy_ratio,
        [(0.45, "taker_sell"), (0.55, "taker_balanced")],
        "taker_buy",
    )
    taker_flow_bucket = _bucket(
        taker_buy_ratio_diff_5_10,
        [(-0.03, "taker_flow_sell"), (0.03, "taker_flow_flat")],
        "taker_flow_buy",
    )
    trend_sum = trend + 0.5 * trend_long
    trend_bucket = _bucket(
        trend_sum,
        [(-0.75, "trend_down"), (-0.20, "trend_soft_down"), (0.20, "trend_mixed"), (0.75, "trend_soft_up")],
        "trend_up",
    )
    volatility_bucket = _bucket(
        volatility_ref,
        [(0.0005, "volatility_low"), (0.0015, "volatility_mid")],
        "volatility_high",
    )
    quote_volume_bucket = _bucket(
        quote_volume_ratio_10,
        [(0.80, "quote_vol_quiet"), (1.20, "quote_vol_normal"), (1.80, "quote_vol_active")],
        "quote_vol_shock",
    )
    trade_count_bucket = _bucket(
        trade_count_ratio_10,
        [(0.80, "trades_quiet"), (1.20, "trades_normal"), (1.80, "trades_active")],
        "trades_shock",
    )
    probability_context = _adaptive_probability_context(prediction)
    return (
        f"{probability_context}|{rsi_bucket}|{ret_bucket}|{ret_short_bucket}|"
        f"{macd_bucket}|{macd_slope_bucket}|{ema_bucket}|{boll_bucket}|{boll_width_bucket}|"
        f"{position_bucket}|{volume_bucket}|{quote_volume_bucket}|{trade_count_bucket}|"
        f"{taker_bucket}|{taker_flow_bucket}|{trend_bucket}|{volatility_bucket}"
    )


def _wilson_lower_bound(wins: int, samples: int, z: float = 1.96) -> float:
    if samples <= 0:
        return 0.0
    p = wins / samples
    denominator = 1 + z * z / samples
    centre = p + z * z / (2 * samples)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * samples)) / samples)
    return max(0.0, (centre - margin) / denominator)


class BaselineDualStrategy:
    name = "baseline_dual"

    def decide(self, features, prediction: dict) -> StrategyDecision:
        p_up = float(prediction.get("up_signal_probability", 0.0))
        p_down = float(prediction.get("down_signal_probability", 0.0))
        edge = float(prediction.get("direction_edge", 0.0))
        if p_up >= LONG_SIGNAL_THRESHOLD and edge > 0:
            rejected = _reject_trap(features, "up", p_up)
            if rejected is not None:
                return rejected
            return StrategyDecision("up", p_up, "p_up_above_threshold")
        if p_down >= 1.0 - SHORT_SIGNAL_THRESHOLD and edge < 0:
            rejected = _reject_trap(features, "down", p_down)
            if rejected is not None:
                return rejected
            return StrategyDecision("down", p_down, "p_down_above_threshold")
        return StrategyDecision("no_trade", max(p_up, p_down), "below_threshold")


class HighConfidenceFilterStrategy:
    name = "high_confidence_filter"

    def decide(self, features, prediction: dict) -> StrategyDecision:
        p_up = float(prediction.get("up_signal_probability", 0.0))
        p_down = float(prediction.get("down_signal_probability", 0.0))
        edge = float(prediction.get("direction_edge", 0.0))
        if p_up > p_down:
            direction = "up"
            confidence = p_up
        elif p_down > p_up:
            direction = "down"
            confidence = p_down
        else:
            return StrategyDecision("no_trade", p_up, "tie")
        ok = passes_high_win_rate_filter(
            X=features.to_frame().T,
            direction=direction,
            up_signal_probability=p_up,
            down_signal_probability=p_down,
            direction_edge=edge,
        )
        if ok:
            rejected = _reject_trap(features, direction, confidence)
            if rejected is not None:
                return rejected
            return StrategyDecision(direction, confidence, "passed_high_confidence_filter")
        return StrategyDecision("no_trade", confidence, "failed_high_confidence_filter")


class ScenarioAwareStrategy:
    name = "scenario_aware"

    def decide(self, features, prediction: dict) -> StrategyDecision:
        p_up = float(prediction.get("up_signal_probability", 0.0))
        p_down = float(prediction.get("down_signal_probability", 0.0))
        edge = float(prediction.get("direction_edge", 0.0))
        confidence = max(p_up, p_down)
        trend = feature_value(features, "trend_agreement")
        macd_hist = feature_value(features, "macd_hist")
        macd_hist_diff = feature_value(features, "macd_hist_diff")
        rsi_14 = feature_value(features, "rsi_14", 50.0)
        boll_position = feature_value(features, "boll_position", 0.5)
        ret_5 = feature_value(features, "ret_5")
        ret_10 = feature_value(features, "ret_10")
        ema_5_20_diff = feature_value(features, "ema_5_20_diff")
        if confidence < 0.85 or abs(edge) < 0.20:
            return StrategyDecision("no_trade", confidence, "low_confidence_or_edge")
        up_scene = (
            p_up > p_down and trend > 0 and macd_hist > 0 and macd_hist_diff >= 0
            and ret_5 > 0 and ret_10 > 0 and ema_5_20_diff > 0
            and rsi_14 < 70 and boll_position < 0.92
        )
        down_scene = (
            p_down > p_up and trend < 0 and macd_hist < 0 and macd_hist_diff <= 0
            and ret_5 < 0 and ret_10 < 0 and ema_5_20_diff < 0
            and rsi_14 > 30 and boll_position > 0.08
        )
        if up_scene:
            rejected = _reject_trap(features, "up", p_up)
            if rejected is not None:
                return rejected
            return StrategyDecision("up", p_up, "trend_continuation_up")
        if down_scene:
            rejected = _reject_trap(features, "down", p_down)
            if rejected is not None:
                return rejected
            return StrategyDecision("down", p_down, "trend_continuation_down")
        return StrategyDecision("no_trade", confidence, "scenario_rejected")


class ConservativeExtremeStrategy:
    name = "conservative_extreme"

    def decide(self, features, prediction: dict) -> StrategyDecision:
        p_up = float(prediction.get("up_signal_probability", 0.0))
        p_down = float(prediction.get("down_signal_probability", 0.0))
        edge = float(prediction.get("direction_edge", 0.0))
        if p_up >= 0.95 and edge >= 0.35:
            rejected = _reject_trap(features, "up", p_up)
            if rejected is not None:
                return rejected
            return StrategyDecision("up", p_up, "extreme_up_confidence")
        if p_down >= 0.95 and edge <= -0.35:
            rejected = _reject_trap(features, "down", p_down)
            if rejected is not None:
                return rejected
            return StrategyDecision("down", p_down, "extreme_down_confidence")
        return StrategyDecision("no_trade", max(p_up, p_down), "not_extreme")


class AdaptiveDualStrategy:
    name = "adaptive_dual"

    def decide(self, features, prediction: dict) -> StrategyDecision:
        p_up = float(prediction.get("up_signal_probability", 0.0))
        p_down = float(prediction.get("down_signal_probability", 0.0))
        edge = float(prediction.get("direction_edge", 0.0))
        confidence = max(p_up, p_down)
        if confidence < ADAPTIVE_DUAL_MIN_CONFIDENCE or abs(edge) < ADAPTIVE_DUAL_MIN_EDGE:
            return StrategyDecision("no_trade", confidence, "adaptive_dual_low_confidence_or_edge")
        direction = "up" if edge > 0 else "down"
        if ADAPTIVE_STRICT_FILTER_ENABLED:
            if direction == "down" and not ADAPTIVE_STRICT_ALLOW_DOWN:
                return StrategyDecision("no_trade", confidence, "adaptive_strict_down_disabled_no_robust_filter")
            if direction == "up":
                ret_30 = feature_value(features, "ret_30")
                boll_position = feature_value(features, "boll_position", 0.5)
                rsi_14 = feature_value(features, "rsi_14", 50.0)
                close_position = feature_value(features, "close_position", 0.5)
                if ret_30 < ADAPTIVE_STRICT_LONG_MIN_RET_30:
                    return StrategyDecision("no_trade", confidence, "adaptive_strict_long_ret30_rejected")
                if boll_position >= ADAPTIVE_STRICT_LONG_MAX_BOLL_POSITION:
                    return StrategyDecision("no_trade", confidence, "adaptive_strict_long_boll_rejected")
                if rsi_14 >= ADAPTIVE_STRICT_LONG_MAX_RSI_14:
                    return StrategyDecision("no_trade", confidence, "adaptive_strict_long_rsi_rejected")
                if close_position >= ADAPTIVE_STRICT_LONG_MAX_CLOSE_POSITION:
                    return StrategyDecision("no_trade", confidence, "adaptive_strict_long_close_position_rejected")
        rejected = _reject_trap(features, direction, confidence)
        if rejected is not None:
            return rejected
        reason = "adaptive_dual_strict_long_pullback_signal" if ADAPTIVE_STRICT_FILTER_ENABLED else "adaptive_dual_edge_signal"
        return StrategyDecision(direction, confidence, reason)


class RelaxedScenarioStrategy:
    name = "relaxed_scenario"

    def decide(self, features, prediction: dict) -> StrategyDecision:
        p_up = float(prediction.get("up_signal_probability", 0.0))
        p_down = float(prediction.get("down_signal_probability", 0.0))
        edge = float(prediction.get("direction_edge", 0.0))
        confidence = max(p_up, p_down)
        trend = feature_value(features, "trend_agreement")
        macd_hist = feature_value(features, "macd_hist")
        rsi_14 = feature_value(features, "rsi_14", 50.0)
        boll_position = feature_value(features, "boll_position", 0.5)
        ret_5 = feature_value(features, "ret_5")
        ema_5_20_diff = feature_value(features, "ema_5_20_diff")
        if confidence < 0.82 or abs(edge) < 0.15:
            return StrategyDecision("no_trade", confidence, "low_confidence_or_edge")
        if (
            p_up > p_down and trend >= 0 and macd_hist >= 0 and ret_5 >= 0
            and ema_5_20_diff >= 0 and rsi_14 < 72 and boll_position < 0.95
        ):
            rejected = _reject_trap(features, "up", p_up)
            if rejected is not None:
                return rejected
            return StrategyDecision("up", p_up, "relaxed_up_scene")
        if (
            p_down > p_up and trend <= 0 and macd_hist <= 0 and ret_5 <= 0
            and ema_5_20_diff <= 0 and rsi_14 > 28 and boll_position > 0.05
        ):
            rejected = _reject_trap(features, "down", p_down)
            if rejected is not None:
                return rejected
            return StrategyDecision("down", p_down, "relaxed_down_scene")
        return StrategyDecision("no_trade", confidence, "relaxed_scene_rejected")


class ShortMomentumStrategy:
    name = "short_momentum"

    def decide(self, features, prediction: dict) -> StrategyDecision:
        p_down = float(prediction.get("down_signal_probability", 0.0))
        edge = float(prediction.get("direction_edge", 0.0))
        trend = feature_value(features, "trend_agreement")
        macd_hist = feature_value(features, "macd_hist")
        ret_5 = feature_value(features, "ret_5")
        ret_10 = feature_value(features, "ret_10")
        ema_5_20_diff = feature_value(features, "ema_5_20_diff")
        rsi_14 = feature_value(features, "rsi_14", 50.0)
        boll_position = feature_value(features, "boll_position", 0.5)
        if p_down < 0.82 or edge > -0.12:
            return StrategyDecision("no_trade", p_down, "low_short_confidence_or_edge")
        if trend > 0:
            return StrategyDecision("no_trade", p_down, "trend_not_short")
        if macd_hist > 0:
            return StrategyDecision("no_trade", p_down, "macd_not_short")
        if ret_5 > 0 and ret_10 > 0:
            return StrategyDecision("no_trade", p_down, "short_return_not_weak")
        if ema_5_20_diff > 0:
            return StrategyDecision("no_trade", p_down, "ema_not_short")
        if rsi_14 < 25:
            return StrategyDecision("no_trade", p_down, "short_oversold_risk")
        if boll_position < 0.03:
            return StrategyDecision("no_trade", p_down, "lower_band_rebound_risk")
        rejected = _reject_trap(features, "down", p_down)
        if rejected is not None:
            return rejected
        return StrategyDecision("down", p_down, "short_momentum_confirmed")


class LongMomentumStrategy:
    name = "long_momentum"

    def decide(self, features, prediction: dict) -> StrategyDecision:
        p_up = float(prediction.get("up_signal_probability", 0.0))
        edge = float(prediction.get("direction_edge", 0.0))
        trend = feature_value(features, "trend_agreement")
        macd_hist = feature_value(features, "macd_hist")
        ret_5 = feature_value(features, "ret_5")
        ret_10 = feature_value(features, "ret_10")
        ema_5_20_diff = feature_value(features, "ema_5_20_diff")
        rsi_14 = feature_value(features, "rsi_14", 50.0)
        boll_position = feature_value(features, "boll_position", 0.5)
        if p_up < 0.90 or edge < 0.25:
            return StrategyDecision("no_trade", p_up, "low_long_confidence_or_edge")
        if trend < 0:
            return StrategyDecision("no_trade", p_up, "trend_not_long")
        if macd_hist < 0:
            return StrategyDecision("no_trade", p_up, "macd_not_long")
        if ret_5 < 0 or ret_10 < 0:
            return StrategyDecision("no_trade", p_up, "long_return_not_strong")
        if ema_5_20_diff < 0:
            return StrategyDecision("no_trade", p_up, "ema_not_long")
        if rsi_14 > 75:
            return StrategyDecision("no_trade", p_up, "long_overheated_risk")
        if boll_position > 0.97:
            return StrategyDecision("no_trade", p_up, "upper_band_chase_risk")
        rejected = _reject_trap(features, "up", p_up)
        if rejected is not None:
            return rejected
        return StrategyDecision("up", p_up, "long_momentum_confirmed")


class AdaptiveRuleSwitchStrategy:
    name = "adaptive_rule_switch"

    def __init__(self):
        self.records: list[dict] = []
        self.cooldowns: dict[str, int] = {}
        self._pending_observations: list[dict] = []

    def decide(self, features, prediction: dict) -> StrategyDecision:
        candidates = self._candidate_rules(features, prediction)
        if not candidates:
            self._pending_observations = []
            return StrategyDecision("no_trade", 0.0, "adaptive_rule_switch_no_candidate")

        regime = classify_market_regime(features)
        session = _beijing_session_from_timestamp(feature_value(features, "timestamp"))
        adaptive_context = _adaptive_feature_context(features, prediction)
        timestamp_ms = int(feature_value(features, "timestamp", 0.0))
        scored = []
        pending = []
        for rule in candidates:
            context_tokens = self._context_tokens(
                rule_name=rule["name"],
                direction=rule["direction"],
                regime=regime,
                session=session,
                adaptive_context=adaptive_context,
            )
            stats = self._select_rule_stats(
                rule["name"],
                regime,
                session,
                adaptive_context,
                context_tokens,
                timestamp_ms,
            )
            state_reason = self._state_reason(features, prediction, rule["direction"])
            scored.append({**rule, **stats, "state_reason": state_reason})
            pending.append({
                "rule": rule["name"],
                "direction": rule["direction"],
                "regime": regime,
                "session": session,
                "context": adaptive_context,
                "tokens": context_tokens,
                "timestamp_ms": timestamp_ms,
                "state_ok": _reason_value(state_reason, "state_ok") == "True",
            })
        self._pending_observations = pending
        shadow_candidates = ",".join(
            f"{item['rule']}:{item['direction']}:{int(bool(item['state_ok']))}"
            for item in pending
        )

        active = [
            item for item in scored
            if item["samples"] >= ADAPTIVE_RULE_SWITCH_MIN_SAMPLES
            and item["win_rate"] >= item.get("min_win_rate", ADAPTIVE_RULE_SWITCH_MIN_WIN_RATE)
            and item.get("wilson_lower", 0.0) >= item.get("min_wilson_lower", 0.0)
            and item["recent_loss_streak"] <= ADAPTIVE_RULE_SWITCH_MAX_RECENT_LOSS_STREAK
            and not item["context_veto"]
        ]
        if active:
            selected = sorted(
                active,
                key=lambda x: (x.get("wilson_lower", 0.0), x["win_rate"], x["samples"], x["confidence"]),
                reverse=True,
            )[0]
            mode = "active"
        else:
            selected = sorted(scored, key=lambda x: (x["samples"], x["confidence"]), reverse=True)[0]
            mode = "explore"

        reason = (
            f"adaptive_rule_switch;adaptive_mode={mode};adaptive_rule={selected['name']};"
            f"adaptive_regime={regime};adaptive_session={session};"
            f"adaptive_context={adaptive_context};rule_scope={selected['scope']};"
            f"rule_samples={selected['samples']};rule_win_rate={selected['win_rate']:.4f};"
            f"rule_wins={selected['wins']};"
            f"rule_wilson_lower={selected.get('wilson_lower', 0.0):.4f};"
            f"rule_recent_loss_streak={selected['recent_loss_streak']};"
            f"context_veto={str(selected['context_veto'])};"
            f"miner_condition={selected.get('condition', '')};"
            f"miner_lookback_days={selected.get('lookback_days', '')};"
            f"shadow_candidates={shadow_candidates};"
            f"adaptive_timestamp_ms={timestamp_ms};"
            f"{selected['state_reason']}"
        )
        return StrategyDecision(selected["direction"], selected["confidence"], reason)

    def _candidate_rules(self, features, prediction: dict) -> list[dict]:
        p_up_raw = float(prediction.get("up_probability", 0.5))
        p_up_signal = float(prediction.get("up_signal_probability", 0.0))
        p_down_signal = float(prediction.get("down_signal_probability", 0.0))
        ret_10 = feature_value(features, "ret_10")
        ret_30 = feature_value(features, "ret_30")
        macd_hist = feature_value(features, "macd_hist")
        boll_position = feature_value(features, "boll_position", 0.5)
        close_position = feature_value(features, "close_position", 0.5)
        trend = feature_value(features, "trend_agreement")
        rsi_14 = feature_value(features, "rsi_14", 50.0)
        volume_ratio_10 = feature_value(features, "volume_ratio_10", 1.0)
        direction_edge = float(prediction.get("direction_edge", 0.0))

        rules = []

        def add(ok: bool, name: str, direction: str, confidence: float):
            if ok:
                rules.append({"name": name, "direction": direction, "confidence": float(confidence)})

        add(p_up_raw <= 0.45, "short_pup_le_045", "down", max(p_down_signal, 1.0 - p_up_raw))
        add(
            p_up_raw <= 0.50 and boll_position > 0.10,
            "short_pup_le_050_not_low",
            "down",
            max(p_down_signal, 1.0 - p_up_raw),
        )
        add(
            p_up_raw <= 0.45 and ret_30 <= 0 and trend < 0,
            "short_pup_le_045_ret30neg_trenddown",
            "down",
            max(p_down_signal, 1.0 - p_up_raw),
        )
        add(
            p_up_raw <= 0.10 and abs(direction_edge) >= 0.60 and ret_30 > -0.001 and volume_ratio_10 <= 1.5,
            "short_xlow_pup_extreme_edge_ret30_floor_vol_ok",
            "down",
            max(p_down_signal, 1.0 - p_up_raw),
        )
        add(
            p_up_raw <= 0.20 and abs(direction_edge) >= 0.60 and rsi_14 > 55 and volume_ratio_10 <= 2.0,
            "short_low_pup_extreme_edge_rsi_warm_vol_ok",
            "down",
            max(p_down_signal, 1.0 - p_up_raw),
        )
        add(
            p_up_raw <= 0.20 and abs(direction_edge) >= 0.35 and rsi_14 >= 60,
            "short_low_pup_hot_rsi_edge",
            "down",
            max(p_down_signal, 1.0 - p_up_raw),
        )
        add(
            p_up_raw <= 0.10
            and abs(direction_edge) >= 0.35
            and rsi_14 >= 60
            and 0.35 < boll_position <= 0.65,
            "short_xlow_pup_hot_rsi_boll_normal",
            "down",
            max(p_down_signal, 1.0 - p_up_raw),
        )
        add(
            p_up_raw <= 0.12 and close_position >= 0.80 and ret_30 >= -0.0005,
            "short_xlow_pup_high_position_not_breakdown",
            "down",
            max(p_down_signal, 1.0 - p_up_raw),
        )
        add(
            p_up_raw <= 0.35 and rsi_14 <= 60 and close_position <= 0.95 and volume_ratio_10 > 0.8,
            "short_midlow_pup_rsi_not_hot_not_extreme_low_volume",
            "down",
            max(p_down_signal, 1.0 - p_up_raw),
        )
        add(
            p_up_raw <= 0.45 and ret_30 > 0.001 and close_position <= 0.8,
            "short_pullback_after_up_move_not_high_close",
            "down",
            max(p_down_signal, 1.0 - p_up_raw),
        )
        add(
            p_up_raw >= 0.98 and boll_position < 0.85,
            "long_pup_ge_098_not_high",
            "up",
            max(p_up_signal, p_up_raw),
        )
        add(
            p_up_raw >= 0.85 and ret_30 >= 0 and macd_hist <= 0 and close_position < 0.95,
            "long_pup_ge_085_ret30pos_macdneg_closeok",
            "up",
            max(p_up_signal, p_up_raw),
        )
        add(
            p_up_raw >= 0.55 and boll_position < 0.85,
            "long_pup_ge_055_not_high",
            "up",
            max(p_up_signal, p_up_raw),
        )
        add(
            p_up_raw >= 0.90 and abs(direction_edge) >= 0.60 and rsi_14 < 45 and volume_ratio_10 <= 1.5,
            "long_high_pup_extreme_edge_rsi_cool_vol_ok",
            "up",
            max(p_up_signal, p_up_raw),
        )
        add(
            p_up_raw >= 0.80 and ret_30 < -0.001 and close_position >= 0.2 and volume_ratio_10 <= 2.0,
            "long_rebound_after_down_move_position_ok",
            "up",
            max(p_up_signal, p_up_raw),
        )
        return rules

    def _state_reason(self, features, prediction: dict, direction: str) -> str:
        p_up_raw = float(prediction.get("up_probability", 0.5))
        ret_30 = feature_value(features, "ret_30")
        rsi_14 = feature_value(features, "rsi_14", 50.0)
        close_position = feature_value(features, "close_position", 0.5)
        volume_ratio_10 = feature_value(features, "volume_ratio_10", 1.0)
        quote_volume_ratio_10 = feature_value(features, "quote_volume_ratio_10", 1.0)
        trade_count_ratio_10 = feature_value(features, "trade_count_ratio_10", 1.0)
        volume_zscore = feature_value(features, "volume_zscore", 0.0)
        volume_change = feature_value(features, "volume_change", 0.0)

        directional_state_ok = (
            direction == "down"
            and p_up_raw <= ADAPTIVE_RULE_SWITCH_MAX_UP_PROBABILITY
            and rsi_14 > ADAPTIVE_RULE_SWITCH_MIN_RSI_14
            and ret_30 > ADAPTIVE_RULE_SWITCH_MIN_RET_30
        )
        volume_shock = (
            volume_ratio_10 >= ADAPTIVE_RULE_SWITCH_MAX_VOLUME_RATIO_10
            or quote_volume_ratio_10 >= ADAPTIVE_RULE_SWITCH_MAX_QUOTE_VOLUME_RATIO_10
            or trade_count_ratio_10 >= ADAPTIVE_RULE_SWITCH_MAX_TRADE_COUNT_RATIO_10
            or volume_zscore >= ADAPTIVE_RULE_SWITCH_MAX_VOLUME_ZSCORE
            or abs(volume_change) >= ADAPTIVE_RULE_SWITCH_MAX_ABS_VOLUME_CHANGE
        )
        state_ok = directional_state_ok and (
            not ADAPTIVE_RULE_SWITCH_VOLUME_GATE_ENABLED or not volume_shock
        )
        return (
            f"state_ok={str(state_ok)};directional_state_ok={str(directional_state_ok)};"
            f"volume_gate_enabled={str(ADAPTIVE_RULE_SWITCH_VOLUME_GATE_ENABLED)};"
            f"volume_shock={str(volume_shock)};"
            f"raw_up_probability={p_up_raw:.4f};"
            f"state_rsi_14={rsi_14:.4f};state_ret_30={ret_30:.6f};"
            f"state_close_position={close_position:.4f};"
            f"volume_ratio_10={volume_ratio_10:.4f};"
            f"quote_volume_ratio_10={quote_volume_ratio_10:.4f};"
            f"trade_count_ratio_10={trade_count_ratio_10:.4f};"
            f"volume_zscore={volume_zscore:.4f};"
            f"volume_change={volume_change:.4f}"
        )

    @staticmethod
    def _empty_stats() -> dict:
        return {
            "samples": 0,
            "wins": 0,
            "win_rate": 0.0,
            "min_win_rate": ADAPTIVE_RULE_SWITCH_MIN_WIN_RATE,
            "wilson_lower": 0.0,
            "min_wilson_lower": ADAPTIVE_RULE_SWITCH_MIN_WILSON_LOWER,
            "recent_loss_streak": 0,
            "context_veto": False,
            "condition": "",
        }

    @staticmethod
    def _context_tokens(
        rule_name: str,
        direction: str,
        regime: str,
        session: str,
        adaptive_context: str,
    ) -> tuple[str, ...]:
        tokens = [
            f"direction={direction}",
            f"regime={regime}",
            f"session={session}",
        ]
        tokens.extend(f"ctx={token}" for token in adaptive_context.split("|") if token)
        return tuple(sorted(set(tokens)))

    def _recent_records(
        self,
        timestamp_ms: float | None = None,
        lookback_days: int | None = None,
    ) -> list[dict]:
        rows = self.records
        if timestamp_ms and timestamp_ms > 0:
            days = lookback_days or ADAPTIVE_RULE_MINER_LOOKBACK_DAYS
            cutoff = int(timestamp_ms) - int(days) * 86_400_000
            rows = [row for row in rows if int(row.get("timestamp_ms") or 0) >= cutoff]
        if ADAPTIVE_RULE_MINER_LOOKBACK_SIGNALS > 0:
            rows = rows[-ADAPTIVE_RULE_MINER_LOOKBACK_SIGNALS:]
        return rows

    @staticmethod
    def _condition_loss_streak(rows: list[dict]) -> int:
        loss_streak = 0
        for row in reversed(rows):
            if bool(row.get("correct")):
                break
            loss_streak += 1
        return loss_streak

    def _condition_in_cooldown(self, condition: str) -> bool:
        return self.cooldowns.get(condition, -1) > len(self.records)

    @staticmethod
    def _miner_search_tokens(
        context_tokens: tuple[str, ...],
        window: list[dict],
        min_samples: int,
    ) -> tuple[str, ...]:
        token_counts: dict[str, int] = {}
        allowed_prefixes = (
            "regime=",
            "session=",
            "ctx=pup_",
            "ctx=edge_",
            "ctx=rsi_",
            "ctx=ret",
            "ctx=macd_",
            "ctx=ema_",
            "ctx=boll_",
            "ctx=pos_",
            "ctx=vol_",
            "ctx=quote_vol_",
            "ctx=trades_",
            "ctx=taker_",
            "ctx=trend_",
        )
        context_set = set(context_tokens)
        for row in window:
            for token in row.get("tokens", ()):
                if token in context_set:
                    token_counts[token] = token_counts.get(token, 0) + 1
        return tuple(
            token for token in context_tokens
            if token_counts.get(token, 0) >= min_samples
            and token.startswith(allowed_prefixes)
        )

    def _mine_rule_stats(
        self,
        rule_name: str,
        context_tokens: tuple[str, ...],
        timestamp_ms: float | None = None,
    ) -> dict | None:
        if not ADAPTIVE_RULE_MINER_ENABLED or not self.records:
            return None

        best = None
        max_clauses = max(1, ADAPTIVE_RULE_MINER_MAX_CLAUSES)
        lookback_days_options = tuple(ADAPTIVE_RULE_MINER_LOOKBACK_DAY_OPTIONS) or (
            ADAPTIVE_RULE_MINER_LOOKBACK_DAYS,
        )
        for lookback_days in lookback_days_options:
            window = [
                row for row in self._recent_records(timestamp_ms, lookback_days=lookback_days)
                if row.get("rule") == rule_name and row.get("state_ok")
            ]
            if len(window) < ADAPTIVE_RULE_MINER_MIN_SAMPLES:
                continue
            search_tokens = self._miner_search_tokens(
                context_tokens,
                window,
                ADAPTIVE_RULE_MINER_MIN_SAMPLES,
            )
            window_sets = [
                (row, set(row.get("tokens", ())))
                for row in window
            ]
            for clause_count in range(1, min(max_clauses, len(search_tokens)) + 1):
                for combo in itertools.combinations(search_tokens, clause_count):
                    condition = ",".join(combo)
                    if self._condition_in_cooldown(condition):
                        continue
                    combo_set = set(combo)
                    rows = [
                        row for row, token_set in window_sets
                        if combo_set.issubset(token_set)
                    ]
                    samples = len(rows)
                    if samples < ADAPTIVE_RULE_MINER_MIN_SAMPLES:
                        continue
                    wins = sum(bool(row.get("correct")) for row in rows)
                    win_rate = wins / samples
                    wilson_lower = _wilson_lower_bound(wins, samples)
                    recent_loss_streak = self._condition_loss_streak(rows)
                    if win_rate < ADAPTIVE_RULE_MINER_MIN_WIN_RATE:
                        continue
                    if recent_loss_streak > ADAPTIVE_RULE_SWITCH_MAX_RECENT_LOSS_STREAK:
                        continue
                    candidate = {
                        "samples": samples,
                        "wins": wins,
                        "win_rate": float(win_rate),
                        "min_win_rate": ADAPTIVE_RULE_MINER_MIN_WIN_RATE,
                        "wilson_lower": float(wilson_lower),
                        "min_wilson_lower": ADAPTIVE_RULE_MINER_MIN_WILSON_LOWER,
                        "recent_loss_streak": recent_loss_streak,
                        "context_veto": False,
                        "condition": condition,
                        "scope": "online_miner",
                        "lookback_days": int(lookback_days),
                    }
                    sort_key = (
                        candidate["wilson_lower"],
                        candidate["win_rate"],
                        candidate["samples"],
                        -clause_count,
                    )
                    if best is None or sort_key > best[0]:
                        best = (sort_key, candidate)

        if best is None:
            return None
        return best[1]

    def _select_rule_stats(
        self,
        rule_name: str,
        regime: str,
        session: str,
        adaptive_context: str,
        context_tokens: tuple[str, ...],
        timestamp_ms: float | None,
    ) -> dict:
        mined_stats = self._mine_rule_stats(rule_name, context_tokens, timestamp_ms)
        if mined_stats is not None:
            return mined_stats

        global_stats = self._rule_stats(rule_name)
        if not ADAPTIVE_RULE_SWITCH_REGIME_ENABLED:
            return {**global_stats, "scope": "global"}

        if ADAPTIVE_RULE_SWITCH_CONTEXT_ENABLED:
            if ADAPTIVE_RULE_SWITCH_MICRO_CONTEXT_ENABLED:
                micro_context_stats = self._rule_stats(
                    rule_name,
                    regime=regime,
                    session=session,
                    adaptive_context=adaptive_context,
                )
                if micro_context_stats["samples"] >= ADAPTIVE_RULE_SWITCH_MICRO_CONTEXT_MIN_SAMPLES:
                    context_veto = (
                        micro_context_stats["win_rate"]
                        < ADAPTIVE_RULE_SWITCH_CONTEXT_DISABLE_WIN_RATE
                    )
                    if context_veto:
                        return {
                            **micro_context_stats,
                            "scope": "micro_context_veto",
                            "context_veto": True,
                        }
                    return {**micro_context_stats, "scope": "micro_context"}

            session_stats = self._rule_stats(rule_name, regime=regime, session=session)
            if session_stats["samples"] >= ADAPTIVE_RULE_SWITCH_CONTEXT_MIN_SAMPLES:
                context_veto = session_stats["win_rate"] < ADAPTIVE_RULE_SWITCH_CONTEXT_DISABLE_WIN_RATE
                if context_veto:
                    return {**session_stats, "scope": "session_context_veto", "context_veto": True}
                return {**session_stats, "scope": "session_context"}

        regime_stats = self._rule_stats(rule_name, regime)
        if regime_stats["samples"] >= ADAPTIVE_RULE_SWITCH_REGIME_MIN_SAMPLES:
            return {**regime_stats, "scope": "regime"}

        if ADAPTIVE_RULE_SWITCH_ALLOW_GLOBAL_FALLBACK:
            return {**global_stats, "scope": "global_fallback"}

        return {**regime_stats, "scope": "regime_cold_start"}

    def _rule_stats(
        self,
        rule_name: str,
        regime: str | None = None,
        session: str | None = None,
        adaptive_context: str | None = None,
    ) -> dict:
        if self.records:
            rows = [
                row for row in self.records
                if row.get("rule") == rule_name
                and row.get("state_ok")
                and (regime is None or row.get("regime") == regime)
                and (session is None or row.get("session") == session)
                and (adaptive_context is None or row.get("context") == adaptive_context)
            ][-ADAPTIVE_RULE_SWITCH_ROLLING_WINDOW:]
            samples = len(rows)
            if samples == 0:
                return self._empty_stats()
            wins = sum(bool(row.get("correct")) for row in rows)
            recent_loss_streak = 0
            for row in reversed(rows):
                if bool(row.get("correct")):
                    break
                recent_loss_streak += 1
            return {
                "samples": samples,
                "wins": wins,
                "win_rate": float(wins / samples),
                "min_win_rate": ADAPTIVE_RULE_SWITCH_MIN_WIN_RATE,
                "wilson_lower": _wilson_lower_bound(wins, samples),
                "min_wilson_lower": ADAPTIVE_RULE_SWITCH_MIN_WILSON_LOWER,
                "recent_loss_streak": recent_loss_streak,
                "context_veto": False,
                "condition": "",
            }
        if not VALIDATED_STRATEGY_SIGNALS.exists() or VALIDATED_STRATEGY_SIGNALS.stat().st_size == 0:
            return self._empty_stats()
        try:
            df = pd.read_csv(VALIDATED_STRATEGY_SIGNALS)
        except Exception:
            return self._empty_stats()
        if df.empty or "reason" not in df.columns:
            return self._empty_stats()
        reason = df["reason"].fillna("").astype(str)
        mask = (
            (df.get("strategy") == self.name)
            & reason.str.contains(f"adaptive_rule={rule_name}", regex=False)
            & reason.str.contains("state_ok=True", regex=False)
        )
        if regime is not None:
            mask &= reason.str.contains(f"adaptive_regime={regime}", regex=False)
        if session is not None:
            mask &= reason.str.contains(f"adaptive_session={session}", regex=False)
        if adaptive_context is not None:
            mask &= reason.str.contains(f"adaptive_context={adaptive_context}", regex=False)
        rows = df[mask].tail(ADAPTIVE_RULE_SWITCH_ROLLING_WINDOW)
        samples = int(len(rows))
        if samples == 0:
            return self._empty_stats()
        correct = rows["correct"].astype(str).str.lower().eq("true").tolist()
        wins = int(sum(correct))
        recent_loss_streak = 0
        for value in reversed(correct):
            if value:
                break
            recent_loss_streak += 1
        return {
            "samples": samples,
            "wins": wins,
            "win_rate": float(wins / samples),
            "min_win_rate": ADAPTIVE_RULE_SWITCH_MIN_WIN_RATE,
            "wilson_lower": _wilson_lower_bound(wins, samples),
            "min_wilson_lower": ADAPTIVE_RULE_SWITCH_MIN_WILSON_LOWER,
            "recent_loss_streak": recent_loss_streak,
            "context_veto": False,
            "condition": "",
        }

    def observe_result(self, decision: StrategyDecision, correct: bool):
        if decision.direction not in {"up", "down"}:
            self._pending_observations = []
            return
        selected_rule = _reason_value(decision.reason, "adaptive_rule")
        if not selected_rule:
            self._pending_observations = []
            return
        actual_direction = decision.direction if correct else ("down" if decision.direction == "up" else "up")
        regime = _reason_value(decision.reason, "adaptive_regime")
        session = _reason_value(decision.reason, "adaptive_session")
        adaptive_context = _reason_value(decision.reason, "adaptive_context")
        condition = _reason_value(decision.reason, "miner_condition")
        scope = _reason_value(decision.reason, "rule_scope")
        try:
            timestamp_ms = int(_reason_value(decision.reason, "adaptive_timestamp_ms") or 0)
        except ValueError:
            timestamp_ms = 0
        pending = self._pending_observations or [{
            "rule": selected_rule,
            "direction": decision.direction,
            "regime": regime,
            "session": session,
            "context": adaptive_context,
            "tokens": self._context_tokens(selected_rule, decision.direction, regime, session, adaptive_context),
            "timestamp_ms": timestamp_ms,
            "state_ok": _reason_value(decision.reason, "state_ok") == "True",
        }]
        for item in pending:
            row_condition = condition if item["rule"] == selected_rule else ""
            row_scope = scope if item["rule"] == selected_rule else "candidate_shadow"
            self.records.append({
                "rule": item["rule"],
                "regime": item.get("regime", regime),
                "session": item.get("session", session),
                "context": item.get("context", adaptive_context),
                "tokens": item.get("tokens") or self._context_tokens(
                    item["rule"],
                    item["direction"],
                    item.get("regime", regime),
                    item.get("session", session),
                    item.get("context", adaptive_context),
                ),
                "condition": row_condition,
                "scope": row_scope,
                "timestamp_ms": item.get("timestamp_ms", timestamp_ms),
                "state_ok": bool(item.get("state_ok")),
                "correct": item["direction"] == actual_direction,
            })
        self._pending_observations = []
        if scope == "online_miner" and condition and not correct:
            rows = [
                row for row in self.records
                if row.get("condition") == condition
            ]
            if self._condition_loss_streak(rows) >= ADAPTIVE_RULE_SWITCH_MAX_RECENT_LOSS_STREAK:
                self.cooldowns[condition] = len(self.records) + ADAPTIVE_RULE_MINER_COOLDOWN_RECORDS


class HistoricalMatchStrategy:
    name = "historical_match"

    def __init__(self, historical_rows: pd.DataFrame | None = None):
        self.historical_rows = historical_rows if historical_rows is not None else pd.DataFrame()
        self.last_match = None

    def update_history(self, historical_rows: pd.DataFrame):
        self.historical_rows = historical_rows

    def decide(self, features, prediction: dict) -> StrategyDecision:
        p_up = float(prediction.get("up_signal_probability", 0.0))
        p_down = float(prediction.get("down_signal_probability", 0.0))
        edge = float(prediction.get("direction_edge", 0.0))
        confidence = max(p_up, p_down)
        if edge > 0.15 and p_up > p_down:
            candidate = "up"
        elif edge < -0.15 and p_down > p_up:
            candidate = "down"
        else:
            self.last_match = None
            return StrategyDecision("no_trade", confidence, "no_directional_candidate")
        rejected = _reject_trap(features, candidate, confidence)
        if rejected is not None:
            self.last_match = None
            return rejected
        result = evaluate_historical_match(self.historical_rows, features, prediction, candidate)
        self.last_match = result
        success = "None" if result.success_rate is None else f"{result.success_rate:.4f}"
        reason = f"{result.reason};matched={result.matched_signals};success_rate={success}"
        if result.accepted:
            return StrategyDecision(candidate, confidence, reason)
        return StrategyDecision("no_trade", confidence, reason)


class HistoricalMatchLongStrategy(HistoricalMatchStrategy):
    name = "historical_match_long"

    def decide(self, features, prediction: dict) -> StrategyDecision:
        p_up = float(prediction.get("up_signal_probability", 0.0))
        p_down = float(prediction.get("down_signal_probability", 0.0))
        edge = float(prediction.get("direction_edge", 0.0))
        if not (edge > 0.15 and p_up > p_down):
            self.last_match = None
            return StrategyDecision("no_trade", p_up, "no_long_candidate")
        rejected = _reject_trap(features, "up", p_up)
        if rejected is not None:
            self.last_match = None
            return rejected
        result = evaluate_historical_match(self.historical_rows, features, prediction, "up")
        self.last_match = result
        success = "None" if result.success_rate is None else f"{result.success_rate:.4f}"
        reason = f"{result.reason};matched={result.matched_signals};success_rate={success}"
        if result.accepted:
            return StrategyDecision("up", p_up, reason)
        return StrategyDecision("no_trade", p_up, reason)


class HistoricalMatchShortStrategy(HistoricalMatchStrategy):
    name = "historical_match_short"

    def decide(self, features, prediction: dict) -> StrategyDecision:
        p_up = float(prediction.get("up_signal_probability", 0.0))
        p_down = float(prediction.get("down_signal_probability", 0.0))
        edge = float(prediction.get("direction_edge", 0.0))
        if not (edge < -0.15 and p_down > p_up):
            self.last_match = None
            return StrategyDecision("no_trade", p_down, "no_short_candidate")
        rejected = _reject_trap(features, "down", p_down)
        if rejected is not None:
            self.last_match = None
            return rejected
        result = evaluate_historical_match(self.historical_rows, features, prediction, "down")
        self.last_match = result
        success = "None" if result.success_rate is None else f"{result.success_rate:.4f}"
        reason = f"{result.reason};matched={result.matched_signals};success_rate={success}"
        if result.accepted:
            return StrategyDecision("down", p_down, reason)
        return StrategyDecision("no_trade", p_down, reason)


class KronosConfirmStrategy:
    name = "kronos_confirm"

    def __init__(self):
        self.kronos_result = None

    def update_kronos_result(self, kronos_result):
        self.kronos_result = kronos_result

    def decide(self, features, prediction: dict) -> StrategyDecision:
        p_up = float(prediction.get("up_signal_probability", 0.0))
        p_down = float(prediction.get("down_signal_probability", 0.0))
        edge = float(prediction.get("direction_edge", 0.0))
        confidence = max(p_up, p_down)
        if self.kronos_result is None:
            return StrategyDecision("no_trade", confidence, "kronos_not_ready")
        if not getattr(self.kronos_result, "available", False):
            return StrategyDecision("no_trade", confidence, getattr(self.kronos_result, "reason", "kronos_unavailable"))
        if edge > 0.15 and p_up > p_down:
            candidate = "up"
        elif edge < -0.15 and p_down > p_up:
            candidate = "down"
        else:
            return StrategyDecision("no_trade", confidence, "no_dual_model_candidate")
        rejected = _reject_trap(features, candidate, confidence)
        if rejected is not None:
            return rejected
        kronos_direction = getattr(self.kronos_result, "direction", "no_trade")
        kronos_confidence = float(getattr(self.kronos_result, "confidence", 0.0))
        if kronos_direction != candidate:
            return StrategyDecision(
                "no_trade", confidence,
                f"kronos_disagree;kronos={kronos_direction};candidate={candidate};kronos_conf={kronos_confidence:.4f}",
            )
        combined_confidence = min(1.0, 0.7 * confidence + 0.3 * kronos_confidence)
        return StrategyDecision(
            candidate,
            combined_confidence,
            f"kronos_confirmed;kronos={kronos_direction};kronos_conf={kronos_confidence:.4f}",
        )


class KronosLeadStrategy:
    name = "kronos_lead"

    def __init__(self):
        self.kronos_result = None

    def update_kronos_result(self, kronos_result):
        self.kronos_result = kronos_result

    def decide(self, features, prediction: dict) -> StrategyDecision:
        p_up = float(prediction.get("up_signal_probability", 0.0))
        p_down = float(prediction.get("down_signal_probability", 0.0))
        edge = float(prediction.get("direction_edge", 0.0))
        confidence = max(p_up, p_down)
        if self.kronos_result is None:
            return StrategyDecision("no_trade", confidence, "kronos_not_ready")
        if not getattr(self.kronos_result, "available", False):
            return StrategyDecision("no_trade", confidence, getattr(self.kronos_result, "reason", "kronos_unavailable"))
        kronos_direction = getattr(self.kronos_result, "direction", "no_trade")
        kronos_confidence = float(getattr(self.kronos_result, "confidence", 0.0))
        if kronos_direction not in {"up", "down"} or kronos_confidence < KRONOS_LEAD_MIN_CONFIDENCE:
            return StrategyDecision("no_trade", confidence, f"kronos_lead_low_confidence;kronos_conf={kronos_confidence:.4f}")
        if kronos_direction == "up" and edge < -KRONOS_LEAD_MAX_OPPOSITE_EDGE:
            return StrategyDecision("no_trade", confidence, f"kronos_lead_dual_opposes_up;edge={edge:.4f}")
        if kronos_direction == "down" and edge > KRONOS_LEAD_MAX_OPPOSITE_EDGE:
            return StrategyDecision("no_trade", confidence, f"kronos_lead_dual_opposes_down;edge={edge:.4f}")
        combined_confidence = min(1.0, 0.5 * confidence + 0.5 * kronos_confidence)
        rejected = _reject_trap(features, kronos_direction, combined_confidence)
        if rejected is not None:
            return rejected
        return StrategyDecision(
            kronos_direction,
            combined_confidence,
            f"kronos_lead_signal;kronos_conf={kronos_confidence:.4f};edge={edge:.4f}",
        )


class FinStarScenarioStrategy:
    name = "finstar_scenario"

    def __init__(self):
        self.historical_rows = pd.DataFrame()
        self.kronos_result = None
        self.last_result = None

    def update_history(self, historical_rows: pd.DataFrame):
        self.historical_rows = historical_rows

    def update_kronos_result(self, kronos_result):
        self.kronos_result = kronos_result

    def decide(self, features, prediction: dict) -> StrategyDecision:
        result = evaluate_finstar_scenario(
            features=features,
            prediction=prediction,
            historical_rows=self.historical_rows,
            kronos_result=self.kronos_result,
        )
        self.last_result = result
        reason = f"finstar_scenario={result.scenario};market_state={result.market_state.market_state};{result.reason}"
        if result.accepted:
            rejected = _reject_trap(features, result.direction, result.confidence)
            if rejected is not None:
                return rejected
            return StrategyDecision(result.direction, result.confidence, reason)
        return StrategyDecision("no_trade", result.confidence, reason)


def default_strategies():
    return [
        BaselineDualStrategy(),
        ConservativeExtremeStrategy(),
        AdaptiveDualStrategy(),
        ShortMomentumStrategy(),
        LongMomentumStrategy(),
        AdaptiveRuleSwitchStrategy(),
        RelaxedScenarioStrategy(),
        HistoricalMatchStrategy(),
        HistoricalMatchLongStrategy(),
        HistoricalMatchShortStrategy(),
        KronosConfirmStrategy(),
        KronosLeadStrategy(),
        FinStarScenarioStrategy(),
        HighConfidenceFilterStrategy(),
        ScenarioAwareStrategy(),
    ]

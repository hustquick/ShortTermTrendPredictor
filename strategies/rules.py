# strategies/rules.py

import pandas as pd

from config import (
    ADAPTIVE_DUAL_MIN_CONFIDENCE,
    ADAPTIVE_DUAL_MIN_EDGE,
    ADAPTIVE_RULE_SWITCH_MIN_SAMPLES,
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
    ret_30 = feature_value(features, "ret_30")
    close_position = feature_value(features, "close_position", 0.5)
    volume_ratio_10 = feature_value(features, "volume_ratio_10", 1.0)
    taker_buy_ratio = feature_value(features, "taker_buy_ratio", 0.5)
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
    probability_context = _adaptive_probability_context(prediction)
    return (
        f"{probability_context}|{rsi_bucket}|{ret_bucket}|{position_bucket}|"
        f"{volume_bucket}|{taker_bucket}|{trend_bucket}|{volatility_bucket}"
    )


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

    def decide(self, features, prediction: dict) -> StrategyDecision:
        candidates = self._candidate_rules(features, prediction)
        if not candidates:
            return StrategyDecision("no_trade", 0.0, "adaptive_rule_switch_no_candidate")

        regime = classify_market_regime(features)
        session = _beijing_session_from_timestamp(feature_value(features, "timestamp"))
        adaptive_context = _adaptive_feature_context(features, prediction)
        scored = []
        for rule in candidates:
            stats = self._select_rule_stats(rule["name"], regime, session, adaptive_context)
            scored.append({**rule, **stats})

        active = [
            item for item in scored
            if item["samples"] >= ADAPTIVE_RULE_SWITCH_MIN_SAMPLES
            and item["win_rate"] >= ADAPTIVE_RULE_SWITCH_MIN_WIN_RATE
            and item["recent_loss_streak"] <= ADAPTIVE_RULE_SWITCH_MAX_RECENT_LOSS_STREAK
            and not item["context_veto"]
        ]
        if active:
            selected = sorted(active, key=lambda x: (x["win_rate"], x["samples"], x["confidence"]), reverse=True)[0]
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
            f"rule_recent_loss_streak={selected['recent_loss_streak']};"
            f"context_veto={str(selected['context_veto'])};"
            f"{self._state_reason(features, prediction, selected['direction'])}"
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
            "recent_loss_streak": 0,
            "context_veto": False,
        }

    def _select_rule_stats(
        self,
        rule_name: str,
        regime: str,
        session: str,
        adaptive_context: str,
    ) -> dict:
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
                "recent_loss_streak": recent_loss_streak,
                "context_veto": False,
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
            "recent_loss_streak": recent_loss_streak,
            "context_veto": False,
        }

    def observe_result(self, decision: StrategyDecision, correct: bool):
        rule = _reason_value(decision.reason, "adaptive_rule")
        if not rule:
            return
        regime = _reason_value(decision.reason, "adaptive_regime")
        session = _reason_value(decision.reason, "adaptive_session")
        adaptive_context = _reason_value(decision.reason, "adaptive_context")
        self.records.append({
            "rule": rule,
            "regime": regime,
            "session": session,
            "context": adaptive_context,
            "state_ok": _reason_value(decision.reason, "state_ok") == "True",
            "correct": bool(correct),
        })


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

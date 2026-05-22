# strategies/rules.py

import pandas as pd

from config import (
    ADAPTIVE_DUAL_MIN_CONFIDENCE,
    ADAPTIVE_DUAL_MIN_EDGE,
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
from strategies.base import StrategyDecision, feature_value


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

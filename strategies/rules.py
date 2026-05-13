# strategies/rules.py

from config import LONG_SIGNAL_THRESHOLD, SHORT_SIGNAL_THRESHOLD
from high_win_rate_filter import passes_high_win_rate_filter
from strategies.base import StrategyDecision, feature_value


class BaselineDualStrategy:
    """双子模型原始方向策略。"""

    name = "baseline_dual"

    def decide(self, features, prediction: dict) -> StrategyDecision:
        p_up = float(prediction.get("up_signal_probability", 0.0))
        p_down = float(prediction.get("down_signal_probability", 0.0))
        edge = float(prediction.get("direction_edge", 0.0))

        if p_up >= LONG_SIGNAL_THRESHOLD and edge > 0:
            return StrategyDecision("up", p_up, "p_up_above_threshold")
        if p_down >= 1.0 - SHORT_SIGNAL_THRESHOLD and edge < 0:
            return StrategyDecision("down", p_down, "p_down_above_threshold")
        return StrategyDecision("no_trade", max(p_up, p_down), "below_threshold")


class HighConfidenceFilterStrategy:
    """高置信方向过滤策略。"""

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
            return StrategyDecision(direction, confidence, "passed_high_confidence_filter")
        return StrategyDecision("no_trade", confidence, "failed_high_confidence_filter")


class ScenarioAwareStrategy:
    """较严格的场景感知方向策略。"""

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
            p_up > p_down
            and trend > 0
            and macd_hist > 0
            and macd_hist_diff >= 0
            and ret_5 > 0
            and ret_10 > 0
            and ema_5_20_diff > 0
            and rsi_14 < 70
            and boll_position < 0.92
        )

        down_scene = (
            p_down > p_up
            and trend < 0
            and macd_hist < 0
            and macd_hist_diff <= 0
            and ret_5 < 0
            and ret_10 < 0
            and ema_5_20_diff < 0
            and rsi_14 > 30
            and boll_position > 0.08
        )

        if up_scene:
            return StrategyDecision("up", p_up, "trend_continuation_up")
        if down_scene:
            return StrategyDecision("down", p_down, "trend_continuation_down")
        return StrategyDecision("no_trade", confidence, "scenario_rejected")


class ConservativeExtremeStrategy:
    """极高置信策略。"""

    name = "conservative_extreme"

    def decide(self, features, prediction: dict) -> StrategyDecision:
        p_up = float(prediction.get("up_signal_probability", 0.0))
        p_down = float(prediction.get("down_signal_probability", 0.0))
        edge = float(prediction.get("direction_edge", 0.0))

        if p_up >= 0.95 and edge >= 0.35:
            return StrategyDecision("up", p_up, "extreme_up_confidence")
        if p_down >= 0.95 and edge <= -0.35:
            return StrategyDecision("down", p_down, "extreme_down_confidence")
        return StrategyDecision("no_trade", max(p_up, p_down), "not_extreme")


class RelaxedScenarioStrategy:
    """放宽版场景策略。

    用于测试“场景过滤”是否本身有效。相比 scenario_aware，
    该策略减少 MACD diff、ret_10 等硬条件，争取更多样本。
    """

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
            p_up > p_down
            and trend >= 0
            and macd_hist >= 0
            and ret_5 >= 0
            and ema_5_20_diff >= 0
            and rsi_14 < 72
            and boll_position < 0.95
        ):
            return StrategyDecision("up", p_up, "relaxed_up_scene")

        if (
            p_down > p_up
            and trend <= 0
            and macd_hist <= 0
            and ret_5 <= 0
            and ema_5_20_diff <= 0
            and rsi_14 > 28
            and boll_position > 0.05
        ):
            return StrategyDecision("down", p_down, "relaxed_down_scene")

        return StrategyDecision("no_trade", confidence, "relaxed_scene_rejected")


class ShortMomentumStrategy:
    """做空动量策略。

    当前实验显示 baseline_dual 的做空准确率明显高于做多准确率。
    该策略只保留高置信做空，并要求短周期趋势和动量同步偏弱。
    """

    name = "short_momentum"

    def decide(self, features, prediction: dict) -> StrategyDecision:
        p_up = float(prediction.get("up_signal_probability", 0.0))
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

        return StrategyDecision("down", p_down, "short_momentum_confirmed")


class LongMomentumStrategy:
    """做多动量策略。

    当前样本中做多弱于做空，因此该策略更严格，仅用于观察是否存在少量高质量做多信号。
    """

    name = "long_momentum"

    def decide(self, features, prediction: dict) -> StrategyDecision:
        p_up = float(prediction.get("up_signal_probability", 0.0))
        p_down = float(prediction.get("down_signal_probability", 0.0))
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

        return StrategyDecision("up", p_up, "long_momentum_confirmed")


def default_strategies():
    return [
        BaselineDualStrategy(),
        ConservativeExtremeStrategy(),
        ShortMomentumStrategy(),
        LongMomentumStrategy(),
        RelaxedScenarioStrategy(),
        HighConfidenceFilterStrategy(),
        ScenarioAwareStrategy(),
    ]

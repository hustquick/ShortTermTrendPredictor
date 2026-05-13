# strategies/rules.py

from config import LONG_SIGNAL_THRESHOLD, SHORT_SIGNAL_THRESHOLD
from high_win_rate_filter import passes_high_win_rate_filter
from strategies.base import StrategyDecision, feature_value


class BaselineDualStrategy:
    """双子模型原始方向策略。

    只根据 up/down 子模型的相对优势和正式阈值判断方向，
    用于作为所有策略的基线。
    """

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
    """高置信方向过滤策略。

    目标只看方向准确率，不使用手续费、收益率或仓位逻辑。
    """

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
    """场景感知方向策略。

    借鉴“先评估场景、再预测方向”的思路：
    趋势延续场景保留顺势信号；反转风险和混沌场景过滤信号。
    """

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
    """极高置信策略。

    只保留概率与方向优势都非常极端的样本，用于测试信号数量极少时的准确率上限。
    """

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


def default_strategies():
    return [
        BaselineDualStrategy(),
        HighConfidenceFilterStrategy(),
        ScenarioAwareStrategy(),
        ConservativeExtremeStrategy(),
    ]

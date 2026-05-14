# finstar_scenario_layer.py

from dataclasses import dataclass

import pandas as pd

from historical_match_filter import evaluate_historical_match
from market_state_assessor import MarketState, assess_market_state


FINSTAR_MIN_EDGE = 0.15
FINSTAR_MIN_CONFIDENCE = 0.55
FINSTAR_STRONG_CONFIDENCE = 0.70


@dataclass
class FinStarScenarioResult:
    accepted: bool
    direction: str
    confidence: float
    scenario: str
    market_state: MarketState
    reason: str


def _candidate_from_dual_model(prediction: dict) -> tuple[str, float, str]:
    p_up = float(prediction.get("up_signal_probability", 0.0))
    p_down = float(prediction.get("down_signal_probability", 0.0))
    edge = float(prediction.get("direction_edge", 0.0))
    confidence = max(p_up, p_down)

    if confidence < FINSTAR_MIN_CONFIDENCE:
        return "no_trade", confidence, "dual_confidence_too_low"
    if edge > FINSTAR_MIN_EDGE and p_up > p_down:
        return "up", p_up, "dual_candidate_up"
    if edge < -FINSTAR_MIN_EDGE and p_down > p_up:
        return "down", p_down, "dual_candidate_down"
    return "no_trade", confidence, "dual_edge_too_small"


def _scenario_from_state(direction: str, state: MarketState) -> tuple[str, bool, str]:
    if direction not in {"up", "down"}:
        return "no_scenario", False, "no_direction"

    if state.market_state == "choppy":
        return "low_confidence_chop", False, "market_choppy"

    if direction == "up":
        if state.market_state == "trend_up":
            return "trend_continuation_up", True, "trend_supports_up"
        if state.market_state == "down_reversal_risk" and state.momentum_state in {"down_fading", "accelerating_up", "mixed"}:
            return "range_reversion_up", True, "downtrend_reversal_supports_up"
        if state.market_state == "up_reversal_risk":
            return "false_breakout_risk", False, "up_reversal_risk_rejects_chase"

    if direction == "down":
        if state.market_state == "trend_down":
            return "trend_continuation_down", True, "trend_supports_down"
        if state.market_state == "up_reversal_risk" and state.momentum_state in {"up_fading", "accelerating_down", "mixed"}:
            return "range_reversion_down", True, "uptrend_reversal_supports_down"
        if state.market_state == "down_reversal_risk":
            return "short_rebound_risk", False, "down_reversal_risk_rejects_short"

    return "scenario_rejected", False, "state_direction_mismatch"


def evaluate_finstar_scenario(
    features: pd.Series,
    prediction: dict,
    historical_rows: pd.DataFrame | None = None,
    kronos_result=None,
) -> FinStarScenarioResult:
    """FinSTaR-inspired scenario-aware decision.

    Engineering translation:
    1. Assess current state deterministically.
    2. Generate a dual-model candidate.
    3. Check whether current scenario supports the candidate.
    4. Optionally confirm with historical matched samples and Kronos.
    """
    state = assess_market_state(features)
    direction, confidence, candidate_reason = _candidate_from_dual_model(prediction)
    if direction == "no_trade":
        return FinStarScenarioResult(False, direction, confidence, "no_scenario", state, candidate_reason)

    scenario, scenario_ok, scenario_reason = _scenario_from_state(direction, state)
    if not scenario_ok:
        return FinStarScenarioResult(False, "no_trade", confidence, scenario, state, scenario_reason)

    match_reason = "historical_match_not_used"
    if historical_rows is not None and not historical_rows.empty:
        match = evaluate_historical_match(historical_rows, features, prediction, direction)
        success = "None" if match.success_rate is None else f"{match.success_rate:.4f}"
        match_reason = f"match={match.reason};matched={match.matched_signals};success_rate={success}"
        if not match.accepted and confidence < FINSTAR_STRONG_CONFIDENCE:
            return FinStarScenarioResult(False, "no_trade", confidence, scenario, state, match_reason)

    kronos_reason = "kronos_not_used"
    if kronos_result is not None:
        if getattr(kronos_result, "available", False):
            kronos_direction = getattr(kronos_result, "direction", "no_trade")
            kronos_reason = f"kronos={kronos_direction};kronos_conf={getattr(kronos_result, 'confidence', 0.0):.4f}"
            if kronos_direction != direction and confidence < FINSTAR_STRONG_CONFIDENCE:
                return FinStarScenarioResult(False, "no_trade", confidence, scenario, state, f"kronos_disagree;{kronos_reason}")
        else:
            kronos_reason = getattr(kronos_result, "reason", "kronos_unavailable")

    reason = (
        f"{candidate_reason};scenario={scenario};{scenario_reason};"
        f"state={state.market_state};{state.reason};{match_reason};{kronos_reason}"
    )
    return FinStarScenarioResult(True, direction, confidence, scenario, state, reason)

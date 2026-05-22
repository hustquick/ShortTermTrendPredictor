# core/learning_gate.py

from dataclasses import dataclass

from config import (
    ENABLE_STRATEGY_SELF_LEARNING,
    STRATEGY_LEARNING_DISABLE_WIN_RATE,
    STRATEGY_LEARNING_ENABLE_WIN_RATE,
    STRATEGY_LEARNING_MIN_SAMPLES,
    STRATEGY_LEARNING_ROLLING_WINDOW,
)


@dataclass
class LearningGateDecision:
    notify: bool
    state: str
    reason: str


class RollingLearningGate:
    """In-memory backtest equivalent of the realtime strategy learning gate."""

    def __init__(self):
        self.records: list[dict] = []

    def decide(self, strategy: str, direction: str) -> LearningGateDecision:
        if not ENABLE_STRATEGY_SELF_LEARNING:
            return LearningGateDecision(True, "disabled_by_config", "self_learning_disabled")
        if direction not in {"up", "down"}:
            return LearningGateDecision(False, "no_trade", "not_directional")

        recent = [
            row for row in self.records
            if row.get("strategy") == strategy and row.get("direction") == direction
        ][-STRATEGY_LEARNING_ROLLING_WINDOW:]
        samples = len(recent)
        if samples < STRATEGY_LEARNING_MIN_SAMPLES:
            return LearningGateDecision(False, "explore", f"learning_explore;samples={samples}")

        wins = sum(bool(row.get("correct")) for row in recent)
        win_rate = wins / samples
        if win_rate < STRATEGY_LEARNING_DISABLE_WIN_RATE:
            return LearningGateDecision(
                False,
                "disabled",
                f"learning_disabled;samples={samples};win_rate={win_rate:.4f}",
            )
        if win_rate >= STRATEGY_LEARNING_ENABLE_WIN_RATE:
            return LearningGateDecision(
                True,
                "active",
                f"learning_active;samples={samples};win_rate={win_rate:.4f}",
            )
        return LearningGateDecision(
            False,
            "probation",
            f"learning_probation_no_notify;samples={samples};win_rate={win_rate:.4f}",
        )

    def observe(self, strategy: str, direction: str, correct: bool):
        if direction not in {"up", "down"}:
            return
        self.records.append(
            {
                "strategy": strategy,
                "direction": direction,
                "correct": bool(correct),
            }
        )

# core/risk_gate.py


class RiskGate:
    """Small decision boundary for official signal eligibility."""

    def is_official(self, final_direction: str, strategy_is_allowed: bool, learning_notify: bool, quality_ok: bool) -> bool:
        return (
            final_direction in {"up", "down"}
            and strategy_is_allowed
            and learning_notify
            and quality_ok
        )

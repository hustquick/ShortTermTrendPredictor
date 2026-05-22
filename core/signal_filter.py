# core/signal_filter.py

from dataclasses import dataclass
from typing import Callable


@dataclass
class ProductionSignalFilter:
    """Adapter around the current production quality gate."""

    quality_gate: Callable

    def evaluate(
        self,
        strategy_name: str,
        raw_direction: str,
        confidence: float,
        prediction: dict,
        reason: str,
        quality_context: dict | None = None,
    ) -> tuple[bool, str]:
        return self.quality_gate(
            strategy_name=strategy_name,
            raw_direction=raw_direction,
            confidence=confidence,
            prediction=prediction,
            reason=reason,
            quality_context=quality_context,
        )

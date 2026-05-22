# signal_quality.py

from __future__ import annotations

try:
    from config import TRADABLE_RETURN_THRESHOLD
except ImportError:  # pragma: no cover
    TRADABLE_RETURN_THRESHOLD = 0.0008


def calc_future_return(current_price, future_price) -> float | None:
    try:
        current = float(current_price)
        future = float(future_price)
    except (TypeError, ValueError):
        return None
    if current <= 0:
        return None
    return future / current - 1.0


def is_direction_correct(direction: str, future_return: float | None) -> bool | None:
    if direction not in {"up", "down"} or future_return is None:
        return None
    if direction == "up":
        return future_return > 0
    return future_return < 0


def is_tradable_correct(
    direction: str,
    future_return: float | None,
    threshold: float = TRADABLE_RETURN_THRESHOLD,
) -> bool | None:
    """Direction is correct and the absolute move is large enough to be worth tracking."""
    if direction not in {"up", "down"} or future_return is None:
        return None
    if direction == "up":
        return future_return >= threshold
    return future_return <= -threshold


def enrich_validation_quality(row: dict) -> dict:
    """Add future_return and tradable correctness without removing legacy fields."""
    out = dict(row)
    future_return = out.get("future_return")
    if future_return in (None, ""):
        future_return = calc_future_return(out.get("current_price"), out.get("future_price"))
    else:
        try:
            future_return = float(future_return)
        except (TypeError, ValueError):
            future_return = None
    out["future_return"] = future_return if future_return is not None else ""

    direction = out.get("raw_direction") or out.get("predicted_direction") or out.get("direction")
    tradable = is_tradable_correct(direction, future_return)
    if tradable is not None:
        out["is_tradable_correct"] = tradable
    return out

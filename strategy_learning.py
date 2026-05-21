# strategy_learning.py

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

csv.field_size_limit(min(sys.maxsize, 2_147_483_647))

from config import (
    ENABLE_STRATEGY_SELF_LEARNING,
    STRATEGY_LEARNING_DISABLE_WIN_RATE,
    STRATEGY_LEARNING_ENABLE_WIN_RATE,
    STRATEGY_LEARNING_FEATURE_BLOCK_MIN_ERRORS,
    STRATEGY_LEARNING_MIN_SAMPLES,
    STRATEGY_LEARNING_ROLLING_WINDOW,
    STRATEGY_LEARNING_STATE_FILE,
)
from strategies.base import feature_value


@dataclass
class LearningDecision:
    notify: bool
    state: str
    reason: str


def _is_true(value) -> bool:
    return str(value).lower() == "true"


def _feature_signature(features) -> str:
    ret_10 = feature_value(features, "ret_10")
    ret_30 = feature_value(features, "ret_30")
    macd_hist = feature_value(features, "macd_hist")
    rsi_14 = feature_value(features, "rsi_14", 50.0)
    boll_position = feature_value(features, "boll_position", 0.5)
    close_position = feature_value(features, "close_position", 0.5)

    parts = [
        "ret10_pos" if ret_10 > 0 else "ret10_neg",
        "ret30_pos" if ret_30 > 0 else "ret30_neg",
        "macd_pos" if macd_hist > 0 else "macd_neg",
        "rsi_low" if rsi_14 < 45 else "rsi_high" if rsi_14 > 65 else "rsi_mid",
        "boll_low" if boll_position <= 0.15 else "boll_high" if boll_position >= 0.85 else "boll_mid",
        "close_low" if close_position <= 0.05 else "close_high" if close_position >= 0.95 else "close_mid",
    ]
    return "|".join(parts)


def _load_validated(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def build_learning_state(validated_csv: Path, state_file: Path = STRATEGY_LEARNING_STATE_FILE) -> dict:
    df = _load_validated(validated_csv)
    state = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rolling_window": STRATEGY_LEARNING_ROLLING_WINDOW,
        "min_samples": STRATEGY_LEARNING_MIN_SAMPLES,
        "strategy_direction": {},
        "feature_errors": {},
    }
    if df.empty:
        _write_state(state_file, state)
        return state

    df = df[df["predicted_direction"].isin(["up", "down"])].copy()
    if df.empty:
        _write_state(state_file, state)
        return state

    df["correct_bool"] = df["correct"].map(_is_true)
    for (strategy, direction), group in df.groupby(["strategy", "predicted_direction"]):
        recent = group.tail(STRATEGY_LEARNING_ROLLING_WINDOW)
        total = int(len(recent))
        wins = int(recent["correct_bool"].sum())
        win_rate = float(wins / total) if total else None
        key = f"{strategy}:{direction}"
        if total < STRATEGY_LEARNING_MIN_SAMPLES:
            status = "explore"
        elif win_rate < STRATEGY_LEARNING_DISABLE_WIN_RATE:
            status = "disabled"
        elif win_rate >= STRATEGY_LEARNING_ENABLE_WIN_RATE:
            status = "active"
        else:
            status = "probation"
        state["strategy_direction"][key] = {
            "strategy": strategy,
            "direction": direction,
            "samples": total,
            "wins": wins,
            "win_rate": win_rate,
            "status": status,
            "last_signal_time": str(recent.iloc[-1].get("signal_time", "")) if total else "",
        }

    _write_state(state_file, state)
    return state


def _write_state(path: Path, state: dict):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _load_state(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def _recent_error_signature_count(validated_csv: Path, strategy: str, direction: str, signature: str) -> int:
    if not validated_csv.exists() or validated_csv.stat().st_size == 0:
        return 0
    count = 0
    with open(validated_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)[-STRATEGY_LEARNING_ROLLING_WINDOW * 4 :]
    for row in rows:
        if row.get("strategy") != strategy:
            continue
        if row.get("predicted_direction") != direction:
            continue
        if _is_true(row.get("correct")):
            continue
        if row.get("feature_signature") == signature:
            count += 1
    return count


def learning_decision(validated_csv: Path, strategy: str, direction: str, features) -> LearningDecision:
    if not ENABLE_STRATEGY_SELF_LEARNING:
        return LearningDecision(True, "disabled_by_config", "self_learning_disabled")
    if direction not in {"up", "down"}:
        return LearningDecision(False, "no_trade", "not_directional")

    state = build_learning_state(validated_csv)
    key = f"{strategy}:{direction}"
    item = state.get("strategy_direction", {}).get(key)
    if item is None:
        return LearningDecision(False, "explore", "learning_no_samples")

    status = item.get("status", "explore")
    samples = int(item.get("samples", 0))
    win_rate = item.get("win_rate")
    if status == "explore":
        return LearningDecision(
            False,
            status,
            f"learning_explore;samples={samples};win_rate={win_rate:.4f}" if win_rate is not None else f"learning_explore;samples={samples}",
        )
    if status == "disabled":
        return LearningDecision(
            False,
            status,
            f"learning_disabled;samples={samples};win_rate={win_rate:.4f}",
        )
    if status == "probation":
        return LearningDecision(
            False,
            status,
            f"learning_probation_no_notify;samples={samples};win_rate={win_rate:.4f}",
        )

    signature = _feature_signature(features)
    feature_errors = _recent_error_signature_count(validated_csv, strategy, direction, signature)
    if feature_errors >= STRATEGY_LEARNING_FEATURE_BLOCK_MIN_ERRORS:
        return LearningDecision(
            False,
            "feature_blocked",
            f"learning_feature_blocked;signature={signature};errors={feature_errors}",
        )

    if win_rate is None:
        return LearningDecision(False, status, "learning_no_win_rate")
    return LearningDecision(
        status == "active",
        status,
        f"learning_{status};samples={samples};win_rate={win_rate:.4f};signature={signature}",
    )


def feature_signature(features) -> str:
    return _feature_signature(features)

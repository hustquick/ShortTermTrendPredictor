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
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def build_learning_state(validated_csv: Path, state_file: Path = STRATEGY_LEARNING_STATE_FILE) -> dict:
    df = _load_validated(validated_csv)
    state = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rolling_window": STRATEGY_LEARNING_ROLLING_WINDOW,
        "min_samples": STRATEGY_LEARNING_MIN_SAMPLES,
        "primary_metric": "correct",
        "scope": "strategy_direction_backtest_equivalent",
        "strategy_direction": {},
        "strategy_direction_regime": {},
        "feature_errors": {},
    }
    required = {"strategy", "predicted_direction", "correct"}
    if df.empty or not required.issubset(set(df.columns)):
        _write_state(state_file, state)
        return state
    df = df[df["predicted_direction"].isin(["up", "down"])].copy()
    if df.empty:
        _write_state(state_file, state)
        return state
    df["correct_bool"] = df["correct"].map(_is_true)
    for (strategy, direction), group in df.groupby(["strategy", "predicted_direction"]):
        _fill_state_item(state["strategy_direction"], f"{strategy}:{direction}", strategy, direction, group)
    _write_state(state_file, state)
    return state


def _fill_state_item(bucket: dict, key: str, strategy: str, direction: str, group: pd.DataFrame):
    recent = group.tail(STRATEGY_LEARNING_ROLLING_WINDOW)
    total = int(len(recent))
    wins = int(recent["correct_bool"].sum())
    win_rate = float(wins / total) if total else None
    if total < STRATEGY_LEARNING_MIN_SAMPLES:
        status = "explore"
    elif win_rate < STRATEGY_LEARNING_DISABLE_WIN_RATE:
        status = "disabled"
    elif win_rate >= STRATEGY_LEARNING_ENABLE_WIN_RATE:
        status = "active"
    else:
        status = "probation"
    bucket[key] = {
        "strategy": strategy,
        "direction": direction,
        "samples": total,
        "wins": wins,
        "win_rate": win_rate,
        "status": status,
        "last_signal_time": str(recent.iloc[-1].get("signal_time", "")) if total else "",
    }


def _write_state(path: Path, state: dict):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def learning_decision(validated_csv: Path, strategy: str, direction: str, features) -> LearningDecision:
    if not ENABLE_STRATEGY_SELF_LEARNING:
        return LearningDecision(True, "disabled_by_config", "self_learning_disabled")
    if direction not in {"up", "down"}:
        return LearningDecision(False, "no_trade", "not_directional")
    state = build_learning_state(validated_csv)
    item = state.get("strategy_direction", {}).get(f"{strategy}:{direction}")
    if item is None:
        return LearningDecision(False, "explore", "learning_explore;samples=0")
    status = item.get("status", "explore")
    samples = int(item.get("samples", 0))
    win_rate = item.get("win_rate")
    if status == "explore":
        return LearningDecision(False, status, f"learning_explore;samples={samples}")
    if status == "disabled":
        return LearningDecision(False, status, f"learning_disabled;samples={samples};win_rate={win_rate:.4f}")
    if status == "probation":
        return LearningDecision(False, status, f"learning_probation_no_notify;samples={samples};win_rate={win_rate:.4f}")
    if win_rate is None:
        return LearningDecision(False, status, "learning_no_win_rate")
    return LearningDecision(status == "active", status, f"learning_{status};samples={samples};win_rate={win_rate:.4f}")


def feature_signature(features) -> str:
    return _feature_signature(features)

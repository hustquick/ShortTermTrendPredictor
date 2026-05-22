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
    DATA_DIR,
    ENABLE_STRATEGY_SELF_LEARNING,
    STRATEGY_LEARNING_DISABLE_WIN_RATE,
    STRATEGY_LEARNING_ENABLE_WIN_RATE,
    STRATEGY_LEARNING_FEATURE_BLOCK_MIN_ERRORS,
    STRATEGY_LEARNING_MIN_SAMPLES,
    STRATEGY_LEARNING_ROLLING_WINDOW,
    STRATEGY_LEARNING_STATE_FILE,
)
from market_regime import classify_market_regime
from strategies.base import feature_value

SIGNAL_FUNNEL_CSV = DATA_DIR / "signal_funnel.csv"


@dataclass
class LearningDecision:
    notify: bool
    state: str
    reason: str


def _is_true(value) -> bool:
    return str(value).lower() == "true"


def _row_correct(row: dict) -> bool:
    for column in ("tradable_correct", "is_tradable_correct", "correct"):
        value = row.get(column)
        if value is None or str(value) == "" or str(value).lower() == "nan":
            continue
        return _is_true(value)
    return False


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


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_validated(path: Path) -> pd.DataFrame:
    df = _load_csv(path)
    funnel_df = _load_csv(SIGNAL_FUNNEL_CSV)
    if not funnel_df.empty:
        funnel_df = funnel_df[funnel_df.get("future_price", "") != ""].copy()
        if "final_direction" in funnel_df.columns:
            funnel_df = funnel_df[funnel_df["final_direction"].isin(["up", "down"])].copy()
        if "raw_direction" in funnel_df.columns:
            funnel_df["predicted_direction"] = funnel_df["raw_direction"]
        if "is_tradable_correct" in funnel_df.columns:
            funnel_df["tradable_correct"] = funnel_df["is_tradable_correct"]
        if "timestamp" in funnel_df.columns and "signal_time" not in funnel_df.columns:
            funnel_df["signal_time"] = funnel_df["timestamp"]
    if df.empty:
        return funnel_df
    if funnel_df.empty:
        return df
    all_cols = sorted(set(df.columns) | set(funnel_df.columns))
    return pd.concat([df.reindex(columns=all_cols), funnel_df.reindex(columns=all_cols)], ignore_index=True)


def _correct_metric_column(df: pd.DataFrame) -> str:
    if "tradable_correct" in df.columns:
        return "tradable_correct"
    if "is_tradable_correct" in df.columns:
        return "is_tradable_correct"
    return "correct"


def _correct_metric_series(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)

    values = pd.Series([""] * len(df), index=df.index, dtype=object)
    for column in ("tradable_correct", "is_tradable_correct", "correct"):
        if column not in df.columns:
            continue
        column_values = df[column]
        usable = column_values.notna() & (column_values.astype(str) != "")
        values.loc[usable & (values.astype(str) == "")] = column_values.loc[usable]
    return values.map(_is_true)


def build_learning_state(validated_csv: Path, state_file: Path = STRATEGY_LEARNING_STATE_FILE) -> dict:
    df = _load_validated(validated_csv)
    state = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rolling_window": STRATEGY_LEARNING_ROLLING_WINDOW,
        "min_samples": STRATEGY_LEARNING_MIN_SAMPLES,
        "primary_metric": "tradable_correct_if_available",
        "strategy_direction": {},
        "strategy_direction_regime": {},
        "feature_errors": {},
    }
    if df.empty or "predicted_direction" not in df.columns:
        _write_state(state_file, state)
        return state

    df = df[df["predicted_direction"].isin(["up", "down"])].copy()
    if df.empty:
        _write_state(state_file, state)
        return state

    df["correct_bool"] = _correct_metric_series(df)
    if "market_regime" not in df.columns:
        df["market_regime"] = "unknown"
    df["market_regime"] = df["market_regime"].fillna("unknown").replace("", "unknown")

    for (strategy, direction), group in df.groupby(["strategy", "predicted_direction"]):
        _fill_state_item(state["strategy_direction"], f"{strategy}:{direction}", strategy, direction, "all", group)

    for (strategy, direction, regime), group in df.groupby(["strategy", "predicted_direction", "market_regime"]):
        key = f"{strategy}:{direction}:{regime}"
        _fill_state_item(state["strategy_direction_regime"], key, strategy, direction, regime, group)

    _write_state(state_file, state)
    return state


def _fill_state_item(bucket: dict, key: str, strategy: str, direction: str, regime: str, group: pd.DataFrame):
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
        "market_regime": regime,
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


def _recent_error_signature_count(validated_csv: Path, strategy: str, direction: str, signature: str) -> int:
    df = _load_validated(validated_csv)
    if df.empty:
        return 0
    rows = df.tail(STRATEGY_LEARNING_ROLLING_WINDOW * 4).to_dict("records")
    count = 0
    for row in rows:
        if row.get("strategy") != strategy:
            continue
        if row.get("predicted_direction") != direction:
            continue
        if _row_correct(row):
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
    regime = classify_market_regime(features)
    regime_key = f"{strategy}:{direction}:{regime}"
    fallback_key = f"{strategy}:{direction}"
    item = state.get("strategy_direction_regime", {}).get(regime_key)
    scope = "regime"
    if item is None:
        item = state.get("strategy_direction", {}).get(fallback_key)
        scope = "strategy_direction"
    if item is None:
        return LearningDecision(False, "explore", f"learning_no_samples;regime={regime}")

    status = item.get("status", "explore")
    samples = int(item.get("samples", 0))
    win_rate = item.get("win_rate")
    if status == "explore":
        return LearningDecision(
            False,
            status,
            f"learning_explore;scope={scope};regime={regime};samples={samples};win_rate={win_rate:.4f}" if win_rate is not None else f"learning_explore;scope={scope};regime={regime};samples={samples}",
        )
    if status == "disabled":
        return LearningDecision(
            False,
            status,
            f"learning_disabled;scope={scope};regime={regime};samples={samples};win_rate={win_rate:.4f}",
        )
    if status == "probation":
        return LearningDecision(
            False,
            status,
            f"learning_probation_no_notify;scope={scope};regime={regime};samples={samples};win_rate={win_rate:.4f}",
        )

    signature = _feature_signature(features)
    feature_errors = _recent_error_signature_count(validated_csv, strategy, direction, signature)
    if feature_errors >= STRATEGY_LEARNING_FEATURE_BLOCK_MIN_ERRORS:
        return LearningDecision(
            False,
            "feature_blocked",
            f"learning_feature_blocked;regime={regime};signature={signature};errors={feature_errors}",
        )

    if win_rate is None:
        return LearningDecision(False, status, f"learning_no_win_rate;regime={regime}")
    return LearningDecision(
        status == "active",
        status,
        f"learning_{status};scope={scope};regime={regime};samples={samples};win_rate={win_rate:.4f};signature={signature}",
    )


def feature_signature(features) -> str:
    return _feature_signature(features)

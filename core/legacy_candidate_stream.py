import json
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

from config import DATA_DIR
from strategies.base import feature_value


LEGACY_CANDIDATE_STREAM_CSV = DATA_DIR / "legacy_recovered_selected_stream_365d_step1_update10080.csv"
LEGACY_ONLINE_CANDIDATE_STREAM_CSV = DATA_DIR / "legacy_online_candidate_stream.csv"
LEGACY_ONLINE_CANDIDATE_RULE_OUTCOMES_CSV = DATA_DIR / "legacy_online_candidate_rule_outcomes.csv"
LEGACY_CANDIDATE_FEATURE_COLUMNS = (
    "ret_5",
    "ret_10",
    "ret_30",
    "ema_10_30_diff",
    "ema_20_60_diff",
    "macd_hist",
    "rsi_14",
    "close_position",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "taker_buy_ratio",
    "trend_agreement",
)
LEGACY_CANDIDATE_STREAM_COLUMNS = [
    "timestamp",
    "current_price",
    "future_price",
    "future_return",
    "predicted_direction",
    "actual_direction",
    "up_probability",
    "confidence",
    "is_valid_signal",
    "is_correct",
    "model_trained_at",
    *LEGACY_CANDIDATE_FEATURE_COLUMNS,
    "dt",
    "rule",
    "direction",
    "correct",
    "prior_rule_win",
    "prior_rule_samples",
    "state_ok",
]
LEGACY_CANDIDATE_RULE_OUTCOME_COLUMNS = [
    "timestamp",
    "rule",
    "direction",
    "actual_direction",
    "correct",
]


def legacy_candidates(features, prediction: dict) -> list[dict]:
    p_up_raw = float(prediction.get("up_probability", 0.5))
    p_up_signal = float(prediction.get("up_signal_probability", 0.0))
    p_down_signal = float(prediction.get("down_signal_probability", 0.0))
    ret_30 = feature_value(features, "ret_30")
    macd_hist = feature_value(features, "macd_hist")
    boll_position = feature_value(features, "boll_position", 0.5)
    close_position = feature_value(features, "close_position", 0.5)
    trend = feature_value(features, "trend_agreement")

    rules = []

    def add(ok: bool, name: str, direction: str, confidence: float) -> None:
        if ok:
            rules.append({"name": name, "direction": direction, "confidence": float(confidence)})

    add(p_up_raw <= 0.45, "short_pup_le_045", "down", max(p_down_signal, 1.0 - p_up_raw))
    add(
        p_up_raw <= 0.50 and boll_position > 0.10,
        "short_pup_le_050_not_low",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        p_up_raw <= 0.45 and ret_30 <= 0 and trend < 0,
        "short_pup_le_045_ret30neg_trenddown",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(p_up_raw >= 0.98 and boll_position < 0.85, "long_pup_ge_098_not_high", "up", max(p_up_signal, p_up_raw))
    add(
        p_up_raw >= 0.85 and ret_30 >= 0 and macd_hist <= 0 and close_position < 0.95,
        "long_pup_ge_085_ret30pos_macdneg_closeok",
        "up",
        max(p_up_signal, p_up_raw),
    )
    add(p_up_raw >= 0.55 and boll_position < 0.85, "long_pup_ge_055_not_high", "up", max(p_up_signal, p_up_raw))
    return rules


def legacy_state_ok(features, prediction: dict, direction: str) -> bool:
    p_up_raw = float(prediction.get("up_probability", 0.5))
    rsi_14 = feature_value(features, "rsi_14", 50.0)
    ret_30 = feature_value(features, "ret_30")
    return direction == "down" and p_up_raw <= 0.285 and rsi_14 > 47.5 and ret_30 > -0.00013


def _stats(records: deque[bool]) -> tuple[int, int, float]:
    samples = len(records)
    if samples == 0:
        return 0, 0, 0.0
    wins = sum(bool(item) for item in records)
    return samples, wins, wins / samples


def _active_candidate(candidates: list[dict], records_by_rule: dict[str, deque[bool]]) -> dict | None:
    scored = []
    for rule in candidates:
        samples, wins, win_rate = _stats(records_by_rule[rule["name"]])
        scored.append({**rule, "prior_rule_samples": samples, "prior_rule_wins": wins, "prior_rule_win": win_rate})
    active = [item for item in scored if item["prior_rule_samples"] >= 5 and item["prior_rule_win"] >= 0.80]
    if not active:
        return None
    return sorted(
        active,
        key=lambda item: (item["prior_rule_win"], item["prior_rule_samples"], item["confidence"]),
        reverse=True,
    )[0]


class LegacyCandidateStreamGenerator:
    def __init__(
        self,
        stream_path: Path | None = None,
        history_path: Path | None = None,
        rule_outcome_path: Path | None = None,
    ):
        self.stream_path = stream_path or LEGACY_ONLINE_CANDIDATE_STREAM_CSV
        self.history_path = history_path or LEGACY_CANDIDATE_STREAM_CSV
        self.rule_outcome_path = rule_outcome_path or LEGACY_ONLINE_CANDIDATE_RULE_OUTCOMES_CSV

    def _records_by_rule(self) -> dict[str, deque[bool]]:
        records_by_rule: dict[str, deque[bool]] = defaultdict(lambda: deque(maxlen=10))
        frames = []
        for path in (self.history_path, self.stream_path):
            if path.exists():
                try:
                    frames.append(pd.read_csv(path, usecols=["timestamp", "rule", "direction", "actual_direction"]))
                except Exception:
                    continue
        if self.rule_outcome_path.exists():
            try:
                frames.append(pd.read_csv(self.rule_outcome_path))
            except Exception:
                pass
        if not frames:
            return records_by_rule
        rows = pd.concat(frames, ignore_index=True)
        rows["timestamp_dt"] = pd.to_datetime(rows["timestamp"], errors="coerce")
        rows = rows[rows["timestamp_dt"].notna()].copy()
        rows = rows.drop_duplicates(subset=["timestamp", "rule", "direction"], keep="last")
        rows = rows.sort_values("timestamp_dt").tail(5000)
        for _, row in rows.iterrows():
            rule = str(row.get("rule", ""))
            if not rule:
                continue
            records_by_rule[rule].append(str(row.get("direction", "")) == str(row.get("actual_direction", "")))
        return records_by_rule

    def selected_candidate(self, features, prediction: dict) -> tuple[dict | None, list[dict]]:
        candidates = legacy_candidates(features, prediction)
        if not candidates:
            return None, []
        return _active_candidate(candidates, self._records_by_rule()), candidates

    def pending_row(
        self,
        features,
        prediction: dict,
        current_price: float,
        signal_time: str,
        model_trained_at: str = "",
    ) -> dict | None:
        selected, candidates = self.selected_candidate(features, prediction)
        if not candidates:
            return None
        state_ok = legacy_state_ok(features, prediction, selected["direction"]) if selected is not None else False
        row = {
            "timestamp": signal_time,
            "current_price": float(current_price),
            "future_price": "",
            "future_return": "",
            "predicted_direction": selected["direction"] if selected is not None and state_ok else "no_trade",
            "actual_direction": "",
            "up_probability": float(prediction.get("up_probability", 0.5)),
            "confidence": selected["confidence"] if selected is not None else "",
            "is_valid_signal": bool(state_ok),
            "is_correct": "",
            "model_trained_at": model_trained_at,
            "dt": signal_time,
            "rule": selected["name"] if selected is not None else "",
            "direction": selected["direction"] if selected is not None else "",
            "correct": "",
            "prior_rule_win": selected["prior_rule_win"] if selected is not None else "",
            "prior_rule_samples": selected["prior_rule_samples"] if selected is not None else "",
            "state_ok": bool(state_ok),
            "all_candidates": json.dumps(
                [
                    {"rule": item["name"], "direction": item["direction"]}
                    for item in candidates
                ],
                separators=(",", ":"),
            ),
        }
        for column in LEGACY_CANDIDATE_FEATURE_COLUMNS:
            value = features.get(column, "") if hasattr(features, "get") else ""
            row[column] = "" if pd.isna(value) else value
        return row

    def append_validated(self, pending_row: dict | None, actual_direction: str, future_price: float) -> None:
        if not pending_row:
            return
        row = dict(pending_row)
        self._append_rule_outcomes(row, actual_direction)
        if not row.get("rule") or str(row.get("direction", "")) not in {"up", "down"}:
            return
        current_price = float(row["current_price"])
        direction = str(row.get("direction", ""))
        state_ok = str(row.get("state_ok", "")).lower() == "true"
        correct = direction == actual_direction
        row["future_price"] = float(future_price)
        row["future_return"] = float(future_price) / current_price - 1
        row["actual_direction"] = actual_direction
        row["is_correct"] = bool(correct) if state_ok else False
        row["correct"] = bool(correct)
        self.stream_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.stream_path.exists()
        with self.stream_path.open("a", encoding="utf-8", newline="") as f:
            import csv

            writer = csv.DictWriter(f, fieldnames=LEGACY_CANDIDATE_STREAM_COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerow({column: row.get(column, "") for column in LEGACY_CANDIDATE_STREAM_COLUMNS})

    def _append_rule_outcomes(self, row: dict, actual_direction: str) -> None:
        raw_candidates = row.get("all_candidates", "")
        if not raw_candidates:
            return
        try:
            candidates = json.loads(raw_candidates)
        except Exception:
            return
        if not isinstance(candidates, list):
            return
        self.rule_outcome_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.rule_outcome_path.exists()
        with self.rule_outcome_path.open("a", encoding="utf-8", newline="") as f:
            import csv

            writer = csv.DictWriter(f, fieldnames=LEGACY_CANDIDATE_RULE_OUTCOME_COLUMNS)
            if not exists:
                writer.writeheader()
            for candidate in candidates:
                rule = str(candidate.get("rule", ""))
                direction = str(candidate.get("direction", ""))
                if not rule or direction not in {"up", "down"}:
                    continue
                writer.writerow(
                    {
                        "timestamp": row.get("timestamp", ""),
                        "rule": rule,
                        "direction": direction,
                        "actual_direction": actual_direction,
                        "correct": direction == actual_direction,
                    }
                )

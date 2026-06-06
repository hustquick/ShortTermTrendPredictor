import json
from collections import defaultdict, deque
import os
from pathlib import Path

import pandas as pd

from config import DATA_DIR
from strategies.base import feature_value


LEGACY_CANDIDATE_STREAM_CSV = (
    Path(os.getenv(
        "LEGACY_CANDIDATE_STREAM",
        str(DATA_DIR / "legacy_recovered_selected_stream_365d_plus_online_causal_delay10_expanded_candidates.csv"),
    ))
)
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
    "mtf_3m_ret_1",
    "mtf_3m_ret_2",
    "mtf_3m_ret_3",
    "mtf_3m_ret_5",
    "mtf_3m_ema_5_20_diff",
    "mtf_3m_ema_10_30_diff",
    "mtf_3m_macd_hist",
    "mtf_3m_macd_hist_diff",
    "mtf_3m_rsi_14",
    "mtf_3m_volatility_5",
    "mtf_3m_trend_agreement",
    "mtf_3m_taker_buy_ratio",
    "mtf_3m_volume_ratio_5",
    "mtf_5m_ret_1",
    "mtf_5m_ret_2",
    "mtf_5m_ret_3",
    "mtf_5m_ret_5",
    "mtf_5m_ema_5_20_diff",
    "mtf_5m_ema_10_30_diff",
    "mtf_5m_macd_hist",
    "mtf_5m_macd_hist_diff",
    "mtf_5m_rsi_14",
    "mtf_5m_volatility_5",
    "mtf_5m_trend_agreement",
    "mtf_5m_taker_buy_ratio",
    "mtf_5m_volume_ratio_5",
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
LEGACY_ACTIVE_RULE_HISTORY = 15
LEGACY_ACTIVE_RULE_MIN_SAMPLES = 5
LEGACY_ACTIVE_RULE_MIN_WIN_RATE = 0.95


def legacy_candidates(features, prediction: dict) -> list[dict]:
    p_up_raw = float(prediction.get("up_probability", 0.5))
    p_up_signal = float(prediction.get("up_signal_probability", 0.0))
    p_down_signal = float(prediction.get("down_signal_probability", 0.0))
    ret_5 = feature_value(features, "ret_5")
    ret_10 = feature_value(features, "ret_10")
    ret_30 = feature_value(features, "ret_30")
    macd_hist = feature_value(features, "macd_hist")
    body_ratio = feature_value(features, "body_ratio")
    ema_10_30_diff = feature_value(features, "ema_10_30_diff")
    ema_20_60_diff = feature_value(features, "ema_20_60_diff")
    rsi_14 = feature_value(features, "rsi_14", 50.0)
    boll_position = feature_value(features, "boll_position", 0.5)
    close_position = feature_value(features, "close_position", 0.5)
    upper_shadow_ratio = feature_value(features, "upper_shadow_ratio")
    lower_shadow_ratio = feature_value(features, "lower_shadow_ratio")
    taker_buy_ratio = feature_value(features, "taker_buy_ratio", 0.5)
    trend_agreement = feature_value(features, "trend_agreement")
    direction_edge = p_up_signal - p_down_signal

    rules = []

    def add(ok: bool, name: str, direction: str, confidence: float) -> None:
        if ok:
            rules.append({"name": name, "direction": direction, "confidence": float(confidence)})

    # Preserve the original 10m short/long anchors; mined 10m holdout checks show
    # this short family still contributes most of the high-frequency edge.
    add(p_up_raw <= 0.45, "short_pup_le_045", "down", max(p_down_signal, 1.0 - p_up_raw))
    add(
        p_up_raw <= 0.50 and boll_position > 0.10,
        "short_pup_le_050_not_low",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        p_up_raw <= 0.45 and ret_30 <= 0 and trend_agreement < 0,
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

    add(
        p_up_raw <= 0.35 and p_up_signal >= 0.95 and ret_5 > 0.0,
        "m10_short_lowpup_conf95_ret5pos",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        p_up_raw <= 0.35 and p_up_signal >= 0.98 and ret_5 > 0.0,
        "m10_short_lowpup_conf98_ret5pos",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        p_up_raw <= 0.35 and p_up_signal >= 0.99 and ret_5 > 0.0,
        "m10_short_lowpup_conf99_ret5pos",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        p_up_raw <= 0.35 and body_ratio <= 0.20 and ret_5 > 0.0,
        "m10_short_lowpup_body20_ret5pos",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        p_up_raw <= 0.20 and body_ratio <= 0.20 and ret_5 > 0.0,
        "m10_short_pup20_body20_ret5pos",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        p_up_raw <= 0.10 and body_ratio <= 0.20 and ret_5 > 0.0,
        "m10_short_pup10_body20_ret5pos",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        p_up_raw <= 0.35 and p_up_signal >= 0.95 and lower_shadow_ratio > 0.50,
        "m10_short_lowpup_conf95_lshadow50",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        p_up_raw <= 0.35 and p_up_signal >= 0.95 and lower_shadow_ratio > 0.70,
        "m10_short_lowpup_conf95_lshadow70",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )

    add(
        p_up_raw > 0.70 and ema_20_60_diff <= -0.003 and rsi_14 <= 47.5 and lower_shadow_ratio <= 0.30,
        "m10_long_ema60neg_rsi47_lshadow30",
        "up",
        max(p_up_signal, p_up_raw),
    )
    add(
        p_up_signal > 0.95 and ema_20_60_diff <= -0.002 and close_position > 0.01 and body_ratio <= 0.70,
        "m10_long_upmodel95_ema60neg_close_body70",
        "up",
        max(p_up_signal, p_up_raw),
    )
    add(
        p_down_signal <= 0.02 and ema_20_60_diff <= -0.002 and macd_hist <= 5 and upper_shadow_ratio > 0.0,
        "m10_long_lowdown_ema60neg_macd5_ushadow",
        "up",
        max(p_up_signal, p_up_raw),
    )
    add(
        p_down_signal <= 0.01 and direction_edge <= 0.90 and ret_10 > -0.001 and taker_buy_ratio > 0.20,
        "m10_short_lowdown_edge90_ret10_taker20",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        ret_5 <= -0.003 and upper_shadow_ratio > 0.10 and lower_shadow_ratio > 0.0 and lower_shadow_ratio <= 0.70,
        "m10_long_ret5drop_ushadow_lshadow",
        "up",
        max(p_up_signal, p_up_raw),
    )
    add(
        p_up_raw > 0.90 and macd_hist <= -30 and lower_shadow_ratio > 0.10,
        "m10_long_pup90_macddeep_lshadow",
        "up",
        max(p_up_signal, p_up_raw),
    )
    add(
        p_up_raw <= 0.65 and p_down_signal <= 0.075 and upper_shadow_ratio <= 0.01 and taker_buy_ratio <= 0.90,
        "m10_short_pup65_lowdown_ushadow01_taker90",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        direction_edge <= -0.95 and ret_5 <= 0 and taker_buy_ratio > 0.45,
        "m10_short_edgeneg95_ret5_taker45",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        direction_edge > 0.95 and ema_20_60_diff <= -0.001 and rsi_14 <= 35 and trend_agreement <= -0.333333,
        "m10_long_edge95_ema60neg_rsi35_trendneg",
        "up",
        max(p_up_signal, p_up_raw),
    )
    add(
        p_up_signal <= 0.05 and direction_edge > -0.10 and ret_30 > -0.0005 and close_position > 0.0,
        "m10_short_lowup_edge_recover_close",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        direction_edge <= -0.95 and macd_hist <= 0 and body_ratio > 0.80 and upper_shadow_ratio <= 0.10,
        "m10_short_edgeneg95_macd_body_ushadow",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        p_up_signal <= 0.95 and p_down_signal <= 0.01 and ema_20_60_diff > -0.002 and ema_20_60_diff <= -0.0005,
        "m10_short_lowdown_ema60_midneg",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        macd_hist <= -30 and close_position > 0.20 and upper_shadow_ratio > 0.20,
        "m10_long_macddeep_close_ushadow",
        "up",
        max(p_up_signal, p_up_raw),
    )
    add(
        rsi_14 <= 15 and close_position > 0.70 and body_ratio > 0.05,
        "m10_long_rsi15_close70_body",
        "up",
        max(p_up_signal, p_up_raw),
    )
    add(
        direction_edge <= -0.95 and ret_10 > 0 and macd_hist <= 5 and taker_buy_ratio <= 0.90,
        "m10_short_edgeneg95_ret10pos_macd5",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        p_up_raw > 0.90 and direction_edge <= 0.30 and ret_30 <= 0.001,
        "m10_long_pup90_edge30_ret30low",
        "up",
        max(p_up_signal, p_up_raw),
    )
    add(
        ret_10 <= 0.001 and rsi_14 > 85 and lower_shadow_ratio <= 0.50 and taker_buy_ratio <= 0.90,
        "m10_short_rsi85_ret10_lshadow",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        direction_edge <= 0.30 and ema_10_30_diff <= -0.002 and taker_buy_ratio <= 0.80,
        "m10_short_edge30_ema30neg_taker80",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        p_up_signal > 0.50 and ema_20_60_diff <= -0.003 and upper_shadow_ratio <= 0.10,
        "m10_long_upmodel50_ema60neg_ushadow10",
        "up",
        max(p_up_signal, p_up_raw),
    )
    add(
        ret_10 > 0.003 and ret_30 <= 0.0005 and lower_shadow_ratio > 0.0,
        "m10_short_ret10surge_ret30flat_lshadow",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        p_down_signal <= 0.01 and macd_hist <= 0 and lower_shadow_ratio <= 0.02,
        "m10_short_lowdown_macd_lshadow02",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        ret_10 > 0.003 and rsi_14 <= 65 and close_position <= 0.90,
        "m10_short_ret10surge_rsi65_close90",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        ret_30 > 0.006 and upper_shadow_ratio > 0.20 and taker_buy_ratio > 0.55,
        "m10_long_ret30strong_ushadow_taker55",
        "up",
        max(p_up_signal, p_up_raw),
    )
    add(
        p_down_signal > 0.95 and ret_5 > 0.001 and lower_shadow_ratio <= 0.01,
        "m10_short_downmodel95_ret5_lshadow01",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        ret_30 <= -0.004 and ema_20_60_diff > -0.0005 and body_ratio > 0.20,
        "m10_long_ret30drop_ema60_recover_body",
        "up",
        max(p_up_signal, p_up_raw),
    )
    add(
        close_position > 0.70 and taker_buy_ratio <= 0.05,
        "m10_short_close70_taker05",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        rsi_14 <= 15 and close_position > 0.50 and taker_buy_ratio <= 0.40,
        "m10_long_rsi15_close50_taker40",
        "up",
        max(p_up_signal, p_up_raw),
    )
    add(
        p_down_signal > 0.90 and ret_10 > 0.003 and ret_30 <= 0.008,
        "m10_short_downmodel90_ret10_ret30cap",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    add(
        ret_30 > 0.004 and rsi_14 <= 55 and taker_buy_ratio > 0.45,
        "m10_long_ret30pos_rsi55_taker45",
        "up",
        max(p_up_signal, p_up_raw),
    )
    add(
        p_up_signal <= 0.03 and body_ratio <= 0.05 and upper_shadow_ratio <= 0.90,
        "m10_short_lowup_body05_ushadow90",
        "down",
        max(p_down_signal, 1.0 - p_up_raw),
    )
    return rules


def legacy_state_ok(features, prediction: dict, direction: str) -> bool:
    return direction in {"up", "down"}


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
    active = [
        item
        for item in scored
        if (
            item["prior_rule_samples"] >= LEGACY_ACTIVE_RULE_MIN_SAMPLES
            and item["prior_rule_win"] >= LEGACY_ACTIVE_RULE_MIN_WIN_RATE
        )
    ]
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
        records_by_rule: dict[str, deque[bool]] = defaultdict(lambda: deque(maxlen=LEGACY_ACTIVE_RULE_HISTORY))
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

from dataclasses import dataclass
import os
from pathlib import Path

import pandas as pd

from config import DATA_DIR
from config import PREDICT_HORIZON_MINUTES
from core.feature_pipeline import FeaturePipeline
from core.legacy_candidate_stream import (
    LEGACY_CANDIDATE_STREAM_CSV,
    LEGACY_ONLINE_CANDIDATE_STREAM_CSV,
    LegacyCandidateStreamGenerator,
    legacy_candidates,
)
from data_download import load_history_csv
from core.rolling_coverage_engine import (
    RollingCoverageConfig,
    build_window_item,
    build_window_items,
    load_candidate_rows,
    matches_condition,
)
from data_download import ms_to_beijing_time
from scripts.direct_feature_signal_stream import FEATURE_EXPORT_COLUMNS
from scripts.stable_rule_generator_walkforward import _discover_stable_rules
from strategies.base import feature_value
from strategies.rules import _adaptive_feature_context


DEFAULT_COVERAGE_REPORT = (
    Path(os.getenv(
        "LEGACY_COVERAGE_REPORT",
        str(DATA_DIR / "rolling_coverage_365d_plus_online_train30_cover7_min10_causal_delay10_expanded_candidates.csv"),
    ))
)
DEFAULT_VALIDATED_SIGNALS = DATA_DIR / "validated_strategy_signals.csv"
DEFAULT_CANDIDATE_STREAM = LEGACY_CANDIDATE_STREAM_CSV
ACTIVE_ORDERFLOW_RULE_ENABLED = os.getenv("ACTIVE_ORDERFLOW_RULE_ENABLED", "1").lower() not in {"0", "false", "no"}
ACTIVE_ORDERFLOW_RULE_CONDITION = "direction=down & mtf_3m_volume_ratio_5>2.0"
ACTIVE_STABLE_RULE_NAME = "active_stable_rule_generator"
ACTIVE_STABLE_RULE_CONFIDENCE = float(os.getenv("ACTIVE_STABLE_RULE_CONFIDENCE", "0.82"))
ACTIVE_STABLE_REDISCOVER_INTERVAL_MS = int(os.getenv("ACTIVE_STABLE_REDISCOVER_INTERVAL_MINUTES", "15")) * 60_000
ACTIVE_STABLE_COVER_MS = int(os.getenv("ACTIVE_STABLE_COVER_MINUTES", "60")) * 60_000
FEATURE_COLUMNS = (
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


@dataclass
class LegacyCoverageDecision:
    accepted: bool
    direction: str
    confidence: float
    rule: str
    condition: str
    reason: str


class LegacyAdaptiveCoverageGate:
    def __init__(
        self,
        report_path: Path | None = None,
        validated_path: Path | None = None,
        candidate_stream_path: Path | None = None,
        enabled: bool = True,
        online_rediscovery_enabled: bool = True,
        rediscover_interval_minutes: int = 10,
        train_days: int = 3,
        max_clauses: int = 2,
        min_samples: int = 10,
        min_signals_per_day: float = 1.0,
        min_win_rate: float = 0.65,
        min_wilson_lower: float = 0.35,
        beam_size: int = 120,
        cover_days: int = 1,
        offline_latest_windows: int = 1,
    ):
        self.report_path = report_path or DEFAULT_COVERAGE_REPORT
        self.validated_path = validated_path or DEFAULT_VALIDATED_SIGNALS
        self.candidate_stream_path = candidate_stream_path or DEFAULT_CANDIDATE_STREAM
        self.enabled = enabled
        self.online_rediscovery_enabled = online_rediscovery_enabled
        self.rediscover_interval_ms = int(rediscover_interval_minutes) * 60_000
        self.train_days = train_days
        self.max_clauses = max_clauses
        self.min_samples = min_samples
        self.min_signals_per_day = min_signals_per_day
        self.min_win_rate = min_win_rate
        self.min_wilson_lower = min_wilson_lower
        self.beam_size = beam_size
        self.cover_days = cover_days
        self.offline_latest_windows = offline_latest_windows
        self._coverage_config = RollingCoverageConfig(
            train_days=train_days,
            cover_days=cover_days,
            step_days=cover_days,
            max_clauses=max_clauses,
            min_samples=min_samples,
            min_signals_per_day=min_signals_per_day,
            min_win_rate=min_win_rate,
            min_wilson_lower=min_wilson_lower,
            beam_size=beam_size,
        )
        self._loaded = False
        self._offline_conditions: list[dict] = []
        self._online_conditions: list[dict] = []
        self._active_window_key: tuple[int, str] | None = None
        self._last_rediscover_ms = 0
        self._candidate_stream = LegacyCandidateStreamGenerator()
        self._active_stable_conditions: list[dict] = []
        self._last_stable_rediscover_ms = 0

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.enabled:
            return
        if not self.report_path.exists():
            return
        try:
            report = pd.read_csv(self.report_path)
        except Exception:
            return
        if report.empty or "condition" not in report.columns:
            return
        if "window" in report.columns:
            report = report.copy()
            report["_window_num"] = pd.to_numeric(report["window"], errors="coerce")
            max_window = report["_window_num"].max()
            if pd.notna(max_window):
                min_window = max_window - max(0, int(self.offline_latest_windows) - 1)
                report = report[report["_window_num"] >= min_window].copy()

        conditions = []
        for _, row in report.iterrows():
            condition = str(row.get("source_condition") or row.get("condition") or "").strip()
            if not condition:
                continue
            conditions.append(
                {
                    "condition": condition,
                    "window": row.get("window", ""),
                    "train_win_rate": row.get("train_win_rate", ""),
                    "train_wilson_lower": row.get("train_wilson_lower", ""),
                    "cover_win_rate": row.get("cover_win_rate", ""),
                    "cover_start": row.get("cover_start", ""),
                    "cover_end": row.get("cover_end", ""),
                }
            )
        seen = set()
        self._offline_conditions = []
        for item in reversed(conditions):
            key = item["condition"]
            if key in seen:
                continue
            seen.add(key)
            item["source"] = "offline_rolling_coverage"
            self._offline_conditions.append(item)

    @staticmethod
    def _session_from_dt(timestamp: pd.Timestamp) -> str:
        hour = int(timestamp.hour)
        if 8 <= hour < 15:
            return "asia_day"
        if 15 <= hour < 21:
            return "europe_overlap"
        if hour >= 21 or hour < 1:
            return "us_open"
        return "late_us"

    @staticmethod
    def _reason_value(reason: str, key: str) -> str:
        prefix = f"{key}="
        for part in str(reason).split(";"):
            if part.startswith(prefix):
                return part[len(prefix):]
        return ""

    def _candidate_training_rows(self) -> pd.DataFrame:
        stream_paths = [self.candidate_stream_path, LEGACY_ONLINE_CANDIDATE_STREAM_CSV]
        df = load_candidate_rows(stream_paths)
        if not df.empty:
            return df
        if not self.validated_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.validated_path)
        except Exception:
            return pd.DataFrame()
        required = {"strategy", "predicted_direction", "correct", "signal_time", "reason", *FEATURE_COLUMNS}
        if df.empty or not required.issubset(df.columns):
            return pd.DataFrame()
        df = df[df["strategy"].eq("adaptive_rule_switch")].copy()
        if df.empty:
            return pd.DataFrame()
        df["timestamp_dt"] = pd.to_datetime(df["signal_time"], errors="coerce")
        df = df[df["timestamp_dt"].notna()].copy()
        reason = df["reason"].fillna("").astype(str)
        df["timestamp"] = df["signal_time"]
        df["direction"] = df["predicted_direction"]
        df["correct_bool"] = df["correct"].astype(str).str.lower().eq("true")
        df["rule"] = [
            self._reason_value(item, "legacy_rule") or self._reason_value(item, "adaptive_rule")
            for item in reason
        ]
        df["adaptive_context"] = [
            self._reason_value(item, "adaptive_context")
            for item in reason
        ]
        df["session"] = df["timestamp_dt"].map(self._session_from_dt)
        for column in FEATURE_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        return df

    def _maybe_rediscover(self, now_ms: int) -> None:
        if not self.online_rediscovery_enabled:
            return
        if now_ms <= 0:
            return
        if self._active_condition_is_current(self._online_conditions[0], now_ms) if self._online_conditions else False:
            return
        if self._last_rediscover_ms and now_ms - self._last_rediscover_ms < self.rediscover_interval_ms:
            return
        self._last_rediscover_ms = now_ms
        df = self._candidate_training_rows()
        if df.empty:
            return
        now_dt = self._now_dt(now_ms)
        items = build_window_items(
            df,
            now_dt,
            self._coverage_config,
            source="online_rolling_coverage",
            limit=5,
            recent_lookback_days=1,
            recent_min_matches=3,
            recent_min_win_rate=self.min_win_rate,
        )
        if not items:
            self._online_conditions = []
            self._active_window_key = (-1, "no_recent_quality_coverage")
            return
        key = (
            int(items[0].get("window", 0)),
            "|".join(str(item.get("condition", "")) for item in items),
        )
        if self._active_window_key == key:
            return
        self._active_window_key = key
        self._online_conditions = items

    @staticmethod
    def _legacy_candidates(features, prediction: dict) -> list[dict]:
        return [
            {"rule": item["name"], "direction": item["direction"], "confidence": item["confidence"]}
            for item in legacy_candidates(features, prediction)
        ]

    @staticmethod
    def _session(timestamp_ms: float) -> str:
        try:
            hour = pd.to_datetime(ms_to_beijing_time(int(timestamp_ms))).hour
        except Exception:
            return "unknown"
        if 8 <= hour < 15:
            return "asia_day"
        if 15 <= hour < 21:
            return "europe_overlap"
        if hour >= 21 or hour < 1:
            return "us_open"
        return "late_us"

    @staticmethod
    def _row_for_condition(features, prediction: dict, candidate: dict) -> dict:
        row = {
            "rule": candidate["rule"],
            "direction": candidate["direction"],
            "session": LegacyAdaptiveCoverageGate._session(feature_value(features, "timestamp", 0.0)),
            "adaptive_context": _adaptive_feature_context(features, prediction),
            "up_probability": float(prediction.get("up_probability", 0.5)),
            "confidence": float(candidate["confidence"]),
        }
        for column in FEATURE_COLUMNS:
            row[column] = feature_value(features, column)
        return row

    @staticmethod
    def _now_dt(now_ms: int) -> pd.Timestamp:
        return pd.to_datetime(ms_to_beijing_time(now_ms))

    def _active_condition_is_current(self, item: dict, now_ms: int) -> bool:
        now_dt = self._now_dt(now_ms)
        cover_start = pd.to_datetime(item.get("cover_start", ""), errors="coerce")
        cover_end = pd.to_datetime(item.get("cover_end", ""), errors="coerce")
        if pd.isna(cover_start) or pd.isna(cover_end):
            return False
        return cover_start <= now_dt < cover_end

    def _active_conditions(self, now_ms: int) -> list[dict]:
        if self._online_conditions and self._active_condition_is_current(self._online_conditions[0], now_ms):
            return self._online_conditions
        if self.online_rediscovery_enabled and self._active_window_key is not None and not self._online_conditions:
            return []
        return [item for item in self._offline_conditions if self._active_condition_is_current(item, now_ms)]

    @staticmethod
    def _matches(condition: str, row: dict) -> bool:
        return matches_condition(condition, row)

    @staticmethod
    def _direction_from_condition(condition: str) -> str:
        for part in str(condition).split("&"):
            item = part.strip()
            if item == "direction=up":
                return "up"
            if item == "direction=down":
                return "down"
        return "no_trade"

    def _active_stable_decision(self, features, now_ms: int) -> LegacyCoverageDecision | None:
        if not ACTIVE_ORDERFLOW_RULE_ENABLED:
            return None
        self._maybe_rediscover_active_stable(now_ms)
        if not self._active_stable_conditions:
            return None
        matched_item = None
        matched_direction = "no_trade"
        for item in self._active_stable_conditions:
            cover_start_ms = int(item.get("cover_start_ms", 0) or 0)
            cover_end_ms = int(item.get("cover_end_ms", 0) or 0)
            if not (cover_start_ms <= now_ms < cover_end_ms):
                continue
            condition = str(item.get("condition", ""))
            direction = self._direction_from_condition(condition)
            directions = (direction,) if direction in {"up", "down"} else ("up", "down")
            for candidate_direction in directions:
                row = {
                    "rule": ACTIVE_STABLE_RULE_NAME,
                    "direction": candidate_direction,
                    "session": self._session(now_ms),
                    "adaptive_context": _adaptive_feature_context(features, {"up_probability": 0.5}),
                    "up_probability": 1.0 if candidate_direction == "up" else 0.0,
                    "confidence": ACTIVE_STABLE_RULE_CONFIDENCE,
                }
                for column in FEATURE_COLUMNS:
                    row[column] = feature_value(features, column)
                if condition and self._matches(condition, row):
                    matched_item = item
                    matched_direction = candidate_direction
                    break
            if matched_item is not None:
                break
        if matched_item is None:
            return None
        condition = str(matched_item.get("condition", ACTIVE_ORDERFLOW_RULE_CONDITION))
        reason = (
            "legacy_coverage_gate=pass;"
            f"legacy_rule={ACTIVE_STABLE_RULE_NAME};"
            f"legacy_condition={condition};"
            "legacy_source=active_stable_rule_generator;"
            "legacy_report=stable_rule_generator_walkforward_current;"
            f"legacy_window=rolling_1h_active_until_{matched_item.get('cover_end', '')};"
            f"legacy_family={matched_item.get('family', '')};"
            f"legacy_train_win_rate={matched_item.get('train_win_rate', '')};"
            f"legacy_train_wilson_lower={matched_item.get('train_wilson_lower', '')};"
            f"legacy_select_win_rate={matched_item.get('select_win_rate', '')};"
            f"legacy_select_samples={matched_item.get('select_samples', '')}"
        )
        return LegacyCoverageDecision(
            True,
            matched_direction,
            ACTIVE_STABLE_RULE_CONFIDENCE,
            ACTIVE_STABLE_RULE_NAME,
            condition,
            reason,
        )

    def _direct_feature_candidate_rows(self) -> pd.DataFrame:
        required_minutes = 3 * 24 * 60 + 6 * 60 + 360 + PREDICT_HORIZON_MINUTES + 30
        raw = load_history_csv().sort_values("timestamp").reset_index(drop=True)
        if raw.empty:
            return pd.DataFrame()
        required_start = int(raw["timestamp"].max()) - required_minutes * 60_000
        raw = raw[raw["timestamp"] >= required_start].copy().reset_index(drop=True)
        features = FeaturePipeline().build(raw)
        close_by_timestamp = raw.set_index("timestamp")["close"]
        features["future_price"] = (
            features["timestamp"] + PREDICT_HORIZON_MINUTES * 60_000
        ).map(close_by_timestamp)
        features["future_return"] = features["future_price"] / features["close"] - 1
        features = features.dropna(subset=["future_return", *FEATURE_EXPORT_COLUMNS]).copy()
        rows = []
        for _, source in features.iterrows():
            actual_direction = "up" if float(source["future_return"]) > 0 else "down"
            base = {
                "timestamp": ms_to_beijing_time(int(source["timestamp"])),
                "timestamp_dt": pd.to_datetime(ms_to_beijing_time(int(source["timestamp"]))),
                "rule": "direct_feature",
                "session": self._session(int(source["timestamp"])),
                "correct_bool": False,
            }
            for column in FEATURE_EXPORT_COLUMNS:
                base[column] = source.get(column)
            for direction in ("up", "down"):
                rows.append(
                    {
                        **base,
                        "direction": direction,
                        "correct_bool": direction == actual_direction,
                    }
                )
        return pd.DataFrame(rows)

    def _maybe_rediscover_active_stable(self, now_ms: int) -> None:
        if self._active_stable_conditions:
            first = self._active_stable_conditions[0]
            if int(first.get("cover_start_ms", 0) or 0) <= now_ms < int(first.get("cover_end_ms", 0) or 0):
                return
        if (
            self._last_stable_rediscover_ms
            and now_ms - self._last_stable_rediscover_ms < ACTIVE_STABLE_REDISCOVER_INTERVAL_MS
        ):
            return
        self._last_stable_rediscover_ms = now_ms
        candidates = self._direct_feature_candidate_rows()
        if candidates.empty:
            self._active_stable_conditions = []
            return
        anchor = candidates["timestamp_dt"].max()
        selected: list[dict] = []
        for family in ("orderflow", "mtf", "reversal", "trend"):
            for train_days in (1, 2, 3):
                for select_hours in (3, 6, 12):
                    select_start = anchor - pd.Timedelta(hours=select_hours)
                    train_start = select_start - pd.Timedelta(days=train_days)
                    selected.extend(
                        _discover_stable_rules(
                            candidates,
                            train_start=train_start,
                            train_end=select_start,
                            select_start=select_start,
                            select_end=anchor,
                            train_days=train_days,
                            max_clauses=2,
                            min_train_samples=15,
                            min_train_wr=0.62,
                            min_train_wilson=0.42,
                            train_subwindows=2,
                            min_subwindow_samples=4,
                            min_subwindow_wr=0.54,
                            min_select_samples=4,
                            min_select_wr=0.62,
                            min_select_wilson=0.15,
                            beam_size=80,
                            candidate_limit=80,
                            selected_limit=3,
                            family=family,
                        )
                    )
        deduped = []
        seen = set()
        for item in sorted(
            selected,
            key=lambda row: (
                row.get("train_min_subwindow_win_rate", 0),
                row.get("select_wilson", 0),
                row.get("select_win_rate", 0),
                row.get("train_wilson_lower", 0),
            ),
            reverse=True,
        ):
            condition = str(item.get("condition", ""))
            if not condition or condition in seen:
                continue
            seen.add(condition)
            cover_start = pd.to_datetime(ms_to_beijing_time(now_ms))
            cover_end = cover_start + pd.Timedelta(milliseconds=ACTIVE_STABLE_COVER_MS)
            deduped.append(
                {
                    **item,
                    "cover_start": str(cover_start),
                    "cover_end": str(cover_end),
                    "cover_start_ms": int(now_ms),
                    "cover_end_ms": int(now_ms + ACTIVE_STABLE_COVER_MS),
                }
            )
            if len(deduped) >= 5:
                break
        self._active_stable_conditions = deduped

    def decide(self, features, prediction: dict) -> LegacyCoverageDecision:
        self._load()
        now_ms = int(feature_value(features, "timestamp", 0.0))
        active_stable = self._active_stable_decision(features, now_ms)
        if active_stable is not None:
            return active_stable
        self._maybe_rediscover(now_ms)
        if not self.enabled:
            return LegacyCoverageDecision(False, "no_trade", 0.0, "", "", "legacy_coverage_gate_disabled")
        conditions = self._active_conditions(now_ms)
        if not conditions:
            return LegacyCoverageDecision(False, "no_trade", 0.0, "", "", "legacy_coverage_no_active_window")

        selected, _ = self._candidate_stream.selected_candidate(features, prediction)
        if selected is None:
            return LegacyCoverageDecision(False, "no_trade", 0.0, "", "", "legacy_coverage_no_selected_candidate")

        candidate = {
            "rule": selected["name"],
            "direction": selected["direction"],
            "confidence": selected["confidence"],
        }
        row = self._row_for_condition(features, prediction, candidate)
        for item in conditions:
            condition = item["condition"]
            if not self._matches(condition, row):
                continue
            reason = (
                "legacy_coverage_gate=pass;"
                f"legacy_rule={candidate['rule']};"
                f"legacy_condition={condition};"
                f"legacy_source={item.get('source', 'offline_rolling_coverage')};"
                f"legacy_report={self.report_path.name};"
                f"legacy_window={item.get('window', '')};"
                f"legacy_cover_start={item.get('cover_start', '')};"
                f"legacy_cover_end={item.get('cover_end', '')};"
                f"legacy_train_win_rate={item.get('train_win_rate', '')};"
                f"legacy_train_wilson_lower={item.get('train_wilson_lower', '')};"
                f"legacy_cover_win_rate={item.get('cover_win_rate', '')}"
            )
            return LegacyCoverageDecision(
                True,
                candidate["direction"],
                float(candidate["confidence"]),
                candidate["rule"],
                condition,
                reason,
            )
        return LegacyCoverageDecision(False, "no_trade", 0.0, "", "", "legacy_coverage_no_match")

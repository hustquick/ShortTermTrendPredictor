from dataclasses import dataclass
import os
from pathlib import Path

import pandas as pd

from config import DATA_DIR
from core.legacy_candidate_stream import (
    LEGACY_CANDIDATE_STREAM_CSV,
    LEGACY_ONLINE_CANDIDATE_STREAM_CSV,
    LegacyCandidateStreamGenerator,
    legacy_candidates,
)
from core.rolling_coverage_engine import (
    RollingCoverageConfig,
    build_window_item,
    load_candidate_rows,
    matches_condition,
)
from data_download import ms_to_beijing_time
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
        rediscover_interval_minutes: int = 30,
        train_days: int = 30,
        max_clauses: int = 3,
        min_samples: int = 60,
        min_signals_per_day: float = 5.0,
        min_win_rate: float = 0.75,
        min_wilson_lower: float = 0.68,
        beam_size: int = 120,
        cover_days: int = 7,
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
        item = build_window_item(
            df,
            now_dt,
            self._coverage_config,
            source="online_rolling_coverage",
        )
        if item is None or not item.get("condition"):
            self._online_conditions = []
            self._active_window_key = None
            return
        key = (int(item.get("window", 0)), str(item.get("condition", "")))
        if self._active_window_key == key:
            return
        self._active_window_key = key
        self._online_conditions = [item]

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
        return [item for item in self._offline_conditions if self._active_condition_is_current(item, now_ms)]

    @staticmethod
    def _matches(condition: str, row: dict) -> bool:
        return matches_condition(condition, row)

    def decide(self, features, prediction: dict) -> LegacyCoverageDecision:
        self._load()
        now_ms = int(feature_value(features, "timestamp", 0.0))
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

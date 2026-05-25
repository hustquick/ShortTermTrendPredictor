from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import DATA_DIR
from data_download import ms_to_beijing_time
from market_regime import classify_market_regime
from scripts.online_signal_filter_walkforward import _search_best_condition
from strategies.base import feature_value
from strategies.rules import _adaptive_feature_context


DEFAULT_COVERAGE_REPORT = (
    DATA_DIR / "rolling_coverage_365d_step1_update10080_train30_cover7_min5_strict_with_provenance.csv"
)
DEFAULT_VALIDATED_SIGNALS = DATA_DIR / "validated_strategy_signals.csv"
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
        offline_latest_windows: int = 1,
        offline_expiry_days: int = 2,
        duplicate_block_minutes: int = 10,
        loss_cooldown_minutes: int = 60,
        feedback_refresh_seconds: int = 60,
    ):
        self.report_path = report_path or DEFAULT_COVERAGE_REPORT
        self.validated_path = validated_path or DEFAULT_VALIDATED_SIGNALS
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
        self.offline_latest_windows = offline_latest_windows
        self.offline_expiry_days = offline_expiry_days
        self.duplicate_block_ms = int(duplicate_block_minutes) * 60_000
        self.loss_cooldown_ms = int(loss_cooldown_minutes) * 60_000
        self.feedback_refresh_ms = int(feedback_refresh_seconds) * 1_000
        self._loaded = False
        self._offline_conditions: list[dict] = []
        self._online_conditions: list[dict] = []
        self._last_rediscover_ms = 0
        self._last_feedback_refresh_ms = 0
        self._condition_cooldowns: dict[tuple[str, str, str], int] = {}
        self._recent_accepts: dict[tuple[str, str, str], int] = {}

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

    def _offline_condition_is_fresh(self, item: dict, now_ms: int) -> bool:
        if item.get("source") != "offline_rolling_coverage":
            return True
        if self.offline_expiry_days <= 0 or now_ms <= 0:
            return True
        cover_end = pd.to_datetime(item.get("cover_end", ""), errors="coerce")
        if pd.isna(cover_end):
            return True
        now_dt = pd.to_datetime(ms_to_beijing_time(now_ms))
        return now_dt <= cover_end + pd.Timedelta(days=self.offline_expiry_days)

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

    def _maybe_rediscover(self, now_ms: int) -> None:
        if not self.online_rediscovery_enabled:
            return
        if now_ms <= 0:
            return
        if self._last_rediscover_ms and now_ms - self._last_rediscover_ms < self.rediscover_interval_ms:
            return
        self._last_rediscover_ms = now_ms
        if not self.validated_path.exists():
            return
        try:
            df = pd.read_csv(self.validated_path)
        except Exception:
            return
        required = {"strategy", "predicted_direction", "correct", "signal_time", "reason", *FEATURE_COLUMNS}
        if df.empty or not required.issubset(df.columns):
            return
        df = df[df["strategy"].eq("adaptive_rule_switch")].copy()
        if df.empty:
            return
        df["timestamp_dt"] = pd.to_datetime(df["signal_time"], errors="coerce")
        df = df[df["timestamp_dt"].notna()].copy()
        now_dt = pd.to_datetime(ms_to_beijing_time(now_ms))
        cutoff = now_dt - pd.Timedelta(days=self.train_days)
        df = df[(df["timestamp_dt"] >= cutoff) & (df["timestamp_dt"] < now_dt)].copy()
        if len(df) < self.min_samples:
            return
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
        selected = _search_best_condition(
            df,
            max_clauses=self.max_clauses,
            min_samples=self.min_samples,
            min_signals_per_day=self.min_signals_per_day,
            min_win_rate=self.min_win_rate,
            min_wilson_lower=self.min_wilson_lower,
            beam_size=self.beam_size,
        )
        if selected is None:
            self._online_conditions = []
            return
        selected = {
            **selected,
            "source": "online_rediscovery",
            "window": "online",
            "cover_win_rate": "",
        }
        self._online_conditions = [selected]

    @staticmethod
    def _legacy_candidates(features, prediction: dict) -> list[dict]:
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
                rules.append({"rule": name, "direction": direction, "confidence": float(confidence)})

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

    @staticmethod
    def _legacy_state_ok(features, prediction: dict, direction: str) -> bool:
        p_up_raw = float(prediction.get("up_probability", 0.5))
        rsi_14 = feature_value(features, "rsi_14", 50.0)
        ret_30 = feature_value(features, "ret_30")
        ret_10 = feature_value(features, "ret_10")
        trend = feature_value(features, "trend_agreement")
        if direction == "down":
            strong_up_continuation = ret_10 > 0 and ret_30 > 0.001 and trend > 0
            return p_up_raw <= 0.285 and rsi_14 > 47.5 and ret_30 > -0.00013 and not strong_up_continuation
        if direction == "up":
            return p_up_raw >= 0.55 and rsi_14 < 75 and ret_30 < 0.004
        return False

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
            "regime": classify_market_regime(features),
            "adaptive_context": _adaptive_feature_context(features, prediction),
            "up_probability": float(prediction.get("up_probability", 0.5)),
            "confidence": float(candidate["confidence"]),
        }
        for column in FEATURE_COLUMNS:
            row[column] = feature_value(features, column)
        return row

    @staticmethod
    def _condition_key(condition: str, direction: str, regime: str) -> tuple[str, str, str]:
        return (condition, direction, regime)

    def _refresh_feedback(self, now_ms: int) -> None:
        if now_ms <= 0:
            return
        if self._last_feedback_refresh_ms and now_ms - self._last_feedback_refresh_ms < self.feedback_refresh_ms:
            return
        self._last_feedback_refresh_ms = now_ms
        if not self.validated_path.exists():
            return
        try:
            df = pd.read_csv(self.validated_path)
        except Exception:
            return
        required = {"strategy", "predicted_direction", "correct", "signal_time", "reason"}
        if df.empty or not required.issubset(df.columns):
            return
        df = df[df["strategy"].eq("adaptive_rule_switch")].copy()
        if df.empty:
            return
        reason = df["reason"].fillna("").astype(str)
        legacy_mask = reason.str.contains("legacy_coverage_gate=pass", regex=False)
        df = df[legacy_mask].copy()
        if df.empty:
            return
        reason = df["reason"].fillna("").astype(str)
        df["legacy_condition"] = [self._reason_value(item, "legacy_condition") for item in reason]
        df["regime"] = [self._reason_value(item, "adaptive_regime") for item in reason]
        df["timestamp_dt"] = pd.to_datetime(df["signal_time"], errors="coerce")
        df = df[df["timestamp_dt"].notna() & df["legacy_condition"].ne("")].copy()
        if df.empty:
            return
        now_dt = pd.to_datetime(ms_to_beijing_time(now_ms))
        cutoff = now_dt - pd.Timedelta(hours=6)
        df = df[df["timestamp_dt"] >= cutoff].sort_values("timestamp_dt")
        for _, row in df.tail(200).iterrows():
            is_correct = str(row.get("correct", "")).lower() == "true"
            if is_correct:
                continue
            signal_ms = int(row["timestamp_dt"].tz_localize("Asia/Shanghai").timestamp() * 1000)
            cooldown_until = signal_ms + self.loss_cooldown_ms
            direction = str(row.get("predicted_direction", ""))
            regime = str(row.get("regime", ""))
            condition = str(row.get("legacy_condition", ""))
            if direction and condition and cooldown_until > now_ms:
                key = self._condition_key(condition, direction, regime)
                self._condition_cooldowns[key] = max(self._condition_cooldowns.get(key, 0), cooldown_until)

    def _block_reason(self, item: dict, row: dict, state_ok: bool, now_ms: int) -> str:
        condition = item["condition"]
        direction = str(row.get("direction", ""))
        regime = str(row.get("regime", ""))
        if not self._offline_condition_is_fresh(item, now_ms):
            return "legacy_coverage_blocked;offline_condition_expired"
        if not state_ok:
            return "legacy_coverage_blocked;legacy_state_not_ok"
        key = self._condition_key(condition, direction, regime)
        cooldown_until = self._condition_cooldowns.get(key, 0)
        if cooldown_until > now_ms:
            return "legacy_coverage_blocked;condition_loss_cooldown"
        last_accept_ms = self._recent_accepts.get(key, 0)
        if last_accept_ms and now_ms - last_accept_ms < self.duplicate_block_ms:
            return "legacy_coverage_blocked;duplicate_condition_window"
        return ""

    @staticmethod
    def _matches(condition: str, row: dict) -> bool:
        for part in condition.split(" & "):
            part = part.strip()
            if not part:
                continue
            if part.startswith("context="):
                token = part.split("=", 1)[1]
                if token not in str(row.get("adaptive_context", "")).split("|"):
                    return False
            elif "=" in part and "<=" not in part and ">" not in part:
                column, value = part.split("=", 1)
                if str(row.get(column, "")) != value:
                    return False
            elif "<=" in part:
                column, value = part.split("<=", 1)
                if float(row.get(column, float("nan"))) > float(value):
                    return False
            elif ">" in part:
                column, value = part.split(">", 1)
                if float(row.get(column, float("nan"))) <= float(value):
                    return False
            else:
                return False
        return True

    def decide(self, features, prediction: dict) -> LegacyCoverageDecision:
        self._load()
        now_ms = int(feature_value(features, "timestamp", 0.0))
        self._maybe_rediscover(now_ms)
        self._refresh_feedback(now_ms)
        if not self.enabled:
            return LegacyCoverageDecision(False, "no_trade", 0.0, "", "", "legacy_coverage_gate_disabled")
        conditions = [*self._online_conditions, *self._offline_conditions]
        if not conditions:
            return LegacyCoverageDecision(False, "no_trade", 0.0, "", "", "legacy_coverage_no_conditions")

        last_block_reason = ""
        for candidate in self._legacy_candidates(features, prediction):
            row = self._row_for_condition(features, prediction, candidate)
            for item in conditions:
                condition = item["condition"]
                if not self._matches(condition, row):
                    continue
                state_ok = self._legacy_state_ok(features, prediction, candidate["direction"])
                block_reason = self._block_reason(item, row, state_ok, now_ms)
                if block_reason:
                    last_block_reason = block_reason
                    continue
                reason = (
                    "legacy_coverage_gate=pass;"
                    f"legacy_rule={candidate['rule']};"
                    f"legacy_condition={condition};"
                    f"legacy_state_ok={state_ok};"
                    f"legacy_source={item.get('source', 'offline_rolling_coverage')};"
                    f"legacy_report={self.report_path.name};"
                    f"legacy_window={item.get('window', '')};"
                    f"legacy_train_win_rate={item.get('train_win_rate', '')};"
                    f"legacy_train_wilson_lower={item.get('train_wilson_lower', '')};"
                    f"legacy_cover_win_rate={item.get('cover_win_rate', '')}"
                )
                key = self._condition_key(condition, candidate["direction"], str(row.get("regime", "")))
                self._recent_accepts[key] = now_ms
                return LegacyCoverageDecision(
                    True,
                    candidate["direction"],
                    float(candidate["confidence"]),
                    candidate["rule"],
                    condition,
                    reason,
                )
        return LegacyCoverageDecision(False, "no_trade", 0.0, "", "", last_block_reason or "legacy_coverage_no_match")

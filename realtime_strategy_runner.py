# realtime_strategy_runner.py

import csv
import json
import os
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

csv.field_size_limit(min(sys.maxsize, 2_147_483_647))

from config import (
    ADAPTIVE_NOTIFY_REQUIRE_CONFIRMATION,
    ADAPTIVE_NOTIFY_MIN_CONFIDENCE,
    ADAPTIVE_NOTIFY_MIN_EDGE,
    DATA_DIR,
    ALL_PREDICTIONS_CSV,
    HISTORICAL_MATCH_NOTIFY_MIN_MATCHED,
    HISTORICAL_MATCH_NOTIFY_MIN_SUCCESS_RATE,
    HISTORICAL_MATCH_CACHE_FILE,
    HISTORICAL_MATCH_CACHE_MAX_AGE_MINUTES,
    HISTORICAL_MATCH_CACHE_STALE_MAX_HOURS,
    HISTORICAL_MATCH_WALK_FORWARD_MODEL_UPDATE_MINUTES,
    KRONOS_NOTIFY_ALLOW_DOWN,
    KRONOS_NOTIFY_MIN_CONFIDENCE,
    KRONOS_RUN_MIN_CONFIDENCE,
    KRONOS_RUN_MIN_EDGE,
    OFFICIAL_SIGNAL_STRATEGY_ALLOWLIST,
    OFFICIAL_SIGNALS_CSV,
    PREDICT_HORIZON_MINUTES,
    PREDICTIONS_CSV,
    REALTIME_INTERVAL_SECONDS,
)
from core.alpha_model import AlphaModelManager
from core.data_feed import RealtimeDataFeed
from core.feature_pipeline import FeaturePipeline
from core.legacy_candidate_stream import LegacyCandidateStreamGenerator
from core.legacy_adaptive_coverage_gate import FEATURE_COLUMNS, LegacyAdaptiveCoverageGate
from core.notifier import EnterpriseWechatNotifier
from core.output_store import PredictionOutputStore
from core.risk_gate import RiskGate
from data_download import ms_to_beijing_time
from historical_match_filter import build_walk_forward_historical_match_rows
from kronos_adapter import KronosAdapter, KronosForecastResult
from strategy_learning import build_learning_state, feature_signature, learning_decision
from strategies.rules import (
    AdaptiveRuleSwitchStrategy,
    AdaptiveDualStrategy,
    FinStarScenarioStrategy,
    HistoricalMatchLongStrategy,
    HistoricalMatchShortStrategy,
    HistoricalMatchStrategy,
    KronosConfirmStrategy,
    KronosLeadStrategy,
    RelaxedScenarioStrategy,
    ShortMomentumStrategy,
)
from strategies.base import StrategyDecision


STRATEGY_MAP = {
    "short_momentum": ShortMomentumStrategy,
    "adaptive_rule_switch": AdaptiveRuleSwitchStrategy,
    "adaptive_dual": AdaptiveDualStrategy,
    "relaxed_scenario": RelaxedScenarioStrategy,
    "historical_match": HistoricalMatchStrategy,
    "historical_match_long": HistoricalMatchLongStrategy,
    "historical_match_short": HistoricalMatchShortStrategy,
    "kronos_confirm": KronosConfirmStrategy,
    "kronos_lead": KronosLeadStrategy,
    "finstar_scenario": FinStarScenarioStrategy,
}

PENDING_STRATEGY_SIGNALS = DATA_DIR / "pending_strategy_signals.jsonl"
VALIDATED_STRATEGY_SIGNALS = DATA_DIR / "validated_strategy_signals.csv"
STRATEGY_PREDICTIONS_CSV = DATA_DIR / "strategy_predictions.csv"
STRATEGY_PREDICTIONS_LATEST_CSV = DATA_DIR / "strategy_predictions_latest.csv"
PER_STRATEGY_PREDICTIONS_DIR = DATA_DIR / "strategy_predictions"
STRATEGY_CHARTS_DIR = DATA_DIR / "strategy_charts"

PREDICTION_COLUMNS = [
    "prediction_id",
    "timestamp",
    "strategy",
    "current_price",
    "predicted_direction",
    "confidence",
    "reason",
    "up_signal_probability",
    "down_signal_probability",
    "direction_edge",
    "validation_timestamp",
    "validation_status",
    "actual_direction",
    "future_price",
    "is_correct",
]

PER_STRATEGY_PREDICTION_COLUMNS = [
    "prediction_id",
    "timestamp",
    "strategy",
    "current_price",
    "raw_direction",
    "final_direction",
    "confidence",
    "reason",
    "up_signal_probability",
    "down_signal_probability",
    "direction_edge",
    "validation_timestamp",
    "validation_status",
    "actual_direction",
    "future_price",
    "is_correct",
    "notify_enabled",
]

OFFICIAL_SIGNAL_STRATEGIES = set(OFFICIAL_SIGNAL_STRATEGY_ALLOWLIST)
OUTPUT_STORE = PredictionOutputStore()
RISK_GATE = RiskGate()
NOTIFIER = EnterpriseWechatNotifier()
LEGACY_CANDIDATE_STREAM = LegacyCandidateStreamGenerator()


def parse_strategy_names(strategy_names: str):
    names = [x.strip() for x in strategy_names.split(",") if x.strip()]
    if not names:
        raise ValueError("at least one strategy is required")
    unknown = [name for name in names if name not in STRATEGY_MAP]
    if unknown:
        raise ValueError(f"unknown strategies: {unknown}; available: {list(STRATEGY_MAP)}")
    return names


def load_pending_signals() -> list[dict]:
    if not PENDING_STRATEGY_SIGNALS.exists():
        return []
    rows = []
    with open(PENDING_STRATEGY_SIGNALS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def save_pending_signals(rows: list[dict]):
    DATA_DIR.mkdir(exist_ok=True)
    tmp_path = PENDING_STRATEGY_SIGNALS.with_name(f".{PENDING_STRATEGY_SIGNALS.name}.{os.getpid()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(PENDING_STRATEGY_SIGNALS)


def _append_csv_row(path, columns: list[str], row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            current_columns = reader.fieldnames or []
            if current_columns != columns:
                rows = list(reader)
                _write_csv_rows(path, columns, rows)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in columns})


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv_rows(path: Path, columns: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    tmp_path.replace(path)


def _upsert_csv_row(path: Path, columns: list[str], row: dict, key: str = "prediction_id"):
    rows = _read_csv_rows(path)
    row_key = str(row.get(key, ""))
    replaced = False
    next_rows = []
    for old_row in rows:
        if row_key and str(old_row.get(key, "")) == row_key:
            next_rows.append({**old_row, **row})
            replaced = True
        else:
            next_rows.append(old_row)
    if not replaced:
        next_rows.append(row)
    _write_csv_rows(path, columns, next_rows)


def _strategy_prediction_csv_path(strategy_name: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in strategy_name)
    return PER_STRATEGY_PREDICTIONS_DIR / f"{safe_name}.csv"


def _load_latest_prediction_rows() -> dict[str, dict]:
    if not STRATEGY_PREDICTIONS_LATEST_CSV.exists():
        return {}
    with open(STRATEGY_PREDICTIONS_LATEST_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["prediction_id"]: row for row in reader if row.get("prediction_id")}


def _save_latest_prediction_rows(rows_by_id: dict[str, dict]):
    DATA_DIR.mkdir(exist_ok=True)
    tmp_path = STRATEGY_PREDICTIONS_LATEST_CSV.with_name(
        f".{STRATEGY_PREDICTIONS_LATEST_CSV.name}.{os.getpid()}.tmp"
    )
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for row in sorted(rows_by_id.values(), key=lambda x: str(x.get("timestamp", ""))):
            writer.writerow({col: row.get(col, "") for col in PREDICTION_COLUMNS})
    tmp_path.replace(STRATEGY_PREDICTIONS_LATEST_CSV)


def _update_latest_prediction(row: dict):
    prediction_id = row.get("prediction_id")
    if not prediction_id:
        return
    rows_by_id = _load_latest_prediction_rows()
    current = rows_by_id.get(prediction_id, {})
    merged = {**current, **row}
    if row.get("validation_status") == "validated":
        merged["validation_status"] = "validated"
    rows_by_id[prediction_id] = merged
    _save_latest_prediction_rows(rows_by_id)


def rebuild_latest_predictions_from_log():
    if not STRATEGY_PREDICTIONS_CSV.exists():
        _save_latest_prediction_rows({})
        return
    rows_by_id = {}
    with open(STRATEGY_PREDICTIONS_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prediction_id = row.get("prediction_id")
            if not prediction_id:
                continue
            current = rows_by_id.get(prediction_id, {})
            merged = {**current, **row}
            if row.get("validation_status") == "validated":
                merged["validation_status"] = "validated"
            rows_by_id[prediction_id] = merged
    _save_latest_prediction_rows(rows_by_id)


def append_prediction_csv(row: dict):
    _append_csv_row(PREDICTIONS_CSV, PREDICTION_COLUMNS, row)
    _append_csv_row(STRATEGY_PREDICTIONS_CSV, PREDICTION_COLUMNS, row)
    _update_latest_prediction(row)


def upsert_per_strategy_prediction(strategy_name: str, row: dict):
    _upsert_csv_row(
        _strategy_prediction_csv_path(strategy_name),
        PER_STRATEGY_PREDICTION_COLUMNS,
        row,
    )


def append_validated_signal(row: dict):
    columns = [
        "prediction_id",
        "strategy",
        "predicted_direction",
        "actual_direction",
        "correct",
        "signal_time",
        "validation_time",
        "signal_price",
        "validation_price",
        "confidence",
        "up_signal_probability",
        "down_signal_probability",
        "direction_edge",
        "reason",
        "feature_signature",
        "learning_state",
        "learning_reason",
        *FEATURE_COLUMNS,
    ]
    _append_csv_row(VALIDATED_STRATEGY_SIGNALS, columns, row)


def _load_strategy_accuracy_before(strategy_name: str) -> tuple[float | None, int, int]:
    if not VALIDATED_STRATEGY_SIGNALS.exists():
        return None, 0, 0
    total = 0
    correct = 0
    with open(VALIDATED_STRATEGY_SIGNALS, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("strategy") != strategy_name:
                continue
            total += 1
            if str(row.get("correct", "")).lower() == "true":
                correct += 1
    if total == 0:
        return None, correct, total
    return correct / total, correct, total


def _strategy_accuracy_after_current(strategy_name: str, current_correct: bool) -> tuple[float, int, int]:
    previous_accuracy, previous_correct, previous_total = _load_strategy_accuracy_before(strategy_name)
    correct = previous_correct + (1 if current_correct else 0)
    total = previous_total + 1
    return correct / total, correct, total


def is_official_signal_strategy(strategy_name: str) -> bool:
    return strategy_name in OFFICIAL_SIGNAL_STRATEGIES


def _extract_reason_float(reason: str, key: str) -> float | None:
    prefix = f"{key}="
    for part in str(reason).split(";"):
        if not part.startswith(prefix):
            continue
        try:
            return float(part[len(prefix):])
        except ValueError:
            return None
    return None


def _extract_reason_value(reason: str, key: str) -> str:
    prefix = f"{key}="
    for part in str(reason).split(";"):
        if part.startswith(prefix):
            return part[len(prefix):]
    return ""


def passes_production_quality_gate(
    strategy_name: str,
    raw_direction: str,
    confidence: float,
    prediction: dict,
    reason: str,
    quality_context: dict | None = None,
) -> tuple[bool, str]:
    if strategy_name == "adaptive_dual":
        edge = abs(float(prediction.get("direction_edge", 0.0)))
        if confidence < ADAPTIVE_NOTIFY_MIN_CONFIDENCE:
            return False, f"production_blocked;adaptive_confidence_below_{ADAPTIVE_NOTIFY_MIN_CONFIDENCE:.2f}"
        if edge < ADAPTIVE_NOTIFY_MIN_EDGE:
            return False, f"production_blocked;adaptive_edge_below_{ADAPTIVE_NOTIFY_MIN_EDGE:.2f}"
        if ADAPTIVE_NOTIFY_REQUIRE_CONFIRMATION:
            confirmations = (quality_context or {}).get("confirmations", {})
            if not confirmations.get(raw_direction, False):
                return False, "production_blocked;adaptive_missing_external_confirmation"
        return True, "production_quality_passed"

    if strategy_name == "adaptive_rule_switch":
        if _extract_reason_value(reason, "legacy_coverage_gate") == "pass":
            return True, "production_quality_passed;legacy_coverage_gate"
        return False, "production_blocked;adaptive_rule_switch_requires_rolling_coverage"

    if strategy_name in {"kronos_confirm", "kronos_lead"}:
        if raw_direction == "down" and not KRONOS_NOTIFY_ALLOW_DOWN:
            return False, "production_blocked;kronos_down_disabled"
        kronos_confidence = _extract_reason_float(reason, "kronos_conf") or 0.0
        if kronos_confidence < KRONOS_NOTIFY_MIN_CONFIDENCE:
            return False, f"production_blocked;kronos_conf_below_{KRONOS_NOTIFY_MIN_CONFIDENCE:.2f}"
        return True, "production_quality_passed"

    if strategy_name in {"historical_match", "historical_match_long", "historical_match_short"}:
        matched = _extract_reason_float(reason, "matched") or 0.0
        success_rate = _extract_reason_float(reason, "success_rate")
        if matched < HISTORICAL_MATCH_NOTIFY_MIN_MATCHED:
            return False, f"production_blocked;historical_match_matched_below_{HISTORICAL_MATCH_NOTIFY_MIN_MATCHED}"
        if success_rate is None or success_rate < HISTORICAL_MATCH_NOTIFY_MIN_SUCCESS_RATE:
            return False, (
                "production_blocked;"
                f"historical_match_success_below_{HISTORICAL_MATCH_NOTIFY_MIN_SUCCESS_RATE:.2f}"
            )
        return True, "production_quality_passed"

    return True, "production_quality_passed"


def _parse_beijing_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _to_float(value, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def render_strategy_chart(strategy_name: str):
    window_rows = _load_strategy_chart_window(strategy_name)
    if not window_rows:
        return

    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[realtime_strategy] chart skipped: matplotlib unavailable: {type(exc).__name__}: {exc}")
        return

    fig, (ax_price, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        dpi=120,
        gridspec_kw={"height_ratios": [3, 1.4]},
    )
    ax_conf = ax_price.twinx()
    _draw_strategy_chart_axes(strategy_name, window_rows, ax_price, ax_conf, mdates)
    _draw_confidence_accuracy_histogram(strategy_name, ax_hist)

    fig.tight_layout()

    STRATEGY_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = STRATEGY_CHARTS_DIR / f"{strategy_name}.png"
    fig.savefig(chart_path)
    plt.close(fig)


def _load_strategy_chart_window(strategy_name: str) -> list[tuple[datetime, float, float, dict]]:
    csv_path = _strategy_prediction_csv_path(strategy_name)
    rows = _read_csv_rows(csv_path)
    parsed_rows = []
    for row in rows:
        timestamp = _parse_beijing_time(row.get("timestamp", ""))
        price = _to_float(row.get("current_price"))
        confidence = _to_float(row.get("confidence"))
        if timestamp is None or price is None or confidence is None:
            continue
        parsed_rows.append((timestamp, price, confidence, row))

    if not parsed_rows:
        return []

    parsed_rows.sort(key=lambda item: item[0])
    latest_time = parsed_rows[-1][0]
    return _filter_chart_window(parsed_rows, latest_time)


def _filter_chart_window(
    parsed_rows: list[tuple[datetime, float, float, dict]],
    right_edge: datetime,
) -> list[tuple[datetime, float, float, dict]]:
    left_edge = right_edge.timestamp() - 30 * 60
    right_ts = right_edge.timestamp()
    return [item for item in parsed_rows if left_edge <= item[0].timestamp() <= right_ts]


def _draw_strategy_chart_axes(strategy_name: str, window_rows, ax_price, ax_conf, mdates, right_edge=None):
    if right_edge is None and window_rows:
        right_edge = window_rows[-1][0]
    times = [item[0] for item in window_rows]
    prices = [item[1] for item in window_rows]
    confidences = [item[2] for item in window_rows]

    if times:
        ax_price.plot(times, prices, color="#1f77b4", linewidth=1.8, label="BTC price")
    ax_price.set_ylabel("BTC price")
    ax_price.grid(False)

    ax_conf.set_ylim(0.0, 1.0)
    ax_conf.set_ylabel("confidence")
    ax_conf.set_yticks([idx / 10 for idx in range(11)])
    ax_conf.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)

    baseline = ax_conf.get_ylim()[0]
    marker_groups = {
        ("up", True): {"x": [], "y": [], "colors": []},
        ("up", False): {"x": [], "y": [], "colors": []},
        ("down", True): {"x": [], "y": [], "colors": []},
        ("down", False): {"x": [], "y": [], "colors": []},
    }

    for timestamp, _, confidence, row in window_rows:
        color = _prediction_status_color(row)
        ax_conf.vlines(
            timestamp,
            baseline,
            confidence,
            colors=color,
            linestyles="dashed",
            linewidth=0.8,
            alpha=0.65,
        )
        direction = row.get("raw_direction") if row.get("raw_direction") in {"up", "down"} else "up"
        is_official = str(row.get("notify_enabled", "")).lower() == "true"
        marker_groups[(direction, is_official)]["x"].append(timestamp)
        marker_groups[(direction, is_official)]["y"].append(confidence)
        marker_groups[(direction, is_official)]["colors"].append(color)

    for (direction, is_official), group in marker_groups.items():
        if not group["x"]:
            continue
        marker = "^" if direction == "up" else "v"
        if is_official:
            ax_conf.scatter(
                group["x"],
                group["y"],
                marker=marker,
                c=group["colors"],
                s=64,
                edgecolors="#111827",
                linewidths=0.45,
                alpha=0.95,
                zorder=5,
            )
        else:
            ax_conf.scatter(
                group["x"],
                group["y"],
                marker=marker,
                facecolors="none",
                edgecolors=group["colors"],
                s=64,
                linewidths=1.3,
                alpha=0.55,
                zorder=5,
            )

    _add_strategy_chart_legend(ax_conf)

    ax_price.set_title(f"{strategy_name} rolling 30m predictions")
    if right_edge is not None:
        ax_price.set_xlim(right_edge - timedelta(minutes=30), right_edge)
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_price.tick_params(axis="x", rotation=30)


def _prediction_status_color(row: dict) -> str:
    status = row.get("validation_status", "")
    is_correct = str(row.get("is_correct", "")).lower()
    if status != "validated":
        return "#8a8f98"
    if is_correct == "true":
        return "#2ca02c"
    return "#d62728"


def _add_strategy_chart_legend(ax_conf):
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#8a8f98", markeredgecolor="#111827", label="official up", markersize=7),
        Line2D([0], [0], marker="v", color="none", markerfacecolor="#8a8f98", markeredgecolor="#111827", label="official down", markersize=7),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="none", markeredgecolor="#8a8f98", label="observe up", markersize=7),
        Line2D([0], [0], marker="v", color="none", markerfacecolor="none", markeredgecolor="#8a8f98", label="observe down", markersize=7),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2ca02c", markeredgecolor="#2ca02c", label="correct", markersize=6),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728", markeredgecolor="#d62728", label="wrong", markersize=6),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#8a8f98", markeredgecolor="#8a8f98", label="pending", markersize=6),
    ]
    ax_conf.legend(handles=handles, loc="upper left", fontsize=8, ncols=2, framealpha=0.85)


def _confidence_accuracy_bins(strategy_name: str, bin_width: float = 0.1):
    rows = _read_csv_rows(_strategy_prediction_csv_path(strategy_name))
    bin_count = int(1.0 / bin_width)
    stats = [{"total": 0, "correct": 0} for _ in range(bin_count)]

    for row in rows:
        if row.get("validation_status") != "validated":
            continue
        if str(row.get("notify_enabled", "")).lower() != "true":
            continue
        confidence = _to_float(row.get("confidence"))
        if confidence is None:
            continue
        idx = min(bin_count - 1, max(0, int(confidence / bin_width)))
        stats[idx]["total"] += 1
        if str(row.get("is_correct", "")).lower() == "true":
            stats[idx]["correct"] += 1

    labels = []
    accuracies = []
    totals = []
    for idx, item in enumerate(stats):
        left = idx * bin_width
        right = left + bin_width
        labels.append(f"{left:.1f}-{right:.1f}")
        totals.append(item["total"])
        if item["total"]:
            accuracies.append(item["correct"] / item["total"])
        else:
            accuracies.append(0.0)
    return labels, accuracies, totals


def _draw_confidence_accuracy_histogram(strategy_name: str, ax_hist):
    labels, accuracies, totals = _confidence_accuracy_bins(strategy_name)
    positions = list(range(len(labels)))
    colors = ["#2ca02c" if total else "#c7cbd1" for total in totals]

    ax_hist.bar(positions, accuracies, color=colors, width=0.82)
    ax_hist.set_ylim(0.0, 1.0)
    ax_hist.set_ylabel("accuracy")
    ax_hist.set_xlabel("confidence bin")
    ax_hist.set_title(f"{strategy_name} notified signal accuracy by confidence")
    ax_hist.set_xticks(positions)
    ax_hist.set_xticklabels(labels, rotation=35, ha="right")
    ax_hist.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)

    for idx, (accuracy, total) in enumerate(zip(accuracies, totals)):
        if total:
            ax_hist.text(idx, min(accuracy + 0.03, 0.98), f"{accuracy:.0%}\n{total}", ha="center", va="bottom", fontsize=8)
        else:
            ax_hist.text(idx, 0.03, "0", ha="center", va="bottom", fontsize=8, color="#6b7280")


def render_strategy_charts(strategy_names: list[str]):
    for strategy_name in strategy_names:
        render_strategy_chart(strategy_name)


class LiveStrategyChartWindow:
    def __init__(self, strategy_names: list[str]):
        self.strategy_names = strategy_names
        self.enabled = False
        self.plt = None
        self.mdates = None
        self.windows = {}

    def start(self):
        try:
            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[realtime_strategy] live chart disabled: matplotlib unavailable: {type(exc).__name__}: {exc}")
            return

        self.plt = plt
        self.mdates = mdates
        plt.ion()

        for strategy_name in self.strategy_names:
            fig, (ax_price, ax_hist) = plt.subplots(
                2,
                1,
                figsize=(12, 7),
                gridspec_kw={"height_ratios": [3, 1.4]},
            )
            manager = getattr(fig.canvas, "manager", None)
            if manager is not None and hasattr(manager, "set_window_title"):
                manager.set_window_title(f"ShortTermTrendPredictor - {strategy_name}")
            ax_conf = ax_price.twinx()
            self.windows[strategy_name] = {
                "fig": fig,
                "ax_price": ax_price,
                "ax_conf": ax_conf,
                "ax_hist": ax_hist,
            }

        self.enabled = True
        self.update()

    def update(self):
        if not self.enabled:
            return

        latest_time = self._latest_time()
        if latest_time is None:
            latest_time = datetime.now()

        for strategy_name in self.strategy_names:
            window = self.windows[strategy_name]
            fig = window["fig"]
            ax_price = window["ax_price"]
            ax_conf = window["ax_conf"]
            ax_hist = window["ax_hist"]
            ax_price.clear()
            ax_conf.clear()
            ax_hist.clear()
            rows = self._load_window_for_right_edge(strategy_name, latest_time)
            _draw_strategy_chart_axes(
                strategy_name,
                rows,
                ax_price,
                ax_conf,
                self.mdates,
                right_edge=latest_time,
            )
            _draw_confidence_accuracy_histogram(strategy_name, ax_hist)
            fig.tight_layout()
            fig.canvas.draw_idle()

        self.plt.pause(0.001)

    def _latest_time(self) -> datetime | None:
        latest = None
        for strategy_name in self.strategy_names:
            rows = _read_csv_rows(_strategy_prediction_csv_path(strategy_name))
            for row in rows:
                timestamp = _parse_beijing_time(row.get("timestamp", ""))
                if timestamp is not None and (latest is None or timestamp > latest):
                    latest = timestamp
        return latest

    def _load_window_for_right_edge(self, strategy_name: str, right_edge: datetime):
        rows = _read_csv_rows(_strategy_prediction_csv_path(strategy_name))
        parsed_rows = []
        for row in rows:
            timestamp = _parse_beijing_time(row.get("timestamp", ""))
            price = _to_float(row.get("current_price"))
            confidence = _to_float(row.get("confidence"))
            if timestamp is None or price is None or confidence is None:
                continue
            parsed_rows.append((timestamp, price, confidence, row))
        parsed_rows.sort(key=lambda item: item[0])
        return _filter_chart_window(parsed_rows, right_edge)


def validate_due_signals(df, now_ms: int):
    pending = load_pending_signals()
    if not pending:
        return

    close_by_timestamp = df.set_index("timestamp")["close"]
    remaining = []
    validated_strategy_names = set()

    for row in pending:
        validation_timestamp = int(row["validation_timestamp"])
        if validation_timestamp > now_ms:
            remaining.append(row)
            continue
        if validation_timestamp not in close_by_timestamp.index:
            remaining.append(row)
            continue

        signal_price = float(row["signal_price"])
        validation_price = float(close_by_timestamp.loc[validation_timestamp])
        actual_direction = "up" if validation_price > signal_price else "down"
        predicted_direction = row.get("raw_direction") or row.get("direction")
        final_direction = row.get("final_direction") or row.get("direction")
        is_correct = predicted_direction == actual_direction
        validation_time = ms_to_beijing_time(validation_timestamp)
        strategy_accuracy, strategy_correct_count, strategy_total_count = _strategy_accuracy_after_current(
            row["strategy"],
            is_correct,
        )

        if row["strategy"] == "adaptive_rule_switch":
            LEGACY_CANDIDATE_STREAM.append_validated(
                row.get("legacy_candidate_stream_row"),
                actual_direction,
                validation_price,
            )

        if final_direction in {"up", "down"}:
            validation_row = {
                "prediction_id": row["prediction_id"],
                "strategy": row["strategy"],
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction,
                "correct": is_correct,
                "signal_time": row["signal_time"],
                "validation_time": validation_time,
                "signal_price": signal_price,
                "validation_price": validation_price,
                "confidence": row["confidence"],
                "up_signal_probability": row.get("up_signal_probability"),
                "down_signal_probability": row.get("down_signal_probability"),
                "direction_edge": row.get("direction_edge"),
                "reason": row.get("reason"),
                "feature_signature": row.get("feature_signature"),
                "learning_state": row.get("learning_state"),
                "learning_reason": row.get("learning_reason"),
                **{column: row.get(column, "") for column in FEATURE_COLUMNS},
            }
            append_validated_signal(validation_row)
            build_learning_state(VALIDATED_STRATEGY_SIGNALS)

        append_prediction_csv(
            {
                "prediction_id": row["prediction_id"],
                "timestamp": row["signal_time"],
                "strategy": row["strategy"],
                "current_price": signal_price,
                "predicted_direction": predicted_direction,
                "confidence": row["confidence"],
                "reason": row.get("reason"),
                "up_signal_probability": row.get("up_signal_probability"),
                "down_signal_probability": row.get("down_signal_probability"),
                "direction_edge": row.get("direction_edge"),
                "validation_timestamp": validation_time,
                "validation_status": "validated",
                "actual_direction": actual_direction,
                "future_price": validation_price,
                "is_correct": is_correct,
            }
        )

        upsert_per_strategy_prediction(
            row["strategy"],
            {
                "prediction_id": row["prediction_id"],
                "timestamp": row["signal_time"],
                "strategy": row["strategy"],
                "current_price": signal_price,
                "raw_direction": predicted_direction,
                "final_direction": final_direction,
                "confidence": row["confidence"],
                "reason": row.get("reason"),
                "up_signal_probability": row.get("up_signal_probability"),
                "down_signal_probability": row.get("down_signal_probability"),
                "direction_edge": row.get("direction_edge"),
                "validation_timestamp": validation_time,
                "validation_status": "validated",
                "actual_direction": actual_direction,
                "future_price": validation_price,
                "is_correct": is_correct,
                "notify_enabled": row.get("notify_enabled"),
            },
        )
        OUTPUT_STORE.record_validation(
            {
                "prediction_id": row["prediction_id"],
                "timestamp": row["signal_time"],
                "strategy": row["strategy"],
                "current_price": signal_price,
                "raw_direction": predicted_direction,
                "final_direction": final_direction,
                "confidence": row["confidence"],
                "reason": row.get("reason"),
                "up_signal_probability": row.get("up_signal_probability"),
                "down_signal_probability": row.get("down_signal_probability"),
                "direction_edge": row.get("direction_edge"),
                "validation_timestamp": validation_time,
                "validation_status": "validated",
                "actual_direction": actual_direction,
                "future_price": validation_price,
                "is_correct": is_correct,
                "notify_enabled": row.get("notify_enabled"),
            }
        )
        validated_strategy_names.add(row["strategy"])

        notify_enabled = str(row.get("notify_enabled", "")).lower() == "true"
        if notify_enabled:
            NOTIFIER.send_validation(
                strategy_name=row["strategy"],
                prediction_id=row["prediction_id"],
                predicted_direction=predicted_direction,
                actual_direction=actual_direction,
                is_correct=is_correct,
                signal_price=signal_price,
                validation_price=validation_price,
                signal_time=row["signal_time"],
                validation_time=validation_time,
                confidence=float(row["confidence"]),
                up_signal_probability=row.get("up_signal_probability"),
                down_signal_probability=row.get("down_signal_probability"),
                direction_edge=row.get("direction_edge"),
                strategy_accuracy=strategy_accuracy,
                strategy_correct_count=strategy_correct_count,
                strategy_total_count=strategy_total_count,
            )
        else:
            print(f"[realtime_strategy] validation notification skipped for observation strategy={row['strategy']}")
        print(
            "[realtime_strategy] validation: "
            f"strategy={row['strategy']}, id={row['prediction_id']}, "
            f"predicted={predicted_direction}, actual={actual_direction}, correct={is_correct}, "
            f"strategy_accuracy={strategy_accuracy:.4f}, "
            f"strategy_samples={strategy_correct_count}/{strategy_total_count}"
        )

    save_pending_signals(remaining)
    render_strategy_charts(sorted(validated_strategy_names))


def register_prediction_signal(
    strategy_name: str,
    decision,
    prediction: dict,
    current_price: float,
    signal_timestamp: int,
    signal_time: str,
    features,
    quality_context: dict | None = None,
):
    prediction_id = f"{strategy_name}-{signal_timestamp}"
    validation_timestamp = signal_timestamp + PREDICT_HORIZON_MINUTES * 60_000
    validation_time = ms_to_beijing_time(validation_timestamp)

    p_up = float(prediction.get("up_signal_probability", 0.0))
    p_down = float(prediction.get("down_signal_probability", 0.0))
    raw_direction = decision.direction if decision.direction in {"up", "down"} else ("up" if p_up >= p_down else "down")
    final_direction = decision.direction
    feature_snapshot = {}
    for column in FEATURE_COLUMNS:
        value = features.get(column, "") if hasattr(features, "get") else ""
        feature_snapshot[column] = "" if pd.isna(value) else value

    learning = learning_decision(
        VALIDATED_STRATEGY_SIGNALS,
        strategy_name,
        raw_direction,
        features,
    )
    if (
        strategy_name == "adaptive_rule_switch"
        and _extract_reason_value(decision.reason, "adaptive_mode") == "active"
    ):
        learning.notify = True
        learning.state = "delegated_to_rule_switch"
        learning.reason = f"learning_delegated_to_adaptive_rule_switch;{learning.reason}"
    if (
        strategy_name == "adaptive_rule_switch"
        and _extract_reason_value(decision.reason, "legacy_coverage_gate") == "pass"
    ):
        learning.notify = True
        learning.state = "delegated_to_legacy_coverage"
        learning.reason = f"learning_delegated_to_legacy_coverage;{learning.reason}"
    reason = f"{decision.reason};{learning.reason}"
    quality_ok, quality_reason = passes_production_quality_gate(
        strategy_name=strategy_name,
        raw_direction=raw_direction,
        confidence=float(decision.confidence),
        prediction=prediction,
        reason=reason,
        quality_context=quality_context,
    )
    notify_enabled = RISK_GATE.is_official(
        final_direction=final_direction,
        strategy_is_allowed=is_official_signal_strategy(strategy_name),
        learning_notify=learning.notify,
        quality_ok=quality_ok,
    )
    reason = f"{reason};{quality_reason}"
    skip_reasons = []
    if final_direction not in {"up", "down"}:
        skip_reasons.append(f"final_direction={final_direction}")
    if not is_official_signal_strategy(strategy_name):
        skip_reasons.append("not_official_strategy")
    if not learning.notify:
        skip_reasons.append(f"learning={learning.state}")
    if not quality_ok:
        skip_reasons.append(f"quality={quality_reason}")

    row = {
        "prediction_id": prediction_id,
        "strategy": strategy_name,
        "direction": raw_direction,
        "raw_direction": raw_direction,
        "final_direction": final_direction,
        "confidence": float(decision.confidence),
        "reason": reason,
        "signal_price": float(current_price),
        "signal_timestamp": int(signal_timestamp),
        "signal_time": signal_time,
        "validation_timestamp": int(validation_timestamp),
        "up_signal_probability": prediction.get("up_signal_probability"),
        "down_signal_probability": prediction.get("down_signal_probability"),
        "direction_edge": prediction.get("direction_edge"),
        "notify_enabled": notify_enabled,
        "feature_signature": feature_signature(features),
        "learning_state": learning.state,
        "learning_reason": learning.reason,
        "legacy_candidate_stream_row": (
            LEGACY_CANDIDATE_STREAM.pending_row(
                features=features,
                prediction=prediction,
                current_price=float(current_price),
                signal_time=signal_time,
            )
            if strategy_name == "adaptive_rule_switch"
            else None
        ),
        **feature_snapshot,
    }

    upsert_per_strategy_prediction(
        strategy_name,
        {
            "prediction_id": prediction_id,
            "timestamp": signal_time,
            "strategy": strategy_name,
            "current_price": float(current_price),
            "raw_direction": raw_direction,
            "final_direction": final_direction,
            "confidence": float(decision.confidence),
            "reason": reason,
            "up_signal_probability": prediction.get("up_signal_probability"),
            "down_signal_probability": prediction.get("down_signal_probability"),
            "direction_edge": prediction.get("direction_edge"),
            "validation_timestamp": validation_time,
            "validation_status": "pending",
            "actual_direction": "",
            "future_price": "",
            "is_correct": "",
            "notify_enabled": notify_enabled,
        },
    )
    OUTPUT_STORE.record_prediction(
        {
            "prediction_id": prediction_id,
            "timestamp": signal_time,
            "strategy": strategy_name,
            "current_price": float(current_price),
            "raw_direction": raw_direction,
            "final_direction": final_direction,
            "confidence": float(decision.confidence),
            "reason": reason,
            "up_signal_probability": prediction.get("up_signal_probability"),
            "down_signal_probability": prediction.get("down_signal_probability"),
            "direction_edge": prediction.get("direction_edge"),
            "validation_timestamp": validation_time,
            "validation_status": "pending",
            "actual_direction": "",
            "future_price": "",
            "is_correct": "",
            "notify_enabled": notify_enabled,
        }
    )

    pending = load_pending_signals()
    already_pending = False
    next_pending = []
    for pending_row in pending:
        if pending_row.get("prediction_id") == prediction_id:
            next_pending.append(row)
            already_pending = True
        else:
            next_pending.append(pending_row)
    if already_pending:
        save_pending_signals(next_pending)
        print(f"[realtime_strategy] prediction already pending, per-strategy row refreshed: {prediction_id}")
        return

    next_pending.append(row)
    save_pending_signals(next_pending)

    append_prediction_csv(
        {
            "prediction_id": prediction_id,
            "timestamp": signal_time,
            "strategy": strategy_name,
            "current_price": float(current_price),
            "predicted_direction": raw_direction,
            "confidence": float(decision.confidence),
            "reason": reason,
            "up_signal_probability": prediction.get("up_signal_probability"),
            "down_signal_probability": prediction.get("down_signal_probability"),
            "direction_edge": prediction.get("direction_edge"),
            "validation_timestamp": validation_time,
            "validation_status": "pending",
            "actual_direction": "",
            "future_price": "",
            "is_correct": "",
        }
    )

    if notify_enabled:
        NOTIFIER.send_prediction(
            strategy_name=strategy_name,
            direction=raw_direction,
            confidence=float(decision.confidence),
            current_price=float(current_price),
            timestamp=signal_time,
            reason=reason,
            prediction_id=prediction_id,
            up_signal_probability=prediction.get("up_signal_probability"),
            down_signal_probability=prediction.get("down_signal_probability"),
            direction_edge=prediction.get("direction_edge"),
            horizon_minutes=PREDICT_HORIZON_MINUTES,
        )
    else:
        print(
            "[realtime_strategy] prediction notification skipped: "
            f"strategy={strategy_name}, reason={';'.join(skip_reasons) or 'unknown'}, "
            f"learning_reason={learning.reason}"
        )


def _refresh_historical_match_rows(df, feature_df, model):
    historical_feature_df = feature_df.iloc[:-PREDICT_HORIZON_MINUTES].copy()
    source_end_ms = int(df.iloc[-1]["timestamp"]) if not df.empty else 0
    cache_payload = _load_historical_match_cache(source_end_ms)
    if cache_payload is not None:
        rows = cache_payload["rows"]
        cache_time = ms_to_beijing_time(int(cache_payload["source_end_ms"]))
        stale_note = " stale_cache" if cache_payload.get("stale") else ""
        print(
            "[realtime_strategy] walk_forward_historical_match_rows="
            f"{len(rows)} loaded_from_cache{stale_note} source_end={cache_time}"
        )
        return rows

    rows = build_walk_forward_historical_match_rows(historical_feature_df, df)
    _save_historical_match_cache(rows, source_end_ms)
    print(f"[realtime_strategy] walk_forward_historical_match_rows={len(rows)}")
    return rows


def _load_historical_match_cache(current_source_end_ms: int) -> dict | None:
    if not HISTORICAL_MATCH_CACHE_FILE.exists() or HISTORICAL_MATCH_CACHE_FILE.stat().st_size == 0:
        return None
    try:
        with open(HISTORICAL_MATCH_CACHE_FILE, "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        print(f"[realtime_strategy] historical match cache ignored: {type(exc).__name__}: {exc}")
        return None

    rows = payload.get("rows")
    source_end_ms = int(payload.get("source_end_ms", 0))
    model_update_minutes = int(payload.get("model_update_minutes", 0))
    if rows is None or source_end_ms <= 0:
        return None
    if model_update_minutes != HISTORICAL_MATCH_WALK_FORWARD_MODEL_UPDATE_MINUTES:
        return None
    max_age_ms = int(HISTORICAL_MATCH_CACHE_MAX_AGE_MINUTES) * 60_000
    cache_age_ms = current_source_end_ms - source_end_ms
    if cache_age_ms > max_age_ms:
        stale_max_ms = int(HISTORICAL_MATCH_CACHE_STALE_MAX_HOURS) * 60 * 60_000
        if cache_age_ms <= stale_max_ms:
            payload["stale"] = True
            return payload
        return None
    payload["stale"] = False
    return payload


def _save_historical_match_cache(rows, source_end_ms: int):
    payload = {
        "source_end_ms": int(source_end_ms),
        "model_update_minutes": int(HISTORICAL_MATCH_WALK_FORWARD_MODEL_UPDATE_MINUTES),
        "rows": rows,
    }
    try:
        HISTORICAL_MATCH_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = HISTORICAL_MATCH_CACHE_FILE.with_name(
            f".{HISTORICAL_MATCH_CACHE_FILE.name}.{os.getpid()}.tmp"
        )
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f)
        tmp_path.replace(HISTORICAL_MATCH_CACHE_FILE)
    except Exception as exc:
        print(f"[realtime_strategy] historical match cache save skipped: {type(exc).__name__}: {exc}")


def _update_historical_strategy_context(strategies: list, historical_rows):
    for strategy in strategies:
        if hasattr(strategy, "update_history"):
            strategy.update_history(historical_rows)


def _update_kronos_strategy_context(strategies: list, kronos_result):
    for strategy in strategies:
        if hasattr(strategy, "update_kronos_result"):
            strategy.update_kronos_result(kronos_result)


def _skipped_kronos_result(reason: str) -> KronosForecastResult:
    return KronosForecastResult(False, "no_trade", 0.0, None, None, reason)


def _should_run_kronos(prediction: dict) -> tuple[bool, str]:
    p_up = float(prediction.get("up_signal_probability", 0.0))
    p_down = float(prediction.get("down_signal_probability", 0.0))
    edge = abs(float(prediction.get("direction_edge", 0.0)))
    confidence = max(p_up, p_down)
    if edge < KRONOS_RUN_MIN_EDGE:
        return False, f"kronos_skipped_edge_below_{KRONOS_RUN_MIN_EDGE:.2f}"
    if confidence < KRONOS_RUN_MIN_CONFIDENCE:
        return False, f"kronos_skipped_confidence_below_{KRONOS_RUN_MIN_CONFIDENCE:.2f}"
    return True, "kronos_run_candidate"


def _build_quality_context(strategy_decisions: list[tuple[str, object]]) -> dict:
    confirmations = {"up": False, "down": False}
    for strategy_name, decision in strategy_decisions:
        direction = getattr(decision, "direction", "no_trade")
        if direction not in {"up", "down"}:
            continue
        reason = getattr(decision, "reason", "")
        if strategy_name in {"historical_match", "historical_match_long", "historical_match_short"}:
            confirmations[direction] = True
        elif strategy_name in {"kronos_confirm", "kronos_lead"}:
            kronos_confidence = _extract_reason_float(reason, "kronos_conf") or 0.0
            if kronos_confidence >= KRONOS_NOTIFY_MIN_CONFIDENCE:
                confirmations[direction] = True
    return {"confirmations": confirmations}


def run_realtime_strategies(
    strategy_names: str = "short_momentum,relaxed_scenario,historical_match,kronos_confirm,finstar_scenario",
    train_minutes: int = 48 * 60,
    once: bool = False,
    update_cache: bool = True,
    live_chart: bool = False,
):
    names = parse_strategy_names(strategy_names)
    strategies = [STRATEGY_MAP[name]() for name in names]
    use_kronos = any(name in {"kronos_confirm", "kronos_lead"} for name in names)
    kronos_adapter = KronosAdapter() if use_kronos else None
    data_feed = RealtimeDataFeed(minutes=train_minutes, update_cache=update_cache)
    feature_pipeline = FeaturePipeline()
    # Match the legacy candidate stream used by the long walk-forward coverage backtest.
    alpha_model = AlphaModelManager(retrain_interval_seconds=10080 * 60)

    print("[realtime_strategy] start")
    print(f"[realtime_strategy] strategies={','.join(names)}")
    print(f"[realtime_strategy] official_notification_allowlist={','.join(sorted(OFFICIAL_SIGNAL_STRATEGIES))}")
    print(f"[realtime_strategy] supported strategies: {','.join(STRATEGY_MAP.keys())}")
    print("[realtime_strategy] objective=high-confidence directional accuracy only")
    print(f"[realtime_strategy] all_predictions_csv={ALL_PREDICTIONS_CSV}")
    print(f"[realtime_strategy] official_signals_csv={OFFICIAL_SIGNALS_CSV}")
    print(f"[realtime_strategy] legacy_predictions_csv={PREDICTIONS_CSV}")
    print(f"[realtime_strategy] strategy_predictions_csv={STRATEGY_PREDICTIONS_CSV}")
    print(f"[realtime_strategy] strategy_predictions_latest_csv={STRATEGY_PREDICTIONS_LATEST_CSV}")
    rebuild_latest_predictions_from_log()

    if alpha_model.load():
        alpha_model.last_train_time = None
        print("[realtime_strategy] loaded saved model; startup retrain required for legacy coverage parity")
    historical_rows = None
    legacy_coverage_gate = LegacyAdaptiveCoverageGate()
    if "adaptive_rule_switch" in names and not legacy_coverage_gate.report_path.exists():
        raise FileNotFoundError(
            "legacy coverage report is required for adaptive_rule_switch official signals: "
            f"{legacy_coverage_gate.report_path}"
        )
    live_chart_window = LiveStrategyChartWindow(names) if live_chart else None
    if live_chart_window is not None:
        live_chart_window.start()
    last_processed_signal_timestamp = None

    while True:
        try:
            df = data_feed.load()
            if df.empty:
                print("[realtime_strategy] empty data")
                if once:
                    return
                time.sleep(REALTIME_INTERVAL_SECONDS)
                continue

            now_ms = int(df.iloc[-1]["timestamp"])
            validate_due_signals(df, now_ms)
            if live_chart_window is not None:
                live_chart_window.update()

            if alpha_model.model is None or alpha_model.last_train_time is None:
                print("[realtime_strategy] training model")
                alpha_model.ensure_trained(df)
                print("[realtime_strategy] model updated")
                historical_rows = None
            elif (
                datetime.now() - alpha_model.last_train_time
            ).total_seconds() >= alpha_model.retrain_interval_seconds:
                print("[realtime_strategy] training model")
                alpha_model.ensure_trained(df)
                print("[realtime_strategy] model updated")
                historical_rows = None

            feature_df = feature_pipeline.build(df, alpha_model.feature_cols)
            if feature_df.empty:
                print("[realtime_strategy] empty features")
                if once:
                    return
                time.sleep(REALTIME_INTERVAL_SECONDS)
                continue

            if historical_rows is None and any(hasattr(s, "update_history") for s in strategies):
                historical_rows = _refresh_historical_match_rows(df, feature_df, alpha_model.model)
                _update_historical_strategy_context(strategies, historical_rows)

            if last_processed_signal_timestamp is None:
                rows_to_process = feature_df.tail(1)
            else:
                rows_to_process = feature_df[
                    pd.to_numeric(feature_df["timestamp"], errors="coerce") > last_processed_signal_timestamp
                ].tail(20)
            if rows_to_process.empty:
                print("[realtime_strategy] no new feature rows")
                if once:
                    return
                time.sleep(REALTIME_INTERVAL_SECONDS)
                continue

            close_by_timestamp = df.set_index("timestamp")["close"]
            for _, feature_row in rows_to_process.iterrows():
                signal_timestamp = int(feature_row["timestamp"])
                latest = feature_row.to_frame().T
                latest_features = latest[alpha_model.feature_cols]
                prediction = alpha_model.predict_one(latest_features)

                current_price = float(close_by_timestamp.loc[signal_timestamp])
                signal_time = ms_to_beijing_time(signal_timestamp)

                print(
                    "[realtime_strategy] dual-model output: "
                    f"time={signal_time}, price={current_price:.2f}, "
                    f"up_model={prediction.get('up_signal_probability'):.4f}, "
                    f"down_model={prediction.get('down_signal_probability'):.4f}, "
                    f"edge_up_minus_down={prediction.get('direction_edge'):.4f}"
                )

                if kronos_adapter is not None:
                    should_run_kronos, kronos_skip_reason = _should_run_kronos(prediction)
                    if should_run_kronos:
                        kronos_df = df[df["timestamp"] <= signal_timestamp].copy()
                        kronos_result = kronos_adapter.forecast_direction(kronos_df)
                    else:
                        kronos_result = _skipped_kronos_result(kronos_skip_reason)
                    _update_kronos_strategy_context(strategies, kronos_result)
                    print(
                        "[realtime_strategy] kronos output: "
                        f"available={kronos_result.available}, direction={kronos_result.direction}, "
                        f"confidence={kronos_result.confidence:.4f}, "
                        f"forecast_close={kronos_result.forecast_close}, reason={kronos_result.reason}"
                    )

                strategy_decisions = []
                legacy_coverage_context = None
                for strategy in strategies:
                    decision = strategy.decide(feature_row, prediction)
                    if strategy.name == "adaptive_rule_switch":
                        legacy_decision = legacy_coverage_gate.decide(feature_row, prediction)
                        if legacy_decision.accepted:
                            legacy_coverage_context = legacy_decision
                            decision = StrategyDecision(
                                legacy_decision.direction,
                                legacy_decision.confidence,
                                f"{decision.reason};{legacy_decision.reason}",
                            )
                    strategy_decisions.append((strategy.name, decision))
                    print(
                        "[realtime_strategy] strategy decision: "
                        f"strategy={strategy.name}, direction={decision.direction}, "
                        f"confidence={decision.confidence:.4f}, reason={decision.reason}, "
                        f"up_model={prediction.get('up_signal_probability'):.4f}, "
                        f"down_model={prediction.get('down_signal_probability'):.4f}, "
                        f"edge={prediction.get('direction_edge'):.4f}"
                    )

                if legacy_coverage_context is not None:
                    peer_reason = (
                        "peer_legacy_coverage_gate=pass;"
                        f"peer_legacy_rule={legacy_coverage_context.rule};"
                        f"peer_legacy_condition={legacy_coverage_context.condition}"
                    )
                    strategy_decisions = [
                        (
                            strategy_name,
                            decision
                            if strategy_name == "adaptive_rule_switch"
                            else StrategyDecision(
                                decision.direction,
                                decision.confidence,
                                f"{decision.reason};{peer_reason}",
                            ),
                        )
                        for strategy_name, decision in strategy_decisions
                    ]

                quality_context = _build_quality_context(strategy_decisions)
                if legacy_coverage_context is not None:
                    quality_context["legacy_coverage"] = {
                        "rule": legacy_coverage_context.rule,
                        "condition": legacy_coverage_context.condition,
                        "direction": legacy_coverage_context.direction,
                    }
                for strategy_name, decision in strategy_decisions:
                    register_prediction_signal(
                        strategy_name=strategy_name,
                        decision=decision,
                        prediction=prediction,
                        current_price=current_price,
                        signal_timestamp=signal_timestamp,
                        signal_time=signal_time,
                        features=feature_row,
                        quality_context=quality_context,
                    )
                last_processed_signal_timestamp = signal_timestamp

            render_strategy_charts(names)
            if live_chart_window is not None:
                live_chart_window.update()

            if once:
                if live_chart_window is not None:
                    input("[realtime_strategy] live chart is open. Press Enter to exit...")
                return

        except KeyboardInterrupt:
            print("[realtime_strategy] stopped")
            return
        except Exception as exc:
            print(f"[realtime_strategy] error: {type(exc).__name__}: {exc}")

        time.sleep(REALTIME_INTERVAL_SECONDS)

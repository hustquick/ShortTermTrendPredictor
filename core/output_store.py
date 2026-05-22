# core/output_store.py

import csv
import os
import sys
from pathlib import Path

from config import ALL_PREDICTIONS_CSV, OFFICIAL_SIGNALS_CSV
from signal_funnel import record_funnel_prediction, record_funnel_validation
from signal_quality import enrich_validation_quality

csv.field_size_limit(min(sys.maxsize, 2_147_483_647))


ALL_PREDICTION_COLUMNS = [
    "prediction_id",
    "timestamp",
    "strategy",
    "market_regime",
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
    "future_return",
    "is_correct",
    "is_tradable_correct",
    "notify_enabled",
]

OFFICIAL_SIGNAL_COLUMNS = [
    "prediction_id",
    "timestamp",
    "strategy",
    "market_regime",
    "current_price",
    "direction",
    "confidence",
    "reason",
    "up_signal_probability",
    "down_signal_probability",
    "direction_edge",
    "validation_timestamp",
    "validation_status",
    "actual_direction",
    "future_price",
    "future_return",
    "is_correct",
    "is_tradable_correct",
]


class PredictionOutputStore:
    """Canonical split outputs: all predictions vs official signals."""

    def __init__(
        self,
        all_predictions_path: Path = ALL_PREDICTIONS_CSV,
        official_signals_path: Path = OFFICIAL_SIGNALS_CSV,
    ):
        self.all_predictions_path = all_predictions_path
        self.official_signals_path = official_signals_path
        self.ensure_files()

    def ensure_files(self):
        _ensure_csv(self.all_predictions_path, ALL_PREDICTION_COLUMNS)
        _ensure_csv(self.official_signals_path, OFFICIAL_SIGNAL_COLUMNS)

    def record_all_prediction(self, row: dict):
        _upsert_csv_row(self.all_predictions_path, ALL_PREDICTION_COLUMNS, row)

    def record_official_signal(self, row: dict):
        _upsert_csv_row(
            self.official_signals_path,
            OFFICIAL_SIGNAL_COLUMNS,
            _official_row(row),
        )

    def record_prediction(self, row: dict):
        self.record_all_prediction(row)
        record_funnel_prediction(row)
        if _is_true(row.get("notify_enabled")):
            self.record_official_signal(row)

    def record_validation(self, row: dict):
        row = enrich_validation_quality(row)
        self.record_all_prediction(row)
        record_funnel_validation(row)
        if _is_true(row.get("notify_enabled")):
            self.record_official_signal(row)


def _official_row(row: dict) -> dict:
    return {
        "prediction_id": row.get("prediction_id"),
        "timestamp": row.get("timestamp"),
        "strategy": row.get("strategy"),
        "market_regime": row.get("market_regime"),
        "current_price": row.get("current_price"),
        "direction": row.get("raw_direction") or row.get("predicted_direction"),
        "confidence": row.get("confidence"),
        "reason": row.get("reason"),
        "up_signal_probability": row.get("up_signal_probability"),
        "down_signal_probability": row.get("down_signal_probability"),
        "direction_edge": row.get("direction_edge"),
        "validation_timestamp": row.get("validation_timestamp"),
        "validation_status": row.get("validation_status"),
        "actual_direction": row.get("actual_direction"),
        "future_price": row.get("future_price"),
        "future_return": row.get("future_return"),
        "is_correct": row.get("is_correct"),
        "is_tradable_correct": row.get("is_tradable_correct"),
    }


def _is_true(value) -> bool:
    return str(value).lower() == "true"


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _ensure_csv(path: Path, columns: list[str]):
    if path.exists() and path.stat().st_size > 0:
        rows = _read_csv_rows(path)
        if not rows:
            return
        current_columns = list(rows[0].keys())
        if current_columns != columns:
            _write_csv_rows(path, columns, rows)
        return
    _write_csv_rows(path, columns, [])


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

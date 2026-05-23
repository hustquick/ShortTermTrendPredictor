# signal_funnel.py

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

from config import DATA_DIR, OFFICIAL_SIGNAL_STRATEGY_ALLOWLIST
from signal_quality import enrich_validation_quality

csv.field_size_limit(min(sys.maxsize, 2_147_483_647))

SIGNAL_FUNNEL_CSV = DATA_DIR / "signal_funnel.csv"
STRATEGY_REGIME_REPORT_CSV = DATA_DIR / "strategy_regime_report.csv"
OFFICIAL_STRATEGIES = set(OFFICIAL_SIGNAL_STRATEGY_ALLOWLIST)

SIGNAL_FUNNEL_COLUMNS = [
    "prediction_id",
    "timestamp",
    "strategy",
    "market_regime",
    "raw_direction",
    "final_direction",
    "candidate_passed",
    "trap_passed",
    "quality_passed",
    "learning_passed",
    "allowlist_passed",
    "notify_enabled",
    "blocked_stage",
    "blocked_reason",
    "confidence",
    "up_signal_probability",
    "down_signal_probability",
    "direction_edge",
    "current_price",
    "future_price",
    "future_return",
    "actual_direction",
    "is_correct",
    "is_tradable_correct",
    "reason",
]

STRATEGY_REGIME_REPORT_COLUMNS = [
    "strategy",
    "raw_direction",
    "market_regime",
    "signals",
    "direction_wins",
    "direction_win_rate",
    "tradable_wins",
    "tradable_win_rate",
    "avg_future_return",
    "avg_abs_future_return",
    "last_timestamp",
]


def _is_true(value) -> bool:
    return str(value).lower() == "true"


def _read_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, columns: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    tmp_path.replace(path)


def _upsert(path: Path, columns: list[str], row: dict, key: str = "prediction_id"):
    rows = _read_rows(path)
    row_key = str(row.get(key, ""))
    next_rows = []
    replaced = False
    for old in rows:
        if row_key and str(old.get(key, "")) == row_key:
            next_rows.append({**old, **row})
            replaced = True
        else:
            next_rows.append(old)
    if not replaced:
        next_rows.append(row)
    _write_rows(path, columns, next_rows)


def _extract_reason_value(reason: str, key: str) -> str:
    prefix = f"{key}="
    for part in str(reason).split(";"):
        if part.startswith(prefix):
            return part[len(prefix):]
    return ""


def _infer_market_regime(row: dict) -> str:
    if row.get("market_regime"):
        return row.get("market_regime")
    reason = row.get("reason") or row.get("learning_reason") or ""
    return _extract_reason_value(reason, "regime") or "unknown"


def infer_blocked_stage(row: dict) -> str:
    if _is_true(row.get("notify_enabled")):
        return "passed"
    if not row.get("final_direction") or row.get("final_direction") == "no_trade":
        return "strategy"
    if not _is_true(row.get("allowlist_passed")):
        return "allowlist"
    if not _is_true(row.get("quality_passed")):
        return "quality"
    if not _is_true(row.get("learning_passed")):
        return "learning_context"
    return "risk_gate"


def _normalize_funnel_row(row: dict) -> dict:
    out = dict(row)
    reason = str(out.get("reason", ""))
    raw_direction = out.get("raw_direction") or out.get("predicted_direction") or out.get("direction")
    final_direction = out.get("final_direction") or raw_direction
    strategy = out.get("strategy", "")
    out["raw_direction"] = raw_direction
    out["final_direction"] = final_direction
    out["market_regime"] = _infer_market_regime(out)
    out.setdefault("candidate_passed", raw_direction in {"up", "down"})
    out.setdefault("trap_passed", not reason.startswith(("long_chase_trap", "short_rebound_trap")))
    out.setdefault("allowlist_passed", strategy in OFFICIAL_STRATEGIES)
    out.setdefault("learning_passed", "learning_explore" not in reason and "learning_disabled" not in reason and "learning_probation" not in reason and "learning_feature_blocked" not in reason)
    out.setdefault("quality_passed", "production_blocked" not in reason)
    out["blocked_stage"] = out.get("blocked_stage") or infer_blocked_stage(out)
    if out.get("blocked_stage") != "passed" and not out.get("blocked_reason"):
        out["blocked_reason"] = reason
    return out


def build_funnel_row(
    *,
    prediction_id: str,
    timestamp: str,
    strategy: str,
    market_regime: str,
    raw_direction: str,
    final_direction: str,
    confidence: float,
    reason: str,
    prediction: dict,
    current_price: float,
    quality_ok: bool,
    learning_notify: bool,
    allowlist_passed: bool,
    notify_enabled: bool,
) -> dict:
    candidate_passed = raw_direction in {"up", "down"}
    trap_passed = not str(reason).startswith(("long_chase_trap", "short_rebound_trap"))
    row = {
        "prediction_id": prediction_id,
        "timestamp": timestamp,
        "strategy": strategy,
        "market_regime": market_regime,
        "raw_direction": raw_direction,
        "final_direction": final_direction,
        "candidate_passed": candidate_passed,
        "trap_passed": trap_passed,
        "quality_passed": quality_ok,
        "learning_passed": learning_notify,
        "allowlist_passed": allowlist_passed,
        "notify_enabled": notify_enabled,
        "confidence": confidence,
        "up_signal_probability": prediction.get("up_signal_probability"),
        "down_signal_probability": prediction.get("down_signal_probability"),
        "direction_edge": prediction.get("direction_edge"),
        "current_price": current_price,
        "reason": reason,
    }
    row["blocked_stage"] = infer_blocked_stage(row)
    row["blocked_reason"] = "" if notify_enabled else reason
    return row


def record_funnel_prediction(row: dict):
    _upsert(SIGNAL_FUNNEL_CSV, SIGNAL_FUNNEL_COLUMNS, _normalize_funnel_row(row))


def record_funnel_validation(row: dict):
    enriched = enrich_validation_quality(_normalize_funnel_row(row))
    _upsert(SIGNAL_FUNNEL_CSV, SIGNAL_FUNNEL_COLUMNS, enriched)
    rebuild_strategy_regime_report()


def rebuild_strategy_regime_report():
    rows = [r for r in _read_rows(SIGNAL_FUNNEL_CSV) if r.get("future_price") not in (None, "")]
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        direction = row.get("raw_direction") or row.get("predicted_direction") or row.get("direction")
        if direction not in {"up", "down"}:
            continue
        key = (row.get("strategy", ""), direction, row.get("market_regime", "unknown"))
        groups.setdefault(key, []).append(row)

    report_rows = []
    for (strategy, direction, regime), items in sorted(groups.items()):
        signals = len(items)
        direction_wins = sum(1 for r in items if _is_true(r.get("is_correct")))
        tradable_wins = sum(1 for r in items if _is_true(r.get("is_tradable_correct")))
        returns = []
        for r in items:
            try:
                returns.append(float(r.get("future_return")))
            except (TypeError, ValueError):
                continue
        report_rows.append(
            {
                "strategy": strategy,
                "raw_direction": direction,
                "market_regime": regime,
                "signals": signals,
                "direction_wins": direction_wins,
                "direction_win_rate": direction_wins / signals if signals else "",
                "tradable_wins": tradable_wins,
                "tradable_win_rate": tradable_wins / signals if signals else "",
                "avg_future_return": sum(returns) / len(returns) if returns else "",
                "avg_abs_future_return": sum(abs(x) for x in returns) / len(returns) if returns else "",
                "last_timestamp": max(str(r.get("timestamp", "")) for r in items),
            }
        )
    _write_rows(STRATEGY_REGIME_REPORT_CSV, STRATEGY_REGIME_REPORT_COLUMNS, report_rows)

# realtime_strategy_runner.py

import csv
import json
import time
from datetime import datetime

from config import (
    DATA_DIR,
    OFFICIAL_SIGNAL_STRATEGY_ALLOWLIST,
    PREDICT_HORIZON_MINUTES,
    PREDICTIONS_CSV,
    REALTIME_INTERVAL_SECONDS,
)
from data_download import get_recent_klines_with_cache, ms_to_beijing_time
from features import build_features
from historical_match_filter import build_walk_forward_historical_match_rows
from kronos_adapter import KronosAdapter
from strategy_notifier import send_prediction_signal, send_validation_signal
from strategies.rules import (
    FinStarScenarioStrategy,
    HistoricalMatchLongStrategy,
    HistoricalMatchShortStrategy,
    HistoricalMatchStrategy,
    KronosConfirmStrategy,
    RelaxedScenarioStrategy,
    ShortMomentumStrategy,
)
from trainer import load_model, save_model, train_validation_model


STRATEGY_MAP = {
    "short_momentum": ShortMomentumStrategy,
    "relaxed_scenario": RelaxedScenarioStrategy,
    "historical_match": HistoricalMatchStrategy,
    "historical_match_long": HistoricalMatchLongStrategy,
    "historical_match_short": HistoricalMatchShortStrategy,
    "kronos_confirm": KronosConfirmStrategy,
    "finstar_scenario": FinStarScenarioStrategy,
}

PENDING_STRATEGY_SIGNALS = DATA_DIR / "pending_strategy_signals.jsonl"
VALIDATED_STRATEGY_SIGNALS = DATA_DIR / "validated_strategy_signals.csv"
STRATEGY_PREDICTIONS_CSV = DATA_DIR / "strategy_predictions.csv"
STRATEGY_PREDICTIONS_LATEST_CSV = DATA_DIR / "strategy_predictions_latest.csv"

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

OFFICIAL_SIGNAL_STRATEGIES = set(OFFICIAL_SIGNAL_STRATEGY_ALLOWLIST)


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
    with open(PENDING_STRATEGY_SIGNALS, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_csv_row(path, columns: list[str], row: dict):
    DATA_DIR.mkdir(exist_ok=True)
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in columns})


def _load_latest_prediction_rows() -> dict[str, dict]:
    if not STRATEGY_PREDICTIONS_LATEST_CSV.exists():
        return {}
    with open(STRATEGY_PREDICTIONS_LATEST_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["prediction_id"]: row for row in reader if row.get("prediction_id")}


def _save_latest_prediction_rows(rows_by_id: dict[str, dict]):
    DATA_DIR.mkdir(exist_ok=True)
    with open(STRATEGY_PREDICTIONS_LATEST_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for row in sorted(rows_by_id.values(), key=lambda x: str(x.get("timestamp", ""))):
            writer.writerow({col: row.get(col, "") for col in PREDICTION_COLUMNS})


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


def validate_due_signals(df, now_ms: int):
    pending = load_pending_signals()
    if not pending:
        return

    close_by_timestamp = df.set_index("timestamp")["close"]
    remaining = []

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
        predicted_direction = row["direction"]
        is_correct = predicted_direction == actual_direction
        validation_time = ms_to_beijing_time(validation_timestamp)
        strategy_accuracy, strategy_correct_count, strategy_total_count = _strategy_accuracy_after_current(
            row["strategy"],
            is_correct,
        )

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
        }
        append_validated_signal(validation_row)
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
        notify_enabled = str(row.get("notify_enabled", "")).lower() == "true" or is_official_signal_strategy(row["strategy"])
        if notify_enabled:
            send_validation_signal(
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


def register_prediction_signal(strategy_name: str, decision, prediction: dict, current_price: float, signal_timestamp: int, signal_time: str):
    prediction_id = f"{strategy_name}-{signal_timestamp}"
    validation_timestamp = signal_timestamp + PREDICT_HORIZON_MINUTES * 60_000
    validation_time = ms_to_beijing_time(validation_timestamp)

    pending = load_pending_signals()
    if any(row.get("prediction_id") == prediction_id for row in pending):
        return

    notify_enabled = is_official_signal_strategy(strategy_name)

    row = {
        "prediction_id": prediction_id,
        "strategy": strategy_name,
        "direction": decision.direction,
        "confidence": float(decision.confidence),
        "reason": decision.reason,
        "signal_price": float(current_price),
        "signal_timestamp": int(signal_timestamp),
        "signal_time": signal_time,
        "validation_timestamp": int(validation_timestamp),
        "up_signal_probability": prediction.get("up_signal_probability"),
        "down_signal_probability": prediction.get("down_signal_probability"),
        "direction_edge": prediction.get("direction_edge"),
        "notify_enabled": notify_enabled,
    }
    pending.append(row)
    save_pending_signals(pending)

    append_prediction_csv(
        {
            "prediction_id": prediction_id,
            "timestamp": signal_time,
            "strategy": strategy_name,
            "current_price": float(current_price),
            "predicted_direction": decision.direction,
            "confidence": float(decision.confidence),
            "reason": decision.reason,
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
        send_prediction_signal(
            strategy_name=strategy_name,
            direction=decision.direction,
            confidence=float(decision.confidence),
            current_price=float(current_price),
            timestamp=signal_time,
            reason=decision.reason,
            prediction_id=prediction_id,
            up_signal_probability=prediction.get("up_signal_probability"),
            down_signal_probability=prediction.get("down_signal_probability"),
            direction_edge=prediction.get("direction_edge"),
            horizon_minutes=PREDICT_HORIZON_MINUTES,
        )
    else:
        print(f"[realtime_strategy] prediction notification skipped for observation strategy={strategy_name}")


def _refresh_historical_match_rows(df, feature_df, model):
    historical_feature_df = feature_df.iloc[:-PREDICT_HORIZON_MINUTES].copy()
    rows = build_walk_forward_historical_match_rows(historical_feature_df, df)
    print(f"[realtime_strategy] walk_forward_historical_match_rows={len(rows)}")
    return rows


def _update_historical_strategy_context(strategies: list, historical_rows):
    for strategy in strategies:
        if hasattr(strategy, "update_history"):
            strategy.update_history(historical_rows)


def _update_kronos_strategy_context(strategies: list, kronos_result):
    for strategy in strategies:
        if hasattr(strategy, "update_kronos_result"):
            strategy.update_kronos_result(kronos_result)


def run_realtime_strategies(
    strategy_names: str = "short_momentum,relaxed_scenario,historical_match,kronos_confirm,finstar_scenario",
    train_minutes: int = 48 * 60,
    once: bool = False,
):
    names = parse_strategy_names(strategy_names)
    strategies = [STRATEGY_MAP[name]() for name in names]
    use_kronos = any(hasattr(strategy, "update_kronos_result") for strategy in strategies)
    kronos_adapter = KronosAdapter() if use_kronos else None

    print("[realtime_strategy] start")
    print(f"[realtime_strategy] strategies={','.join(names)}")
    print(f"[realtime_strategy] official_notification_allowlist={','.join(sorted(OFFICIAL_SIGNAL_STRATEGIES))}")
    print(f"[realtime_strategy] supported strategies: {','.join(STRATEGY_MAP.keys())}")
    print("[realtime_strategy] objective=high-confidence directional accuracy only")
    print(f"[realtime_strategy] predictions_csv={PREDICTIONS_CSV}")
    print(f"[realtime_strategy] strategy_predictions_csv={STRATEGY_PREDICTIONS_CSV}")
    print(f"[realtime_strategy] strategy_predictions_latest_csv={STRATEGY_PREDICTIONS_LATEST_CSV}")
    rebuild_latest_predictions_from_log()

    model = load_model()
    last_train_time = None
    historical_rows = None

    while True:
        try:
            df = get_recent_klines_with_cache(minutes=train_minutes, update_if_needed=True)
            if df.empty:
                print("[realtime_strategy] empty data")
                if once:
                    return
                time.sleep(REALTIME_INTERVAL_SECONDS)
                continue

            now_ms = int(df.iloc[-1]["timestamp"])
            validate_due_signals(df, now_ms)

            need_train = model is None
            if last_train_time is None:
                need_train = True
            elif (datetime.now() - last_train_time).total_seconds() >= 30 * 60:
                need_train = True

            if need_train:
                print("[realtime_strategy] training model")
                model = train_validation_model(df)
                save_model(model)
                last_train_time = datetime.now()
                print("[realtime_strategy] model updated")
                historical_rows = None

            feature_df = build_features(df).dropna(subset=model.feature_cols).copy()
            if feature_df.empty:
                print("[realtime_strategy] empty features")
                if once:
                    return
                time.sleep(REALTIME_INTERVAL_SECONDS)
                continue

            if historical_rows is None and any(hasattr(s, "update_history") for s in strategies):
                historical_rows = _refresh_historical_match_rows(df, feature_df, model)
                _update_historical_strategy_context(strategies, historical_rows)

            if kronos_adapter is not None:
                kronos_result = kronos_adapter.forecast_direction(df)
                _update_kronos_strategy_context(strategies, kronos_result)
                print(
                    "[realtime_strategy] kronos output: "
                    f"available={kronos_result.available}, direction={kronos_result.direction}, "
                    f"confidence={kronos_result.confidence:.4f}, "
                    f"forecast_close={kronos_result.forecast_close}, reason={kronos_result.reason}"
                )

            latest = feature_df.iloc[[-1]].copy()
            latest_features = latest[model.feature_cols]
            feature_row = latest.iloc[0]
            prediction = model.predict_one(latest_features, signal_filter=None)

            current_price = float(df.iloc[-1]["close"])
            signal_timestamp = int(df.iloc[-1]["timestamp"])
            signal_time = ms_to_beijing_time(signal_timestamp)

            print(
                "[realtime_strategy] dual-model output: "
                f"time={signal_time}, price={current_price:.2f}, "
                f"up_model={prediction.get('up_signal_probability'):.4f}, "
                f"down_model={prediction.get('down_signal_probability'):.4f}, "
                f"edge_up_minus_down={prediction.get('direction_edge'):.4f}"
            )

            for strategy in strategies:
                decision = strategy.decide(feature_row, prediction)
                print(
                    "[realtime_strategy] strategy decision: "
                    f"strategy={strategy.name}, direction={decision.direction}, "
                    f"confidence={decision.confidence:.4f}, reason={decision.reason}, "
                    f"up_model={prediction.get('up_signal_probability'):.4f}, "
                    f"down_model={prediction.get('down_signal_probability'):.4f}, "
                    f"edge={prediction.get('direction_edge'):.4f}"
                )
                if decision.direction in {"up", "down"}:
                    register_prediction_signal(
                        strategy_name=strategy.name,
                        decision=decision,
                        prediction=prediction,
                        current_price=current_price,
                        signal_timestamp=signal_timestamp,
                        signal_time=signal_time,
                    )

            if once:
                return

        except KeyboardInterrupt:
            print("[realtime_strategy] stopped")
            return
        except Exception as exc:
            print(f"[realtime_strategy] error: {type(exc).__name__}: {exc}")

        time.sleep(REALTIME_INTERVAL_SECONDS)

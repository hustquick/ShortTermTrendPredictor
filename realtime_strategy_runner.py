# realtime_strategy_runner.py

import json
import time
from datetime import datetime
from pathlib import Path

from config import DATA_DIR, PREDICT_HORIZON_MINUTES, REALTIME_INTERVAL_SECONDS
from data_download import get_recent_klines_with_cache, ms_to_beijing_time
from features import build_features
from strategy_notifier import send_prediction_signal, send_validation_signal
from strategies.rules import RelaxedScenarioStrategy, ShortMomentumStrategy
from trainer import load_model, save_model, train_validation_model


STRATEGY_MAP = {
    "short_momentum": ShortMomentumStrategy,
    "relaxed_scenario": RelaxedScenarioStrategy,
}

PENDING_STRATEGY_SIGNALS = DATA_DIR / "pending_strategy_signals.jsonl"
VALIDATED_STRATEGY_SIGNALS = DATA_DIR / "validated_strategy_signals.csv"


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


def append_validated_signal(row: dict):
    DATA_DIR.mkdir(exist_ok=True)
    exists = VALIDATED_STRATEGY_SIGNALS.exists()
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
    ]
    with open(VALIDATED_STRATEGY_SIGNALS, "a", encoding="utf-8") as f:
        if not exists:
            f.write(",".join(columns) + "\n")
        f.write(",".join(str(row.get(col, "")) for col in columns) + "\n")


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
        }
        append_validated_signal(validation_row)
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
        )
        print(
            "[realtime_strategy] validation: "
            f"strategy={row['strategy']}, id={row['prediction_id']}, "
            f"predicted={predicted_direction}, actual={actual_direction}, correct={is_correct}"
        )

    save_pending_signals(remaining)


def register_prediction_signal(
    strategy_name: str,
    decision,
    prediction: dict,
    current_price: float,
    signal_timestamp: int,
    signal_time: str,
):
    prediction_id = f"{strategy_name}-{signal_timestamp}"
    validation_timestamp = signal_timestamp + PREDICT_HORIZON_MINUTES * 60_000

    pending = load_pending_signals()
    if any(row.get("prediction_id") == prediction_id for row in pending):
        return

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
    }
    pending.append(row)
    save_pending_signals(pending)

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


def run_realtime_strategies(
    strategy_names: str = "short_momentum,relaxed_scenario",
    train_minutes: int = 48 * 60,
    once: bool = False,
):
    names = parse_strategy_names(strategy_names)
    strategies = [STRATEGY_MAP[name]() for name in names]

    print("[realtime_strategy] start")
    print(f"[realtime_strategy] strategies={','.join(names)}")
    print("[realtime_strategy] objective=high-confidence directional accuracy only")

    model = load_model()
    last_train_time = None

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

            feature_df = build_features(df).dropna(subset=model.feature_cols).copy()
            if feature_df.empty:
                print("[realtime_strategy] empty features")
                if once:
                    return
                time.sleep(REALTIME_INTERVAL_SECONDS)
                continue

            latest = feature_df.iloc[[-1]].copy()
            latest_features = latest[model.feature_cols]
            feature_row = latest.iloc[0]
            prediction = model.predict_one(latest_features, signal_filter=None)

            current_price = float(df.iloc[-1]["close"])
            signal_timestamp = int(df.iloc[-1]["timestamp"])
            signal_time = ms_to_beijing_time(signal_timestamp)

            print(
                "[realtime_strategy] model output: "
                f"time={signal_time}, price={current_price:.2f}, "
                f"up={prediction.get('up_signal_probability'):.4f}, "
                f"down={prediction.get('down_signal_probability'):.4f}, "
                f"edge={prediction.get('direction_edge'):.4f}"
            )

            for strategy in strategies:
                decision = strategy.decide(feature_row, prediction)
                print(
                    "[realtime_strategy] decision: "
                    f"strategy={strategy.name}, direction={decision.direction}, "
                    f"confidence={decision.confidence:.4f}, reason={decision.reason}"
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

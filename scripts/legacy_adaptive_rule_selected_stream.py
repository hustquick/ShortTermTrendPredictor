import argparse
import contextlib
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

from config import (
    BACKTEST_MIN_TRAIN_SAMPLES,
    BACKTEST_TRAIN_WINDOW_MINUTES,
    DATA_DIR,
    PREDICT_HORIZON_MINUTES,
)
from core.feature_pipeline import FeaturePipeline
from data_download import get_recent_klines_with_cache, ms_to_beijing_time
from trainer import train_validation_model


def _feature_value(features, name: str, default: float = 0.0) -> float:
    try:
        value = features.get(name, default)
    except AttributeError:
        return default
    if pd.isna(value):
        return default
    return float(value)


def _legacy_candidates(features, prediction: dict) -> list[dict]:
    p_up_raw = float(prediction.get("up_probability", 0.5))
    p_up_signal = float(prediction.get("up_signal_probability", 0.0))
    p_down_signal = float(prediction.get("down_signal_probability", 0.0))
    ret_30 = _feature_value(features, "ret_30")
    macd_hist = _feature_value(features, "macd_hist")
    boll_position = _feature_value(features, "boll_position", 0.5)
    close_position = _feature_value(features, "close_position", 0.5)
    trend = _feature_value(features, "trend_agreement")

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
    add(
        p_up_raw >= 0.98 and boll_position < 0.85,
        "long_pup_ge_098_not_high",
        "up",
        max(p_up_signal, p_up_raw),
    )
    add(
        p_up_raw >= 0.85 and ret_30 >= 0 and macd_hist <= 0 and close_position < 0.95,
        "long_pup_ge_085_ret30pos_macdneg_closeok",
        "up",
        max(p_up_signal, p_up_raw),
    )
    add(
        p_up_raw >= 0.55 and boll_position < 0.85,
        "long_pup_ge_055_not_high",
        "up",
        max(p_up_signal, p_up_raw),
    )
    return rules


def _legacy_state_ok(features, prediction: dict, direction: str) -> bool:
    p_up_raw = float(prediction.get("up_probability", 0.5))
    rsi_14 = _feature_value(features, "rsi_14", 50.0)
    ret_30 = _feature_value(features, "ret_30")
    return direction == "down" and p_up_raw <= 0.285 and rsi_14 > 47.5 and ret_30 > -0.00013


def _prediction_from_row(row: pd.Series) -> dict:
    p_up = _feature_value(row, "up_probability", 0.5)
    p_up_signal = _feature_value(row, "up_signal_probability", p_up)
    p_down_signal = _feature_value(row, "down_signal_probability", 1.0 - p_up)
    return {
        "up_probability": p_up,
        "up_signal_probability": p_up_signal,
        "down_signal_probability": p_down_signal,
    }


def _stats(records: deque[bool]) -> tuple[int, int, float]:
    samples = len(records)
    if samples == 0:
        return 0, 0, 0.0
    wins = sum(bool(item) for item in records)
    return samples, wins, wins / samples


def _selected_candidate(candidates: list[dict], records_by_rule: dict[str, deque[bool]]) -> dict:
    scored = []
    for rule in candidates:
        samples, wins, win_rate = _stats(records_by_rule[rule["name"]])
        scored.append({**rule, "prior_rule_samples": samples, "prior_rule_wins": wins, "prior_rule_win": win_rate})
    active = [
        item for item in scored
        if item["prior_rule_samples"] >= 5 and item["prior_rule_win"] >= 0.80
    ]
    if active:
        return sorted(
            active,
            key=lambda item: (item["prior_rule_win"], item["prior_rule_samples"], item["confidence"]),
            reverse=True,
        )[0]
    return sorted(
        scored,
        key=lambda item: (item["prior_rule_samples"], item["confidence"]),
        reverse=True,
    )[0]


def _active_candidate(candidates: list[dict], records_by_rule: dict[str, deque[bool]]) -> dict | None:
    scored = []
    for rule in candidates:
        samples, wins, win_rate = _stats(records_by_rule[rule["name"]])
        scored.append({**rule, "prior_rule_samples": samples, "prior_rule_wins": wins, "prior_rule_win": win_rate})
    active = [
        item for item in scored
        if item["prior_rule_samples"] >= 5 and item["prior_rule_win"] >= 0.80
    ]
    if not active:
        return None
    return sorted(
        active,
        key=lambda item: (item["prior_rule_win"], item["prior_rule_samples"], item["confidence"]),
        reverse=True,
    )[0]


def build_stream_from_predictions(
    predictions: pd.DataFrame,
    output: Path,
    emit_all_candidates: bool = False,
) -> pd.DataFrame:
    df = predictions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    records_by_rule: dict[str, deque[bool]] = defaultdict(lambda: deque(maxlen=10))
    rows = []
    for _, row in df.iterrows():
        prediction = _prediction_from_row(row)
        candidates = _legacy_candidates(row, prediction)
        if not candidates:
            continue

        selected = _selected_candidate(candidates, records_by_rule) if emit_all_candidates else _active_candidate(candidates, records_by_rule)
        if selected is None:
            actual_direction = str(row["actual_direction"])
            for candidate in candidates:
                records_by_rule[candidate["name"]].append(candidate["direction"] == actual_direction)
            continue

        actual_direction = str(row["actual_direction"])
        correct = selected["direction"] == actual_direction
        state_ok = _legacy_state_ok(row, prediction, selected["direction"])
        is_valid_signal = (
            selected["prior_rule_samples"] >= 5
            and selected["prior_rule_win"] >= 0.80
            and state_ok
        )
        out_row = row.to_dict()
        timestamp_text = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        out_row.update({
            "timestamp": timestamp_text,
            "predicted_direction": selected["direction"] if is_valid_signal else "no_trade",
            "confidence": selected["confidence"],
            "is_valid_signal": bool(is_valid_signal),
            "is_correct": bool(correct) if is_valid_signal else False,
            "dt": timestamp_text,
            "rule": selected["name"],
            "direction": selected["direction"],
            "correct": bool(correct),
            "prior_rule_win": selected["prior_rule_win"],
            "prior_rule_samples": selected["prior_rule_samples"],
            "state_ok": bool(state_ok),
        })
        rows.append(out_row)
        for candidate in candidates:
            records_by_rule[candidate["name"]].append(candidate["direction"] == actual_direction)

    out = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    return out


def build_stream(
    df: pd.DataFrame,
    days: int,
    step_minutes: int,
    model_update_minutes: int,
    train_window_minutes: int,
    max_steps: int | None,
    output: Path,
    progress_every_steps: int,
) -> pd.DataFrame:
    print("[legacy_stream] building features...")
    df = df.sort_values("timestamp").reset_index(drop=True)
    close_by_timestamp = df.set_index("timestamp")["close"]
    feature_pipeline = FeaturePipeline()
    feature_df = feature_pipeline.build(df)
    print(f"[legacy_stream] features ready: rows={len(feature_df)}")

    test_minutes = days * 24 * 60
    horizon = PREDICT_HORIZON_MINUTES
    test_end = len(df) - horizon - 1
    test_start = max(train_window_minutes, test_end - test_minutes)
    candidate_indices = list(range(test_start, test_end, step_minutes))
    if max_steps is not None:
        candidate_indices = candidate_indices[-max_steps:]
    print(
        "[legacy_stream] generation plan: "
        f"days={days}, step_minutes={step_minutes}, model_update_minutes={model_update_minutes}, "
        f"train_window_minutes={train_window_minutes}, candidate_steps={len(candidate_indices)}"
    )

    model = None
    next_model_update_idx = None
    model_trained_at_time = None
    records_by_rule: dict[str, deque[bool]] = defaultdict(lambda: deque(maxlen=10))
    rows = []
    started = time.time()
    total = len(candidate_indices)
    for step_no, idx in enumerate(candidate_indices, start=1):
        point_time = ms_to_beijing_time(int(df.iloc[idx]["timestamp"]))
        if model is None or next_model_update_idx is None or idx >= next_model_update_idx:
            train_start = max(0, idx - train_window_minutes)
            train_df = df.iloc[train_start:idx].copy()
            if len(train_df) < BACKTEST_MIN_TRAIN_SAMPLES:
                continue
            train_started = time.time()
            print(
                "[legacy_stream] model update start: "
                f"step={step_no}/{total}, point_time={point_time}, train_rows={len(train_df)}"
            )
            model = train_validation_model(train_df)
            print(
                "[legacy_stream] model update done: "
                f"step={step_no}/{total}, elapsed={time.time() - train_started:.1f}s"
            )
            next_model_update_idx = idx + model_update_minutes
            model_trained_at_time = point_time

        latest = feature_df.iloc[[idx]].copy()
        if latest[model.feature_cols].isna().any(axis=None):
            continue
        prediction = model.predict_one(latest[model.feature_cols], signal_filter=None)
        feature_row = latest.iloc[0]
        candidates = _legacy_candidates(feature_row, prediction)
        if not candidates:
            continue

        current_row = df.iloc[idx]
        current_price = float(current_row["close"])
        future_ms = int(current_row["timestamp"]) + horizon * 60_000
        if future_ms not in close_by_timestamp.index:
            continue
        future_price = float(close_by_timestamp.loc[future_ms])
        future_return = future_price / current_price - 1
        actual_direction = "up" if future_price > current_price else "down"

        selected = _active_candidate(candidates, records_by_rule)
        if selected is None:
            for candidate in candidates:
                records_by_rule[candidate["name"]].append(candidate["direction"] == actual_direction)
            continue
        correct = selected["direction"] == actual_direction
        state_ok = _legacy_state_ok(feature_row, prediction, selected["direction"])
        is_valid_signal = (
            selected["prior_rule_samples"] >= 5
            and selected["prior_rule_win"] >= 0.80
            and state_ok
        )
        rows.append({
            "timestamp": point_time,
            "current_price": current_price,
            "future_price": future_price,
            "future_return": future_return,
            "predicted_direction": selected["direction"] if is_valid_signal else "no_trade",
            "actual_direction": actual_direction,
            "up_probability": prediction.get("up_probability"),
            "confidence": selected["confidence"],
            "is_valid_signal": bool(is_valid_signal),
            "is_correct": bool(correct) if is_valid_signal else False,
            "model_trained_at": model_trained_at_time,
            "ret_5": feature_row.get("ret_5"),
            "ret_10": feature_row.get("ret_10"),
            "ret_30": feature_row.get("ret_30"),
            "ema_10_30_diff": feature_row.get("ema_10_30_diff"),
            "ema_20_60_diff": feature_row.get("ema_20_60_diff"),
            "macd_hist": feature_row.get("macd_hist"),
            "rsi_14": feature_row.get("rsi_14"),
            "close_position": feature_row.get("close_position"),
            "body_ratio": feature_row.get("body_ratio"),
            "upper_shadow_ratio": feature_row.get("upper_shadow_ratio"),
            "lower_shadow_ratio": feature_row.get("lower_shadow_ratio"),
            "taker_buy_ratio": feature_row.get("taker_buy_ratio"),
            "trend_agreement": feature_row.get("trend_agreement"),
            "dt": point_time,
            "rule": selected["name"],
            "direction": selected["direction"],
            "correct": bool(correct),
            "prior_rule_win": selected["prior_rule_win"],
            "prior_rule_samples": selected["prior_rule_samples"],
            "state_ok": bool(state_ok),
        })
        for candidate in candidates:
            records_by_rule[candidate["name"]].append(candidate["direction"] == actual_direction)
        if step_no % progress_every_steps == 0:
            elapsed = time.time() - started
            print(f"[legacy_stream] {step_no}/{total} rows={len(rows)} elapsed={elapsed:.1f}s")

    out = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    metadata = {
        "source": "legacy_adaptive_rule_selected_stream",
        "days": days,
        "step_minutes": step_minutes,
        "model_update_minutes": model_update_minutes,
        "train_window_minutes": train_window_minutes,
        "max_steps": max_steps,
        "candidate_steps": len(candidate_indices),
        "rows": len(out),
        "output": str(output),
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the legacy adaptive_rule_switch selected candidate stream.")
    parser.add_argument("--input-predictions", type=Path, default=None)
    parser.add_argument("--emit-all-candidates", action="store_true")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--step-minutes", type=int, default=1)
    parser.add_argument("--model-update-minutes", type=int, default=120)
    parser.add_argument("--train-window-minutes", type=int, default=BACKTEST_TRAIN_WINDOW_MINUTES)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--progress-every-steps", type=int, default=1000)
    parser.add_argument("--no-update-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "legacy_adaptive_rule_selected_stream.csv")
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    if args.input_predictions is not None:
        predictions = pd.read_csv(args.input_predictions)
        out = build_stream_from_predictions(
            predictions,
            args.output,
            emit_all_candidates=args.emit_all_candidates,
        )
        valid = out[out["is_valid_signal"] == True]
        print("[legacy_stream] summary:")
        print(f"  input={args.input_predictions}")
        print(f"  rows={len(out)}")
        print(f"  valid_signals={len(valid)}")
        print(f"  valid_win_rate={float(valid['is_correct'].mean()) if not valid.empty else None}")
        print(f"  output={args.output}")
        return

    required_minutes = (
        args.train_window_minutes
        + args.days * 24 * 60
        + PREDICT_HORIZON_MINUTES
        + 5
    )
    df = get_recent_klines_with_cache(
        minutes=required_minutes,
        update_if_needed=not args.no_update_cache,
    )
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("w", encoding="utf-8") as log_file, contextlib.redirect_stdout(log_file):
            out = build_stream(
                df=df,
                days=args.days,
                step_minutes=args.step_minutes,
                model_update_minutes=args.model_update_minutes,
                train_window_minutes=args.train_window_minutes,
                max_steps=args.max_steps,
                output=args.output,
                progress_every_steps=args.progress_every_steps,
            )
    else:
        out = build_stream(
            df=df,
            days=args.days,
            step_minutes=args.step_minutes,
            model_update_minutes=args.model_update_minutes,
            train_window_minutes=args.train_window_minutes,
            max_steps=args.max_steps,
            output=args.output,
            progress_every_steps=args.progress_every_steps,
        )
    valid = out[out["is_valid_signal"] == True]
    print("[legacy_stream] summary:")
    print(f"  rows={len(out)}")
    print(f"  valid_signals={len(valid)}")
    print(f"  valid_win_rate={float(valid['is_correct'].mean()) if not valid.empty else None}")
    print(f"  output={args.output}")


if __name__ == "__main__":
    main()

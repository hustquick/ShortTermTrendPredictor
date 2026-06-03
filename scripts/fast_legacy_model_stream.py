import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import BACKTEST_TRAIN_WINDOW_MINUTES, DATA_DIR, PREDICT_HORIZON_MINUTES
from core.feature_pipeline import FeaturePipeline
from core.legacy_adaptive_coverage_gate import FEATURE_COLUMNS
from core.legacy_candidate_stream import (
    _active_candidate as legacy_active_candidate,
    legacy_candidates,
    legacy_state_ok,
)
from data_download import get_recent_klines_with_cache, ms_to_beijing_time
from features import add_dual_future_labels, get_feature_columns


class FastDualModel:
    def __init__(self, up_model, down_model, feature_cols: list[str]):
        self.up_model = up_model
        self.down_model = down_model
        self.feature_cols = feature_cols

    def predict_one(self, features: pd.DataFrame) -> dict:
        x = features[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        p_up_signal = float(self.up_model.predict_proba(x)[:, 1][0])
        p_down_signal = float(self.down_model.predict_proba(x)[:, 1][0])
        return _prediction_payload(p_up_signal, p_down_signal)

    def predict_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        x = features[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        p_up_signal = self.up_model.predict_proba(x)[:, 1]
        p_down_signal = self.down_model.predict_proba(x)[:, 1]
        score_sum = p_up_signal + p_down_signal
        p_up_relative = np.where(score_sum <= 1e-12, 0.5, p_up_signal / score_sum)
        return pd.DataFrame(
            {
                "up_probability": p_up_relative,
                "up_signal_probability": p_up_signal,
                "down_signal_probability": p_down_signal,
                "direction_edge": p_up_signal - p_down_signal,
                "confidence": np.maximum(p_up_signal, p_down_signal),
            },
            index=features.index,
        )


def _prediction_payload(p_up_signal: float, p_down_signal: float) -> dict:
        score_sum = p_up_signal + p_down_signal
        p_up_relative = 0.5 if score_sum <= 1e-12 else p_up_signal / score_sum
        return {
            "predicted_direction": "no_trade",
            "up_probability": p_up_relative,
            "up_signal_probability": p_up_signal,
            "down_signal_probability": p_down_signal,
            "direction_edge": p_up_signal - p_down_signal,
            "confidence": max(p_up_signal, p_down_signal),
            "high_win_rate_signal": False,
            "is_valid_signal": False,
        }


def _fit_fast_model(
    labeled: pd.DataFrame,
    train_start: int,
    train_end: int,
    feature_cols: list[str],
    label_mode: str,
    args: argparse.Namespace,
) -> FastDualModel:
    train = labeled.iloc[train_start:train_end].copy()
    if label_mode == "relative":
        valid_future = train["future_price"].notna()
        train.loc[valid_future, "up_label"] = (train.loc[valid_future, "future_return"] > 0).astype(int)
        train.loc[valid_future, "down_label"] = (train.loc[valid_future, "future_return"] <= 0).astype(int)
    train = train.dropna(subset=feature_cols + ["up_label", "down_label"])
    x = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_up = train["up_label"].astype(int)
    y_down = train["down_label"].astype(int)
    if y_up.nunique() < 2 or y_down.nunique() < 2:
        raise ValueError("fast model training labels must contain both classes")
    if args.model_type == "sgd":
        common = dict(
            loss="log_loss",
            penalty="l2",
            alpha=args.sgd_alpha,
            max_iter=args.sgd_max_iter,
            tol=1e-3,
            random_state=7,
        )
        up_model = make_pipeline(StandardScaler(), SGDClassifier(**common))
        down_model = make_pipeline(StandardScaler(), SGDClassifier(**common))
    elif args.model_type == "extra_trees":
        common = dict(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=7,
            n_jobs=-1,
        )
        up_model = ExtraTreesClassifier(**common)
        down_model = ExtraTreesClassifier(**common)
    else:
        common = dict(
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2_regularization,
            random_state=7,
        )
        up_model = HistGradientBoostingClassifier(**common)
        down_model = HistGradientBoostingClassifier(**common)
    up_model.fit(x, y_up)
    down_model.fit(x, y_down)
    return FastDualModel(up_model, down_model, feature_cols)


def _stats(records: deque[bool]) -> tuple[int, int, float]:
    samples = len(records)
    wins = sum(bool(item) for item in records)
    return samples, wins, wins / samples if samples else 0.0


def _selected_candidate(candidates: list[dict], records_by_rule: dict[str, deque[bool]], emit_all_candidates: bool):
    if not emit_all_candidates:
        return legacy_active_candidate(candidates, records_by_rule)
    scored = []
    for rule in candidates:
        samples, wins, win_rate = _stats(records_by_rule[rule["name"]])
        scored.append({**rule, "prior_rule_samples": samples, "prior_rule_wins": wins, "prior_rule_win": win_rate})
    active = [item for item in scored if item["prior_rule_samples"] >= 5 and item["prior_rule_win"] >= 0.80]
    if active:
        return sorted(active, key=lambda item: (item["prior_rule_win"], item["prior_rule_samples"], item["confidence"]), reverse=True)[0]
    return sorted(scored, key=lambda item: (item["prior_rule_samples"], item["confidence"]), reverse=True)[0]


def build_fast_stream(args: argparse.Namespace) -> pd.DataFrame:
    required_minutes = args.train_window_minutes + args.days * 24 * 60 + PREDICT_HORIZON_MINUTES + 5
    df = get_recent_klines_with_cache(minutes=required_minutes, update_if_needed=not args.no_update_cache)
    df = df.sort_values("timestamp").reset_index(drop=True)
    close_by_timestamp = df.set_index("timestamp")["close"]
    feature_df = FeaturePipeline().build(df)
    labeled = add_dual_future_labels(feature_df)
    feature_cols = get_feature_columns(labeled)
    labeled_features = labeled[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    horizon = PREDICT_HORIZON_MINUTES
    test_end = len(df) - horizon - 1
    test_start = max(args.train_window_minutes, test_end - args.days * 24 * 60)
    candidate_indices = list(range(test_start, test_end, args.step_minutes))
    if args.max_steps is not None:
        candidate_indices = candidate_indices[-args.max_steps:]
    print(
        "[fast_legacy_stream] generation plan: "
        f"days={args.days}, steps={len(candidate_indices)}, model_update_minutes={args.model_update_minutes}, "
        f"train_window_minutes={args.train_window_minutes}",
        flush=True,
    )

    model = None
    next_model_update_idx = None
    model_trained_at_time = ""
    records_by_rule: dict[str, deque[bool]] = defaultdict(lambda: deque(maxlen=10))
    pending_outcomes = []
    rows = []
    rule_outcome_rows = []
    started = time.time()

    prediction_cache = pd.DataFrame()
    prediction_cache_end_idx = -1
    for step_no, idx in enumerate(candidate_indices, start=1):
        point_time = ms_to_beijing_time(int(df.iloc[idx]["timestamp"]))
        point_dt = pd.to_datetime(point_time)
        if args.causal_validation_delay_minutes > 0:
            ready = []
            still_pending = []
            for due_time, outcomes in pending_outcomes:
                if due_time <= point_dt:
                    ready.append(outcomes)
                else:
                    still_pending.append((due_time, outcomes))
            pending_outcomes = still_pending
            for outcomes in ready:
                for rule_name, direction, actual_direction in outcomes:
                    records_by_rule[rule_name].append(direction == actual_direction)

        if model is None or next_model_update_idx is None or idx >= next_model_update_idx:
            train_start = max(0, idx - args.train_window_minutes)
            fit_started = time.time()
            model = _fit_fast_model(labeled, train_start, idx, feature_cols, args.label_mode, args)
            next_model_update_idx = idx + args.model_update_minutes
            model_trained_at_time = point_time
            segment_end = min(next_model_update_idx, candidate_indices[-1] + args.step_minutes)
            segment_positions = list(range(idx, segment_end, args.step_minutes))
            prediction_cache = model.predict_frame(labeled_features.iloc[segment_positions])
            prediction_cache_end_idx = segment_end
            print(
                "[fast_legacy_stream] model update done: "
                f"step={step_no}/{len(candidate_indices)}, anchor={point_time}, "
                f"elapsed={time.time() - fit_started:.1f}s",
                flush=True,
            )

        if idx >= prediction_cache_end_idx or idx not in prediction_cache.index:
            prediction = model.predict_one(labeled_features.iloc[[idx]].copy())
        else:
            pred_row = prediction_cache.loc[idx]
            prediction = _prediction_payload(
                float(pred_row["up_signal_probability"]),
                float(pred_row["down_signal_probability"]),
            )
        feature_row = feature_df.iloc[idx]
        candidates = legacy_candidates(feature_row, prediction)
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
        candidate_outcomes = [(candidate["name"], candidate["direction"], actual_direction) for candidate in candidates]
        rule_outcome_rows.extend(
            {
                "timestamp": point_time,
                "rule": rule_name,
                "direction": direction,
                "actual_direction": actual,
                "correct": direction == actual,
            }
            for rule_name, direction, actual in candidate_outcomes
        )

        selected = _selected_candidate(candidates, records_by_rule, args.emit_all_candidates)
        if selected is None:
            if args.causal_validation_delay_minutes > 0:
                pending_outcomes.append((point_dt + pd.Timedelta(minutes=args.causal_validation_delay_minutes), candidate_outcomes))
            else:
                for rule_name, direction, actual in candidate_outcomes:
                    records_by_rule[rule_name].append(direction == actual)
            continue

        correct = selected["direction"] == actual_direction
        state_ok = legacy_state_ok(feature_row, prediction, selected["direction"])
        is_valid_signal = selected["prior_rule_samples"] >= 5 and selected["prior_rule_win"] >= 0.80 and state_ok
        row = {
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
            "dt": point_time,
            "rule": selected["name"],
            "direction": selected["direction"],
            "correct": bool(correct),
            "prior_rule_win": selected["prior_rule_win"],
            "prior_rule_samples": selected["prior_rule_samples"],
            "state_ok": bool(state_ok),
        }
        for column in FEATURE_COLUMNS:
            row[column] = feature_row.get(column)
        rows.append(row)

        if args.causal_validation_delay_minutes > 0:
            pending_outcomes.append((point_dt + pd.Timedelta(minutes=args.causal_validation_delay_minutes), candidate_outcomes))
        else:
            for rule_name, direction, actual in candidate_outcomes:
                records_by_rule[rule_name].append(direction == actual)

        if step_no % args.progress_every_steps == 0:
            print(f"[fast_legacy_stream] progress {step_no}/{len(candidate_indices)} rows={len(rows)} elapsed={time.time() - started:.1f}s", flush=True)

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    if args.rule_outcome_output is not None:
        args.rule_outcome_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rule_outcome_rows).to_csv(args.rule_outcome_output, index=False)
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(
            {
                "source": "fast_legacy_model_stream",
                "days": args.days,
                "step_minutes": args.step_minutes,
                "model_update_minutes": args.model_update_minutes,
                "train_window_minutes": args.train_window_minutes,
                "label_mode": args.label_mode,
                "model_type": args.model_type,
                "max_iter": args.max_iter,
                "learning_rate": args.learning_rate,
                "max_leaf_nodes": args.max_leaf_nodes,
                "l2_regularization": args.l2_regularization,
                "sgd_alpha": args.sgd_alpha,
                "sgd_max_iter": args.sgd_max_iter,
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "max_steps": args.max_steps,
                "rows": len(out),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast experimental legacy candidate stream.")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--step-minutes", type=int, default=1)
    parser.add_argument("--model-update-minutes", type=int, default=1440)
    parser.add_argument("--train-window-minutes", type=int, default=BACKTEST_TRAIN_WINDOW_MINUTES)
    parser.add_argument("--label-mode", choices=("dual", "relative"), default="dual")
    parser.add_argument("--model-type", choices=("hgb", "sgd", "extra_trees"), default="hgb")
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--max-leaf-nodes", type=int, default=7)
    parser.add_argument("--l2-regularization", type=float, default=5.0)
    parser.add_argument("--sgd-alpha", type=float, default=0.0005)
    parser.add_argument("--sgd-max-iter", type=int, default=1000)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--emit-all-candidates", action="store_true")
    parser.add_argument("--causal-validation-delay-minutes", type=int, default=PREDICT_HORIZON_MINUTES)
    parser.add_argument("--progress-every-steps", type=int, default=1000)
    parser.add_argument("--no-update-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "fast_legacy_candidate_stream.csv")
    parser.add_argument("--rule-outcome-output", type=Path, default=None)
    args = parser.parse_args()
    out = build_fast_stream(args)
    valid = out[out["is_valid_signal"] == True]
    print("[fast_legacy_stream] summary:")
    print(f"  rows={len(out)}")
    print(f"  valid_signals={len(valid)}")
    print(f"  valid_win_rate={float(valid['is_correct'].mean()) if not valid.empty else None}")
    print(f"  output={args.output}")


if __name__ == "__main__":
    main()

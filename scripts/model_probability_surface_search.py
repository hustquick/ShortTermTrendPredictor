import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from config import BACKTEST_TRAIN_WINDOW_MINUTES, LABEL_NEUTRAL_THRESHOLD, RANDOM_STATE
from data_download import get_recent_klines_with_cache, load_history_csv, ms_to_beijing_time
from features import build_features, get_feature_columns


@dataclass
class FoldPrediction:
    fold: int
    timestamp: pd.Series
    future_return: np.ndarray
    p_up: np.ndarray
    p_down: np.ndarray


def _prepare_frame(
    days: int,
    horizon_minutes: int,
    train_days: int,
    label_threshold: float,
    update_cache: bool,
) -> pd.DataFrame:
    minutes = days * 24 * 60 + train_days * 24 * 60 + BACKTEST_TRAIN_WINDOW_MINUTES + horizon_minutes + 5
    if update_cache:
        raw = get_recent_klines_with_cache(
            minutes=minutes,
            update_if_needed=True,
        ).sort_values("timestamp").reset_index(drop=True)
    else:
        raw = load_history_csv().sort_values("timestamp").reset_index(drop=True)
        if raw.empty:
            raise RuntimeError("local history cache is empty; rerun with --update-cache")
        required_start = int(raw["timestamp"].max()) - minutes * 60_000
        raw = raw[raw["timestamp"] >= required_start].copy().reset_index(drop=True)

    feature_df = build_features(raw)
    close_by_timestamp = raw.set_index("timestamp")["close"]
    future_timestamp = feature_df["timestamp"] + horizon_minutes * 60_000
    feature_df["future_price"] = future_timestamp.map(close_by_timestamp)
    feature_df["future_return"] = feature_df["future_price"] / feature_df["close"] - 1
    feature_df["up_label"] = (feature_df["future_return"] > label_threshold).astype(float)
    feature_df["down_label"] = (feature_df["future_return"] < -label_threshold).astype(float)
    feature_df.loc[feature_df["future_price"].isna(), ["up_label", "down_label"]] = np.nan

    feature_cols = get_feature_columns(feature_df)
    data = feature_df.dropna(subset=[*feature_cols, "future_return", "up_label", "down_label"]).copy()
    data.attrs["feature_cols"] = feature_cols
    return data.reset_index(drop=True)


def _make_model(max_iter: int, learning_rate: float, l2_regularization: float):
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=l2_regularization,
        early_stopping=True,
        validation_fraction=0.12,
        random_state=RANDOM_STATE,
    )


def _fit_predict_fold(
    data: pd.DataFrame,
    feature_cols: list[str],
    fold: int,
    fold_start_ts: int,
    fold_end_ts: int,
    train_days: int,
    max_train_rows: int,
    max_iter: int,
    learning_rate: float,
    l2_regularization: float,
) -> FoldPrediction | None:
    train_start_ts = fold_start_ts - train_days * 24 * 60 * 60_000
    train = data[(data["timestamp"] >= train_start_ts) & (data["timestamp"] < fold_start_ts)].copy()
    valid = data[(data["timestamp"] >= fold_start_ts) & (data["timestamp"] < fold_end_ts)].copy()
    if len(train) > max_train_rows:
        train = train.tail(max_train_rows).copy()
    if len(train) < 1000 or len(valid) == 0:
        print(
            f"[model_probability_surface] fold={fold} skipped train_rows={len(train)} "
            f"valid_rows={len(valid)}"
        )
        return None

    X_train = train[feature_cols]
    X_valid = valid[feature_cols]
    y_up = train["up_label"].astype(int)
    y_down = train["down_label"].astype(int)
    if y_up.nunique() < 2 or y_down.nunique() < 2:
        print(f"[model_probability_surface] fold={fold} skipped single-class labels")
        return None

    up_model = _make_model(max_iter, learning_rate, l2_regularization)
    down_model = _make_model(max_iter, learning_rate, l2_regularization)
    up_model.fit(X_train, y_up)
    down_model.fit(X_train, y_down)

    p_up = up_model.predict_proba(X_valid)[:, 1]
    p_down = down_model.predict_proba(X_valid)[:, 1]
    print(
        f"[model_probability_surface] fold={fold} "
        f"{ms_to_beijing_time(int(fold_start_ts))}->{ms_to_beijing_time(int(fold_end_ts))} "
        f"train_rows={len(train)} valid_rows={len(valid)} "
        f"p_up_mean={p_up.mean():.4f} p_down_mean={p_down.mean():.4f}"
    )
    return FoldPrediction(
        fold=fold,
        timestamp=valid["timestamp"],
        future_return=valid["future_return"].to_numpy(),
        p_up=p_up,
        p_down=p_down,
    )


def _scan_thresholds(
    predictions: list[FoldPrediction],
    days: int,
    min_signals_per_day: float,
    thresholds: list[float],
    edge_thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    min_signals_per_fold = days / len(predictions) * min_signals_per_day
    for threshold in thresholds:
        for edge_threshold in edge_thresholds:
            for direction in ["up", "down", "both"]:
                fold_rows = []
                for pred in predictions:
                    edge = pred.p_up - pred.p_down
                    up_mask = (pred.p_up >= threshold) & (edge >= edge_threshold)
                    down_mask = (pred.p_down >= threshold) & (edge <= -edge_threshold)
                    if direction == "up":
                        mask = up_mask
                        correct = pred.future_return > 0
                    elif direction == "down":
                        mask = down_mask
                        correct = pred.future_return < 0
                    else:
                        mask = up_mask | down_mask
                        correct = np.where(up_mask, pred.future_return > 0, pred.future_return < 0)
                    signals = int(mask.sum())
                    wins = int(correct[mask].sum()) if signals else 0
                    win_rate = wins / signals if signals else np.nan
                    fold_rows.append((signals, wins, win_rate))

                counts = [item[0] for item in fold_rows]
                if min(counts) < min_signals_per_fold:
                    continue
                wins = [item[1] for item in fold_rows]
                total_signals = sum(counts)
                total_wins = sum(wins)
                row = {
                    "direction": direction,
                    "threshold": threshold,
                    "edge_threshold": edge_threshold,
                    "signals": total_signals,
                    "signals_per_day": total_signals / days,
                    "weighted_win_rate": total_wins / total_signals,
                    "min_fold_win_rate": min(item[2] for item in fold_rows),
                }
                for idx, (signals, _, win_rate) in enumerate(fold_rows, start=1):
                    row[f"fold{idx}_signals"] = signals
                    row[f"fold{idx}_win_rate"] = win_rate
                rows.append(row)

    report = pd.DataFrame(rows)
    if report.empty:
        return report
    return report.sort_values(
        ["min_fold_win_rate", "weighted_win_rate", "signals"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Strict multi-fold probability surface search for lightweight dual-direction models."
    )
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--horizon-minutes", type=int, default=10)
    parser.add_argument("--label-threshold", type=float, default=LABEL_NEUTRAL_THRESHOLD)
    parser.add_argument("--train-days", type=int, default=30)
    parser.add_argument("--max-train-rows", type=int, default=80_000)
    parser.add_argument("--max-iter", type=int, default=140)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--l2-regularization", type=float, default=0.10)
    parser.add_argument("--min-signals-per-day", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", default=None)
    parser.add_argument("--update-cache", action="store_true")
    args = parser.parse_args()

    data = _prepare_frame(
        days=args.days,
        horizon_minutes=args.horizon_minutes,
        train_days=args.train_days,
        label_threshold=args.label_threshold,
        update_cache=args.update_cache,
    )
    feature_cols = data.attrs["feature_cols"]
    test_end = data["timestamp"].max() - args.horizon_minutes * 60_000
    test_start = test_end - args.days * 24 * 60 * 60_000
    fold_size_ms = args.days * 24 * 60 * 60_000 // args.folds

    predictions = []
    for fold in range(1, args.folds + 1):
        fold_start_ts = test_start + (fold - 1) * fold_size_ms
        fold_end_ts = test_start + (fold * fold_size_ms if fold < args.folds else args.days * 24 * 60 * 60_000)
        pred = _fit_predict_fold(
            data=data,
            feature_cols=feature_cols,
            fold=fold,
            fold_start_ts=fold_start_ts,
            fold_end_ts=fold_end_ts,
            train_days=args.train_days,
            max_train_rows=args.max_train_rows,
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            l2_regularization=args.l2_regularization,
        )
        if pred is not None:
            predictions.append(pred)

    if len(predictions) != args.folds:
        raise RuntimeError(f"only {len(predictions)}/{args.folds} folds produced predictions")

    thresholds = [round(x, 2) for x in np.arange(0.50, 0.91, 0.02)]
    edge_thresholds = [round(x, 2) for x in np.arange(0.00, 0.31, 0.02)]
    report = _scan_thresholds(
        predictions=predictions,
        days=args.days,
        min_signals_per_day=args.min_signals_per_day,
        thresholds=thresholds,
        edge_thresholds=edge_thresholds,
    )

    if report.empty:
        print("[model_probability_surface] no threshold surface met density requirements")
    else:
        print(report.head(args.top).to_string(index=False))

    if args.output:
        report.head(args.top).to_csv(args.output, index=False)
        print(f"[model_probability_surface] saved: {args.output}")


if __name__ == "__main__":
    main()

import argparse
import itertools

import numpy as np
import pandas as pd

from config import BACKTEST_TRAIN_WINDOW_MINUTES, PREDICT_HORIZON_MINUTES
from data_download import get_recent_klines_with_cache
from features import build_features


FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_30",
    "rsi_14",
    "boll_position",
    "macd_hist",
    "macd_hist_diff",
    "trend_agreement",
    "ema_5_20_diff",
    "ema_10_30_diff",
    "volume_ratio_10",
    "trade_count_ratio_10",
    "quote_volume_ratio_10",
    "taker_buy_ratio",
    "taker_buy_ratio_diff_5_10",
    "close_position",
    "atr_14",
]


def _add_condition(conditions: list[tuple[str, np.ndarray]], name: str, mask: pd.Series):
    conditions.append((name, mask.fillna(False).to_numpy()))


def _build_conditions(feature_df: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    conditions: list[tuple[str, np.ndarray]] = []

    for col in [
        "ret_1",
        "ret_5",
        "ret_10",
        "ret_30",
        "macd_hist",
        "macd_hist_diff",
        "ema_5_20_diff",
        "ema_10_30_diff",
        "taker_buy_ratio_diff_5_10",
    ]:
        _add_condition(conditions, f"{col}>0", feature_df[col] > 0)
        _add_condition(conditions, f"{col}<0", feature_df[col] < 0)

    for threshold in [0.25, 0.40, 0.50, 0.60, 0.75]:
        _add_condition(conditions, f"boll<{threshold}", feature_df["boll_position"] < threshold)
        _add_condition(conditions, f"boll>{threshold}", feature_df["boll_position"] > threshold)

    for threshold in [35, 45, 50, 55, 65]:
        _add_condition(conditions, f"rsi<{threshold}", feature_df["rsi_14"] < threshold)
        _add_condition(conditions, f"rsi>{threshold}", feature_df["rsi_14"] > threshold)

    for threshold in [-0.333, 0.0, 0.333, 0.667]:
        _add_condition(conditions, f"trend>{threshold}", feature_df["trend_agreement"] > threshold)
        _add_condition(conditions, f"trend<{threshold}", feature_df["trend_agreement"] < threshold)

    for threshold in [0.80, 1.00, 1.20, 1.50]:
        _add_condition(conditions, f"volratio<{threshold}", feature_df["volume_ratio_10"] < threshold)
        _add_condition(conditions, f"volratio>{threshold}", feature_df["volume_ratio_10"] > threshold)

    for threshold in [0.45, 0.50, 0.55, 0.60]:
        _add_condition(conditions, f"taker>{threshold}", feature_df["taker_buy_ratio"] > threshold)
        _add_condition(conditions, f"taker<{threshold}", feature_df["taker_buy_ratio"] < threshold)

    for threshold in [0.20, 0.50, 0.80]:
        _add_condition(conditions, f"closepos>{threshold}", feature_df["close_position"] > threshold)
        _add_condition(conditions, f"closepos<{threshold}", feature_df["close_position"] < threshold)

    return conditions


def _prepare_feature_frame(days: int, horizon_minutes: int, update_cache: bool) -> pd.DataFrame:
    minutes = days * 24 * 60 + BACKTEST_TRAIN_WINDOW_MINUTES + horizon_minutes + 5
    raw = get_recent_klines_with_cache(
        minutes=minutes,
        update_if_needed=update_cache,
    ).sort_values("timestamp").reset_index(drop=True)

    feature_df = build_features(raw)
    close_by_timestamp = raw.set_index("timestamp")["close"]
    feature_df["future_price"] = (
        feature_df["timestamp"] + horizon_minutes * 60_000
    ).map(close_by_timestamp)
    feature_df["future_return"] = feature_df["future_price"] / feature_df["close"] - 1
    feature_df = feature_df.dropna(subset=["future_return", *FEATURE_COLUMNS]).copy()

    test_end = feature_df["timestamp"].max() - horizon_minutes * 60_000
    test_start = test_end - days * 24 * 60 * 60_000
    return feature_df[
        (feature_df["timestamp"] >= test_start)
        & (feature_df["timestamp"] <= test_end)
    ].copy().reset_index(drop=True)


def _fold_results(mask: np.ndarray, target: np.ndarray, fold_ids: np.ndarray, folds: int):
    rows = []
    for fold in range(folds):
        fold_mask = mask & (fold_ids == fold)
        signals = int(fold_mask.sum())
        win_rate = float(target[fold_mask].mean()) if signals else np.nan
        rows.append((signals, win_rate))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Search simple feature-rule surfaces with strict multi-fold density checks."
    )
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--horizon-minutes", type=int, default=PREDICT_HORIZON_MINUTES)
    parser.add_argument("--max-clauses", type=int, default=3)
    parser.add_argument("--min-signals-per-day", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", default=None)
    parser.add_argument("--update-cache", action="store_true")
    args = parser.parse_args()

    feature_df = _prepare_feature_frame(
        args.days,
        horizon_minutes=args.horizon_minutes,
        update_cache=args.update_cache,
    )
    days = (feature_df["timestamp"].max() - feature_df["timestamp"].min()) / 86_400_000
    print(
        f"[rule_surface_search] rows={len(feature_df)} days={days:.2f} "
        f"horizon_minutes={args.horizon_minutes}"
    )

    conditions = _build_conditions(feature_df)
    n = len(feature_df)
    fold_size = max(1, n // args.folds)
    fold_ids = np.minimum(np.arange(n) // fold_size, args.folds - 1)
    min_signals_per_fold = (args.days / args.folds) * args.min_signals_per_day
    actual = {
        "up": (feature_df["future_return"] > 0).to_numpy(),
        "down": (feature_df["future_return"] < 0).to_numpy(),
    }

    candidates = []
    for clause_count in range(1, args.max_clauses + 1):
        for combo in itertools.combinations(range(len(conditions)), clause_count):
            mask = np.ones(n, dtype=bool)
            names = []
            for condition_idx in combo:
                name, condition_mask = conditions[condition_idx]
                names.append(name)
                mask &= condition_mask

            total_signals = int(mask.sum())
            if total_signals < min_signals_per_fold * args.folds or total_signals > n * 0.5:
                continue

            for direction, target in actual.items():
                folds = _fold_results(mask, target, fold_ids, args.folds)
                fold_counts = [row[0] for row in folds]
                if min(fold_counts) < min_signals_per_fold:
                    continue
                fold_win_rates = [row[1] for row in folds]
                weighted_win_rate = float(
                    sum(signals * win_rate for signals, win_rate in folds) / sum(fold_counts)
                )
                candidates.append(
                    {
                        "direction": direction,
                        "rule": " & ".join(names),
                        "signals": sum(fold_counts),
                        "min_fold_win_rate": min(fold_win_rates),
                        "weighted_win_rate": weighted_win_rate,
                        "folds": folds,
                    }
                )

    candidates.sort(
        key=lambda row: (
            row["min_fold_win_rate"],
            row["weighted_win_rate"],
            row["signals"],
        ),
        reverse=True,
    )

    rows = []
    for row in candidates[: args.top]:
        flat = {
            "direction": row["direction"],
            "rule": row["rule"],
            "signals": row["signals"],
            "min_fold_win_rate": row["min_fold_win_rate"],
            "weighted_win_rate": row["weighted_win_rate"],
        }
        for idx, (signals, win_rate) in enumerate(row["folds"], start=1):
            flat[f"fold{idx}_signals"] = signals
            flat[f"fold{idx}_win_rate"] = win_rate
        rows.append(flat)

    report = pd.DataFrame(rows)
    if report.empty:
        print("[rule_surface_search] no candidates met density requirements")
    else:
        print(report.to_string(index=False))

    if args.output:
        output_path = pd.io.common.stringify_path(args.output)
        report.to_csv(output_path, index=False)
        print(f"[rule_surface_search] saved: {output_path}")


if __name__ == "__main__":
    main()

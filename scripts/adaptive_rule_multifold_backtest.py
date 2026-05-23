import argparse
import contextlib
from pathlib import Path

import pandas as pd

import run_strategy
from config import (
    BACKTEST_MIN_TRAIN_SAMPLES,
    BACKTEST_TRAIN_WINDOW_MINUTES,
    DATA_DIR,
    PREDICT_HORIZON_MINUTES,
)
from data_download import get_recent_klines_with_cache, ms_to_beijing_time
from run_strategy import high_confidence_report, strict_walk_forward_backtest


def _fold_summary(df: pd.DataFrame, fold: int, fold_start: str, fold_end: str) -> dict:
    report = high_confidence_report(df)
    signals = df[df["is_valid_signal"] == True].copy()
    avg_gap = None
    if len(signals) >= 2:
        times = pd.to_datetime(signals["timestamp"])
        avg_gap = float(times.diff().dt.total_seconds().dropna().mean() / 60.0)
    fold_days = (
        pd.to_datetime(fold_end) - pd.to_datetime(fold_start)
    ).total_seconds() / 86400.0
    signals_per_day = report["valid_signals"] / fold_days if fold_days > 0 else None
    return {
        "fold": fold,
        "fold_start": fold_start,
        "fold_end": fold_end,
        "rows": report["total_rows"],
        "signals": report["valid_signals"],
        "signals_per_day": signals_per_day,
        "signal_rate": report["valid_signal_ratio"],
        "win_rate": report["valid_win_rate"],
        "long_signals": report["long_signals"],
        "long_win_rate": report["long_win_rate"],
        "short_signals": report["short_signals"],
        "short_win_rate": report["short_win_rate"],
        "avg_signal_gap_minutes": avg_gap,
    }


def main():
    parser = argparse.ArgumentParser(description="Strict multi-fold backtest for adaptive_rule_switch only.")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--step-minutes", type=int, default=1)
    parser.add_argument("--model-update-minutes", type=int, default=30)
    parser.add_argument("--train-window-minutes", type=int, default=BACKTEST_TRAIN_WINDOW_MINUTES)
    parser.add_argument("--max-steps-per-fold", type=int, default=None)
    parser.add_argument("--min-win-rate", type=float, default=0.75)
    parser.add_argument("--min-signals-per-day", type=float, default=10.0)
    parser.add_argument("--no-update-cache", action="store_true")
    parser.add_argument("--output-prefix", default="adaptive_rule_switch_multifold")
    args = parser.parse_args()

    if args.folds < 1:
        raise ValueError("--folds must be >= 1")

    test_minutes = args.days * 24 * 60
    required_minutes = args.train_window_minutes + test_minutes + PREDICT_HORIZON_MINUTES + 5
    df = get_recent_klines_with_cache(
        minutes=required_minutes,
        update_if_needed=not args.no_update_cache,
    ).sort_values("timestamp").reset_index(drop=True)
    if len(df) < args.train_window_minutes + test_minutes + PREDICT_HORIZON_MINUTES:
        raise RuntimeError(f"not enough rows for requested backtest: rows={len(df)}")

    run_strategy.BACKTEST_STRATEGIES = ("adaptive_rule_switch",)

    test_end = len(df) - PREDICT_HORIZON_MINUTES - 1
    test_start = test_end - test_minutes
    fold_size = test_minutes // args.folds
    result_parts = []
    summaries = []

    for fold in range(1, args.folds + 1):
        fold_start = test_start + (fold - 1) * fold_size
        fold_end = test_start + (fold * fold_size if fold < args.folds else test_minutes)
        slice_start = max(0, fold_start - args.train_window_minutes)
        slice_end = min(len(df), fold_end + PREDICT_HORIZON_MINUTES + 1)
        fold_df = df.iloc[slice_start:slice_end].copy().reset_index(drop=True)
        fold_start_time = ms_to_beijing_time(int(df.iloc[fold_start]["timestamp"]))
        fold_end_time = ms_to_beijing_time(int(df.iloc[fold_end - 1]["timestamp"]))
        print(f"[adaptive_rule_multifold] fold={fold}/{args.folds} {fold_start_time} -> {fold_end_time}")

        result = strict_walk_forward_backtest(
            fold_df,
            train_window_minutes=args.train_window_minutes,
            step_minutes=args.step_minutes,
            model_update_minutes=args.model_update_minutes,
            min_train_samples=BACKTEST_MIN_TRAIN_SAMPLES,
            max_steps=args.max_steps_per_fold,
            progress_every=500,
            use_walk_forward_match_pool=False,
        )
        if result.empty:
            continue
        result["fold"] = fold
        result["fold_start"] = fold_start_time
        result["fold_end"] = fold_end_time
        result_parts.append(result)
        summaries.append(_fold_summary(result, fold, fold_start_time, fold_end_time))

        fold_path = DATA_DIR / f"{args.output_prefix}_fold{fold}.csv"
        result.to_csv(fold_path, index=False)
        print(f"[adaptive_rule_multifold] fold result saved: {fold_path}")

    combined = pd.concat(result_parts, ignore_index=True) if result_parts else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)

    combined_path = DATA_DIR / f"{args.output_prefix}_{args.days}d_{args.folds}fold.csv"
    summary_path = DATA_DIR / f"{args.output_prefix}_{args.days}d_{args.folds}fold_summary.csv"
    combined.to_csv(combined_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("[adaptive_rule_multifold] summary:")
    if summary_df.empty:
        print("  no results")
    else:
        print(summary_df.to_string(index=False))
        total_report = high_confidence_report(combined)
        combined_start = pd.to_datetime(combined["timestamp"]).min()
        combined_end = pd.to_datetime(combined["timestamp"]).max()
        combined_days = (combined_end - combined_start).total_seconds() / 86400.0
        combined_signals_per_day = (
            total_report["valid_signals"] / combined_days if combined_days > 0 else None
        )
        fold_pass = (
            (summary_df["win_rate"].fillna(0.0) >= args.min_win_rate)
            & (summary_df["signals_per_day"].fillna(0.0) >= args.min_signals_per_day)
        )
        combined_pass = (
            (total_report["valid_win_rate"] or 0.0) >= args.min_win_rate
            and (combined_signals_per_day or 0.0) >= args.min_signals_per_day
        )
        print("[adaptive_rule_multifold] combined:")
        print(f"  rows={total_report['total_rows']}")
        print(f"  signals={total_report['valid_signals']}")
        print(f"  signals_per_day={combined_signals_per_day}")
        print(f"  signal_rate={total_report['valid_signal_ratio']}")
        print(f"  win_rate={total_report['valid_win_rate']}")
        print(f"  short_signals={total_report['short_signals']}")
        print(f"  short_win_rate={total_report['short_win_rate']}")
        print(f"  long_signals={total_report['long_signals']}")
        print(f"  long_win_rate={total_report['long_win_rate']}")
        print("[adaptive_rule_multifold] acceptance:")
        print(f"  min_win_rate={args.min_win_rate}")
        print(f"  min_signals_per_day={args.min_signals_per_day}")
        print(f"  fold_pass={bool(fold_pass.all())}")
        print(f"  combined_pass={bool(combined_pass)}")
        print(f"  accepted={bool(fold_pass.all() and combined_pass)}")
    print(f"[adaptive_rule_multifold] combined saved: {combined_path}")
    print(f"[adaptive_rule_multifold] summary saved: {summary_path}")


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()

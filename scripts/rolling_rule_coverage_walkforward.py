import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from online_signal_filter_walkforward import (
    _apply_condition,
    _load_rows,
    _search_best_condition,
)


def run_rolling_coverage(
    df: pd.DataFrame,
    train_days: int,
    cover_days: int,
    step_days: int,
    max_clauses: int,
    min_samples: int,
    min_signals_per_day: float,
    min_win_rate: float,
    min_wilson_lower: float,
    beam_size: int,
    source_strategy: str,
    source_layer: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = df["timestamp_dt"].min() + pd.Timedelta(days=train_days)
    end = df["timestamp_dt"].max()
    current = start
    rows = []
    signal_frames = []
    window_no = 1

    while current < end:
        train_start = current - pd.Timedelta(days=train_days)
        cover_end = min(current + pd.Timedelta(days=cover_days), end)
        train = df[(df["timestamp_dt"] >= train_start) & (df["timestamp_dt"] < current)].copy()
        cover = df[(df["timestamp_dt"] >= current) & (df["timestamp_dt"] < cover_end)].copy()
        if train.empty or cover.empty:
            current += pd.Timedelta(days=step_days)
            window_no += 1
            continue

        selected = _search_best_condition(
            train,
            max_clauses=max_clauses,
            min_samples=min_samples,
            min_signals_per_day=min_signals_per_day,
            min_win_rate=min_win_rate,
            min_wilson_lower=min_wilson_lower,
            beam_size=beam_size,
        )
        if selected is None:
            rows.append({
                "window": window_no,
                "source_strategy": source_strategy,
                "source_layer": source_layer,
                "train_start": train_start,
                "train_end": current,
                "cover_start": current,
                "cover_end": cover_end,
                "condition": "",
                "cover_signals": 0,
                "cover_wins": 0,
                "cover_win_rate": np.nan,
                "cover_signals_per_day": 0.0,
            })
            current += pd.Timedelta(days=step_days)
            window_no += 1
            continue

        cover_mask = _apply_condition(cover, selected["condition"])
        cover_signals = cover[cover_mask].copy()
        cover_count = int(len(cover_signals))
        cover_wins = int(cover_signals["correct_bool"].sum()) if cover_count else 0
        cover_span_days = max((cover_end - current).total_seconds() / 86_400, 1e-9)
        if cover_count:
            cover_signals["rolling_window"] = window_no
            cover_signals["rolling_condition"] = selected["condition"]
            cover_signals["source_strategy"] = source_strategy
            cover_signals["source_layer"] = source_layer
            cover_signals["source_rule"] = cover_signals.get("rule", "")
            cover_signals["source_condition"] = selected["condition"]
            signal_frames.append(cover_signals)

        rows.append({
            "window": window_no,
            "source_strategy": source_strategy,
            "source_layer": source_layer,
            "source_condition": selected["condition"],
            **selected,
            "train_start": train_start,
            "train_end": current,
            "cover_start": current,
            "cover_end": cover_end,
            "cover_signals": cover_count,
            "cover_wins": cover_wins,
            "cover_win_rate": cover_wins / cover_count if cover_count else np.nan,
            "cover_signals_per_day": cover_count / cover_span_days,
        })
        current += pd.Timedelta(days=step_days)
        window_no += 1

    report = pd.DataFrame(rows)
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    return report, signals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rolling coverage validation: rediscover rules from the past window and cover the next window."
    )
    parser.add_argument("csv", type=Path)
    parser.add_argument("--train-days", type=int, default=30)
    parser.add_argument("--cover-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--max-clauses", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=60)
    parser.add_argument("--min-signals-per-day", type=float, default=5.0)
    parser.add_argument("--min-win-rate", type=float, default=0.75)
    parser.add_argument("--min-wilson-lower", type=float, default=0.68)
    parser.add_argument("--beam-size", type=int, default=80)
    parser.add_argument("--require-step-minutes", type=int, default=None)
    parser.add_argument("--source-strategy", default="adaptive_rule_switch")
    parser.add_argument("--source-layer", default="legacy_adaptive_candidate_stream")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    df = _load_rows(args.csv)
    if df.empty:
        raise RuntimeError(f"empty csv: {args.csv}")
    if args.require_step_minutes is not None:
        observed_step = df["timestamp_dt"].sort_values().diff().dropna().dt.total_seconds().median() / 60
        if pd.isna(observed_step) or abs(float(observed_step) - args.require_step_minutes) > 1e-6:
            raise RuntimeError(
                f"candidate stream step mismatch: expected {args.require_step_minutes} minutes, "
                f"observed median {observed_step} minutes"
            )

    report, signals = run_rolling_coverage(
        df=df,
        train_days=args.train_days,
        cover_days=args.cover_days,
        step_days=args.step_days,
        max_clauses=args.max_clauses,
        min_samples=args.min_samples,
        min_signals_per_day=args.min_signals_per_day,
        min_win_rate=args.min_win_rate,
        min_wilson_lower=args.min_wilson_lower,
        beam_size=args.beam_size,
        source_strategy=args.source_strategy,
        source_layer=args.source_layer,
    )
    print(report.to_string(index=False) if not report.empty else "no windows")
    total_signals = int(len(signals))
    total_wins = int(signals["correct_bool"].sum()) if total_signals else 0
    total_days = (
        (pd.to_datetime(report["cover_end"]).max() - pd.to_datetime(report["cover_start"]).min()).total_seconds()
        / 86_400
        if not report.empty
        else 0.0
    )
    print("[rolling_rule_coverage_walkforward] combined:")
    print(f"  windows={len(report)}")
    print(f"  signals={total_signals}")
    print(f"  signals_per_day={total_signals / total_days if total_days > 0 else 0.0}")
    print(f"  win_rate={(total_wins / total_signals) if total_signals else None}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
        if not signals.empty:
            signals.to_csv(args.output.with_name(args.output.stem + "_signals.csv"), index=False)
        print(f"[rolling_rule_coverage_walkforward] saved: {args.output}")


if __name__ == "__main__":
    main()

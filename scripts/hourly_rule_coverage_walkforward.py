import argparse
import math
from pathlib import Path

import pandas as pd

from nested_rule_coverage_walkforward import _candidate_pool
from online_signal_filter_walkforward import _apply_condition, _load_rows


def _best_condition(train: pd.DataFrame, args: argparse.Namespace) -> dict | None:
    train_days = max((train["timestamp_dt"].max() - train["timestamp_dt"].min()).total_seconds() / 86_400, 1e-9)
    candidates = _candidate_pool(
        train,
        max_clauses=args.max_clauses,
        min_samples=args.min_samples,
        min_win_rate=args.min_win_rate,
        beam_size=args.beam_size,
        max_primitives=args.max_primitives,
    )
    viable = [
        item
        for item in candidates
        if item["fit_samples"] / train_days >= args.min_train_signals_per_day
        and item["fit_wilson_lower"] >= args.min_wilson_lower
    ]
    if not viable:
        return None
    return sorted(
        viable,
        key=lambda item: (
            item["fit_wilson_lower"],
            item["fit_win_rate"],
            item["fit_samples"],
            -item["condition"].count("&"),
            item["condition"],
        ),
        reverse=True,
    )[0]


def run(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    end = df["timestamp_dt"].max()
    start = max(
        df["timestamp_dt"].min() + pd.Timedelta(days=args.train_days),
        end - pd.Timedelta(days=args.latest_days),
    )
    current = start.floor("h")
    rows = []
    signal_frames = []
    while current < end:
        train_start = current - pd.Timedelta(days=args.train_days)
        cover_end = min(current + pd.Timedelta(minutes=args.cover_minutes), end)
        train = df[(df["timestamp_dt"] >= train_start) & (df["timestamp_dt"] < current)].copy().reset_index(drop=True)
        cover = df[(df["timestamp_dt"] >= current) & (df["timestamp_dt"] < cover_end)].copy().reset_index(drop=True)
        if train.empty or cover.empty:
            current += pd.Timedelta(minutes=args.step_minutes)
            continue
        item = _best_condition(train, args)
        condition = item["condition"] if item else ""
        if item:
            mask = _apply_condition(cover, condition)
        else:
            mask = pd.Series(False, index=cover.index)
        signals = int(mask.sum())
        wins = int(cover.loc[mask, "correct_bool"].sum()) if signals else 0
        cover_days = max((cover_end - current).total_seconds() / 86_400, 1e-9)
        if signals:
            selected = cover[mask].copy()
            selected["hourly_condition"] = condition
            selected["hourly_cover_start"] = current
            selected["hourly_cover_end"] = cover_end
            signal_frames.append(selected)
        rows.append(
            {
                "cover_start": current,
                "cover_end": cover_end,
                "condition": condition,
                "fit_samples": item.get("fit_samples", 0) if item else 0,
                "fit_win_rate": item.get("fit_win_rate", math.nan) if item else math.nan,
                "fit_wilson_lower": item.get("fit_wilson_lower", math.nan) if item else math.nan,
                "cover_signals": signals,
                "cover_wins": wins,
                "cover_win_rate": wins / signals if signals else math.nan,
                "cover_signals_per_day": signals / cover_days,
            }
        )
        current += pd.Timedelta(minutes=args.step_minutes)
    hourly = pd.DataFrame(rows)
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    if signals.empty:
        daily = pd.DataFrame(columns=["date", "signals", "wins", "win_rate"])
    else:
        signals["date"] = signals["timestamp_dt"].dt.date
        daily = signals.groupby("date").agg(signals=("correct_bool", "size"), wins=("correct_bool", "sum")).reset_index()
        daily["win_rate"] = daily["wins"] / daily["signals"]
    return hourly, signals, daily


def main() -> None:
    parser = argparse.ArgumentParser(description="Hourly rolling rule coverage validation.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--train-days", type=int, default=30)
    parser.add_argument("--latest-days", type=int, default=10)
    parser.add_argument("--cover-minutes", type=int, default=60)
    parser.add_argument("--step-minutes", type=int, default=60)
    parser.add_argument("--max-clauses", type=int, default=2)
    parser.add_argument("--min-samples", type=int, default=60)
    parser.add_argument("--min-win-rate", type=float, default=0.72)
    parser.add_argument("--min-wilson-lower", type=float, default=0.66)
    parser.add_argument("--min-train-signals-per-day", type=float, default=10.0)
    parser.add_argument("--beam-size", type=int, default=80)
    parser.add_argument("--max-primitives", type=int, default=120)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    df = _load_rows(args.csv)
    hourly, signals, daily = run(df, args)
    total = len(signals)
    wins = int(signals["correct_bool"].sum()) if total else 0
    days = len(daily)
    print(hourly.tail(48).to_string(index=False) if not hourly.empty else "no hourly windows")
    print("[hourly_rule_coverage_walkforward] daily:")
    print(daily.to_string(index=False) if not daily.empty else "no signals")
    print("[hourly_rule_coverage_walkforward] combined:")
    print(f"  hourly_windows={len(hourly)}")
    print(f"  active_days={days}")
    print(f"  signals={total}")
    print(f"  signals_per_active_day={total / days if days else 0.0}")
    print(f"  win_rate={wins / total if total else None}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        hourly.to_csv(args.output, index=False)
        if not signals.empty:
            signals.to_csv(args.output.with_name(args.output.stem + "_signals.csv"), index=False)
        daily.to_csv(args.output.with_name(args.output.stem + "_daily.csv"), index=False)
        print(f"[hourly_rule_coverage_walkforward] saved: {args.output}")


if __name__ == "__main__":
    main()

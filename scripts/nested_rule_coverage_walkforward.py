import argparse
from pathlib import Path

import math
import numpy as np
import pandas as pd

from online_signal_filter_walkforward import _apply_condition, _build_conditions, _load_rows, _wilson_lower_bound


def _candidate_pool(
    train: pd.DataFrame,
    max_clauses: int,
    min_samples: int,
    min_win_rate: float,
    beam_size: int,
    max_primitives: int,
):
    primitive = _build_conditions(train)
    correct = train["correct_bool"].to_numpy()
    pool = []
    beam = []

    if max_primitives > 0 and len(primitive) > max_primitives:
        ranked_primitive = []
        for idx, (name, mask) in enumerate(primitive):
            samples = int(mask.sum())
            if samples < min_samples:
                continue
            wins = int(correct[mask].sum())
            win_rate = wins / samples
            wilson = _wilson_lower_bound(wins, samples)
            ranked_primitive.append((wilson, win_rate, samples, idx, name, mask))
        ranked_primitive = sorted(ranked_primitive, reverse=True)[:max_primitives]
        primitive = [(name, mask) for _, _, _, _, name, mask in ranked_primitive]

    def score(names, mask):
        samples = int(mask.sum())
        if samples < min_samples:
            return None
        wins = int(correct[mask].sum())
        win_rate = wins / samples
        wilson = _wilson_lower_bound(wins, samples)
        item = {
            "condition": " & ".join(names),
            "fit_samples": samples,
            "fit_wins": wins,
            "fit_win_rate": win_rate,
            "fit_wilson_lower": wilson,
            "mask": mask,
        }
        if win_rate >= min_win_rate:
            pool.append(item)
        return item

    def key(item, clauses):
        return item["fit_wilson_lower"], item["fit_win_rate"], item["fit_samples"], -clauses

    for idx, (name, mask) in enumerate(primitive):
        item = score([name], mask)
        if item is not None and item["fit_win_rate"] >= min_win_rate - 0.15:
            beam.append(((idx,), [name], mask, item))
    beam = sorted(beam, key=lambda row: key(row[3], 1), reverse=True)[:beam_size]
    seen = {row[0] for row in beam}
    for clause_count in range(2, max_clauses + 1):
        next_beam = []
        for indices, names, mask, _ in beam:
            for idx in range(indices[-1] + 1, len(primitive)):
                combo = (*indices, idx)
                if combo in seen:
                    continue
                seen.add(combo)
                name, other = primitive[idx]
                item = score([*names, name], mask & other)
                if item is not None:
                    next_beam.append((combo, [*names, name], mask & other, item))
        if not next_beam:
            break
        beam = sorted(next_beam, key=lambda row: key(row[3], clause_count), reverse=True)[:beam_size]
    unique = {item["condition"]: item for item in pool}
    return list(unique.values())


def _evaluate(df: pd.DataFrame, condition: str, days: float, prefix: str) -> dict:
    mask = _apply_condition(df, condition)
    samples = int(mask.sum())
    wins = int(df.loc[mask, "correct_bool"].sum()) if samples else 0
    return {
        f"{prefix}_signals": samples,
        f"{prefix}_wins": wins,
        f"{prefix}_win_rate": wins / samples if samples else math.nan,
        f"{prefix}_signals_per_day": samples / max(days, 1e-9),
    }


def run(
    df: pd.DataFrame,
    train_days: int,
    select_days: int,
    cover_days: int,
    step_days: int,
    max_clauses: int,
    min_samples: int,
    fit_min_win_rate: float,
    select_min_signals_per_day: float,
    select_min_win_rate: float,
    beam_size: int,
    max_primitives: int,
    latest_windows: int,
):
    start = df["timestamp_dt"].min() + pd.Timedelta(days=train_days)
    end = df["timestamp_dt"].max()
    schedule = []
    current = start
    window = 1
    while current < end:
        schedule.append((window, current))
        current += pd.Timedelta(days=step_days)
        window += 1
    if latest_windows > 0:
        schedule = schedule[-latest_windows:]

    rows = []
    signal_frames = []
    for window, current in schedule:
        fit_start = current - pd.Timedelta(days=train_days)
        select_start = current - pd.Timedelta(days=select_days)
        cover_end = min(current + pd.Timedelta(days=cover_days), end)
        fit = df[(df["timestamp_dt"] >= fit_start) & (df["timestamp_dt"] < select_start)].copy().reset_index(drop=True)
        select = df[(df["timestamp_dt"] >= select_start) & (df["timestamp_dt"] < current)].copy().reset_index(drop=True)
        cover = df[(df["timestamp_dt"] >= current) & (df["timestamp_dt"] < cover_end)].copy().reset_index(drop=True)
        if fit.empty or select.empty or cover.empty:
            current += pd.Timedelta(days=step_days)
            window += 1
            continue
        candidates = _candidate_pool(fit, max_clauses, min_samples, fit_min_win_rate, beam_size, max_primitives)
        ranked = []
        for item in candidates:
            try:
                select_stats = _evaluate(select, item["condition"], select_days, "select")
            except Exception:
                continue
            if (
                select_stats["select_signals_per_day"] >= select_min_signals_per_day
                and select_stats["select_win_rate"] >= select_min_win_rate
            ):
                ranked.append((select_stats["select_win_rate"], select_stats["select_signals_per_day"], item, select_stats))
        if ranked:
            _, _, item, select_stats = sorted(
                ranked,
                key=lambda row: (
                    row[0],
                    row[1],
                    row[2]["fit_wilson_lower"],
                    row[2]["fit_win_rate"],
                    row[2]["fit_samples"],
                    row[2]["condition"],
                ),
                reverse=True,
            )[0]
            condition = item["condition"]
            cover_stats = _evaluate(cover, condition, (cover_end - current).total_seconds() / 86_400, "cover")
            if cover_stats["cover_signals"]:
                sig = cover[_apply_condition(cover, condition)].copy()
                sig["nested_window"] = window
                sig["nested_condition"] = condition
                signal_frames.append(sig)
            rows.append(
                {
                    "window": window,
                    "condition": condition,
                    "fit_start": fit_start,
                    "fit_end": select_start,
                    "select_start": select_start,
                    "select_end": current,
                    "cover_start": current,
                    "cover_end": cover_end,
                    **{k: v for k, v in item.items() if k != "mask"},
                    **select_stats,
                    **cover_stats,
                }
            )
        else:
            rows.append(
                {
                    "window": window,
                    "condition": "",
                    "fit_start": fit_start,
                    "fit_end": select_start,
                    "select_start": select_start,
                    "select_end": current,
                    "cover_start": current,
                    "cover_end": cover_end,
                    "cover_signals": 0,
                    "cover_wins": 0,
                    "cover_win_rate": np.nan,
                    "cover_signals_per_day": 0.0,
                }
            )
        if rows:
            last = rows[-1]
            print(
                "[nested_rule_coverage_walkforward] "
                f"window={window} cover={last.get('cover_start')}->{last.get('cover_end')} "
                f"signals={last.get('cover_signals')} win_rate={last.get('cover_win_rate')}",
                flush=True,
            )
    report = pd.DataFrame(rows)
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    return report, signals


def main():
    parser = argparse.ArgumentParser(description="Nested walk-forward rule selection.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--train-days", type=int, default=30)
    parser.add_argument("--select-days", type=int, default=7)
    parser.add_argument("--cover-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--max-clauses", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--fit-min-win-rate", type=float, default=0.70)
    parser.add_argument("--select-min-signals-per-day", type=float, default=10.0)
    parser.add_argument("--select-min-win-rate", type=float, default=0.75)
    parser.add_argument("--beam-size", type=int, default=160)
    parser.add_argument("--max-primitives", type=int, default=120)
    parser.add_argument("--latest-windows", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    df = _load_rows(args.csv)
    report, signals = run(
        df,
        args.train_days,
        args.select_days,
        args.cover_days,
        args.step_days,
        args.max_clauses,
        args.min_samples,
        args.fit_min_win_rate,
        args.select_min_signals_per_day,
        args.select_min_win_rate,
        args.beam_size,
        args.max_primitives,
        args.latest_windows,
    )
    total = len(signals)
    wins = int(signals["correct_bool"].sum()) if total else 0
    days = (
        (pd.to_datetime(report["cover_end"]).max() - pd.to_datetime(report["cover_start"]).min()).total_seconds()
        / 86_400
        if not report.empty
        else 0.0
    )
    print(report.to_string(index=False) if not report.empty else "no windows")
    print("[nested_rule_coverage_walkforward] combined:")
    print(f"  windows={len(report)}")
    print(f"  signals={total}")
    print(f"  signals_per_day={total / days if days > 0 else 0.0}")
    print(f"  win_rate={wins / total if total else None}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
        if not signals.empty:
            signals.to_csv(args.output.with_name(args.output.stem + "_signals.csv"), index=False)
        print(f"[nested_rule_coverage_walkforward] saved: {args.output}")


if __name__ == "__main__":
    main()

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from online_signal_filter_walkforward import (
    _apply_condition,
    _build_conditions,
    _load_rows,
    _wilson_lower_bound,
)


def _condition_candidates(
    train: pd.DataFrame,
    *,
    max_clauses: int,
    min_samples: int,
    min_win_rate: float,
    min_wilson_lower: float,
    beam_size: int,
    max_primitives: int,
) -> list[dict]:
    primitive = _build_conditions(train)
    correct = train["correct_bool"].to_numpy()
    candidates: list[dict] = []
    beam: list[tuple[tuple[int, ...], list[str], np.ndarray, dict]] = []

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

    def score(names: list[str], mask: np.ndarray) -> dict | None:
        samples = int(mask.sum())
        if samples < min_samples:
            return None
        wins = int(correct[mask].sum())
        win_rate = wins / samples
        wilson = _wilson_lower_bound(wins, samples)
        item = {
            "condition": " & ".join(names),
            "train_samples": samples,
            "train_wins": wins,
            "train_win_rate": win_rate,
            "train_wilson_lower": wilson,
            "mask": mask,
        }
        if win_rate >= min_win_rate and wilson >= min_wilson_lower:
            candidates.append(item)
        return item

    def sort_key(item: dict, clauses: int) -> tuple[float, float, int, int]:
        return (
            float(item["train_wilson_lower"]),
            float(item["train_win_rate"]),
            int(item["train_samples"]),
            -clauses,
        )

    for idx, (name, mask) in enumerate(primitive):
        item = score([name], mask)
        if item is not None and item["train_win_rate"] >= min_win_rate - 0.12:
            beam.append(((idx,), [name], mask, item))

    beam = sorted(beam, key=lambda row: sort_key(row[3], 1), reverse=True)[:beam_size]
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
                combo_mask = mask & other
                item = score([*names, name], combo_mask)
                if item is not None:
                    next_beam.append((combo, [*names, name], combo_mask, item))
        if not next_beam:
            break
        beam = sorted(next_beam, key=lambda row: sort_key(row[3], clause_count), reverse=True)[:beam_size]

    unique = {}
    for item in candidates:
        key = item["condition"]
        if key not in unique or sort_key(item, item["condition"].count("&") + 1) > sort_key(
            unique[key],
            unique[key]["condition"].count("&") + 1,
        ):
            unique[key] = item
    return sorted(
        unique.values(),
        key=lambda item: (item["train_wilson_lower"], item["train_win_rate"], item["train_samples"]),
        reverse=True,
    )


def _select_rule_set(
    train: pd.DataFrame,
    candidates: list[dict],
    *,
    min_added_samples: int,
    min_added_win_rate: float,
    min_added_wilson_lower: float,
    target_signals_per_day: float,
    max_rules: int,
) -> list[dict]:
    selected: list[dict] = []
    covered = np.zeros(len(train), dtype=bool)
    correct = train["correct_bool"].to_numpy()
    train_days = max((train["timestamp_dt"].max() - train["timestamp_dt"].min()).total_seconds() / 86_400, 1e-9)

    for _ in range(max_rules):
        best = None
        for item in candidates:
            if any(item["condition"] == row["condition"] for row in selected):
                continue
            added = item["mask"] & ~covered
            samples = int(added.sum())
            if samples < min_added_samples:
                continue
            wins = int(correct[added].sum())
            win_rate = wins / samples
            wilson = _wilson_lower_bound(wins, samples)
            if win_rate < min_added_win_rate or wilson < min_added_wilson_lower:
                continue
            key = (wilson, win_rate, samples)
            if best is None or key > best[0]:
                best = (
                    key,
                    {
                        "condition": item["condition"],
                        "added_train_samples": samples,
                        "added_train_wins": wins,
                        "added_train_win_rate": win_rate,
                        "added_train_wilson_lower": wilson,
                    },
                    added,
                )
        if best is None:
            break
        selected.append(best[1])
        covered |= best[2]
        if int(covered.sum()) / train_days >= target_signals_per_day:
            break
    return selected


def _union_mask(df: pd.DataFrame, conditions: list[str]) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for condition in conditions:
        mask |= _apply_condition(df, condition)
    return mask


def run(
    df: pd.DataFrame,
    *,
    train_days: int,
    cover_days: int,
    step_days: int,
    max_clauses: int,
    min_samples: int,
    min_win_rate: float,
    min_wilson_lower: float,
    min_added_samples: int,
    min_added_win_rate: float,
    min_added_wilson_lower: float,
    target_signals_per_day: float,
    max_rules: int,
    beam_size: int,
    max_primitives: int,
    latest_windows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = df["timestamp_dt"].min() + pd.Timedelta(days=train_days)
    end = df["timestamp_dt"].max()
    schedule = []
    current = start
    window_no = 1
    while current < end:
        schedule.append((window_no, current))
        current += pd.Timedelta(days=step_days)
        window_no += 1
    if latest_windows > 0:
        schedule = schedule[-latest_windows:]

    rows = []
    signal_frames = []

    for window_no, current in schedule:
        train_start = current - pd.Timedelta(days=train_days)
        cover_end = min(current + pd.Timedelta(days=cover_days), end)
        train = df[(df["timestamp_dt"] >= train_start) & (df["timestamp_dt"] < current)].copy().reset_index(drop=True)
        cover = df[(df["timestamp_dt"] >= current) & (df["timestamp_dt"] < cover_end)].copy().reset_index(drop=True)
        if train.empty or cover.empty:
            current += pd.Timedelta(days=step_days)
            window_no += 1
            continue

        candidates = _condition_candidates(
            train,
            max_clauses=max_clauses,
            min_samples=min_samples,
            min_win_rate=min_win_rate,
            min_wilson_lower=min_wilson_lower,
            beam_size=beam_size,
            max_primitives=max_primitives,
        )
        selected = _select_rule_set(
            train,
            candidates,
            min_added_samples=min_added_samples,
            min_added_win_rate=min_added_win_rate,
            min_added_wilson_lower=min_added_wilson_lower,
            target_signals_per_day=target_signals_per_day,
            max_rules=max_rules,
        )
        conditions = [item["condition"] for item in selected]
        train_mask = _union_mask(train, conditions) if conditions else pd.Series(False, index=train.index)
        cover_mask = _union_mask(cover, conditions) if conditions else pd.Series(False, index=cover.index)
        train_count = int(train_mask.sum())
        cover_count = int(cover_mask.sum())
        train_wins = int(train.loc[train_mask, "correct_bool"].sum()) if train_count else 0
        cover_wins = int(cover.loc[cover_mask, "correct_bool"].sum()) if cover_count else 0
        train_span_days = max((current - train_start).total_seconds() / 86_400, 1e-9)
        cover_span_days = max((cover_end - current).total_seconds() / 86_400, 1e-9)

        if cover_count:
            signals = cover[cover_mask].copy()
            signals["rolling_window"] = window_no
            signals["multi_rule_conditions"] = " || ".join(conditions)
            signal_frames.append(signals)

        rows.append(
            {
                "window": window_no,
                "rule_count": len(conditions),
                "conditions": " || ".join(conditions),
                "train_start": train_start,
                "train_end": current,
                "cover_start": current,
                "cover_end": cover_end,
                "train_signals": train_count,
                "train_wins": train_wins,
                "train_win_rate": train_wins / train_count if train_count else math.nan,
                "train_signals_per_day": train_count / train_span_days,
                "cover_signals": cover_count,
                "cover_wins": cover_wins,
                "cover_win_rate": cover_wins / cover_count if cover_count else math.nan,
                "cover_signals_per_day": cover_count / cover_span_days,
            }
        )
        print(
            "[multi_rule_coverage_walkforward] "
            f"window={window_no} cover={current}->{cover_end} "
            f"rules={len(conditions)} signals={cover_count} win_rate={cover_wins / cover_count if cover_count else math.nan}",
            flush=True,
        )

    report = pd.DataFrame(rows)
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    return report, signals


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward multi-rule coverage validation.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--extra-csv", type=Path, action="append", default=[])
    parser.add_argument("--train-days", type=int, default=30)
    parser.add_argument("--cover-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--max-clauses", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=60)
    parser.add_argument("--min-win-rate", type=float, default=0.75)
    parser.add_argument("--min-wilson-lower", type=float, default=0.68)
    parser.add_argument("--min-added-samples", type=int, default=20)
    parser.add_argument("--min-added-win-rate", type=float, default=0.75)
    parser.add_argument("--min-added-wilson-lower", type=float, default=0.62)
    parser.add_argument("--target-signals-per-day", type=float, default=10.0)
    parser.add_argument("--max-rules", type=int, default=5)
    parser.add_argument("--beam-size", type=int, default=120)
    parser.add_argument("--max-primitives", type=int, default=120)
    parser.add_argument("--latest-windows", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    frames = [_load_rows(args.csv)]
    frames.extend(_load_rows(path) for path in args.extra_csv)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp", "rule", "direction"], keep="last")
    df = df.sort_values("timestamp_dt").reset_index(drop=True)

    report, signals = run(
        df,
        train_days=args.train_days,
        cover_days=args.cover_days,
        step_days=args.step_days,
        max_clauses=args.max_clauses,
        min_samples=args.min_samples,
        min_win_rate=args.min_win_rate,
        min_wilson_lower=args.min_wilson_lower,
        min_added_samples=args.min_added_samples,
        min_added_win_rate=args.min_added_win_rate,
        min_added_wilson_lower=args.min_added_wilson_lower,
        target_signals_per_day=args.target_signals_per_day,
        max_rules=args.max_rules,
        beam_size=args.beam_size,
        max_primitives=args.max_primitives,
        latest_windows=args.latest_windows,
    )
    total = int(len(signals))
    wins = int(signals["correct_bool"].sum()) if total else 0
    days = (
        (pd.to_datetime(report["cover_end"]).max() - pd.to_datetime(report["cover_start"]).min()).total_seconds()
        / 86_400
        if not report.empty
        else 0.0
    )
    print(report.to_string(index=False) if not report.empty else "no windows")
    print("[multi_rule_coverage_walkforward] combined:")
    print(f"  windows={len(report)}")
    print(f"  signals={total}")
    print(f"  signals_per_day={total / days if days > 0 else 0.0}")
    print(f"  win_rate={wins / total if total else None}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
        if not signals.empty:
            signals.to_csv(args.output.with_name(args.output.stem + "_signals.csv"), index=False)
        print(f"[multi_rule_coverage_walkforward] saved: {args.output}")


if __name__ == "__main__":
    main()

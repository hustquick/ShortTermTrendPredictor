import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core.rolling_coverage_engine import load_candidate_rows
from scripts.online_signal_filter_walkforward import (
    _apply_condition,
    _build_conditions,
    _load_rows,
    _search_ranked_conditions,
    _wilson_lower_bound,
)


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _eval(df: pd.DataFrame, condition: str) -> dict:
    mask = _apply_condition(df, condition)
    samples = int(mask.sum())
    wins = int(df.loc[mask, "correct_bool"].sum()) if samples else 0
    return {
        "samples": samples,
        "wins": wins,
        "win_rate": wins / samples if samples else np.nan,
        "wilson": _wilson_lower_bound(wins, samples) if samples else 0.0,
    }


def _family_keywords(family: str) -> list[str]:
    if family == "trend":
        return ["direction=", "ret_", "ema_", "macd", "trend_agreement", "rsi_14"]
    if family == "reversal":
        return ["direction=", "rsi_14", "boll", "close_position", "upper_shadow", "lower_shadow", "body_ratio", "ret_"]
    if family == "orderflow":
        return ["direction=", "taker_buy_ratio", "volume_ratio", "trade_count", "quote_volume", "body_ratio", "close_position"]
    if family == "mtf":
        return ["direction=", "mtf_3m", "mtf_5m", "trend_agreement", "ret_30", "ema_20_60"]
    return []


def _search_ranked_family_conditions(
    train: pd.DataFrame,
    *,
    family: str,
    max_clauses: int,
    min_samples: int,
    min_signals_per_day: float,
    min_win_rate: float,
    min_wilson_lower: float,
    beam_size: int,
    limit: int,
) -> list[dict]:
    keywords = _family_keywords(family)
    if not keywords:
        return _search_ranked_conditions(
            train,
            max_clauses=max_clauses,
            min_samples=min_samples,
            min_signals_per_day=min_signals_per_day,
            min_win_rate=min_win_rate,
            min_wilson_lower=min_wilson_lower,
            beam_size=beam_size,
            limit=limit,
        )
    conditions = [
        item
        for item in _build_conditions(train)
        if any(keyword in item[0] for keyword in keywords)
    ]
    days = max((train["timestamp_dt"].max() - train["timestamp_dt"].min()).total_seconds() / 86_400, 1e-9)
    min_count = max(min_samples, int(np.ceil(days * min_signals_per_day)))
    correct = train["correct_bool"].to_numpy()
    ranked: dict[str, tuple[tuple[float, float, int, int], dict]] = {}
    beam = []

    def score(names: list[str], mask: np.ndarray) -> dict | None:
        samples = int(mask.sum())
        if samples < min_count:
            return None
        wins = int(correct[mask].sum())
        wr = wins / samples
        wilson = _wilson_lower_bound(wins, samples)
        return {
            "condition": " & ".join(names),
            "train_samples": samples,
            "train_wins": wins,
            "train_win_rate": wr,
            "train_wilson_lower": wilson,
            "train_signals_per_day": samples / days,
            "family": family,
        }

    def key(item: dict, clauses: int) -> tuple[float, float, int, int]:
        return (item["train_wilson_lower"], item["train_win_rate"], item["train_samples"], -clauses)

    def add(item: dict, clauses: int) -> None:
        if item["train_win_rate"] < min_win_rate or item["train_wilson_lower"] < min_wilson_lower:
            return
        condition = item["condition"]
        item_key = key(item, clauses)
        if condition not in ranked or item_key > ranked[condition][0]:
            ranked[condition] = (item_key, item)

    for idx, (name, mask) in enumerate(conditions):
        item = score([name], mask)
        if item is None:
            continue
        if item["train_win_rate"] >= min_win_rate - 0.10:
            beam.append(((idx,), [name], mask, item))
            add(item, 1)
    beam = sorted(beam, key=lambda row: key(row[3], len(row[0])), reverse=True)[:beam_size]
    seen = {row[0] for row in beam}
    for clauses in range(2, max_clauses + 1):
        next_beam = []
        for indices, names, mask, _ in beam:
            for idx in range(indices[-1] + 1, len(conditions)):
                combo = (*indices, idx)
                if combo in seen:
                    continue
                seen.add(combo)
                name, condition_mask = conditions[idx]
                item = score([*names, name], mask & condition_mask)
                if item is None:
                    continue
                next_beam.append((combo, [*names, name], mask & condition_mask, item))
                add(item, clauses)
        if not next_beam:
            break
        beam = sorted(next_beam, key=lambda row: key(row[3], len(row[0])), reverse=True)[:beam_size]
    return [item for _, item in sorted(ranked.values(), key=lambda row: row[0], reverse=True)[:limit]]


def _subwindow_stats(
    df: pd.DataFrame,
    condition: str,
    *,
    parts: int,
    min_samples: int,
    min_win_rate: float,
) -> dict:
    if df.empty:
        return {"passed": False, "subwindows": 0, "passed_subwindows": 0, "min_wr": np.nan, "min_samples": 0}
    start = df["timestamp_dt"].min()
    end = df["timestamp_dt"].max()
    span = (end - start) / max(parts, 1)
    rows = []
    for idx in range(parts):
        sub_start = start + span * idx
        sub_end = start + span * (idx + 1) if idx < parts - 1 else end + pd.Timedelta(minutes=1)
        sub = df[(df["timestamp_dt"] >= sub_start) & (df["timestamp_dt"] < sub_end)].copy()
        stats = _eval(sub, condition)
        if stats["samples"] > 0:
            rows.append(stats)
    if not rows:
        return {"passed": False, "subwindows": 0, "passed_subwindows": 0, "min_wr": np.nan, "min_samples": 0}
    passed = [
        row
        for row in rows
        if row["samples"] >= min_samples and row["win_rate"] >= min_win_rate
    ]
    return {
        "passed": len(passed) == len(rows) and len(rows) >= max(2, parts - 1),
        "subwindows": len(rows),
        "passed_subwindows": len(passed),
        "min_wr": min(row["win_rate"] for row in rows),
        "min_samples": min(row["samples"] for row in rows),
    }


def _discover_stable_rules(
    df: pd.DataFrame,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    select_start: pd.Timestamp,
    select_end: pd.Timestamp,
    train_days: int,
    max_clauses: int,
    min_train_samples: int,
    min_train_wr: float,
    min_train_wilson: float,
    train_subwindows: int,
    min_subwindow_samples: int,
    min_subwindow_wr: float,
    min_select_samples: int,
    min_select_wr: float,
    min_select_wilson: float,
    beam_size: int,
    candidate_limit: int,
    selected_limit: int,
    family: str,
) -> list[dict]:
    train = df[(df["timestamp_dt"] >= train_start) & (df["timestamp_dt"] < train_end)].copy()
    select = df[(df["timestamp_dt"] >= select_start) & (df["timestamp_dt"] < select_end)].copy()
    if train.empty or select.empty:
        return []
    train_span_days = max((train["timestamp_dt"].max() - train["timestamp_dt"].min()).total_seconds() / 86_400, 1e-9)
    ranked = _search_ranked_family_conditions(
        train,
        family=family,
        max_clauses=max_clauses,
        min_samples=min_train_samples,
        min_signals_per_day=min_train_samples / max(train_days, train_span_days),
        min_win_rate=min_train_wr,
        min_wilson_lower=min_train_wilson,
        beam_size=beam_size,
        limit=candidate_limit,
    )
    selected = []
    for item in ranked:
        condition = item["condition"]
        sub = _subwindow_stats(
            train,
            condition,
            parts=train_subwindows,
            min_samples=min_subwindow_samples,
            min_win_rate=min_subwindow_wr,
        )
        if not sub["passed"]:
            continue
        select_stats = _eval(select, condition)
        if select_stats["samples"] < min_select_samples:
            continue
        if select_stats["win_rate"] < min_select_wr:
            continue
        if select_stats["wilson"] < min_select_wilson:
            continue
        selected.append(
            {
                **item,
                "train_subwindows": sub["subwindows"],
                "train_passed_subwindows": sub["passed_subwindows"],
                "train_min_subwindow_win_rate": sub["min_wr"],
                "train_min_subwindow_samples": sub["min_samples"],
                "select_samples": select_stats["samples"],
                "select_wins": select_stats["wins"],
                "select_win_rate": select_stats["win_rate"],
                "select_wilson": select_stats["wilson"],
            }
        )
    return sorted(
        selected,
        key=lambda row: (
            row["train_min_subwindow_win_rate"],
            row["select_wilson"],
            row["select_win_rate"],
            row["train_wilson_lower"],
            row["train_samples"],
        ),
        reverse=True,
    )[:selected_limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable rule generator with train subwindow + recent select validation.")
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--cover-hours", type=int, default=6)
    parser.add_argument("--step-hours", type=int, default=6)
    parser.add_argument("--train-days", default="3,5,7")
    parser.add_argument("--select-hours", default="6,12,24")
    parser.add_argument("--max-clauses", type=int, default=3)
    parser.add_argument("--min-train-samples", type=int, default=40)
    parser.add_argument("--min-train-wr", type=float, default=0.70)
    parser.add_argument("--min-train-wilson", type=float, default=0.58)
    parser.add_argument("--train-subwindows", type=int, default=3)
    parser.add_argument("--min-subwindow-samples", type=int, default=8)
    parser.add_argument("--min-subwindow-wr", type=float, default=0.62)
    parser.add_argument("--min-select-samples", type=int, default=6)
    parser.add_argument("--min-select-wr", type=float, default=0.70)
    parser.add_argument("--min-select-wilson", type=float, default=0.35)
    parser.add_argument("--beam-size", type=int, default=120)
    parser.add_argument("--candidate-limit", type=int, default=120)
    parser.add_argument("--selected-limit", type=int, default=5)
    parser.add_argument("--families", default="trend,reversal,orderflow,mtf")
    parser.add_argument("--output", type=Path, default=Path("data/stable_rule_generator_walkforward.csv"))
    args = parser.parse_args()

    df = _load_rows(args.csv[0]) if len(args.csv) == 1 else load_candidate_rows(args.csv)
    if df.empty:
        raise RuntimeError(f"empty input: {args.csv}")
    end = df["timestamp_dt"].max()
    start = end - pd.Timedelta(days=args.lookback_days)
    train_days_values = _parse_ints(args.train_days)
    select_hours_values = _parse_ints(args.select_hours)
    families = [item.strip() for item in args.families.split(",") if item.strip()]

    rows = []
    signal_frames = []
    current = start
    while current < end:
        cover_start = current
        cover_end = min(current + pd.Timedelta(hours=args.cover_hours), end)
        cover = df[(df["timestamp_dt"] >= cover_start) & (df["timestamp_dt"] < cover_end)].copy()
        fold_selected = []
        for family in families:
            for train_days in train_days_values:
                for select_hours in select_hours_values:
                    select_start = cover_start - pd.Timedelta(hours=select_hours)
                    train_start = select_start - pd.Timedelta(days=train_days)
                    selected = _discover_stable_rules(
                        df,
                        train_start=train_start,
                        train_end=select_start,
                        select_start=select_start,
                        select_end=cover_start,
                        train_days=train_days,
                        max_clauses=args.max_clauses,
                        min_train_samples=args.min_train_samples,
                        min_train_wr=args.min_train_wr,
                        min_train_wilson=args.min_train_wilson,
                        train_subwindows=args.train_subwindows,
                        min_subwindow_samples=args.min_subwindow_samples,
                        min_subwindow_wr=args.min_subwindow_wr,
                        min_select_samples=args.min_select_samples,
                        min_select_wr=args.min_select_wr,
                        min_select_wilson=args.min_select_wilson,
                        beam_size=args.beam_size,
                        candidate_limit=args.candidate_limit,
                        selected_limit=args.selected_limit,
                        family=family,
                    )
                    for item in selected:
                        fold_selected.append(
                            {
                                **item,
                                "family": family,
                                "train_days": train_days,
                                "select_hours": select_hours,
                                "train_start": train_start,
                                "train_end": select_start,
                                "select_start": select_start,
                                "select_end": cover_start,
                            }
                        )
        seen = set()
        deduped = []
        for item in sorted(
            fold_selected,
            key=lambda row: (
                row["train_min_subwindow_win_rate"],
                row["select_wilson"],
                row["select_win_rate"],
                row["train_wilson_lower"],
            ),
            reverse=True,
        ):
            if item["condition"] in seen:
                continue
            seen.add(item["condition"])
            deduped.append(item)
            if len(deduped) >= args.selected_limit:
                break

        fold_mask = pd.Series(False, index=cover.index)
        for item in deduped:
            condition = item["condition"]
            stats = _eval(cover, condition)
            rows.append(
                {
                    **item,
                    "cover_start": cover_start,
                    "cover_end": cover_end,
                    "cover_condition": condition,
                    "cover_signals": stats["samples"],
                    "cover_wins": stats["wins"],
                    "cover_win_rate": stats["win_rate"],
                }
            )
            if stats["samples"]:
                fold_mask |= _apply_condition(cover, condition)
        if not deduped:
            rows.append({"cover_start": cover_start, "cover_end": cover_end, "cover_condition": "", "cover_signals": 0, "cover_wins": 0, "cover_win_rate": np.nan})
        signals = cover[fold_mask].copy()
        if not signals.empty:
            signals["cover_start"] = cover_start
            signals["cover_end"] = cover_end
            signals["stable_conditions"] = " || ".join(item["condition"] for item in deduped)
            signal_frames.append(signals)
        current += pd.Timedelta(hours=args.step_hours)

    report = pd.DataFrame(rows)
    all_signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    signals_path = args.output.with_name(args.output.stem + "_signals.csv")
    if not all_signals.empty:
        all_signals.to_csv(signals_path, index=False)

    days = max((end - start).total_seconds() / 86_400, 1e-9)
    total = len(all_signals)
    wins = int(all_signals["correct_bool"].sum()) if total else 0
    print("[stable_rule_generator] combined")
    print(f"  data_range={df['timestamp_dt'].min()} -> {df['timestamp_dt'].max()}")
    print(f"  validation_range={start} -> {end}")
    print(f"  folds={report['cover_start'].nunique() if not report.empty else 0}")
    print(f"  signals={total}")
    print(f"  wins={wins}")
    print(f"  win_rate={(wins / total) if total else None}")
    print(f"  signals_per_day={total / days}")
    if total:
        by_day = all_signals.assign(day=all_signals["timestamp_dt"].dt.date).groupby("day")["correct_bool"].agg(["count", "sum"])
        by_day["win_rate"] = by_day["sum"] / by_day["count"]
        print("[stable_rule_generator] by day")
        print(by_day.tail(20).to_string())
    print(f"[stable_rule_generator] saved: {args.output}")
    if not all_signals.empty:
        print(f"[stable_rule_generator] saved: {signals_path}")


if __name__ == "__main__":
    main()

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _reason_value(reason: str, key: str) -> str:
    match = re.search(fr"{key}=([^;]+)", str(reason))
    return match.group(1) if match else ""


def _reason_float(reason: str, key: str) -> float:
    value = _reason_value(reason, key)
    try:
        return float(value)
    except ValueError:
        return np.nan


def _session(hour: int) -> str:
    if 8 <= hour < 15:
        return "asia_day"
    if 15 <= hour < 21:
        return "europe_overlap"
    if hour >= 21 or hour < 1:
        return "us_open"
    return "late_us"


def _add_condition(conditions: list[tuple[str, np.ndarray]], name: str, mask: pd.Series):
    conditions.append((name, mask.fillna(False).to_numpy()))


def _prepare_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    reason = df["reason"].fillna("").astype(str)
    df["adaptive_rule"] = [_reason_value(item, "adaptive_rule") for item in reason]
    df["adaptive_mode"] = [_reason_value(item, "adaptive_mode") for item in reason]
    df["state_ok"] = [_reason_value(item, "state_ok") for item in reason]
    for key in [
        "raw_up_probability",
        "state_rsi_14",
        "state_ret_30",
        "state_close_position",
        "volume_ratio_10",
        "quote_volume_ratio_10",
        "trade_count_ratio_10",
        "volume_zscore",
        "volume_change",
    ]:
        df[key] = [_reason_float(item, key) for item in reason]
    timestamp = pd.to_datetime(df["timestamp"])
    df["session"] = timestamp.dt.hour.map(_session)
    df["timestamp_dt"] = timestamp
    return df


def _build_conditions(df: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    conditions: list[tuple[str, np.ndarray]] = []
    for rule in sorted(value for value in df["adaptive_rule"].dropna().unique() if value):
        _add_condition(conditions, f"rule={rule}", df["adaptive_rule"].eq(rule))
    for session in sorted(df["session"].dropna().unique()):
        _add_condition(conditions, f"session={session}", df["session"].eq(session))
    for threshold in [0.10, 0.15, 0.20, 0.25, 0.285, 0.35, 0.45, 0.50]:
        _add_condition(conditions, f"pup<={threshold}", df["raw_up_probability"] <= threshold)
        _add_condition(conditions, f"pup>{threshold}", df["raw_up_probability"] > threshold)
    for threshold in [35, 40, 45, 47.5, 50, 55, 60, 65]:
        _add_condition(conditions, f"rsi>{threshold}", df["state_rsi_14"] > threshold)
        _add_condition(conditions, f"rsi<={threshold}", df["state_rsi_14"] <= threshold)
    for threshold in [-0.002, -0.001, -0.0005, -0.00013, 0, 0.0005, 0.001, 0.002]:
        _add_condition(conditions, f"ret30>{threshold}", df["state_ret_30"] > threshold)
        _add_condition(conditions, f"ret30<={threshold}", df["state_ret_30"] <= threshold)
    for threshold in [0.20, 0.50, 0.80, 0.95]:
        _add_condition(conditions, f"closepos>{threshold}", df["state_close_position"] > threshold)
        _add_condition(conditions, f"closepos<={threshold}", df["state_close_position"] <= threshold)
    for threshold in [0.80, 1.00, 1.20, 1.50, 2.00]:
        _add_condition(conditions, f"volratio<={threshold}", df["volume_ratio_10"] <= threshold)
        _add_condition(conditions, f"volratio>{threshold}", df["volume_ratio_10"] > threshold)
    return conditions


def main():
    parser = argparse.ArgumentParser(
        description="Search rule switching conditions over strict walk-forward output rows."
    )
    parser.add_argument("csv", type=Path)
    parser.add_argument("--direction", choices=["up", "down"], default="down")
    parser.add_argument("--max-clauses", type=int, default=4)
    parser.add_argument("--min-signals-per-day", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    df = _prepare_frame(args.csv)
    if df.empty:
        raise RuntimeError(f"empty csv: {args.csv}")

    base = df[df["final_direction"].eq(args.direction)].copy()
    if base.empty:
        raise RuntimeError(f"no {args.direction} rows in {args.csv}")

    days = (base["timestamp_dt"].max() - base["timestamp_dt"].min()).total_seconds() / 86_400
    min_count = int(days * args.min_signals_per_day)
    correct = base["raw_is_correct"].astype(bool).to_numpy()
    conditions = _build_conditions(base)

    rows = []
    for clause_count in range(1, args.max_clauses + 1):
        for combo in itertools.combinations(range(len(conditions)), clause_count):
            mask = np.ones(len(base), dtype=bool)
            names = []
            for idx in combo:
                name, condition = conditions[idx]
                names.append(name)
                mask &= condition
            signals = int(mask.sum())
            if signals < min_count:
                continue
            win_rate = float(correct[mask].mean())
            rows.append(
                {
                    "win_rate": win_rate,
                    "signals": signals,
                    "signals_per_day": signals / days,
                    "condition": " & ".join(names),
                }
            )

    report = pd.DataFrame(rows).sort_values(
        ["win_rate", "signals"],
        ascending=[False, False],
    )
    if report.empty:
        print("[strict_output_rule_condition_search] no condition met density")
    else:
        print(report.head(args.top).to_string(index=False))
    if args.output:
        report.head(args.top).to_csv(args.output, index=False)
        print(f"[strict_output_rule_condition_search] saved: {args.output}")


if __name__ == "__main__":
    main()

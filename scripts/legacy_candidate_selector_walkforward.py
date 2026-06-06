import argparse
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_bool(value) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _load_candidate_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "rule", "direction", "actual_direction"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"missing columns: {sorted(missing)}")
    df = df[df["direction"].isin(["up", "down"])].copy()
    df = df[df["actual_direction"].isin(["up", "down"])].copy()
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp_dt"].notna()].copy()
    if "correct" in df:
        df["raw_correct_bool"] = df["correct"].map(_parse_bool)
    elif "is_correct" in df:
        df["raw_correct_bool"] = df["is_correct"].map(_parse_bool)
    else:
        df["raw_correct_bool"] = df["direction"].eq(df["actual_direction"])
    return df.sort_values(["timestamp_dt", "rule", "direction"]).reset_index(drop=True)


def _stats(history: deque[bool]) -> tuple[int, int, float]:
    samples = len(history)
    wins = sum(1 for item in history if item)
    return samples, wins, wins / samples if samples else 0.0


def _feature(row: pd.Series, column: str, default: float = 0.0) -> float:
    if column not in row:
        return default
    value = pd.to_numeric(row[column], errors="coerce")
    return default if pd.isna(value) else float(value)


def _strong_up_state(row: pd.Series) -> bool:
    return (
        _feature(row, "ret_10") > 0.003
        and _feature(row, "ret_30") > 0.004
        and _feature(row, "rsi_14", 50.0) > 58.0
        and _feature(row, "trend_agreement") > 0.0
        and _feature(row, "mtf_3m_ret_3") > 0.003
        and _feature(row, "mtf_5m_ret_3") > 0.003
    )


def _strong_down_state(row: pd.Series) -> bool:
    return (
        _feature(row, "ret_10") < -0.003
        and _feature(row, "ret_30") < -0.004
        and _feature(row, "rsi_14", 50.0) < 42.0
        and _feature(row, "trend_agreement") < 0.0
        and _feature(row, "mtf_3m_ret_3") < -0.003
        and _feature(row, "mtf_5m_ret_3") < -0.003
    )


def _state_veto(row: pd.Series, direction: str, *, enabled: bool) -> bool:
    if not enabled:
        return False
    if direction == "down" and _strong_up_state(row):
        return True
    if direction == "up" and _strong_down_state(row):
        return True
    return False


def _choose_candidate(
    group: pd.DataFrame,
    histories: dict[tuple[str, str], deque[bool]],
    *,
    min_samples: int,
    min_win_rate: float,
    allow_inverse: bool,
    inverse_max_win_rate: float,
    state_veto: bool,
) -> dict | None:
    candidates = []
    for _, row in group.iterrows():
        rule = str(row["rule"])
        direction = str(row["direction"])
        samples, wins, win_rate = _stats(histories[(rule, direction)])
        if samples < min_samples:
            continue
        effective_direction = direction
        inverted = False
        effective_win_rate = win_rate
        if win_rate >= min_win_rate:
            pass
        elif allow_inverse and win_rate <= inverse_max_win_rate:
            effective_direction = "down" if direction == "up" else "up"
            inverted = True
            effective_win_rate = 1.0 - win_rate
        else:
            continue
        if _state_veto(row, effective_direction, enabled=state_veto):
            continue
        candidates.append(
            {
                "timestamp": row["timestamp"],
                "timestamp_dt": row["timestamp_dt"],
                "rule": rule,
                "raw_direction": direction,
                "direction": effective_direction,
                "actual_direction": str(row["actual_direction"]),
                "correct_bool": effective_direction == str(row["actual_direction"]),
                "raw_correct_bool": bool(row["raw_correct_bool"]),
                "inverted": inverted,
                "prior_samples": samples,
                "prior_wins": wins,
                "prior_win_rate": win_rate,
                "effective_win_rate": effective_win_rate,
                "confidence": _feature(row, "confidence", 0.0),
                "strong_up_state": _strong_up_state(row),
                "strong_down_state": _strong_down_state(row),
            }
        )
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            item["effective_win_rate"],
            item["prior_samples"],
            item["confidence"],
        ),
        reverse=True,
    )[0]


def run(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    histories: dict[tuple[str, str], deque[bool]] = defaultdict(lambda: deque(maxlen=args.history))
    signals = []
    report_rows = []
    start = df["timestamp_dt"].max() - pd.Timedelta(days=args.lookback_days)

    for timestamp, group in df.groupby("timestamp_dt", sort=True):
        if timestamp >= start:
            chosen = _choose_candidate(
                group,
                histories,
                min_samples=args.min_samples,
                min_win_rate=args.min_win_rate,
                allow_inverse=args.allow_inverse,
                inverse_max_win_rate=args.inverse_max_win_rate,
                state_veto=args.state_veto,
            )
            if chosen is not None:
                signals.append(chosen)

        for _, row in group.iterrows():
            histories[(str(row["rule"]), str(row["direction"]))].append(bool(row["raw_correct_bool"]))

        if timestamp >= start and timestamp.floor("6h") == timestamp:
            report_rows.append({"timestamp": timestamp, "known_rule_directions": len(histories)})

    return pd.DataFrame(signals), pd.DataFrame(report_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict chronological selector validation for legacy candidate rows.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--min-win-rate", type=float, default=0.80)
    parser.add_argument("--allow-inverse", action="store_true")
    parser.add_argument("--inverse-max-win-rate", type=float, default=0.20)
    parser.add_argument("--state-veto", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/legacy_candidate_selector_walkforward.csv"))
    args = parser.parse_args()

    df = _load_candidate_rows(args.csv)
    signals, report = run(df, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(args.output.with_name(args.output.stem + "_signals.csv"), index=False)
    report.to_csv(args.output, index=False)

    start = df["timestamp_dt"].max() - pd.Timedelta(days=args.lookback_days)
    days = max((df["timestamp_dt"].max() - start).total_seconds() / 86_400, 1e-9)
    total = len(signals)
    wins = int(signals["correct_bool"].sum()) if total else 0
    print("[legacy_candidate_selector_walkforward] combined")
    print(f"  data_range={df['timestamp_dt'].min()} -> {df['timestamp_dt'].max()}")
    print(f"  validation_range={start} -> {df['timestamp_dt'].max()}")
    print(f"  signals={total}")
    print(f"  wins={wins}")
    print(f"  win_rate={(wins / total) if total else None}")
    print(f"  signals_per_day={total / days}")
    if total:
        by_day = signals.assign(day=signals["timestamp_dt"].dt.date).groupby("day")["correct_bool"].agg(["count", "sum"])
        by_day["win_rate"] = by_day["sum"] / by_day["count"]
        print("[legacy_candidate_selector_walkforward] by day")
        print(by_day.tail(20).to_string())
        print("[legacy_candidate_selector_walkforward] by rule")
        by_rule = signals.groupby(["rule", "direction", "inverted"])["correct_bool"].agg(["count", "sum", "mean"])
        print(by_rule.sort_values(["mean", "count"], ascending=[False, False]).head(30).to_string())
    print(f"[legacy_candidate_selector_walkforward] saved: {args.output}")
    print(f"[legacy_candidate_selector_walkforward] saved: {args.output.with_name(args.output.stem + '_signals.csv')}")


if __name__ == "__main__":
    main()

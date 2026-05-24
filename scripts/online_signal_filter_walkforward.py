import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _reason_value(reason: str, key: str) -> str:
    prefix = f"{key}="
    for part in str(reason).split(";"):
        if part.startswith(prefix):
            return part[len(prefix):]
    return ""


def _wilson_lower_bound(wins: int, samples: int, z: float = 1.96) -> float:
    if samples <= 0:
        return 0.0
    p = wins / samples
    denominator = 1 + z * z / samples
    centre = p + z * z / (2 * samples)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * samples)) / samples)
    return max(0.0, (centre - margin) / denominator)


def _session(hour: int) -> str:
    if 8 <= hour < 15:
        return "asia_day"
    if 15 <= hour < 21:
        return "europe_overlap"
    if hour >= 21 or hour < 1:
        return "us_open"
    return "late_us"


def _load_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "timestamp" not in df:
        raise RuntimeError("input must include timestamp column")
    if "direction" not in df:
        if "final_direction" in df:
            df["direction"] = df["final_direction"]
        elif "raw_predicted_direction" in df:
            df["direction"] = df["raw_predicted_direction"]
        elif "predicted_direction" in df:
            df["direction"] = df["predicted_direction"]
        else:
            raise RuntimeError(
                "input must include direction, final_direction, raw_predicted_direction, or predicted_direction column"
            )
    if "correct" not in df:
        if "raw_is_correct" in df:
            df["correct"] = df["raw_is_correct"]
        elif "filtered_is_correct" in df:
            df["correct"] = df["filtered_is_correct"]
        elif "is_correct" in df:
            df["correct"] = df["is_correct"]
        else:
            raise RuntimeError("input must include correct, raw_is_correct, filtered_is_correct, or is_correct column")
    df = df[df["direction"].isin(["up", "down"])].copy()
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp_dt"].notna()].copy()
    df["session"] = df["timestamp_dt"].dt.hour.map(_session)
    df["correct_bool"] = df["correct"].astype(str).str.lower().eq("true")
    if "reason" in df:
        reason = df["reason"].fillna("").astype(str)
        if "rule" not in df:
            df["rule"] = [_reason_value(item, "adaptive_rule") for item in reason]
        df["adaptive_context"] = [_reason_value(item, "adaptive_context") for item in reason]
    return df.sort_values("timestamp_dt").reset_index(drop=True)


def _add_condition(conditions: list[tuple[str, np.ndarray]], name: str, mask) -> None:
    arr = np.asarray(mask.fillna(False) if hasattr(mask, "fillna") else mask, dtype=bool)
    if 0 < arr.sum() < len(arr):
        conditions.append((name, arr))


def _build_conditions(df: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    conditions: list[tuple[str, np.ndarray]] = []
    for column in ["rule", "direction", "session"]:
        if column not in df:
            continue
        for value in sorted(v for v in df[column].dropna().unique() if str(v)):
            _add_condition(conditions, f"{column}={value}", df[column].eq(value))
    if "adaptive_context" in df:
        context_tokens = sorted({
            token
            for value in df["adaptive_context"].fillna("").astype(str)
            for token in value.split("|")
            if token
        })
        context_parts = df["adaptive_context"].fillna("").astype(str).str.split("|")
        for token in context_tokens:
            _add_condition(conditions, f"context={token}", context_parts.apply(lambda parts: token in parts))

    thresholds = {
        "up_probability": [0.10, 0.20, 0.35, 0.45, 0.55, 0.70, 0.85, 0.95, 0.98, 0.99],
        "confidence": [0.55, 0.70, 0.85, 0.90, 0.95, 0.98, 0.99],
        "ret_5": [-0.002, -0.001, -0.0005, 0.0, 0.0005, 0.001, 0.002],
        "ret_10": [-0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003],
        "ret_30": [-0.006, -0.004, -0.002, -0.001, -0.0005, 0.0, 0.001, 0.002],
        "ema_10_30_diff": [-0.002, -0.001, -0.0005, 0.0, 0.0005, 0.001],
        "ema_20_60_diff": [-0.003, -0.002, -0.001, -0.0005, 0.0, 0.0005, 0.001],
        "macd_hist": [-20, -10, -5, 0, 5, 10, 20],
        "rsi_14": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
        "close_position": [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95],
        "body_ratio": [0.10, 0.20, 0.30, 0.50, 0.70, 0.90],
        "upper_shadow_ratio": [0.05, 0.10, 0.20, 0.30, 0.50, 0.70],
        "lower_shadow_ratio": [0.05, 0.10, 0.20, 0.30, 0.50, 0.70],
        "taker_buy_ratio": [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70],
        "trend_agreement": [-1.0, -0.333333, 0.0, 0.333333, 1.0],
    }
    for column, values in thresholds.items():
        if column not in df:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        for value in values:
            _add_condition(conditions, f"{column}<={value}", series <= value)
            _add_condition(conditions, f"{column}>{value}", series > value)
    return conditions


def _search_best_condition(
    train: pd.DataFrame,
    max_clauses: int,
    min_samples: int,
    min_signals_per_day: float,
    min_win_rate: float,
    min_wilson_lower: float,
    beam_size: int,
) -> dict | None:
    days = max((train["timestamp_dt"].max() - train["timestamp_dt"].min()).total_seconds() / 86_400, 1e-9)
    min_count = max(min_samples, math.ceil(days * min_signals_per_day))
    if len(train) < min_count:
        return None
    conditions = _build_conditions(train)
    correct = train["correct_bool"].to_numpy()
    best: tuple[tuple[float, float, int, int], dict] | None = None
    beam: list[tuple[tuple[int, ...], list[str], np.ndarray, dict]] = []

    def score_mask(names: list[str], mask: np.ndarray) -> dict | None:
        samples = int(mask.sum())
        if samples < min_count:
            return None
        wins = int(correct[mask].sum())
        win_rate = wins / samples
        wilson = _wilson_lower_bound(wins, samples)
        return {
            "condition": " & ".join(names),
            "train_samples": samples,
            "train_wins": wins,
            "train_win_rate": win_rate,
            "train_wilson_lower": wilson,
            "train_signals_per_day": samples / days,
        }

    def sort_key(candidate: dict, clause_count: int) -> tuple[float, float, int, int]:
        return (
            candidate["train_wilson_lower"],
            candidate["train_win_rate"],
            candidate["train_samples"],
            -clause_count,
        )

    for idx, (name, mask) in enumerate(conditions):
        count = int(mask.sum())
        if count >= min_count:
            wins = int(correct[mask].sum())
            win_rate = wins / count
            if win_rate >= min_win_rate - 0.10:
                candidate = score_mask([name], mask)
                if candidate is None:
                    continue
                beam.append(((idx,), [name], mask, candidate))
                if (
                    candidate["train_win_rate"] >= min_win_rate
                    and candidate["train_wilson_lower"] >= min_wilson_lower
                ):
                    key = sort_key(candidate, 1)
                    if best is None or key > best[0]:
                        best = (key, candidate)

    beam = sorted(
        beam,
        key=lambda item: sort_key(item[3], len(item[0])),
        reverse=True,
    )[:beam_size]
    seen = {item[0] for item in beam}

    for clause_count in range(2, max_clauses + 1):
        next_beam: list[tuple[tuple[int, ...], list[str], np.ndarray, dict]] = []
        for indices, names, mask, _ in beam:
            for idx in range(indices[-1] + 1, len(conditions)):
                combo_indices = (*indices, idx)
                if combo_indices in seen:
                    continue
                seen.add(combo_indices)
                name, condition_mask = conditions[idx]
                combo_mask = mask & condition_mask
                candidate = score_mask([*names, name], combo_mask)
                if candidate is None:
                    continue
                next_beam.append((combo_indices, [*names, name], combo_mask, candidate))
                if (
                    candidate["train_win_rate"] < min_win_rate
                    or candidate["train_wilson_lower"] < min_wilson_lower
                ):
                    continue
                key = sort_key(candidate, clause_count)
                if best is None or key > best[0]:
                    best = (key, candidate)
        if not next_beam:
            break
        beam = sorted(
            next_beam,
            key=lambda item: sort_key(item[3], len(item[0])),
            reverse=True,
        )[:beam_size]
    return None if best is None else best[1]


def _apply_condition(df: pd.DataFrame, condition: str) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for part in condition.split(" & "):
        if not part:
            continue
        if part.startswith("context="):
            token = part.split("=", 1)[1]
            context_parts = df["adaptive_context"].fillna("").astype(str).str.split("|")
            mask &= context_parts.apply(lambda parts: token in parts)
        elif "=" in part and "<=" not in part and ">" not in part:
            column, value = part.split("=", 1)
            mask &= df[column].astype(str).eq(value)
        elif "<=" in part:
            column, value = part.split("<=", 1)
            mask &= pd.to_numeric(df[column], errors="coerce") <= float(value)
        elif ">" in part:
            column, value = part.split(">", 1)
            mask &= pd.to_numeric(df[column], errors="coerce") > float(value)
        else:
            raise RuntimeError(f"unsupported condition: {part}")
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward online rediscovery over candidate signal rows.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--train-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--max-clauses", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=60)
    parser.add_argument("--min-signals-per-day", type=float, default=5.0)
    parser.add_argument("--min-win-rate", type=float, default=0.75)
    parser.add_argument("--min-wilson-lower", type=float, default=0.68)
    parser.add_argument("--beam-size", type=int, default=80)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    df = _load_rows(args.csv)
    if df.empty:
        raise RuntimeError(f"empty csv: {args.csv}")

    start = df["timestamp_dt"].min() + pd.Timedelta(days=args.train_days)
    end = df["timestamp_dt"].max()
    rows = []
    test_signal_frames = []
    current = start
    while current < end:
        train_start = current - pd.Timedelta(days=args.train_days)
        test_end = min(current + pd.Timedelta(days=args.test_days), end)
        train = df[(df["timestamp_dt"] >= train_start) & (df["timestamp_dt"] < current)].copy()
        test = df[(df["timestamp_dt"] >= current) & (df["timestamp_dt"] < test_end)].copy()
        if train.empty or test.empty:
            current += pd.Timedelta(days=args.step_days)
            continue
        selected = _search_best_condition(
            train,
            max_clauses=args.max_clauses,
            min_samples=args.min_samples,
            min_signals_per_day=args.min_signals_per_day,
            min_win_rate=args.min_win_rate,
            min_wilson_lower=args.min_wilson_lower,
            beam_size=args.beam_size,
        )
        if selected is None:
            rows.append({
                "train_start": train_start,
                "test_start": current,
                "test_end": test_end,
                "condition": "",
                "test_signals": 0,
                "test_win_rate": np.nan,
                "test_signals_per_day": 0.0,
            })
            current += pd.Timedelta(days=args.step_days)
            continue
        test_mask = _apply_condition(test, selected["condition"])
        test_signals = test[test_mask].copy()
        test_days = max((test_end - current).total_seconds() / 86_400, 1e-9)
        if not test_signals.empty:
            test_signals["online_condition"] = selected["condition"]
            test_signal_frames.append(test_signals)
        test_wins = int(test_signals["correct_bool"].sum()) if not test_signals.empty else 0
        test_count = int(len(test_signals))
        rows.append({
            **selected,
            "train_start": train_start,
            "test_start": current,
            "test_end": test_end,
            "test_signals": test_count,
            "test_wins": test_wins,
            "test_win_rate": test_wins / test_count if test_count else np.nan,
            "test_signals_per_day": test_count / test_days,
        })
        current += pd.Timedelta(days=args.step_days)

    report = pd.DataFrame(rows)
    if report.empty:
        print("no folds")
        return
    print(report.to_string(index=False))
    all_signals = pd.concat(test_signal_frames, ignore_index=True) if test_signal_frames else pd.DataFrame()
    total_signals = int(len(all_signals))
    total_wins = int(all_signals["correct_bool"].sum()) if total_signals else 0
    total_days = max((end - start).total_seconds() / 86_400, 1e-9)
    print("[online_signal_filter_walkforward] combined:")
    print(f"  folds={len(report)}")
    print(f"  signals={total_signals}")
    print(f"  signals_per_day={total_signals / total_days}")
    print(f"  win_rate={(total_wins / total_signals) if total_signals else None}")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output, index=False)
        if not all_signals.empty:
            all_signals.to_csv(output.with_name(output.stem + "_signals.csv"), index=False)
        print(f"[online_signal_filter_walkforward] saved: {output}")


if __name__ == "__main__":
    main()

import argparse
import itertools
import math
from pathlib import Path

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


def _shadow_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        reason = row.get("reason", "")
        context = _reason_value(reason, "adaptive_context")
        regime = _reason_value(reason, "adaptive_regime")
        session = _reason_value(reason, "adaptive_session")
        timestamp = row.get("timestamp")
        actual = str(row.get("actual_direction", ""))
        for item in _reason_value(reason, "shadow_candidates").split(","):
            if not item:
                continue
            parts = item.split(":")
            if len(parts) != 3:
                continue
            rule, direction, state_ok_text = parts
            if direction not in {"up", "down"}:
                continue
            tokens = [f"direction={direction}", f"regime={regime}", f"session={session}"]
            tokens.extend(f"ctx={token}" for token in context.split("|") if token)
            rows.append({
                "timestamp": timestamp,
                "rule": rule,
                "direction": direction,
                "state_ok": state_ok_text == "1",
                "correct": direction == actual,
                "regime": regime,
                "session": session,
                "context": context,
                "tokens": tuple(sorted(set(tokens))),
            })
    return pd.DataFrame(rows)


def _condition_summary(rows: pd.DataFrame, max_clauses: int, min_samples: int) -> pd.DataFrame:
    summaries = []
    for rule, rule_df in rows[rows["state_ok"]].groupby("rule"):
        token_counts = {}
        for tokens in rule_df["tokens"]:
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        useful_tokens = [
            token
            for token, count in token_counts.items()
            if count >= min_samples
        ]
        for clause_count in range(1, min(max_clauses, len(useful_tokens)) + 1):
            for combo in itertools.combinations(useful_tokens, clause_count):
                combo_set = set(combo)
                mask = rule_df["tokens"].apply(lambda tokens: combo_set.issubset(tokens))
                matched = rule_df[mask]
                samples = int(len(matched))
                if samples < min_samples:
                    continue
                wins = int(matched["correct"].sum())
                win_rate = wins / samples
                summaries.append({
                    "rule": rule,
                    "condition": ",".join(combo),
                    "clauses": clause_count,
                    "samples": samples,
                    "wins": wins,
                    "win_rate": win_rate,
                    "wilson_lower": _wilson_lower_bound(wins, samples),
                })
    return pd.DataFrame(summaries)


def _python_tuple(condition: str) -> str:
    tokens = [token for token in condition.split(",") if token]
    return "(" + ", ".join(repr(token) for token in tokens) + ",)"


def _explicit_filter_recommendations(
    conditions: pd.DataFrame,
    min_samples: int,
    min_win_rate: float,
    min_wilson_lower: float,
) -> pd.DataFrame:
    if conditions.empty:
        return pd.DataFrame()
    candidates = conditions[
        (conditions["samples"] >= min_samples)
        & (conditions["win_rate"] >= min_win_rate)
        & (conditions["wilson_lower"] >= min_wilson_lower)
    ].copy()
    if candidates.empty:
        return candidates
    candidates = candidates[
        ~candidates["condition"].str.contains("direction=", regex=False)
    ].copy()
    if candidates.empty:
        return candidates
    candidates["filter_tuple"] = candidates["condition"].map(_python_tuple)
    candidates = candidates.sort_values(
        ["wilson_lower", "win_rate", "samples", "clauses"],
        ascending=[False, False, False, True],
    )
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Analyze adaptive rule shadow candidate outcomes.")
    parser.add_argument("paths", nargs="+", help="CSV files or glob patterns from adaptive_rule_multifold_backtest.")
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--min-win-rate", type=float, default=0.75)
    parser.add_argument("--min-wilson-lower", type=float, default=0.68)
    parser.add_argument("--max-clauses", type=int, default=3)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", default=None)
    parser.add_argument("--explicit-filter-output", default=None)
    parser.add_argument("--explicit-filter-min-samples", type=int, default=None)
    parser.add_argument("--explicit-filter-min-win-rate", type=float, default=None)
    parser.add_argument("--explicit-filter-min-wilson-lower", type=float, default=None)
    args = parser.parse_args()

    paths = []
    for pattern in args.paths:
        matched = sorted(Path().glob(pattern))
        if matched:
            paths.extend(matched)
        else:
            paths.append(Path(pattern))
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        raise RuntimeError("no input rows found")

    df = pd.concat(frames, ignore_index=True)
    shadows = _shadow_rows(df)
    if shadows.empty:
        print("no shadow_candidates found; rerun backtest after telemetry is enabled")
        return

    by_rule = shadows[shadows["state_ok"]].groupby(["rule", "direction"]).agg(
        samples=("correct", "size"),
        wins=("correct", "sum"),
        win_rate=("correct", "mean"),
    ).reset_index()
    by_rule["wilson_lower"] = [
        _wilson_lower_bound(int(wins), int(samples))
        for wins, samples in zip(by_rule["wins"], by_rule["samples"])
    ]
    by_rule = by_rule.sort_values(["wilson_lower", "win_rate", "samples"], ascending=[False, False, False])

    explicit_filter_min_samples = args.explicit_filter_min_samples or args.min_samples
    explicit_filter_min_win_rate = args.explicit_filter_min_win_rate or args.min_win_rate
    explicit_filter_min_wilson_lower = args.explicit_filter_min_wilson_lower or args.min_wilson_lower
    condition_min_samples = min(args.min_samples, explicit_filter_min_samples)

    conditions = _condition_summary(shadows, args.max_clauses, condition_min_samples)
    if not conditions.empty:
        conditions = conditions.sort_values(
            ["wilson_lower", "win_rate", "samples"],
            ascending=[False, False, False],
        )
    accepted = conditions[
        (conditions["samples"] >= args.min_samples)
        & (conditions["win_rate"] >= args.min_win_rate)
        & (conditions["wilson_lower"] >= args.min_wilson_lower)
    ] if not conditions.empty else pd.DataFrame()
    explicit_filters = _explicit_filter_recommendations(
        conditions,
        min_samples=explicit_filter_min_samples,
        min_win_rate=explicit_filter_min_win_rate,
        min_wilson_lower=explicit_filter_min_wilson_lower,
    )

    print("[shadow] by rule:")
    print(by_rule.head(args.top).to_string(index=False))
    print("[shadow] top conditions:")
    if conditions.empty:
        print("  none")
    else:
        print(conditions.head(args.top).to_string(index=False))
    print("[shadow] accepted conditions:")
    if accepted.empty:
        print("  none")
    else:
        print(accepted.head(args.top).to_string(index=False))
    print("[shadow] explicit filter recommendations:")
    if explicit_filters.empty:
        print("  none")
    else:
        print(explicit_filters.head(args.top).to_string(index=False))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        conditions.to_csv(out, index=False)
        print(f"[shadow] conditions saved: {out}")
    if args.explicit_filter_output:
        out = Path(args.explicit_filter_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        explicit_filters.to_csv(out, index=False)
        print(f"[shadow] explicit filter recommendations saved: {out}")


if __name__ == "__main__":
    main()

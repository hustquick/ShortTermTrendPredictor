import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from scripts.online_signal_filter_walkforward import (
    _apply_condition,
    _load_rows,
    _search_ranked_conditions,
    _wilson_lower_bound,
)


DEFAULT_REGIME_FEATURES = [
    "ret_5",
    "ret_10",
    "ret_30",
    "ema_10_30_diff",
    "ema_20_60_diff",
    "macd_hist",
    "rsi_14",
    "close_position",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "taker_buy_ratio",
    "trend_agreement",
    "mtf_3m_ret_1",
    "mtf_3m_ret_3",
    "mtf_3m_ema_10_30_diff",
    "mtf_3m_macd_hist",
    "mtf_3m_rsi_14",
    "mtf_3m_taker_buy_ratio",
    "mtf_3m_volume_ratio_5",
    "mtf_5m_ret_1",
    "mtf_5m_ret_3",
    "mtf_5m_ema_10_30_diff",
    "mtf_5m_macd_hist",
    "mtf_5m_rsi_14",
    "mtf_5m_taker_buy_ratio",
    "mtf_5m_volume_ratio_5",
]


def _parse_features(value: str | None) -> list[str]:
    if not value:
        return DEFAULT_REGIME_FEATURES
    return [item.strip() for item in value.split(",") if item.strip()]


def _fit_regime_model(train: pd.DataFrame, feature_cols: list[str], n_clusters: int, random_state: int):
    cols = [column for column in feature_cols if column in train.columns]
    if not cols:
        raise RuntimeError("no regime feature columns are present in input")
    x_train = train[cols].apply(pd.to_numeric, errors="coerce")
    medians = x_train.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_train = x_train.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    if len(x_train) < n_clusters:
        raise RuntimeError(f"train rows fewer than n_clusters: rows={len(x_train)}, n_clusters={n_clusters}")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    model.fit(x_scaled)
    return cols, medians, scaler, model


def _assign_regime(df: pd.DataFrame, cols: list[str], medians: pd.Series, scaler: StandardScaler, model: KMeans) -> pd.DataFrame:
    out = df.copy()
    x = out[cols].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    out["unsup_regime"] = model.predict(scaler.transform(x)).astype(int)
    return out


def _stats(df: pd.DataFrame, mask: pd.Series) -> dict:
    samples = int(mask.sum())
    wins = int(df.loc[mask, "correct_bool"].sum()) if samples else 0
    return {
        "samples": samples,
        "wins": wins,
        "win_rate": wins / samples if samples else np.nan,
        "wilson": _wilson_lower_bound(wins, samples) if samples else 0.0,
    }


def _build_rule_regime_table(
    train: pd.DataFrame,
    ranked_conditions: list[dict],
    *,
    min_regime_samples: int,
    min_regime_win_rate: float,
    min_regime_wilson: float,
) -> pd.DataFrame:
    rows = []
    for condition in ranked_conditions:
        condition_mask = _apply_condition(train, condition["condition"])
        for regime in sorted(train["unsup_regime"].dropna().unique()):
            regime_mask = train["unsup_regime"].eq(regime)
            mask = condition_mask & regime_mask
            stats = _stats(train, mask)
            if stats["samples"] < min_regime_samples:
                continue
            if stats["win_rate"] < min_regime_win_rate:
                continue
            if stats["wilson"] < min_regime_wilson:
                continue
            rows.append(
                {
                    **condition,
                    "regime": int(regime),
                    "regime_train_samples": stats["samples"],
                    "regime_train_wins": stats["wins"],
                    "regime_train_win_rate": stats["win_rate"],
                    "regime_train_wilson": stats["wilson"],
                }
            )
    return pd.DataFrame(rows)


def _select_rule_regimes(
    select: pd.DataFrame,
    table: pd.DataFrame,
    *,
    min_select_samples: int,
    min_select_win_rate: float,
    min_select_wilson: float,
    limit: int,
) -> list[dict]:
    selected = []
    for _, row in table.iterrows():
        condition = str(row["condition"])
        regime = int(row["regime"])
        mask = _apply_condition(select, condition) & select["unsup_regime"].eq(regime)
        stats = _stats(select, mask)
        if stats["samples"] < min_select_samples:
            continue
        if stats["win_rate"] < min_select_win_rate:
            continue
        if stats["wilson"] < min_select_wilson:
            continue
        selected.append(
            {
                **row.to_dict(),
                "select_samples": stats["samples"],
                "select_wins": stats["wins"],
                "select_win_rate": stats["win_rate"],
                "select_wilson": stats["wilson"],
            }
        )
    selected.sort(
        key=lambda item: (
            item["select_wilson"],
            item["select_win_rate"],
            item["regime_train_wilson"],
            item["regime_train_samples"],
        ),
        reverse=True,
    )
    return selected[:limit]


def run(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = _parse_features(args.regime_features)
    start = df["timestamp_dt"].min() + pd.Timedelta(days=args.train_days) + pd.Timedelta(hours=args.select_hours)
    end = df["timestamp_dt"].max()
    current = start
    window_no = 1
    report_rows = []
    signal_frames = []
    table_frames = []

    while current < end:
        train_end = current - pd.Timedelta(hours=args.select_hours)
        train_start = train_end - pd.Timedelta(days=args.train_days)
        select_start = train_end
        select_end = current
        cover_start = current
        cover_end = min(current + pd.Timedelta(hours=args.cover_hours), end)

        train_raw = df[(df["timestamp_dt"] >= train_start) & (df["timestamp_dt"] < train_end)].copy()
        select_raw = df[(df["timestamp_dt"] >= select_start) & (df["timestamp_dt"] < select_end)].copy()
        cover_raw = df[(df["timestamp_dt"] >= cover_start) & (df["timestamp_dt"] < cover_end)].copy()
        if train_raw.empty or select_raw.empty or cover_raw.empty:
            current += pd.Timedelta(hours=args.step_hours)
            window_no += 1
            continue

        try:
            cols, medians, scaler, regime_model = _fit_regime_model(
                train_raw,
                feature_cols,
                args.n_clusters,
                args.random_state,
            )
        except RuntimeError as exc:
            report_rows.append(
                {
                    "window": window_no,
                    "train_start": train_start,
                    "train_end": train_end,
                    "select_start": select_start,
                    "select_end": select_end,
                    "cover_start": cover_start,
                    "cover_end": cover_end,
                    "selected": 0,
                    "cover_signals": 0,
                    "cover_wins": 0,
                    "cover_win_rate": np.nan,
                    "skip_reason": str(exc),
                }
            )
            current += pd.Timedelta(hours=args.step_hours)
            window_no += 1
            continue

        train = _assign_regime(train_raw, cols, medians, scaler, regime_model)
        select = _assign_regime(select_raw, cols, medians, scaler, regime_model)
        cover = _assign_regime(cover_raw, cols, medians, scaler, regime_model)

        train_days = max((train_end - train_start).total_seconds() / 86_400, 1e-9)
        ranked = _search_ranked_conditions(
            train,
            max_clauses=args.max_clauses,
            min_samples=args.min_train_samples,
            min_signals_per_day=args.min_train_samples / train_days,
            min_win_rate=args.min_train_win_rate,
            min_wilson_lower=args.min_train_wilson,
            beam_size=args.beam_size,
            limit=args.candidate_limit,
        )
        table = _build_rule_regime_table(
            train,
            ranked,
            min_regime_samples=args.min_regime_samples,
            min_regime_win_rate=args.min_regime_win_rate,
            min_regime_wilson=args.min_regime_wilson,
        )
        if not table.empty:
            table = table.copy()
            table["window"] = window_no
            table["train_start"] = train_start
            table["train_end"] = train_end
            table_frames.append(table)

        selected = (
            _select_rule_regimes(
                select,
                table,
                min_select_samples=args.min_select_samples,
                min_select_win_rate=args.min_select_win_rate,
                min_select_wilson=args.min_select_wilson,
                limit=args.selected_limit,
            )
            if not table.empty
            else []
        )

        cover_masks = []
        selected_descriptions = []
        for item in selected:
            condition = str(item["condition"])
            regime = int(item["regime"])
            cover_masks.append(_apply_condition(cover, condition) & cover["unsup_regime"].eq(regime))
            selected_descriptions.append(f"{condition} @ regime={regime}")
        if cover_masks:
            cover_mask = cover_masks[0].copy()
            for mask in cover_masks[1:]:
                cover_mask |= mask
        else:
            cover_mask = pd.Series(False, index=cover.index)

        cover_signals = cover[cover_mask].copy()
        cover_count = int(len(cover_signals))
        cover_wins = int(cover_signals["correct_bool"].sum()) if cover_count else 0
        cover_days = max((cover_end - cover_start).total_seconds() / 86_400, 1e-9)
        if cover_count:
            cover_signals["window"] = window_no
            cover_signals["unsup_signal_direction"] = cover_signals["direction"]
            cover_signals["unsup_selected_rules"] = " || ".join(selected_descriptions)
            signal_frames.append(cover_signals)

        report_rows.append(
            {
                "window": window_no,
                "train_start": train_start,
                "train_end": train_end,
                "select_start": select_start,
                "select_end": select_end,
                "cover_start": cover_start,
                "cover_end": cover_end,
                "regime_features": ",".join(cols),
                "n_clusters": args.n_clusters,
                "candidate_rules": len(ranked),
                "rule_regime_rows": int(len(table)),
                "selected": len(selected),
                "selected_rules": " || ".join(selected_descriptions),
                "cover_signals": cover_count,
                "cover_wins": cover_wins,
                "cover_win_rate": cover_wins / cover_count if cover_count else np.nan,
                "cover_signals_per_day": cover_count / cover_days,
                "skip_reason": "",
            }
        )
        current += pd.Timedelta(hours=args.step_hours)
        window_no += 1

    report = pd.DataFrame(report_rows)
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    tables = pd.concat(table_frames, ignore_index=True) if table_frames else pd.DataFrame()
    return report, signals, tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strict walk-forward unsupervised regime + rule win-rate table. "
            "Each fold fits scaler/KMeans on train only, then freezes it for select/cover."
        )
    )
    parser.add_argument("csv", type=Path)
    parser.add_argument("--train-days", type=int, default=3)
    parser.add_argument("--select-hours", type=int, default=6)
    parser.add_argument("--cover-hours", type=int, default=1)
    parser.add_argument("--step-hours", type=int, default=1)
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--regime-features", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-clauses", type=int, default=2)
    parser.add_argument("--min-train-samples", type=int, default=20)
    parser.add_argument("--min-train-win-rate", type=float, default=0.60)
    parser.add_argument("--min-train-wilson", type=float, default=0.35)
    parser.add_argument("--min-regime-samples", type=int, default=8)
    parser.add_argument("--min-regime-win-rate", type=float, default=0.68)
    parser.add_argument("--min-regime-wilson", type=float, default=0.35)
    parser.add_argument("--min-select-samples", type=int, default=4)
    parser.add_argument("--min-select-win-rate", type=float, default=0.70)
    parser.add_argument("--min-select-wilson", type=float, default=0.20)
    parser.add_argument("--beam-size", type=int, default=80)
    parser.add_argument("--candidate-limit", type=int, default=80)
    parser.add_argument("--selected-limit", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    df = _load_rows(args.csv)
    if df.empty:
        raise RuntimeError(f"empty csv: {args.csv}")
    report, signals, tables = run(df, args)

    total = int(len(signals))
    wins = int(signals["correct_bool"].sum()) if total else 0
    if not report.empty:
        total_days = max(
            (pd.to_datetime(report["cover_end"]).max() - pd.to_datetime(report["cover_start"]).min()).total_seconds()
            / 86_400,
            1e-9,
        )
    else:
        total_days = 0.0
    print("[unsupervised_regime_rule_walkforward] combined:")
    print(f"  windows={len(report)}")
    print(f"  signals={total}")
    print(f"  wins={wins}")
    print(f"  win_rate={wins / total if total else None}")
    print(f"  signals_per_day={total / total_days if total_days else 0.0}")
    if not signals.empty:
        by_day = signals.assign(day=signals["timestamp_dt"].dt.date).groupby("day")["correct_bool"].agg(["count", "sum"])
        by_day["win_rate"] = by_day["sum"] / by_day["count"]
        print("[unsupervised_regime_rule_walkforward] by day")
        print(by_day.to_string())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
        if not signals.empty:
            signals.to_csv(args.output.with_name(args.output.stem + "_signals.csv"), index=False)
        if not tables.empty:
            tables.to_csv(args.output.with_name(args.output.stem + "_rule_regime_table.csv"), index=False)
        print(f"[unsupervised_regime_rule_walkforward] saved: {args.output}")


if __name__ == "__main__":
    main()

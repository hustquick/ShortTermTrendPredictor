from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.online_signal_filter_walkforward import (
    _apply_condition,
    _load_rows,
    _search_best_condition,
    _search_ranked_conditions,
)


@dataclass(frozen=True)
class RollingCoverageConfig:
    train_days: int = 30
    cover_days: int = 7
    step_days: int = 7
    max_clauses: int = 3
    min_samples: int = 60
    min_signals_per_day: float = 5.0
    min_win_rate: float = 0.75
    min_wilson_lower: float = 0.68
    beam_size: int = 120


def load_candidate_rows(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        try:
            frames.append(_load_rows(path))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return df
    if {"timestamp", "rule", "direction"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["timestamp", "rule", "direction"], keep="last")
    else:
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
    return df.sort_values("timestamp_dt").reset_index(drop=True)


def rolling_origin(df: pd.DataFrame, config: RollingCoverageConfig) -> pd.Timestamp | None:
    if df.empty or "timestamp_dt" not in df:
        return None
    start = df["timestamp_dt"].min()
    if pd.isna(start):
        return None
    return start + pd.Timedelta(days=config.train_days)


def active_window_bounds(
    df: pd.DataFrame,
    now_dt: pd.Timestamp,
    config: RollingCoverageConfig,
) -> tuple[int, pd.Timestamp, pd.Timestamp, pd.Timestamp] | None:
    origin = rolling_origin(df, config)
    if origin is None or now_dt < origin:
        return None
    step = pd.Timedelta(days=config.step_days)
    cover = pd.Timedelta(days=config.cover_days)
    elapsed_steps = int((now_dt - origin) // step)
    cover_start = origin + elapsed_steps * step
    cover_end = cover_start + cover
    if not (cover_start <= now_dt < cover_end):
        return None
    train_start = cover_start - pd.Timedelta(days=config.train_days)
    return elapsed_steps + 1, train_start, cover_start, cover_end


def discover_window_condition(
    df: pd.DataFrame,
    cover_start: pd.Timestamp,
    config: RollingCoverageConfig,
) -> dict | None:
    train_start = cover_start - pd.Timedelta(days=config.train_days)
    train = df[(df["timestamp_dt"] >= train_start) & (df["timestamp_dt"] < cover_start)].copy()
    if train.empty:
        return None
    return _search_best_condition(
        train,
        max_clauses=config.max_clauses,
        min_samples=config.min_samples,
        min_signals_per_day=config.min_signals_per_day,
        min_win_rate=config.min_win_rate,
        min_wilson_lower=config.min_wilson_lower,
        beam_size=config.beam_size,
    )


def discover_window_conditions(
    df: pd.DataFrame,
    cover_start: pd.Timestamp,
    config: RollingCoverageConfig,
    *,
    limit: int = 10,
    recent_start: pd.Timestamp | None = None,
    recent_min_matches: int = 0,
    recent_min_win_rate: float | None = None,
) -> list[dict]:
    train_start = cover_start - pd.Timedelta(days=config.train_days)
    train = df[(df["timestamp_dt"] >= train_start) & (df["timestamp_dt"] < cover_start)].copy()
    if train.empty:
        return []
    ranked = _search_ranked_conditions(
        train,
        max_clauses=config.max_clauses,
        min_samples=config.min_samples,
        min_signals_per_day=config.min_signals_per_day,
        min_win_rate=config.min_win_rate,
        min_wilson_lower=config.min_wilson_lower,
        beam_size=config.beam_size,
        limit=max(limit * 20, limit),
    )
    if not ranked:
        return []
    if recent_start is None or recent_min_matches <= 0:
        return ranked[:limit]

    recent = df[df["timestamp_dt"] >= recent_start].copy()
    if recent.empty:
        return []
    selected: list[dict] = []
    for item in ranked:
        mask = _apply_condition(recent, item["condition"])
        matches = int(mask.sum())
        if matches < recent_min_matches:
            continue
        wins = int(recent.loc[mask, "correct_bool"].sum())
        win_rate = wins / matches if matches else 0.0
        if recent_min_win_rate is not None and win_rate < recent_min_win_rate:
            continue
        selected.append(
            {
                **item,
                "recent_start": str(recent_start),
                "recent_matches": matches,
                "recent_wins": wins,
                "recent_win_rate": win_rate,
            }
        )
        if len(selected) >= limit:
            break
    return selected


def build_window_item(
    df: pd.DataFrame,
    now_dt: pd.Timestamp,
    config: RollingCoverageConfig,
    source: str,
) -> dict | None:
    bounds = active_window_bounds(df, now_dt, config)
    if bounds is None:
        return None
    window_no, train_start, cover_start, cover_end = bounds
    selected = discover_window_condition(df, cover_start, config)
    if selected is None:
        return {
            "source": source,
            "window": window_no,
            "condition": "",
            "train_start": str(train_start),
            "train_end": str(cover_start),
            "cover_start": str(cover_start),
            "cover_end": str(cover_end),
        }
    return {
        **selected,
        "source": source,
        "window": window_no,
        "train_start": str(train_start),
        "train_end": str(cover_start),
        "cover_start": str(cover_start),
        "cover_end": str(cover_end),
    }


def build_window_items(
    df: pd.DataFrame,
    now_dt: pd.Timestamp,
    config: RollingCoverageConfig,
    source: str,
    *,
    limit: int = 10,
    recent_lookback_days: float | None = None,
    recent_min_matches: int = 0,
    recent_min_win_rate: float | None = None,
) -> list[dict]:
    bounds = active_window_bounds(df, now_dt, config)
    if bounds is None:
        return []
    window_no, train_start, cover_start, cover_end = bounds
    recent_start = None
    if recent_lookback_days is not None and recent_lookback_days > 0:
        recent_start = now_dt - pd.Timedelta(days=float(recent_lookback_days))
    selected = discover_window_conditions(
        df,
        cover_start,
        config,
        limit=limit,
        recent_start=recent_start,
        recent_min_matches=recent_min_matches,
        recent_min_win_rate=recent_min_win_rate,
    )
    return [
        {
            **item,
            "source": source,
            "window": window_no,
            "train_start": str(train_start),
            "train_end": str(cover_start),
            "cover_start": str(cover_start),
            "cover_end": str(cover_end),
        }
        for item in selected
    ]


def matches_condition(condition: str, row: pd.Series | dict) -> bool:
    df = pd.DataFrame([dict(row)])
    return bool(_apply_condition(df, condition).iloc[0])

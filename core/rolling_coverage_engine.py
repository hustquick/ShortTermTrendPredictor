from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.online_signal_filter_walkforward import (
    _apply_condition,
    _load_rows,
    _search_best_condition,
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


def matches_condition(condition: str, row: pd.Series | dict) -> bool:
    df = pd.DataFrame([dict(row)])
    return bool(_apply_condition(df, condition).iloc[0])

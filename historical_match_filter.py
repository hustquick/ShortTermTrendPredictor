# historical_match_filter.py

from dataclasses import dataclass

import pandas as pd

from config import PREDICT_HORIZON_MINUTES


MATCH_LOOKBACK_DAYS = 60
MATCH_MIN_SAMPLES = 30
MATCH_MIN_SUCCESS_RATE = 0.70
MATCH_MIN_EDGE = 0.15
MATCH_PROBA_BUCKET = 0.10
MATCH_EDGE_BUCKET = 0.15
MATCH_RSI_BUCKET = 10.0


@dataclass
class MatchResult:
    accepted: bool
    direction: str
    matched_signals: int
    success_rate: float | None
    reason: str


def _bucket(value: float, width: float) -> tuple[float, float]:
    lower = value - width
    upper = value + width
    return lower, upper


def _same_sign_or_zero(series: pd.Series, value: float) -> pd.Series:
    if value > 0:
        return series >= 0
    if value < 0:
        return series <= 0
    return series.abs() < 1e-12


def _actual_direction(current_price: pd.Series, future_price: pd.Series) -> pd.Series:
    return pd.Series(
        ["up" if f > c else "down" for c, f in zip(current_price, future_price)],
        index=current_price.index,
    )


def evaluate_historical_match(
    historical_rows: pd.DataFrame,
    current_features: pd.Series,
    prediction: dict,
    candidate_direction: str,
) -> MatchResult:
    """Evaluate whether historical similar samples support the candidate direction.

    Objective is directional accuracy only. No fee, return, PnL, or position logic is used.
    """
    if candidate_direction not in {"up", "down"}:
        return MatchResult(False, "no_trade", 0, None, "invalid_direction")

    if historical_rows.empty:
        return MatchResult(False, candidate_direction, 0, None, "empty_history")

    p_up = float(prediction.get("up_signal_probability", 0.0))
    p_down = float(prediction.get("down_signal_probability", 0.0))
    edge = float(prediction.get("direction_edge", 0.0))

    if abs(edge) < MATCH_MIN_EDGE:
        return MatchResult(False, candidate_direction, 0, None, "edge_too_small")

    df = historical_rows.copy()
    required = [
        "close",
        "future_close",
        "up_signal_probability",
        "down_signal_probability",
        "direction_edge",
        "trend_agreement",
        "macd_hist",
        "ret_5",
        "ema_5_20_diff",
        "rsi_14",
        "boll_position",
    ]
    for col in required:
        if col not in df.columns:
            return MatchResult(False, candidate_direction, 0, None, f"missing_{col}")

    up_lo, up_hi = _bucket(p_up, MATCH_PROBA_BUCKET)
    down_lo, down_hi = _bucket(p_down, MATCH_PROBA_BUCKET)
    edge_lo, edge_hi = _bucket(edge, MATCH_EDGE_BUCKET)

    trend = float(current_features.get("trend_agreement", 0.0))
    macd = float(current_features.get("macd_hist", 0.0))
    ret_5 = float(current_features.get("ret_5", 0.0))
    ema = float(current_features.get("ema_5_20_diff", 0.0))
    rsi = float(current_features.get("rsi_14", 50.0))
    boll = float(current_features.get("boll_position", 0.5))

    mask = (
        df["up_signal_probability"].between(up_lo, up_hi)
        & df["down_signal_probability"].between(down_lo, down_hi)
        & df["direction_edge"].between(edge_lo, edge_hi)
        & _same_sign_or_zero(df["trend_agreement"], trend)
        & _same_sign_or_zero(df["macd_hist"], macd)
        & _same_sign_or_zero(df["ret_5"], ret_5)
        & _same_sign_or_zero(df["ema_5_20_diff"], ema)
        & df["rsi_14"].between(rsi - MATCH_RSI_BUCKET, rsi + MATCH_RSI_BUCKET)
        & df["boll_position"].between(boll - 0.15, boll + 0.15)
    )

    matched = df[mask].copy()
    matched_signals = int(len(matched))

    if matched_signals < MATCH_MIN_SAMPLES:
        return MatchResult(False, candidate_direction, matched_signals, None, "not_enough_matched_signals")

    actual = _actual_direction(matched["close"], matched["future_close"])
    success_rate = float((actual == candidate_direction).mean())

    if success_rate < MATCH_MIN_SUCCESS_RATE:
        return MatchResult(False, candidate_direction, matched_signals, success_rate, "success_rate_too_low")

    return MatchResult(True, candidate_direction, matched_signals, success_rate, "historical_match_confirmed")


def build_historical_match_rows(feature_df: pd.DataFrame, model, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build historical rows with model outputs and future direction labels."""
    if feature_df.empty:
        return pd.DataFrame()

    close_by_timestamp = raw_df.set_index("timestamp")["close"]
    rows = []
    horizon_ms = PREDICT_HORIZON_MINUTES * 60_000

    for idx, row in feature_df.iterrows():
        ts = int(row["timestamp"])
        future_ts = ts + horizon_ms
        if future_ts not in close_by_timestamp.index:
            continue
        X = row[model.feature_cols].to_frame().T
        if X.isna().any(axis=None):
            continue
        pred = model.predict_one(X, signal_filter=None)
        rows.append(
            {
                "timestamp": ts,
                "close": float(row["close"]),
                "future_close": float(close_by_timestamp.loc[future_ts]),
                "up_signal_probability": pred.get("up_signal_probability"),
                "down_signal_probability": pred.get("down_signal_probability"),
                "direction_edge": pred.get("direction_edge"),
                "trend_agreement": row.get("trend_agreement"),
                "macd_hist": row.get("macd_hist"),
                "ret_5": row.get("ret_5"),
                "ema_5_20_diff": row.get("ema_5_20_diff"),
                "rsi_14": row.get("rsi_14"),
                "boll_position": row.get("boll_position"),
            }
        )

    return pd.DataFrame(rows)

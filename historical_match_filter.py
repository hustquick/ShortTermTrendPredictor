# historical_match_filter.py

from dataclasses import dataclass

import pandas as pd

from config import (
    BACKTEST_MIN_TRAIN_SAMPLES,
    BACKTEST_TRAIN_WINDOW_MINUTES,
    HISTORICAL_MATCH_WALK_FORWARD_MODEL_UPDATE_MINUTES,
    PREDICT_HORIZON_MINUTES,
)
from trainer import _ensemble_predict_proba, train_validation_model


MATCH_LOOKBACK_DAYS = 60
MATCH_MIN_SAMPLES = 15
MATCH_MIN_SUCCESS_RATE = 0.72
MATCH_MIN_EDGE = 0.15
MATCH_PROBA_BUCKET = 0.10
MATCH_EDGE_BUCKET = 0.15
MATCH_RSI_BUCKET = 10.0
HISTORICAL_MATCH_MAX_ROWS = 6000


@dataclass
class MatchResult:
    accepted: bool
    direction: str
    matched_signals: int
    success_rate: float | None
    reason: str


def _bucket(value: float, width: float) -> tuple[float, float]:
    return value - width, value + width


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
    """Build historical rows with vectorized model outputs and future labels."""
    if feature_df.empty:
        return pd.DataFrame()

    required_feature_cols = [
        "timestamp",
        "close",
        "trend_agreement",
        "macd_hist",
        "ret_5",
        "ema_5_20_diff",
        "rsi_14",
        "boll_position",
    ]
    for col in required_feature_cols:
        if col not in feature_df.columns:
            return pd.DataFrame()

    close_by_timestamp = raw_df.set_index("timestamp")["close"]
    horizon_ms = PREDICT_HORIZON_MINUTES * 60_000

    data = feature_df.dropna(subset=model.feature_cols + required_feature_cols).copy()
    if len(data) > HISTORICAL_MATCH_MAX_ROWS:
        data = data.tail(HISTORICAL_MATCH_MAX_ROWS).copy()

    data["future_timestamp"] = data["timestamp"].astype(int) + horizon_ms
    data["future_close"] = data["future_timestamp"].map(close_by_timestamp)
    data = data.dropna(subset=["future_close"]).copy()
    if data.empty:
        return pd.DataFrame()

    X = data[model.feature_cols]
    p_up = _ensemble_predict_proba(model.up_models, X)
    p_down = _ensemble_predict_proba(model.down_models, X)

    out = data[
        [
            "timestamp",
            "close",
            "future_close",
            "trend_agreement",
            "macd_hist",
            "ret_5",
            "ema_5_20_diff",
            "rsi_14",
            "boll_position",
        ]
    ].copy()
    out["up_signal_probability"] = p_up
    out["down_signal_probability"] = p_down
    out["direction_edge"] = out["up_signal_probability"] - out["down_signal_probability"]
    return out.reset_index(drop=True)


def build_walk_forward_historical_match_rows(
    feature_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    train_window_minutes: int = BACKTEST_TRAIN_WINDOW_MINUTES,
    model_update_minutes: int = HISTORICAL_MATCH_WALK_FORWARD_MODEL_UPDATE_MINUTES,
    min_train_samples: int = BACKTEST_MIN_TRAIN_SAMPLES,
) -> pd.DataFrame:
    """Build historical match rows with out-of-sample model scores.

    Each historical row is scored by a model trained only on candles strictly
    before the row's time bucket. This avoids the optimistic "current model
    scores all history" leakage that made matched success rates too good.
    """
    if feature_df.empty or raw_df.empty:
        return pd.DataFrame()

    required_feature_cols = [
        "timestamp",
        "close",
        "trend_agreement",
        "macd_hist",
        "ret_5",
        "ema_5_20_diff",
        "rsi_14",
        "boll_position",
    ]
    for col in required_feature_cols:
        if col not in feature_df.columns:
            return pd.DataFrame()

    raw = raw_df.sort_values("timestamp").copy()
    features = feature_df.sort_values("timestamp").copy()
    close_by_timestamp = raw.set_index("timestamp")["close"]
    horizon_ms = PREDICT_HORIZON_MINUTES * 60_000
    update_ms = model_update_minutes * 60_000
    train_window_ms = train_window_minutes * 60_000

    data = features.dropna(subset=required_feature_cols).copy()
    if len(data) > HISTORICAL_MATCH_MAX_ROWS:
        data = data.tail(HISTORICAL_MATCH_MAX_ROWS).copy()

    data["future_timestamp"] = data["timestamp"].astype(int) + horizon_ms
    data["future_close"] = data["future_timestamp"].map(close_by_timestamp)
    data = data.dropna(subset=["future_close"]).copy()
    if data.empty:
        return pd.DataFrame()

    data["score_bucket"] = (data["timestamp"].astype(int) // update_ms) * update_ms
    rows = []

    for bucket_start, bucket_df in data.groupby("score_bucket", sort=True):
        bucket_start = int(bucket_start)
        train_start = bucket_start - train_window_ms
        train_df = raw[(raw["timestamp"] >= train_start) & (raw["timestamp"] < bucket_start)].copy()
        if len(train_df) < min_train_samples:
            continue

        try:
            bucket_model = train_validation_model(train_df)
        except Exception as exc:
            print(
                "[historical_match] skip walk-forward bucket: "
                f"bucket={bucket_start}, error={type(exc).__name__}: {exc}"
            )
            continue

        usable = bucket_df.dropna(subset=bucket_model.feature_cols).copy()
        if usable.empty:
            continue

        X = usable[bucket_model.feature_cols]
        out = usable[
            [
                "timestamp",
                "close",
                "future_close",
                "trend_agreement",
                "macd_hist",
                "ret_5",
                "ema_5_20_diff",
                "rsi_14",
                "boll_position",
            ]
        ].copy()
        out["up_signal_probability"] = _ensemble_predict_proba(bucket_model.up_models, X)
        out["down_signal_probability"] = _ensemble_predict_proba(bucket_model.down_models, X)
        out["direction_edge"] = out["up_signal_probability"] - out["down_signal_probability"]
        out["scored_model_time"] = int(bucket_start)
        rows.append(out)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True).tail(HISTORICAL_MATCH_MAX_ROWS).reset_index(drop=True)

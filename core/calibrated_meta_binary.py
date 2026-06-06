from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from config import DATA_DIR, PREDICT_HORIZON_MS
from features import get_feature_columns


MODEL_PATH = DATA_DIR / "calibrated_meta_binary_model.pkl"


def _make_classifier(random_state: int = 17):
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=260,
            learning_rate=0.035,
            num_leaves=24,
            max_depth=6,
            min_child_samples=60,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.15,
            reg_lambda=0.6,
            objective="binary",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        return ExtraTreesClassifier(
            n_estimators=320,
            max_depth=12,
            min_samples_leaf=12,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )


def _clean_x(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return df.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _safe_binary_proba(model, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.shape[1] == 1:
        only_class = int(getattr(model, "classes_", [0])[0])
        return np.ones(len(x)) if only_class == 1 else np.zeros(len(x))
    classes = list(getattr(model, "classes_", [0, 1]))
    one_idx = classes.index(1) if 1 in classes else 1
    return proba[:, one_idx].astype(float)


def _binary_labels(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy().sort_values("timestamp").reset_index(drop=True)
    close_by_timestamp = df.set_index("timestamp")["close"]
    future_timestamp = df["timestamp"] + PREDICT_HORIZON_MS
    df["future_price"] = future_timestamp.map(close_by_timestamp)
    df["future_return"] = df["future_price"] / df["close"] - 1.0
    valid = df["future_price"].notna()
    df["binary_label"] = np.nan
    df.loc[valid, "binary_label"] = (df.loc[valid, "future_return"] > 0).astype(int)
    df["label_mature_timestamp"] = df["timestamp"] + PREDICT_HORIZON_MS
    return df


def prepare_calibrated_binary_frame(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    labeled = _binary_labels(feature_df)
    feature_cols = get_feature_columns(labeled)
    feature_cols = [c for c in feature_cols if c not in {"binary_label", "label_mature_timestamp"}]
    data = labeled.dropna(subset=feature_cols + ["binary_label", "future_return"]).copy()
    data["binary_label"] = data["binary_label"].astype(int)
    return data, feature_cols


def _fit_calibrator(raw_proba: np.ndarray, y: np.ndarray):
    if len(np.unique(y)) < 2 or len(np.unique(np.round(raw_proba, 6))) < 4:
        return None
    try:
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(raw_proba, y)
        return calibrator
    except Exception:
        lr = LogisticRegression(max_iter=500)
        lr.fit(raw_proba.reshape(-1, 1), y)
        return lr


def _apply_calibrator(calibrator, raw_proba: np.ndarray) -> np.ndarray:
    raw_proba = np.asarray(raw_proba, dtype=float)
    if calibrator is None:
        return raw_proba
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
    return np.asarray(calibrator.predict(raw_proba), dtype=float)


def _meta_feature_frame(source: pd.DataFrame, calibrated_up: np.ndarray) -> pd.DataFrame:
    p_up = np.asarray(calibrated_up, dtype=float)
    primary_direction_up = (p_up >= 0.5).astype(int)
    out = pd.DataFrame(index=source.index)
    out["cmb_up_probability"] = p_up
    out["cmb_primary_direction_up"] = primary_direction_up
    out["cmb_primary_margin"] = np.abs(p_up - 0.5)
    out["cmb_primary_signed_margin"] = p_up - 0.5
    for col in [
        "ret_1",
        "ret_3",
        "ret_5",
        "ret_10",
        "ret_30",
        "ret_60",
        "volatility_10",
        "volatility_30",
        "volatility_60",
        "atr_14",
        "rsi_6",
        "rsi_14",
        "macd",
        "macd_hist",
        "macd_hist_diff",
        "boll_position",
        "boll_width",
        "close_position",
        "volume_ratio_10",
        "volume_ratio_30",
        "volume_zscore",
        "trade_count_ratio_10",
        "quote_volume_ratio_10",
        "taker_buy_ratio",
        "taker_buy_ratio_diff_5_10",
        "trend_agreement",
        "trend_agreement_long",
        "mtf_3_close_ret_1",
        "mtf_3_ema_5_20_diff",
        "mtf_3_rsi_14",
        "mtf_5_close_ret_1",
        "mtf_5_ema_5_20_diff",
        "mtf_5_rsi_14",
        "minute_sin",
        "minute_cos",
        "dow_sin",
        "dow_cos",
    ]:
        if col in source.columns:
            out[col] = pd.to_numeric(source[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _max_loss_streak(correct: pd.Series) -> int:
    worst = 0
    current = 0
    for value in correct.astype(bool).tolist():
        if value:
            current = 0
        else:
            current += 1
            worst = max(worst, current)
    return worst


@dataclass
class ThresholdSelection:
    threshold: float
    win_rate: float
    signals: int
    density_per_day: float


def select_threshold(
    validation: pd.DataFrame,
    *,
    min_win_rate: float = 0.75,
    min_signals: int = 8,
    allow_fallback: bool = False,
    threshold_grid: list[float] | None = None,
) -> ThresholdSelection:
    if threshold_grid is None:
        threshold_grid = [round(x, 2) for x in np.arange(0.50, 0.951, 0.01)]
    if validation.empty:
        return ThresholdSelection(0.99, 0.0, 0, 0.0)
    days = max(
        1.0 / 24.0,
        (float(validation["timestamp"].max()) - float(validation["timestamp"].min())) / 86_400_000.0,
    )
    rows = []
    for threshold in threshold_grid:
        selected = validation[validation["meta_probability"] >= threshold]
        signals = int(len(selected))
        if signals == 0:
            wr = 0.0
        else:
            wr = float(selected["correct"].mean())
        rows.append((threshold, wr, signals, signals / days))
    eligible = [r for r in rows if r[2] >= min_signals and r[1] >= min_win_rate]
    if not eligible:
        if not allow_fallback:
            return ThresholdSelection(1.01, 0.0, 0, 0.0)
        fallback = [r for r in rows if r[2] > 0]
        if not fallback:
            return ThresholdSelection(1.01, 0.0, 0, 0.0)
        best = sorted(fallback, key=lambda r: (r[1], r[2], -r[0]), reverse=True)[0]
    else:
        best = sorted(eligible, key=lambda r: (r[3], r[1], -r[0]), reverse=True)[0]
    return ThresholdSelection(best[0], best[1], best[2], best[3])


@dataclass
class CalibratedMetaBinaryModel:
    primary_model: object
    calibrator: object | None
    meta_model: object
    feature_cols: list[str]
    meta_feature_cols: list[str]
    threshold: float
    threshold_win_rate: float
    threshold_signals: int
    trained_at_timestamp: int
    trained_at: str

    def predict_one(self, features: pd.DataFrame | pd.Series) -> dict:
        if isinstance(features, pd.Series):
            row = features.to_frame().T
        else:
            row = features.copy()
        x = _clean_x(row, self.feature_cols)
        raw_up = _safe_binary_proba(self.primary_model, x)
        calibrated_up = np.clip(_apply_calibrator(self.calibrator, raw_up), 0.0, 1.0)
        meta_x = _meta_feature_frame(row, calibrated_up).reindex(columns=self.meta_feature_cols).fillna(0.0)
        meta_probability = np.clip(_safe_binary_proba(self.meta_model, meta_x), 0.0, 1.0)
        p_up = float(calibrated_up[0])
        meta_p = float(meta_probability[0])
        primary_direction = "up" if p_up >= 0.5 else "down"
        gate = "pass" if meta_p >= self.threshold else "block"
        direction = primary_direction if gate == "pass" else "no_trade"
        return {
            "predicted_direction": direction,
            "up_probability": p_up,
            "up_signal_probability": p_up,
            "down_signal_probability": 1.0 - p_up,
            "direction_edge": p_up - (1.0 - p_up),
            "confidence": meta_p,
            "meta_probability": meta_p,
            "primary_direction": primary_direction,
            "primary_margin": abs(p_up - 0.5),
            "calibrated_meta_binary_gate": gate,
            "calibrated_meta_binary_threshold": self.threshold,
            "calibrated_meta_binary_threshold_win_rate": self.threshold_win_rate,
            "calibrated_meta_binary_threshold_signals": self.threshold_signals,
            "is_valid_signal": gate == "pass",
            "high_win_rate_signal": gate == "pass",
        }

    def predict_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        row = features.copy()
        x = _clean_x(row, self.feature_cols)
        raw_up = _safe_binary_proba(self.primary_model, x)
        calibrated_up = np.clip(_apply_calibrator(self.calibrator, raw_up), 0.0, 1.0)
        meta_x = _meta_feature_frame(row, calibrated_up).reindex(columns=self.meta_feature_cols).fillna(0.0)
        meta_probability = np.clip(_safe_binary_proba(self.meta_model, meta_x), 0.0, 1.0)
        primary_direction = np.where(calibrated_up >= 0.5, "up", "down")
        gate = meta_probability >= self.threshold
        out = pd.DataFrame(index=row.index)
        out["up_probability"] = calibrated_up
        out["down_probability"] = 1.0 - calibrated_up
        out["direction_edge"] = calibrated_up - (1.0 - calibrated_up)
        out["meta_probability"] = meta_probability
        out["primary_direction"] = primary_direction
        out["predicted_direction"] = np.where(gate, primary_direction, "no_trade")
        out["threshold"] = self.threshold
        out["gate"] = np.where(gate, "pass", "block")
        return out


def train_calibrated_meta_binary_model(
    feature_df: pd.DataFrame,
    *,
    anchor_timestamp: int | None = None,
    primary_train_fraction: float = 0.70,
    meta_fit_fraction: float = 0.55,
    min_win_rate: float = 0.75,
    min_threshold_signals: int = 8,
    random_state: int = 17,
) -> CalibratedMetaBinaryModel:
    data, feature_cols = prepare_calibrated_binary_frame(feature_df)
    if anchor_timestamp is None:
        anchor_timestamp = int(data["timestamp"].max())
    mature = data[pd.to_numeric(data["label_mature_timestamp"], errors="coerce") <= anchor_timestamp].copy()
    mature = mature.sort_values("timestamp").reset_index(drop=True)
    if len(mature) < 600:
        raise ValueError(f"calibrated_meta_binary requires at least 600 mature rows, got {len(mature)}")
    split = int(len(mature) * primary_train_fraction)
    split = min(max(split, 300), len(mature) - 200)
    primary_train = mature.iloc[:split].copy()
    holdout = mature.iloc[split:].copy()
    if primary_train["binary_label"].nunique() < 2 or holdout["binary_label"].nunique() < 2:
        raise ValueError("calibrated_meta_binary labels must contain both classes")

    primary_model = _make_classifier(random_state)
    primary_model.fit(_clean_x(primary_train, feature_cols), primary_train["binary_label"].astype(int))

    raw_holdout = _safe_binary_proba(primary_model, _clean_x(holdout, feature_cols))
    calibrator = _fit_calibrator(raw_holdout, holdout["binary_label"].astype(int).to_numpy())
    calibrated_holdout = np.clip(_apply_calibrator(calibrator, raw_holdout), 0.0, 1.0)
    primary_direction = (calibrated_holdout >= 0.5).astype(int)
    correct = (primary_direction == holdout["binary_label"].astype(int).to_numpy()).astype(int)
    meta_features = _meta_feature_frame(holdout, calibrated_holdout)
    meta_feature_cols = list(meta_features.columns)
    meta_split = int(len(holdout) * meta_fit_fraction)
    meta_split = min(max(meta_split, 100), len(holdout) - 80)
    meta_train_x = meta_features.iloc[:meta_split]
    meta_train_y = correct[:meta_split]
    threshold_validation = holdout.iloc[meta_split:].copy()
    threshold_validation["meta_probability"] = 0.0
    threshold_validation["correct"] = correct[meta_split:]
    if len(np.unique(meta_train_y)) < 2:
        meta_model = ExtraTreesClassifier(
            n_estimators=80,
            max_depth=4,
            min_samples_leaf=20,
            random_state=random_state + 1,
            n_jobs=-1,
        )
        meta_model.fit(meta_train_x, np.zeros(len(meta_train_x), dtype=int))
    else:
        meta_model = _make_classifier(random_state + 1)
        meta_model.fit(meta_train_x, meta_train_y)
    threshold_validation["meta_probability"] = _safe_binary_proba(
        meta_model,
        meta_features.iloc[meta_split:].reindex(columns=meta_feature_cols).fillna(0.0),
    )
    selected = select_threshold(
        threshold_validation,
        min_win_rate=min_win_rate,
        min_signals=min_threshold_signals,
    )
    return CalibratedMetaBinaryModel(
        primary_model=primary_model,
        calibrator=calibrator,
        meta_model=meta_model,
        feature_cols=feature_cols,
        meta_feature_cols=meta_feature_cols,
        threshold=selected.threshold,
        threshold_win_rate=selected.win_rate,
        threshold_signals=selected.signals,
        trained_at_timestamp=int(anchor_timestamp),
        trained_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


class CalibratedMetaBinaryModelManager:
    def __init__(
        self,
        *,
        retrain_interval_seconds: int = 30 * 60,
        model_path: Path = MODEL_PATH,
        min_win_rate: float = 0.75,
    ):
        self.retrain_interval_seconds = retrain_interval_seconds
        self.model_path = model_path
        self.min_win_rate = min_win_rate
        self.model: CalibratedMetaBinaryModel | None = None
        self.last_train_time: datetime | None = None

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        self.last_train_time = datetime.now()
        return True

    def save(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

    def ensure_trained(self, feature_df: pd.DataFrame, *, anchor_timestamp: int | None = None) -> bool:
        if self.model is not None and self.last_train_time is not None:
            elapsed = (datetime.now() - self.last_train_time).total_seconds()
            if elapsed < self.retrain_interval_seconds:
                return False
        self.model = train_calibrated_meta_binary_model(
            feature_df,
            anchor_timestamp=anchor_timestamp,
            min_win_rate=self.min_win_rate,
        )
        self.last_train_time = datetime.now()
        self.save()
        return True

    @property
    def feature_cols(self) -> list[str]:
        if self.model is None:
            return []
        return list(self.model.feature_cols)

    def predict_one(self, features: pd.DataFrame | pd.Series) -> dict:
        if self.model is None:
            raise RuntimeError("calibrated_meta_binary model is not trained")
        return self.model.predict_one(features)

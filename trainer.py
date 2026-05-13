# trainer.py

import json
import pickle
from collections import deque

import numpy as np
import pandas as pd

from config import (
    CAT_WEIGHT,
    DUAL_DIRECTION_MIN_EDGE,
    DUAL_MODEL_PARAMS_FILE,
    ENABLE_HIGH_WIN_RATE_FILTER,
    ENABLE_LONG_SIGNALS,
    ENABLE_LONG_REGIME_FILTER,
    ENABLE_SHORT_REGIME_FILTER,
    ENABLE_SHORT_SIGNALS,
    ENABLE_SIGNAL_STABILITY_FILTER,
    ENABLE_SIGNAL_QUALITY_GATE,
    LGB_WEIGHT,
    LONG_SIGNAL_THRESHOLD,
    LONG_REGIME_FULL_HIGH_MIN_BODY_RATIO,
    LONG_REGIME_FULL_HIGH_MIN_CLOSE_POSITION,
    LONG_REGIME_MAX_RET_5,
    LONG_REGIME_MAX_RSI_14,
    LONG_REGIME_MIN_CLOSE_POSITION,
    LONG_REGIME_MIN_RET_30,
    LONG_REGIME_REQUIRE_EMA_20_60_POSITIVE,
    LONG_REGIME_REQUIRE_MACD_HIST_POSITIVE,
    LONG_REGIME_SKIP_FULL_HIGH_BODY,
    LONG_STABILITY_MIN_COUNT,
    LONG_STABILITY_THRESHOLD,
    LONG_STABILITY_WINDOW,
    MODEL_FILE,
    RANDOM_STATE,
    SHORT_SIGNAL_THRESHOLD,
    SHORT_REGIME_AGGRESSIVE_BUY_MIN_BODY_RATIO,
    SHORT_REGIME_AGGRESSIVE_BUY_MIN_TAKER_RATIO,
    SHORT_REGIME_AGGRESSIVE_BUY_MIN_TREND,
    SHORT_REGIME_MAX_RET_30,
    SHORT_REGIME_MAX_RSI_14,
    SHORT_REGIME_MIN_CLOSE_POSITION,
    SHORT_REGIME_MIN_RET_10,
    SHORT_REGIME_MIN_RET_30,
    SHORT_REGIME_REQUIRE_MACD_HIST_NEGATIVE,
    SHORT_REGIME_SKIP_AGGRESSIVE_BUY_CANDLE,
    SHORT_REGIME_SKIP_WEAK_MIXED_BULLISH_TREND,
    SHORT_REGIME_WEAK_TREND_MAX_RSI_6,
    SHORT_STABILITY_MIN_COUNT,
    SHORT_STABILITY_THRESHOLD,
    SHORT_STABILITY_WINDOW,
    SIGNAL_MIN_INTERVAL_MINUTES,
    SIGNAL_MIN_TREND_AGREEMENT,
    TIME_DECAY_STRENGTH,
    XGB_WEIGHT,
)
from features import make_dual_train_dataset, make_train_dataset
from high_win_rate_filter import passes_high_win_rate_filter


class ProbabilityStabilityFilter:
    """按最近概率序列过滤高置信信号。"""

    def __init__(self):
        max_window = max(LONG_STABILITY_WINDOW, SHORT_STABILITY_WINDOW)
        self.recent_p_up = deque(maxlen=max_window)
        self.steps_since_signal = SIGNAL_MIN_INTERVAL_MINUTES

    def update(self, p_up: float):
        self.recent_p_up.append(float(p_up))
        self.steps_since_signal += 1

    def _count_long(self) -> int:
        recent = list(self.recent_p_up)[-LONG_STABILITY_WINDOW:]
        return sum(p >= LONG_STABILITY_THRESHOLD for p in recent)

    def _count_short(self) -> int:
        recent = list(self.recent_p_up)[-SHORT_STABILITY_WINDOW:]
        return sum(p <= SHORT_STABILITY_THRESHOLD for p in recent)

    def accept(self, direction: str, p_up: float) -> bool:
        if not ENABLE_SIGNAL_STABILITY_FILTER:
            return True

        if direction == "up":
            if len(self.recent_p_up) < LONG_STABILITY_WINDOW:
                return False
            if self.steps_since_signal < SIGNAL_MIN_INTERVAL_MINUTES:
                return False
            return p_up >= LONG_STABILITY_THRESHOLD and self._count_long() >= LONG_STABILITY_MIN_COUNT

        if direction == "down":
            if len(self.recent_p_up) < SHORT_STABILITY_WINDOW:
                return False
            if self.steps_since_signal < SIGNAL_MIN_INTERVAL_MINUTES:
                return False
            return p_up <= SHORT_STABILITY_THRESHOLD and self._count_short() >= SHORT_STABILITY_MIN_COUNT

        return False

    def register_signal(self):
        self.steps_since_signal = 0


def make_time_decay_weights(n: int) -> np.ndarray:
    if n <= 0:
        return np.array([])
    x = np.linspace(0, 1, n)
    weights = np.exp(TIME_DECAY_STRENGTH * x)
    return weights / weights.mean()


def load_tuned_dual_params(path=DUAL_MODEL_PARAMS_FILE) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    return params if isinstance(params, dict) else {}


def _validate_binary_target(y: pd.Series, name: str, min_positive: int = 20):
    if y.nunique() < 2:
        raise ValueError(f"{name} 只有一个类别，无法训练方向子模型。")
    positive_count = int((y == 1).sum())
    if positive_count < min_positive:
        raise ValueError(f"{name} 正样本过少：{positive_count}，至少需要 {min_positive} 条。")


def _ensemble_predict_proba(models: tuple, X: pd.DataFrame) -> np.ndarray:
    lgb_model, xgb_model, cat_model = models
    p_lgb = lgb_model.predict_proba(X)[:, 1]
    p_xgb = xgb_model.predict_proba(X)[:, 1]
    p_cat = cat_model.predict_proba(X)[:, 1]
    probability = LGB_WEIGHT * p_lgb + XGB_WEIGHT * p_xgb + CAT_WEIGHT * p_cat
    return np.clip(probability, 0.0, 1.0)


def _default_params(mode: str) -> dict:
    if mode in {"overfit", "validation"}:
        return {
            "n_estimators": 800,
            "learning_rate": 0.035,
            "max_depth": 8,
            "num_leaves": 63,
            "min_child_samples": 10,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
        }
    return {
        "n_estimators": 220,
        "learning_rate": 0.04,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 60,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.15,
        "reg_lambda": 1.5,
    }


def _resolve_params(mode: str, side: str, override_params: dict | None = None) -> dict:
    params = _default_params(mode)
    tuned_params = override_params if override_params is not None else load_tuned_dual_params()
    side_params = tuned_params.get(side, {}) if isinstance(tuned_params, dict) else {}
    if side_params:
        params.update(side_params)
    return params


class SingleDirectionModel:
    def __init__(self, lgb_model, xgb_model, cat_model, feature_cols):
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.cat_model = cat_model
        self.feature_cols = feature_cols

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_cols]
        p_up = _ensemble_predict_proba((self.lgb_model, self.xgb_model, self.cat_model), X)
        return np.vstack([1 - p_up, p_up]).T

    @staticmethod
    def _feature_value(row: pd.Series, name: str, default: float = 0.0) -> float:
        value = row.get(name, default)
        if pd.isna(value):
            return default
        return float(value)

    def _passes_signal_quality_gate(self, X: pd.DataFrame, direction: str) -> bool:
        if not ENABLE_SIGNAL_QUALITY_GATE:
            return True
        row = X.iloc[0]
        trend_agreement = self._feature_value(row, "trend_agreement")
        ret_5 = self._feature_value(row, "ret_5")
        ret_10 = self._feature_value(row, "ret_10")
        ema_5_20_diff = self._feature_value(row, "ema_5_20_diff")
        macd_hist = self._feature_value(row, "macd_hist")
        if direction == "up":
            return (
                trend_agreement >= SIGNAL_MIN_TREND_AGREEMENT
                and ret_5 > 0
                and ret_10 > 0
                and ema_5_20_diff > 0
                and macd_hist > 0
            )
        if direction == "down":
            return (
                trend_agreement <= -SIGNAL_MIN_TREND_AGREEMENT
                and ret_5 < 0
                and ret_10 < 0
                and ema_5_20_diff < 0
                and macd_hist < 0
            )
        return False

    def _passes_long_regime_filter(self, X: pd.DataFrame) -> bool:
        if not ENABLE_LONG_REGIME_FILTER:
            return True
        row = X.iloc[0]
        if LONG_REGIME_REQUIRE_EMA_20_60_POSITIVE and self._feature_value(row, "ema_20_60_diff") <= 0:
            return False
        if LONG_REGIME_REQUIRE_MACD_HIST_POSITIVE and self._feature_value(row, "macd_hist") <= 0:
            return False
        if LONG_REGIME_MAX_RET_5 is not None and self._feature_value(row, "ret_5") >= LONG_REGIME_MAX_RET_5:
            return False
        if LONG_REGIME_MIN_CLOSE_POSITION is not None and self._feature_value(row, "close_position") <= LONG_REGIME_MIN_CLOSE_POSITION:
            return False
        if LONG_REGIME_MIN_RET_30 is not None and self._feature_value(row, "ret_30") <= LONG_REGIME_MIN_RET_30:
            return False
        if LONG_REGIME_MAX_RSI_14 is not None and self._feature_value(row, "rsi_14") >= LONG_REGIME_MAX_RSI_14:
            return False
        if LONG_REGIME_SKIP_FULL_HIGH_BODY:
            body_ratio = self._feature_value(row, "body_ratio")
            close_position = self._feature_value(row, "close_position")
            if body_ratio > LONG_REGIME_FULL_HIGH_MIN_BODY_RATIO and close_position > LONG_REGIME_FULL_HIGH_MIN_CLOSE_POSITION:
                return False
        return True

    def _passes_short_regime_filter(self, X: pd.DataFrame) -> bool:
        if not ENABLE_SHORT_REGIME_FILTER:
            return True
        row = X.iloc[0]
        if SHORT_REGIME_MIN_CLOSE_POSITION is not None and self._feature_value(row, "close_position") <= SHORT_REGIME_MIN_CLOSE_POSITION:
            return False
        if SHORT_REGIME_REQUIRE_MACD_HIST_NEGATIVE and self._feature_value(row, "macd_hist") >= 0:
            return False
        if SHORT_REGIME_MIN_RET_30 is not None and self._feature_value(row, "ret_30") <= SHORT_REGIME_MIN_RET_30:
            return False
        if SHORT_REGIME_MAX_RET_30 is not None and self._feature_value(row, "ret_30") >= SHORT_REGIME_MAX_RET_30:
            return False
        if SHORT_REGIME_MIN_RET_10 is not None and self._feature_value(row, "ret_10") <= SHORT_REGIME_MIN_RET_10:
            return False
        if SHORT_REGIME_MAX_RSI_14 is not None and self._feature_value(row, "rsi_14") >= SHORT_REGIME_MAX_RSI_14:
            return False
        if SHORT_REGIME_SKIP_AGGRESSIVE_BUY_CANDLE:
            taker_buy_ratio = self._feature_value(row, "taker_buy_ratio")
            body_ratio = self._feature_value(row, "body_ratio")
            trend_agreement = self._feature_value(row, "trend_agreement")
            if (
                taker_buy_ratio >= SHORT_REGIME_AGGRESSIVE_BUY_MIN_TAKER_RATIO
                and body_ratio >= SHORT_REGIME_AGGRESSIVE_BUY_MIN_BODY_RATIO
                and trend_agreement > SHORT_REGIME_AGGRESSIVE_BUY_MIN_TREND
            ):
                return False
        if SHORT_REGIME_SKIP_WEAK_MIXED_BULLISH_TREND:
            trend_agreement = self._feature_value(row, "trend_agreement")
            rsi_6 = self._feature_value(row, "rsi_6")
            if 0 < trend_agreement < 1 and rsi_6 < SHORT_REGIME_WEAK_TREND_MAX_RSI_6:
                return False
        return True


class DualDirectionModel(SingleDirectionModel):
    def __init__(self, up_models: tuple, down_models: tuple, feature_cols):
        self.up_models = up_models
        self.down_models = down_models
        self.feature_cols = feature_cols

    def predict_direction_scores(self, X: pd.DataFrame) -> tuple[float, float, float, float]:
        X = X[self.feature_cols]
        p_up_signal = float(_ensemble_predict_proba(self.up_models, X)[0])
        p_down_signal = float(_ensemble_predict_proba(self.down_models, X)[0])
        direction_edge = p_up_signal - p_down_signal
        score_sum = p_up_signal + p_down_signal
        p_up_relative = 0.5 if score_sum <= 1e-12 else p_up_signal / score_sum
        return p_up_signal, p_down_signal, p_up_relative, direction_edge

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        _, _, p_up_relative, _ = self.predict_direction_scores(X)
        return np.array([[1 - p_up_relative, p_up_relative]])

    def predict_one(self, X: pd.DataFrame, signal_filter: ProbabilityStabilityFilter | None = None):
        p_up_signal, p_down_signal, p_up_relative, direction_edge = self.predict_direction_scores(X)

        if signal_filter is not None:
            signal_filter.update(p_up_relative)

        high_win_rate_signal = False

        if p_up_signal >= LONG_SIGNAL_THRESHOLD and direction_edge >= DUAL_DIRECTION_MIN_EDGE:
            signal = "up"
            confidence = p_up_signal
            is_valid_signal = ENABLE_LONG_SIGNALS and self._passes_signal_quality_gate(X, signal)
        elif p_down_signal >= 1.0 - SHORT_SIGNAL_THRESHOLD and direction_edge <= -DUAL_DIRECTION_MIN_EDGE:
            signal = "down"
            confidence = p_down_signal
            is_valid_signal = ENABLE_SHORT_SIGNALS and self._passes_signal_quality_gate(X, signal)
        else:
            signal = "no_trade"
            confidence = max(p_up_signal, p_down_signal)
            is_valid_signal = False

        if signal in {"up", "down"} and is_valid_signal and signal_filter is not None:
            is_valid_signal = signal_filter.accept(signal, p_up_relative)

        if signal == "up" and is_valid_signal:
            is_valid_signal = self._passes_long_regime_filter(X)
        if signal == "down" and is_valid_signal:
            is_valid_signal = self._passes_short_regime_filter(X)

        if signal in {"up", "down"} and is_valid_signal and ENABLE_HIGH_WIN_RATE_FILTER:
            high_win_rate_signal = passes_high_win_rate_filter(
                X=X,
                direction=signal,
                up_signal_probability=p_up_signal,
                down_signal_probability=p_down_signal,
                direction_edge=direction_edge,
            )
            is_valid_signal = high_win_rate_signal
        elif signal in {"up", "down"} and is_valid_signal:
            high_win_rate_signal = True

        if signal in {"up", "down"} and is_valid_signal and signal_filter is not None:
            signal_filter.register_signal()

        if signal in {"up", "down"} and not is_valid_signal:
            signal = "no_trade"

        return {
            "predicted_direction": signal,
            "up_probability": p_up_relative,
            "up_signal_probability": p_up_signal,
            "down_signal_probability": p_down_signal,
            "direction_edge": direction_edge,
            "confidence": confidence,
            "high_win_rate_signal": bool(high_win_rate_signal),
            "is_valid_signal": bool(is_valid_signal),
        }


def _make_lgb_model(mode: str, params: dict | None = None):
    import lightgbm as lgb
    p = _default_params(mode) if params is None else params
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(p["n_estimators"]),
        learning_rate=float(p["learning_rate"]),
        max_depth=int(p["max_depth"]),
        num_leaves=int(p["num_leaves"]),
        min_child_samples=int(p["min_child_samples"]),
        subsample=float(p["subsample"]),
        colsample_bytree=float(p["colsample_bytree"]),
        reg_alpha=float(p["reg_alpha"]),
        reg_lambda=float(p["reg_lambda"]),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def _make_xgb_model(mode: str, params: dict | None = None):
    import xgboost as xgb
    p = _default_params(mode) if params is None else params
    return xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=int(p["n_estimators"]),
        learning_rate=float(p["learning_rate"]),
        max_depth=int(p["max_depth"]),
        min_child_weight=max(1, int(p["min_child_samples"] // 4)),
        subsample=float(p["subsample"]),
        colsample_bytree=float(p["colsample_bytree"]),
        reg_alpha=float(p["reg_alpha"]),
        reg_lambda=float(p["reg_lambda"]),
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


def _make_cat_model(mode: str, params: dict | None = None):
    from catboost import CatBoostClassifier
    p = _default_params(mode) if params is None else params
    return CatBoostClassifier(
        iterations=int(p["n_estimators"]),
        learning_rate=float(p["learning_rate"]),
        depth=max(3, min(10, int(p["max_depth"]))),
        l2_leaf_reg=max(1e-6, float(p["reg_lambda"])),
        loss_function="Logloss",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def _fit_ensemble(X: pd.DataFrame, y: pd.Series, mode: str, params: dict | None = None):
    weights = make_time_decay_weights(len(X))
    lgb_model = _make_lgb_model(mode, params=params)
    xgb_model = _make_xgb_model(mode, params=params)
    cat_model = _make_cat_model(mode, params=params)
    lgb_model.fit(X, y, sample_weight=weights)
    xgb_model.fit(X, y, sample_weight=weights)
    cat_model.fit(X, y, sample_weight=weights)
    return lgb_model, xgb_model, cat_model


def train_single_direction_model(df: pd.DataFrame, mode: str = "validation") -> SingleDirectionModel:
    try:
        import lightgbm  # noqa: F401
        import xgboost  # noqa: F401
        import catboost  # noqa: F401
    except ImportError as exc:
        raise ImportError("缺少模型依赖，请先执行：pip install -r requirements.txt") from exc

    X, y, data, feature_cols = make_train_dataset(df)
    min_samples = 250 if mode == "validation" else 200
    if len(X) < min_samples:
        raise ValueError(f"训练样本过少：{len(X)}，至少需要 {min_samples} 条有效样本。")
    _validate_binary_target(y, "label")
    params = _resolve_params(mode, "up")
    lgb_model, xgb_model, cat_model = _fit_ensemble(X, y, mode, params=params)
    return SingleDirectionModel(lgb_model, xgb_model, cat_model, feature_cols)


def train_dual_direction_model(
    df: pd.DataFrame,
    mode: str = "validation",
    params_by_side: dict | None = None,
) -> DualDirectionModel:
    try:
        import lightgbm  # noqa: F401
        import xgboost  # noqa: F401
        import catboost  # noqa: F401
    except ImportError as exc:
        raise ImportError("缺少模型依赖，请先执行：pip install -r requirements.txt") from exc

    X, y_up, y_down, data, feature_cols = make_dual_train_dataset(df)
    min_samples = 250 if mode == "validation" else 200
    if len(X) < min_samples:
        raise ValueError(f"训练样本过少：{len(X)}，至少需要 {min_samples} 条有效样本。")
    _validate_binary_target(y_up, "up_label")
    _validate_binary_target(y_down, "down_label")
    up_params = _resolve_params(mode, "up", override_params=params_by_side)
    down_params = _resolve_params(mode, "down", override_params=params_by_side)
    up_models = _fit_ensemble(X, y_up, mode, params=up_params)
    down_models = _fit_ensemble(X, y_down, mode, params=down_params)
    return DualDirectionModel(up_models, down_models, feature_cols)


def train_validation_model(df: pd.DataFrame) -> DualDirectionModel:
    return train_dual_direction_model(df, mode="validation")


def train_overfit_model(df: pd.DataFrame) -> DualDirectionModel:
    return train_dual_direction_model(df, mode="overfit")


def save_model(model: SingleDirectionModel, path=MODEL_FILE):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path=MODEL_FILE):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def train_and_save(df: pd.DataFrame, path=MODEL_FILE, mode: str = "overfit"):
    model = train_dual_direction_model(df, mode=mode)
    save_model(model, path=path)
    return model
